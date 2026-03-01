import os
import d4rl
import gym
import hydra
import numpy as np
import torch
import imageio

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed


@hydra.main(config_path="../configs/diffuser/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLAntmazeDataset(
        env.get_dataset(), horizon=args.task.horizon, discount=args.discount,
        noreaching_penalty=args.noreaching_penalty,)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier)
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    classifier = CumRewClassifier(nn_classifier, device=args.device)

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    print("\n--- INFERENCE & RENDERING MODE ---")
    print("Loading 500k-step brains into the agent...")
    
    # Load the specific 500k checkpoints you generated
    agent.load("results/diffuser_d4rl_antmaze/antmaze-medium-play-v2/diffusion_ckpt_latest.pt")
    agent.classifier.load("results/diffuser_d4rl_antmaze/antmaze-medium-play-v2/classifier_ckpt_latest.pt")
    agent.eval()

    # Setup a SINGLE visual environment (not vectorized)
    print("Resetting Maze...")
    env_eval = gym.make(args.task.env_name)
    normalizer = dataset.get_normalizer()
    
    # Initialize variables
    frames = []
    obs = env_eval.reset()
    prior = torch.zeros((1, args.task.horizon, obs_dim + act_dim), device=args.device)
    
    print("Simulating 300 steps of Diffuser rollout...")
    for t in range(300):
        # Capture the MuJoCo screen
        frame = env_eval.unwrapped.sim.render(width=500, height=500, mode='offscreen')
        frame = frame[::-1, :, :]  # mujoco_py renders upside down natively, so we flip it
        frames.append(frame)
        if frame is not None:
            frames.append(frame)

        # Normalize observation exactly like the authors do
        norm_obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
        prior[:, 0, :obs_dim] = norm_obs

        # Agent imagines multiple trajectory paths
        traj, log = agent.sample(
            prior.repeat(args.num_candidates, 1, 1),
            solver=args.solver,
            n_samples=args.num_candidates,
            sample_steps=args.sampling_steps,
            use_ema=args.use_ema, 
            w_cg=args.task.w_cg, 
            temperature=args.temperature
        )

        # The Classifier (Value Function) selects the best plan
        logp = log["log_p"].sum(-1)
        idx = logp.argmax(0)
        
        # Extract the first physical action from the winning plan
        act = traj.view(args.num_candidates, 1, args.task.horizon, -1)[idx, 0, 0, obs_dim:]
        act = act.clip(-1., 1.).cpu().numpy()

        # Step the environment forward physically
        obs, rew, done, info = env_eval.step(act)
        
        if done:
            print(f"Goal reached or episode ended at step {t}!")
            break

    if len(frames) > 0:
        imageio.mimsave('antmaze_500k_rollout.gif', frames, fps=30)
        print("Done")
    else:
        print("Error: No frames were captured.")


if __name__ == "__main__":
    pipeline()
