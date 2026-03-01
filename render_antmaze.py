import os
import gym
import d4rl
import torch
import imageio
import numpy as np

# We import the exact same agent building block used in your training script
from cleandiffuser.diffusion import DiscreteDiffusionSDE 

def render_video():
    env_name = 'antmaze-medium-play-v2'
    print(f"Loading {env_name} Environment...")
    env = gym.make(env_name)
    
    # ---------------------------------------------------------
    # TODO: Agent Initialization
    # You will need to copy the 'agent = DiscreteDiffusionSDE(...)' 
    # block from pipelines/diffuser_d4rl_antmaze.py and paste it here
    # so the architecture exactly matches your training config.
    # ---------------------------------------------------------
    
    # Example placeholder for loading your 500k checkpoints:
    diffusion_weight_path = "results/diffuser_d4rl_antmaze/antmaze-medium-play-v2/diffusion_ckpt_latest.pt"
    classifier_weight_path = "results/diffuser_d4rl_antmaze/antmaze-medium-play-v2/classifier_ckpt_latest.pt"
    
    print("Loading 500,000-step checkpoints...")
    # agent.model.load_state_dict(torch.load(diffusion_weight_path))
    # agent.classifier.load_state_dict(torch.load(classifier_weight_path))
    
    state = env.reset()
    frames = []
    
    print("Simulating Diffuser rollout...")
    for step in range(300): # 300 steps is a solid test run
        # Capture the screen
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        # 1. The agent "imagines" a full trajectory from the current state
        # 2. It executes only the very first action of that imagined plan
        # action = agent.sample(state) 
        action = env.action_space.sample() # <-- Remove this random action once agent is loaded
        
        state, reward, done, info = env.step(action)
        if done:
            break

    print("Saving video to antmaze_500k_rollout.mp4...")
    imageio.mimsave('antmaze_500k_rollout.mp4', frames, fps=30)
    print("Done! Right-click the mp4 file in VS Code to download/view it.")

if __name__ == "__main__":
    render_video()