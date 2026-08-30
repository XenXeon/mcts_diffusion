import numpy as np
import gym, d4rl
from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset

env = gym.make("maze2d-umaze-v1")
dataset = DV_D4RLMaze2DSeqDataset(
    env.get_dataset(), horizon=32, stride=15,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")
normalizer = dataset.get_normalizer()

# Check A: dataset normalisation state
sample = dataset.seq_obs[0, 0]
print("dataset.seq_obs[0,0]:", sample)
print("  abs-max:", np.abs(sample).max(), "  mean:", sample.mean(), "  std:", sample.std())

# Compare to env observation (which IS raw)
raw_obs = env.reset()
print("env.reset() raw obs:", raw_obs)
normed = normalizer.normalize(raw_obs[None])
print("normalised env obs:", normed)
print("  abs-max:", np.abs(normed).max())

# Sanity: re-normalising an already-normalised sample should look broken
double_normed = normalizer.normalize(sample[None])
print("dataset sample double-normalised (should look weird):", double_normed)

# Check B: trajectory length structure
print("\nseq_obs shape:", dataset.seq_obs.shape)
for attr in ["seq_lengths", "path_lengths", "lengths", "valid_lengths"]:
    if hasattr(dataset, attr):
        print(f"  dataset.{attr}[:5]:", getattr(dataset, attr)[:5])


raw = env.get_dataset()
# D4RL maze2d uses 'timeouts' for episode boundaries (no real terminations)
boundaries = np.where(raw['timeouts'] | raw.get('terminals', np.zeros_like(raw['timeouts'])))[0]
# lengths between consecutive boundaries
lengths = np.diff(np.concatenate([[-1], boundaries])).astype(int)
print("n_trajectories:", len(lengths), "  mean length:", lengths.mean())
print("first 5:", lengths[:5], "  shortest:", lengths.min(), "  longest:", lengths.max())