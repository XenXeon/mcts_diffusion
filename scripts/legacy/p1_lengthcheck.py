import numpy as np
import gym, d4rl, os
from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset

env = gym.make("maze2d-umaze-v1")
dataset = DV_D4RLMaze2DSeqDataset(
    env.get_dataset(), horizon=32, stride=15,
    learn_policy=False, center_mapping=False,
    discount=1.0, continous_reward_at_done=True, reward_tune="iql")

# Extract the padded tensor
obs = dataset.seq_obs

# 1. Calculate the absolute difference between consecutive timesteps
# If the dataset pads by repeating the last state, this difference will be 0.
step_diffs = np.abs(obs[:, 1:] - obs[:, :-1]).sum(axis=-1)

# 2. Flag transitions where the state actually changed 
# (using 1e-5 to account for floating point inaccuracies)
is_active = step_diffs > 1e-5

# 3. Find the last active step for every trajectory
true_lengths = np.zeros(obs.shape[0], dtype=int)

for i in range(obs.shape[0]):
    active_indices = np.where(is_active[i])[0]
    if len(active_indices) > 0:
        # The index of the last change, +2 to convert to length 
        # (+1 because 0-indexed, +1 because a transition involves 2 states)
        true_lengths[i] = active_indices[-1] + 2 
    else:
        # Failsafe if a trajectory is entirely static
        true_lengths[i] = 1

print("seq_obs shape:", obs.shape)
print("Empirical lengths array shape:", true_lengths.shape)
print("Mean true physical length:", true_lengths.mean())
print("First 5 physical lengths:", true_lengths[:5])
print("Shortest:", true_lengths.min(), " Longest:", true_lengths.max())

# Save this for your Phase 1 verification script
os.makedirs("results/phase1", exist_ok=True)

# Save this for your Phase 1 verification script
np.save("results/phase1/path_lengths.npy", true_lengths)
print("Saved true lengths to disk.")

# 1. How many trajectories hit the 800 cap? Are they actually different at the end?
at_cap = (true_lengths == 800)
print(f"trajectories at length 800: {at_cap.sum()} / {len(true_lengths)}")

# 2. Spot-check one short trajectory: does it actually look padded after its detected length?
i = np.argmin(true_lengths)
L = true_lengths[i]
print(f"trajectory {i}, detected length {L}")
print(f"state at L-1: {obs[i, L-1]}")
print(f"state at L:   {obs[i, L]}")
print(f"state at L+5: {obs[i, L+5]}")
print(f"state at end: {obs[i, -1]}")

# 3. Spot-check a typical mid-length trajectory
i = 0  # length 221 per your earlier output
L = true_lengths[i]
print(f"\ntrajectory 0, detected length {L}")
print(f"state at L-1: {obs[i, L-1]}")
print(f"state at L:   {obs[i, L]}")
print(f"state at L+5: {obs[i, L+5]}")

heldout_frac = 0.10
rng = np.random.default_rng(0)
n_traj = len(true_lengths)
ids = rng.permutation(n_traj)
n_heldout = int(n_traj * heldout_frac)
heldout_ids = ids[-n_heldout:]
heldout_lens = true_lengths[heldout_ids]

usable = heldout_lens >= 480
print(f"held-out trajectories: {n_heldout}")
print(f"  usable (length >= 480): {usable.sum()}")
print(f"  total valid offset slots: {(heldout_lens[usable] - 480).sum()}")