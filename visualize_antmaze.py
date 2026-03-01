import gym
import d4rl
import torch
import numpy as np
import imageio
from cleandiffuser.diffusion import DiscreteDiffusionSDE
# Note: You may need to adjust the import path based on exactly where CleanDiffuser saves its model checkpoints.

def render_video():
    print("Loading AntMaze Environment...")
    env = gym.make('antmaze-umaze-v0')
    
    # Standard evaluation loop setup
    state = env.reset()
    frames = []
    
    print("Simulating rollout...")
    for step in range(200):  # Run for 200 physical steps
        # Render the current frame from the MuJoCo physics engine
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        # Random action fallback (Replace this block with your loaded Diffuser model later)
        # action = loaded_agent.sample(state) 
        action = env.action_space.sample() 
        
        state, reward, done, info = env.step(action)
        if done:
            break

    print("Saving video to antmaze_rollout.mp4...")
    imageio.mimsave('antmaze_rollout.mp4', frames, fps=30)
    print("Done! Check your folder for the video.")

if __name__ == "__main__":
    render_video()