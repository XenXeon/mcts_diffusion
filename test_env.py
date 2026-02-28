import sys
import torch
import cleandiffuser

def run_diagnostics():
    print("="*50)
    print("CLEAN-DIFFUSER ENVIRONMENT DIAGNOSTICS")
    print("="*50)

    # 1. Test CleanDiffuser Installation
    print(f"[✓] CleanDiffuser successfully imported from: {cleandiffuser.__path__[0]}")

    # 2. Test PyTorch & GPU Hardware
    print("\n--- GPU & PyTorch Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"Active GPU:      {device_name}")
        
        # Perform a dummy calculation on the GPU
        try:
            device = torch.device("cuda")
            x = torch.randn(1000, 1000).to(device)
            y = x @ x.T
            print(f"[✓] GPU Matrix Multiplication successful! Tensor device: {y.device}")
        except Exception as e:
            print(f"[X] GPU Calculation failed: {e}")
    else:
        print("[X] WARNING: PyTorch cannot see your NVIDIA GPU.")

    # 3. Test Physics Engines
    print("\n--- Physics Engine Check ---")
    try:
        import mujoco_py
        print("[✓] OpenAI mujoco_py wrapper loaded successfully.")
    except Exception as e:
        print(f"[X] mujoco_py failed: {e}")

    try:
        import mujoco
        print("[✓] DeepMind mujoco official bindings loaded successfully.")
    except Exception as e:
        print(f"[X] mujoco failed: {e}")

    try:
        import gym
        print(f"[✓] Gym (v{gym.__version__}) loaded successfully.")
    except Exception as e:
        print(f"[X] Gym failed: {e}")

    print("="*50)
    print("If all checks have a [✓], your sandbox is 100% ready.")
    print("="*50)

if __name__ == "__main__":
    run_diagnostics()