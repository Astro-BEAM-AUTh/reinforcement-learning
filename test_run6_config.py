"""Test if Run 6 configuration still works."""
import numpy as np
from src.sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

# Run 6 config: 3° init
env = GaussianBeamDishEnv(
    dt=0.5,
    v_max_deg_s=10.0,
    tau=0.3,
    noise_std=0.01,
    horizon_s=90.0,
    seed=42
)

print("="*60)
print("TESTING RUN 6 CONFIGURATION (3° initialization)")
print("="*60)

# Temporarily override max_offset to 3.0 to match Run 6
import src.sun_tracking.envs.dish_env_gaussian as env_module

# Reset with 3° offset
obs, info = env.reset(seed=42)

# Check if current implementation uses 1° or 3°
print(f"\nChecking max_offset_deg in current code...")

# Read the reset method to see what max_offset_deg is
import inspect
source = inspect.getsource(env.reset)
if "max_offset_deg = 1.0" in source:
    print("❌ Current code uses 1° offset")
    print("   Run 6 used 3° offset")
    print("   This is likely why recent runs fail!\n")
elif "max_offset_deg = 3.0" in source:
    print("✓ Current code uses 3° offset (matches Run 6)")
else:
    print("⚠ Could not determine max_offset_deg from code")

print(f"\nInitial state after reset:")
print(f"  Power: {obs[4]:.4f}")
print(f"  dP: {obs[5]:.4f}")

# Take a few small random actions
print(f"\nTaking 5 steps with small random actions...")
for i in range(5):
    action = np.random.uniform(-0.1, 0.1, size=2)  # Small actions
    obs, reward, term, trunc, info = env.step(action)
    print(f"  Step {i+1}: action={action}, power={info['power']:.4f}, reward={reward:.2f}")
