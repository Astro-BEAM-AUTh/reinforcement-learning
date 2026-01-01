"""Debug script to manually test the Gaussian environment."""
import numpy as np
from src.sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

# Create environment
env = GaussianBeamDishEnv(
    dt=0.5,
    v_max_deg_s=10.0,
    tau=0.3,
    noise_std=0.0,  # No noise for clean debugging
    horizon_s=90.0,
    seed=42
)

print("="*60)
print("DEBUGGING GAUSSIAN BEAM ENVIRONMENT")
print("="*60)

# Reset
obs, info = env.reset(seed=42)
print(f"\nInitial observation: {obs}")
print(f"Initial power: {obs[4]:.4f}")
print(f"Initial dP: {obs[5]:.4f}")

# Test 1: Do nothing (should get small negative reward due to sun movement)
print("\n" + "="*60)
print("TEST 1: Do nothing (action = [0, 0])")
print("="*60)
for i in range(5):
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    print(f"Step {i+1}: Power={info['power']:.6f}, Reward={reward:.2f}, dP={info['power_improvement']:.6f}")

# Reset for next test
obs, info = env.reset(seed=42)
print(f"\nReset - Initial power: {obs[4]:.4f}")

# Test 2: Try random actions
print("\n" + "="*60)
print("TEST 2: Random actions")
print("="*60)
for i in range(5):
    action = np.random.uniform(-1, 1, size=2)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: Action={action}, Power={info['power']:.6f}, Reward={reward:.2f}")

# Reset for next test
obs, info = env.reset(seed=42)
print(f"\nReset - Initial power: {obs[4]:.4f}")

# Test 3: Constant positive command (should eventually improve or worsen power)
print("\n" + "="*60)
print("TEST 3: Constant action [0.5, 0.5]")
print("="*60)
for i in range(10):
    obs, reward, terminated, truncated, info = env.step([0.5, 0.5])
    print(f"Step {i+1}: Power={info['power']:.6f}, Reward={reward:.2f}, AngDist={info['angular_distance']:.3f}°")

# Test 4: Check if we ever see positive rewards
print("\n" + "="*60)
print("TEST 4: Look for any positive rewards in random exploration")
print("="*60)
obs, info = env.reset(seed=123)
print(f"Initial power: {obs[4]:.4f}")

positive_count = 0
negative_count = 0
max_reward = -float('inf')
min_reward = float('inf')
max_power = 0.0

for i in range(100):
    action = np.random.uniform(-1, 1, size=2)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if reward > 0:
        positive_count += 1
    else:
        negative_count += 1
    
    max_reward = max(max_reward, reward)
    min_reward = min(min_reward, reward)
    max_power = max(max_power, info['power'])

print(f"\nAfter 100 random steps:")
print(f"  Positive rewards: {positive_count}")
print(f"  Negative rewards: {negative_count}")
print(f"  Max reward seen: {max_reward:.2f}")
print(f"  Min reward seen: {min_reward:.2f}")
print(f"  Max power achieved: {max_power:.4f}")
print(f"  Final power: {info['power']:.4f}")
