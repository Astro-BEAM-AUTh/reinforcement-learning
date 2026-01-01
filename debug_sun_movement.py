"""Debug script to check if sun moves between reset and first step."""
import numpy as np
from src.sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

env = GaussianBeamDishEnv(dt=0.5, seed=42, v_max_deg_s=10.0, tau=0.3)

# Reset environment
obs, info = env.reset(seed=42)
print(f"After reset:")
print(f"  sim.t = {env.sim.t:.2f}s")
sun1 = env.sim.sun_vec()
print(f"  sun_vec = [{sun1[0]:.6f}, {sun1[1]:.6f}, {sun1[2]:.6f}]")
print(f"  Initial power = {env.P:.6f}")
print(f"  Dish angles = az:{np.degrees(env.az):.4f}°, el:{np.degrees(env.el):.4f}°")

# Take one step with zero action (should do nothing)
print(f"\nTaking step with action=[0, 0]...")
obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

print(f"\nAfter first step:")
print(f"  sim.t = {env.sim.t:.2f}s")
sun2 = env.sim.sun_vec()
print(f"  sun_vec = [{sun2[0]:.6f}, {sun2[1]:.6f}, {sun2[2]:.6f}]")
print(f"  Power = {info['power']:.6f}")
print(f"  Dish angles = az:{np.degrees(env.az):.4f}°, el:{np.degrees(env.el):.4f}°")
print(f"  Angular distance = {info['angular_distance']:.6f}°")
print(f"  Reward = {reward:.2f}")

# Calculate sun movement
sun_movement = np.linalg.norm(np.array(sun2) - np.array(sun1))
print(f"\n🔍 Sun vector changed by: {sun_movement:.6f}")
print(f"🔍 Time advanced by: {env.sim.dt}s")
print(f"🔍 This means sun moved during the step!")
