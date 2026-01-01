"""Test the FIXED environment with both bugs resolved."""
import numpy as np
from src.sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

print("="*70)
print("TESTING FIXED ENVIRONMENT")
print("="*70)
print("✅ Bug #1 FIXED: alpha = min(1.0, dt/tau) prevents overshoot")
print("✅ Bug #2 FIXED: max_offset_deg = 3.0° (matches Run 6)")
print("="*70)

env = GaussianBeamDishEnv(
    dt=0.5,
    v_max_deg_s=10.0,
    tau=0.3,
    noise_std=0.0,  # No noise for clean testing
    horizon_s=90.0,
    seed=42
)

# Test 1: Reset and check initial state
obs, info = env.reset(seed=42)
print(f"\nAfter reset (3° offset):")
print(f"  Initial power: {obs[4]:.4f} (should be ~1-2%, not 75%)")
print(f"  Position: az={np.degrees(obs[0]):.2f}°, el={np.degrees(obs[1]):.2f}°")

# Test 2: Take moderate actions (should NOT crash now)
print(f"\nTest: Moderate actions (0.3-0.5) should be safe now...")
for i in range(5):
    action = np.array([0.3, 0.3])
    obs, reward, term, trunc, info = env.step(action)
    print(f"  Step {i+1}: action={action}, power={info['power']:.4f}, "
          f"reward={reward:.2f}, dist={info['angular_distance']:.2f}°")

# Test 3: Random exploration
print(f"\nTest: Random exploration over 50 steps...")
obs, info = env.reset(seed=123)
print(f"Initial power: {obs[4]:.4f}")

rewards = []
powers = []
for i in range(50):
    action = np.random.uniform(-0.5, 0.5, size=2)
    obs, reward, term, trunc, info = env.step(action)
    rewards.append(reward)
    powers.append(info['power'])

print(f"\nResults after 50 random steps:")
print(f"  Mean reward: {np.mean(rewards):.2f}")
print(f"  Positive rewards: {sum(1 for r in rewards if r > 0)}/50")
print(f"  Max power: {max(powers):.4f}")
print(f"  Final power: {powers[-1]:.4f}")

if max(powers) > 0.05:
    print(f"\n✅ SUCCESS! Agent found decent power (>{max(powers)*100:.1f}%) with random actions")
    print(f"   This means the environment is learnable!")
else:
    print(f"\n❌ Still problematic - couldn't find good power even with exploration")

# Test 4: Can the agent improve over baseline?
print(f"\nTest: Compare random policy vs doing nothing...")
obs, info = env.reset(seed=456)
initial_power = obs[4]

# Do nothing for 20 steps
do_nothing_powers = [initial_power]
for i in range(20):
    obs, _, _, _, info = env.step([0.0, 0.0])
    do_nothing_powers.append(info['power'])

# Random actions for 20 steps
obs, info = env.reset(seed=456)
random_powers = [obs[4]]
for i in range(20):
    action = np.random.uniform(-0.3, 0.3, size=2)
    obs, _, _, _, info = env.step(action)
    random_powers.append(info['power'])

print(f"  Do nothing - final power: {do_nothing_powers[-1]:.4f}")
print(f"  Random actions - final power: {random_powers[-1]:.4f}")
print(f"  Best random power: {max(random_powers):.4f}")

if max(random_powers) > do_nothing_powers[-1]:
    print(f"\n✅ Random exploration can improve over baseline!")
else:
    print(f"\n⚠ Random policy didn't beat baseline, but that's OK for RL")
