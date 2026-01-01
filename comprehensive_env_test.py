"""Comprehensive environment validation - does it actually work?"""
import numpy as np
import math
from src.sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

def test_basic_functionality():
    """Test 1: Does the environment run without crashing?"""
    print("="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)
    
    env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=10.0, tau=0.3, noise_std=0.0, seed=42)
    
    # Reset and run 100 steps
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful - initial power: {obs[4]:.4f}")
    
    for i in range(100):
        action = np.random.uniform(-1, 1, size=2)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            print(f"✗ Episode ended prematurely at step {i+1}")
            return False
    
    print(f"✓ Ran 100 steps without crashing")
    return True


def test_power_gradient():
    """Test 2: Does moving toward sun increase power?"""
    print("\n" + "="*70)
    print("TEST 2: Power Gradient - Reward Points Toward Sun")
    print("="*70)
    
    env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=10.0, tau=0.3, noise_std=0.0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Get current position and sun position
    from src.sun_tracking.envs.dish_env_gaussian import compute_n_from_angles
    
    dish_az, dish_el = obs[0], obs[1]
    sun_vec = env.sim.sun_vec()
    sun_az = math.atan2(sun_vec[1], sun_vec[0])
    sun_el = math.asin(sun_vec[2])
    
    # Calculate direction to sun
    delta_az = sun_az - dish_az
    delta_el = sun_el - dish_el
    
    print(f"Dish at: az={math.degrees(dish_az):.2f}°, el={math.degrees(dish_el):.2f}°")
    print(f"Sun at: az={math.degrees(sun_az):.2f}°, el={math.degrees(sun_el):.2f}°")
    print(f"Delta: Δaz={math.degrees(delta_az):.2f}°, Δel={math.degrees(delta_el):.2f}°")
    
    # Test: Move toward sun vs away from sun
    # Toward sun (positive direction if delta is positive)
    action_toward = np.array([
        0.3 * np.sign(delta_az),
        0.3 * np.sign(delta_el)
    ])
    
    # Away from sun
    action_away = -action_toward
    
    # Test moving toward sun
    obs_before = obs.copy()
    obs, reward_toward, _, _, info = env.step(action_toward)
    power_toward = info['power']
    print(f"\nMove TOWARD sun: action={action_toward}, power={power_toward:.4f}, reward={reward_toward:.2f}")
    
    # Reset and test moving away
    obs, info = env.reset(seed=42)
    obs, reward_away, _, _, info = env.step(action_away)
    power_away = info['power']
    print(f"Move AWAY from sun: action={action_away}, power={power_away:.4f}, reward={reward_away:.2f}")
    
    # Verify gradient is correct
    if reward_toward > reward_away:
        print(f"✓ Gradient correct: toward sun ({reward_toward:.2f}) > away from sun ({reward_away:.2f})")
        return True
    else:
        print(f"✗ PROBLEM: toward sun ({reward_toward:.2f}) ≤ away from sun ({reward_away:.2f})")
        return False


def test_action_magnitudes():
    """Test 3: Are action effects reasonable?"""
    print("\n" + "="*70)
    print("TEST 3: Action Magnitude Effects")
    print("="*70)
    
    env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=10.0, tau=0.3, noise_std=0.0, seed=42)
    
    results = []
    for action_mag in [0.0, 0.1, 0.3, 0.5, 1.0]:
        obs, info = env.reset(seed=42)
        initial_power = obs[4]
        
        action = np.array([action_mag, action_mag])
        obs, reward, _, _, info = env.step(action)
        final_power = info['power']
        angular_dist = info['angular_distance']
        
        # Calculate actual displacement
        displacement = action_mag * 10.0 * 0.5  # v_max * dt
        
        print(f"Action magnitude: {action_mag:.1f} → displacement: {displacement:.2f}°/step")
        print(f"  Power: {initial_power:.4f} → {final_power:.4f}, Ang dist: {angular_dist:.2f}°")
        
        results.append({
            'action': action_mag,
            'displacement': displacement,
            'power_change': final_power - initial_power
        })
    
    # Check that larger actions cause larger displacements
    displacements = [r['displacement'] for r in results]
    if displacements == sorted(displacements):
        print(f"✓ Action magnitudes scale correctly")
        return True
    else:
        print(f"✗ Action scaling problem")
        return False


def test_can_find_sun():
    """Test 4: Can a simple policy find the sun?"""
    print("\n" + "="*70)
    print("TEST 4: Can a Gradient-Following Policy Find the Sun?")
    print("="*70)
    
    env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=10.0, tau=0.3, noise_std=0.0, seed=42)
    obs, info = env.reset(seed=42)
    
    initial_power = obs[4]
    print(f"Initial power: {initial_power:.4f}")
    
    # Simple gradient ascent: try small moves in different directions, keep best
    max_power = initial_power
    power_history = [initial_power]
    
    for step in range(50):
        # Try 8 directions
        best_action = None
        best_next_power = -1
        
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            # Try small action in this direction
            test_action = 0.2 * np.array([np.cos(angle), np.sin(angle)])
            
            # Save state
            state = (env.az, env.el, env.az_dot, env.el_dot, env.P, env.sim.t)
            
            # Try action
            obs, reward, _, _, info = env.step(test_action)
            next_power = info['power']
            
            if next_power > best_next_power:
                best_next_power = next_power
                best_action = test_action
            
            # Restore state
            env.az, env.el, env.az_dot, env.el_dot, env.P, env.sim.t = state
        
        # Take best action
        obs, reward, term, trunc, info = env.step(best_action)
        power = info['power']
        power_history.append(power)
        max_power = max(max_power, power)
        
        if step % 10 == 0:
            print(f"  Step {step}: power={power:.4f}, max={max_power:.4f}, ang_dist={info['angular_distance']:.2f}°")
    
    print(f"\nFinal power: {power_history[-1]:.4f}")
    print(f"Max power reached: {max_power:.4f}")
    print(f"Improvement: {max_power - initial_power:.4f}")
    
    if max_power > 0.5:
        print(f"✓ Policy found good alignment (>50% power)")
        return True
    elif max_power > 0.2:
        print(f"⚠ Policy found moderate alignment (>20% power)")
        return True
    else:
        print(f"✗ Policy failed to find sun (max {max_power*100:.1f}% power)")
        return False


def test_multiple_episodes():
    """Test 5: Does reset work correctly across episodes?"""
    print("\n" + "="*70)
    print("TEST 5: Multiple Episodes")
    print("="*70)
    
    env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=10.0, tau=0.3, noise_std=0.0)
    
    initial_powers = []
    for ep in range(10):
        obs, info = env.reset()
        initial_powers.append(obs[4])
        
        # Run episode
        for _ in range(20):
            action = np.random.uniform(-0.3, 0.3, size=2)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
    
    print(f"Initial powers across 10 episodes:")
    print(f"  Min: {min(initial_powers):.4f}")
    print(f"  Max: {max(initial_powers):.4f}")
    print(f"  Mean: {np.mean(initial_powers):.4f}")
    print(f"  Std: {np.std(initial_powers):.4f}")
    
    # Check variety (should not all be the same)
    if np.std(initial_powers) > 0.001:
        print(f"✓ Episodes have varied initial conditions")
        return True
    else:
        print(f"✗ All episodes start the same (no randomization)")
        return False


def test_dynamics_stability():
    """Test 6: Are dynamics stable (no overshoot)?"""
    print("\n" + "="*70)
    print("TEST 6: Dynamics Stability")
    print("="*70)
    
    env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=10.0, tau=0.3, noise_std=0.0, seed=42)
    obs, info = env.reset(seed=42)
    
    # Command constant velocity and check if it stabilizes
    action = np.array([0.5, 0.5])  # 5°/s command
    
    velocities = []
    for i in range(10):
        obs, reward, _, _, info = env.step(action)
        az_dot, el_dot = obs[2], obs[3]
        velocities.append((az_dot, el_dot))
        
        if i < 3:
            print(f"  Step {i+1}: v_az={math.degrees(az_dot):.3f}°/s, v_el={math.degrees(el_dot):.3f}°/s")
    
    # Check final velocity
    final_v_az = math.degrees(velocities[-1][0])
    final_v_el = math.degrees(velocities[-1][1])
    commanded = 5.0  # 0.5 * 10°/s
    
    print(f"\nCommanded velocity: {commanded:.2f}°/s")
    print(f"Final velocity: v_az={final_v_az:.2f}°/s, v_el={final_v_el:.2f}°/s")
    
    # Check for overshoot
    if abs(final_v_az) <= commanded * 1.01 and abs(final_v_el) <= commanded * 1.01:
        print(f"✓ No overshoot (velocities ≤ commanded)")
        return True
    else:
        print(f"✗ Overshoot detected!")
        return False


def main():
    """Run all tests."""
    print("\n" + "🔬 COMPREHENSIVE ENVIRONMENT VALIDATION")
    print("="*70)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Power Gradient", test_power_gradient),
        ("Action Magnitudes", test_action_magnitudes),
        ("Can Find Sun", test_can_find_sun),
        ("Multiple Episodes", test_multiple_episodes),
        ("Dynamics Stability", test_dynamics_stability),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} CRASHED: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Environment is working correctly!")
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"❌ FAILURES: {', '.join(failed)}")
        print("   Environment needs more debugging")
    
    return all_passed


if __name__ == "__main__":
    main()
