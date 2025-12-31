"""Quick test script for Gaussian beam environment."""
import numpy as np

from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv, compute_angular_offset, compute_gaussian_power


def test_power_calculation():
    """Test that power calculation makes sense."""
    print("Testing power calculation...")
    
    # Test perfect alignment
    p_perfect = compute_gaussian_power(0.0, 0.0, noise_std=0.0)
    print(f"  Perfect alignment (d=0): P = {p_perfect:.4f} (should be ~1.0)")
    
    # Test small offset
    p_small = compute_gaussian_power(0.3, 0.0, noise_std=0.0)
    print(f"  Small offset (d=0.3°): P = {p_small:.4f} (should be ~0.9)")
    
    # Test moderate offset
    p_moderate = compute_gaussian_power(1.2, 0.0, noise_std=0.0)
    print(f"  Moderate offset (d=1.2°): P = {p_moderate:.4f} (should be ~0.5)")
    
    # Test large offset
    p_large = compute_gaussian_power(3.0, 0.0, noise_std=0.0)
    print(f"  Large offset (d=3.0°): P = {p_large:.4f} (should be <0.1)")
    
    print()


def test_environment():
    """Test that the environment can be created and stepped."""
    print("Testing Gaussian beam environment...")
    
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=10.0,
        tau=0.3,
        noise_std=0.0,  # No noise for testing
        horizon_s=10.0,  # Short episode
        seed=42,
    )
    
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\n  Initial observation: {obs}")
    print(f"  Initial power: {obs[4]:.4f}")
    
    # Check sun position
    import math
    from sun_tracking.envs.dish_env_gaussian import compute_n_from_angles, compute_angular_offset
    
    sun_vec = env.sim.sun_vec()
    dish_vec = compute_n_from_angles(env.az, env.el)
    
    print(f"\n  Sun position vector: ({sun_vec[0]:.3f}, {sun_vec[1]:.3f}, {sun_vec[2]:.3f})")
    print(f"  Sun above horizon: {sun_vec[2] > 0}")
    print(f"  Dish pointing vector: ({dish_vec[0]:.3f}, {dish_vec[1]:.3f}, {dish_vec[2]:.3f})")
    
    d_x, d_y = compute_angular_offset(dish_vec, sun_vec)
    offset_deg = math.sqrt(d_x**2 + d_y**2)
    print(f"  Angular offset (d_x, d_y): ({d_x:.2f}°, {d_y:.2f}°)")
    print(f"  Total offset: {offset_deg:.2f}° (random start - dish needs to find sun!)")
    
    # Now manually point dish at sun to verify power calculation works
    print(f"\n  Manually pointing dish at sun to verify power calculation...")
    
    # Calculate sun azimuth and elevation from vector
    sun_az = math.atan2(sun_vec[1], sun_vec[0])
    sun_el = math.asin(sun_vec[2])
    
    env.az = sun_az
    env.el = sun_el
    env.az_dot = 0.0
    env.el_dot = 0.0
    
    # Recalculate power
    dish_vec = compute_n_from_angles(env.az, env.el)
    d_x, d_y = compute_angular_offset(dish_vec, sun_vec)
    offset_deg = math.sqrt(d_x**2 + d_y**2)
    
    from sun_tracking.envs.dish_env_gaussian import compute_gaussian_power
    power = compute_gaussian_power(d_x, d_y, noise_std=0.0)
    
    print(f"  After pointing at sun:")
    print(f"    Angular offset: {offset_deg:.4f}° (should be ~0)")
    print(f"    Power: {power:.4f} (should be ~1.0)")
    
    # Take a few steps with dish pointed at sun
    print("\n  Taking 5 steps (starting near sun)...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate offset after step
        dish_vec = compute_n_from_angles(env.az, env.el)
        sun_vec = env.sim.sun_vec()
        d_x, d_y = compute_angular_offset(dish_vec, sun_vec)
        offset_deg = math.sqrt(d_x**2 + d_y**2)
        
        print(f"    Step {i+1}: offset={offset_deg:.2f}°, power={obs[4]:.4f}, reward={reward:.4f}")
        
        if terminated or truncated:
            break
    
    print("\n  Environment test passed!")
    print()


def test_angular_offset():
    """Test angular offset calculation."""
    print("Testing angular offset calculation...")
    
    # Two identical vectors (perfect alignment)
    v1 = (1.0, 0.0, 0.0)
    v2 = (1.0, 0.0, 0.0)
    d_x, d_y = compute_angular_offset(v1, v2)
    print(f"  Same vector: offset = ({d_x:.6f}, {d_y:.6f}) degrees (should be ~0)")
    
    # Small angle difference
    import math
    v1 = (math.cos(0.01), math.sin(0.01), 0.0)  # 0.01 rad ≈ 0.57 deg
    v2 = (1.0, 0.0, 0.0)
    d_x, d_y = compute_angular_offset(v1, v2)
    offset_mag = math.sqrt(d_x**2 + d_y**2)
    print(f"  Small angle: offset magnitude = {offset_mag:.4f} degrees (should be ~0.57)")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Gaussian Beam Environment Tests")
    print("=" * 60)
    print()
    
    test_power_calculation()
    test_angular_offset()
    test_environment()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
