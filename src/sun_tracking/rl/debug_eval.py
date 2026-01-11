"""Debug script to diagnose 0% power issue with RecurrentPPO evaluation."""
import numpy as np
from sb3_contrib import RecurrentPPO

from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv


# ========== FIX 1: CORRECT RECURRENT EVALUATION ==========

def evaluate_recurrent_policy_correct(
    model: RecurrentPPO,
    env: GaussianBeamDishEnv,
    n_episodes: int = 10,
    verbose: bool = True
) -> dict:
    """
    Correct evaluation for RecurrentPPO with LSTM.
    
    CRITICAL: Must reset LSTM state at start of each episode!
    """
    episode_rewards = []
    episode_tail_powers = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        # CRITICAL: Reset LSTM state for new episode
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)  # Mark first step
        
        episode_reward = 0.0
        power_history = []
        step = 0
        
        while True:
            # CORRECT: Pass LSTM state and episode_start flag
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # After first step, episode_start is always False
            episode_starts = np.zeros((1,), dtype=bool)
            
            episode_reward += reward
            power_history.append(info["power"])
            step += 1
            
            if verbose and ep == 0 and step <= 5:
                print(f"  Step {step}: action={action}, P={info['power']:.4f}, reward={reward:.4f}")
            
            if terminated or truncated:
                break
        
        # Compute tail power (last 100 steps)
        tail_power = np.mean(power_history[-100:]) if len(power_history) >= 100 else np.mean(power_history)
        
        episode_rewards.append(episode_reward)
        episode_tail_powers.append(tail_power)
        episode_lengths.append(step)
        
        if verbose:
            print(f"Episode {ep+1}: len={step}, reward={episode_reward:.2f}, tail_power={tail_power:.4f}")
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "mean_tail_power": np.mean(episode_tail_powers),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean([p > 0.95 for p in episode_tail_powers]),
    }
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Mean Episode Reward: {results['mean_reward']:.2f}")
    print(f"Mean Tail Power: {results['mean_tail_power']:.4f}")
    print(f"Mean Episode Length: {results['mean_length']:.0f}")
    print(f"Success Rate (>95%): {results['success_rate']*100:.1f}%")
    
    return results


# ========== FIX 2: VERIFY REWARD = P ==========

def test_reward_calculation():
    """Test that reward actually equals P (minus small penalty)."""
    print("\n=== FIX 2: TESTING REWARD CALCULATION ===")
    
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,
        noise_std=0.0,
        horizon_s=10.0,
        seed=42
    )
    
    obs, info = env.reset()
    print(f"Initial P: {info['power']:.4f}")
    
    # Take 5 random actions
    for step in range(5):
        action = np.array([0.1, 0.1])  # Small action
        obs, reward, terminated, truncated, info = env.step(action)
        
        power = info["power"]
        expected_reward = power - 0.01 * (0.1**2 + 0.1**2)  # P - action_penalty
        
        print(f"Step {step+1}:")
        print(f"  Power: {power:.6f}")
        print(f"  Reward: {reward:.6f}")
        print(f"  Expected: {expected_reward:.6f}")
        print(f"  Match: {abs(reward - expected_reward) < 1e-6}")
    
    env.close()


# ========== FIX 3: VERIFY INFO["POWER"] IS CORRECT ==========

def test_power_logging():
    """Verify that info['power'] is returned on every step."""
    print("\n=== FIX 3: TESTING POWER LOGGING ===")
    
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,
        noise_std=0.0,
        horizon_s=20.0,
        seed=42
    )
    
    obs, info = env.reset()
    print(f"After reset - info keys: {list(info.keys())}")
    print(f"  info['power']: {info.get('power', 'MISSING')}")
    
    powers = []
    for step in range(10):
        action = np.random.uniform(-0.5, 0.5, size=2)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1} - info keys: {list(info.keys())}")
        power = info.get("power", None)
        print(f"  info['power']: {power}")
        
        if power is None:
            print(f"  ERROR: Power is None at step {step+1}!")
        else:
            powers.append(power)
    
    print(f"\nPower statistics over {len(powers)} steps:")
    print(f"  Min: {min(powers):.4f}")
    print(f"  Max: {max(powers):.4f}")
    print(f"  Mean: {np.mean(powers):.4f}")
    print(f"  All zero?: {all(p == 0 for p in powers)}")
    
    env.close()


# ========== FIX 4: BASELINE SANITY CHECKS ==========

def test_do_nothing_policy():
    """Test A: Do-nothing policy should achieve nonzero power."""
    print("\n=== FIX 4A: DO-NOTHING POLICY TEST ===")
    
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,
        noise_std=0.0,
        horizon_s=90.0,
        seed=42
    )
    
    obs, info = env.reset()
    print(f"Initial power: {info['power']:.4f}")
    
    powers = []
    action = np.array([0.0, 0.0])  # Do nothing
    
    for step in range(180):  # 90 seconds at 0.5s dt
        obs, reward, terminated, truncated, info = env.step(action)
        powers.append(info["power"])
        
        if terminated or truncated:
            break
    
    tail_power = np.mean(powers[-100:])
    
    print(f"Total steps: {len(powers)}")
    print(f"Power range: [{min(powers):.4f}, {max(powers):.4f}]")
    print(f"Mean power: {np.mean(powers):.4f}")
    print(f"Tail power (last 100 steps): {tail_power:.4f}")
    
    if tail_power == 0.0:
        print("ERROR: Do-nothing policy gives 0 power - environment is broken!")
    else:
        print("OK: Do-nothing achieves nonzero power")
    
    env.close()


def test_random_policy():
    """Test B: Random small actions should fluctuate power."""
    print("\n=== FIX 4B: RANDOM POLICY TEST ===")
    
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,
        noise_std=0.0,
        horizon_s=90.0,
        seed=42
    )
    
    obs, info = env.reset()
    print(f"Initial power: {info['power']:.4f}")
    
    powers = []
    np.random.seed(99)
    
    for step in range(180):
        action = np.random.uniform(-0.1, 0.1, size=2)  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        powers.append(info["power"])
        
        if terminated or truncated:
            break
    
    tail_power = np.mean(powers[-100:])
    power_std = np.std(powers)
    
    print(f"Total steps: {len(powers)}")
    print(f"Power range: [{min(powers):.4f}, {max(powers):.4f}]")
    print(f"Mean power: {np.mean(powers):.4f}")
    print(f"Std dev: {power_std:.4f}")
    print(f"Tail power: {tail_power:.4f}")
    
    if power_std == 0.0:
        print("ERROR: Power never changes - environment is frozen!")
    else:
        print("OK: Power fluctuates with random actions")
    
    env.close()


# ========== MAIN DIAGNOSTIC SCRIPT ==========

def main():
    """Run all diagnostic tests."""
    print("="*70)
    print("DIAGNOSTIC SCRIPT FOR 0% POWER ISSUE")
    print("="*70)
    
    # Test 1: Verify reward calculation
    test_reward_calculation()
    
    # Test 2: Verify power logging
    test_power_logging()
    
    # Test 3: Baseline sanity checks
    test_do_nothing_policy()
    test_random_policy()
    
    print("\n" + "="*70)
    print("Now testing with actual RecurrentPPO model...")
    print("="*70)
    
    # Load the model
    try:
        model = RecurrentPPO.load("models/recurrent_ppo_run16_final")
        print("Model loaded successfully")
    except FileNotFoundError:
        print("ERROR: Model not found. Using random policy instead.")
        # Create dummy model for testing evaluation logic
        env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=5.0, seed=42)
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0)
        env.close()
    
    # Create eval environment
    eval_env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,
        noise_std=0.0,
        horizon_s=180.0,
        seed=999
    )
    
    # Run correct evaluation
    results = evaluate_recurrent_policy_correct(model, eval_env, n_episodes=10, verbose=True)
    
    eval_env.close()
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    print("\nIf tail_power is still 0.0000, the likely causes are:")
    print("1. Model checkpoint is corrupted or not properly saved")
    print("2. Model was trained with wrong observation space")
    print("3. LSTM states are not being properly maintained")
    print("4. Environment dynamics have changed since training")


if __name__ == "__main__":
    main()
