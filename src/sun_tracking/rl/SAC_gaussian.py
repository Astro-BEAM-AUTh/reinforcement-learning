"""SAC training for Gaussian beam sun tracking environment."""
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

# Constants
SUCCESS_POWER_THRESHOLD = 0.95  # Power threshold for considering sun tracking successful

# ---------- ENV FACTORIES ----------

def make_train_env(seed: int = 0) -> Monitor:
    """Environment used for training (no noise for curriculum learning)."""
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=1.0,  # Slow speed to prevent overshooting at 3° offset
        tau=0.3,
        noise_std=0.0,   # no noise - learn from clean gradients first
        horizon_s=90.0,  # 90 s time limit (faster training)
        seed=seed,
    )
    return Monitor(env)


def make_eval_env(seed: int = 0, noise_std: float = 0.0) -> Monitor:
    """Environment used for evaluation (optionally noise-free)."""
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=1.0,  # Slow speed to prevent overshooting at 3° offset
        tau=0.3,
        noise_std=noise_std,  # usually 0.0 for clean eval
        horizon_s=90.0,  # 90 s time limit (faster training)
        seed=seed,
    )
    return Monitor(env)


# ---------- EVALUATION FUNCTION ----------

def evaluate_policy_multi_episodes(
    model: SAC,
    n_episodes: int = 50,
    noise_std: float = 0.0,
    seed: int = 123,
    tail_k: int = 100,
) -> tuple[float, float, float]:
    """
    Evaluate a SAC policy over multiple episodes.
    
    Returns:
        (avg_total_reward, avg_last_power, avg_tail_power)
    """
    eval_env = make_eval_env(seed=seed, noise_std=noise_std)
    
    total_rewards = []
    last_powers = []
    tail_powers = []
    
    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        powers = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            powers.append(info.get("power", 0.0))
        
        total_rewards.append(episode_reward)
        last_powers.append(powers[-1] if powers else 0.0)
        
        # Tail power: average of last k steps
        tail = powers[-tail_k:] if len(powers) >= tail_k else powers
        tail_powers.append(sum(tail) / len(tail) if tail else 0.0)
    
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_last_power = sum(last_powers) / len(last_powers)
    avg_tail_power = sum(tail_powers) / len(tail_powers)
    
    success_count = sum(1 for tp in tail_powers if tp > SUCCESS_POWER_THRESHOLD)
    success_rate = success_count / len(tail_powers)
    
    print(f"Eval over {n_episodes} episodes (noise_std={noise_std}):")
    print(f"  Avg total reward            : {avg_total_reward:.2f}")
    print(f"  Avg last power (single step): {avg_last_power:.4f}")
    print(f"  Avg tail power (last {tail_k} steps): {avg_tail_power:.4f}")
    print(f"  Success (tail_avg > {SUCCESS_POWER_THRESHOLD})   : {success_rate:.1%}")
    
    eval_env.close()
    return avg_total_reward, avg_last_power, avg_tail_power


# ---------- MAIN TRAINING ----------

def main():
    """
    Run 2 (SAC): 90s episodes for faster training
    - 3° exact initialization (fixed from PPO runs)
    - v_max = 1.0 deg/s (prevents overshoot)
    - 90s episodes (2× faster than Run 1)
    - No noise (clean gradients)
    - SAC with replay buffer and entropy regularization
    """
    
    # Create training environment
    train_env = make_train_env(seed=42)
    
    # SAC hyperparameters
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,  # Replay buffer for off-policy learning
        learning_starts=1000,  # Collect data before training
        batch_size=256,
        tau=0.005,  # Soft update coefficient for target networks
        gamma=0.99,  # Discount factor
        train_freq=1,  # Train after every step
        gradient_steps=1,  # One gradient update per step
        ent_coef='auto',  # Automatic entropy tuning
        policy_kwargs=dict(net_arch=[256, 256]),  # Larger network than PPO
        verbose=1,
        seed=42,
    )
    
    # Training configuration
    total_steps = 100_000
    
    # Checkpoint callback (save at end only)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models/checkpoints/",
        name_prefix="sac_gaussian_run2",
        save_replay_buffer=True,  # SAC can save replay buffer
        save_vecnormalize=False,
    )
    
    model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

    os.makedirs("models", exist_ok=True)
    model.save("models/sac_gaussian_run2_final")

    # --- Multi-episode evaluation ---
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    evaluate_policy_multi_episodes(model, n_episodes=200, noise_std=0.0, seed=999)


if __name__ == "__main__":
    main()
