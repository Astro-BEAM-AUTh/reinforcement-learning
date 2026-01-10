"""Evaluate SAC Run 1 checkpoint at 600K steps."""
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv


def make_eval_env(seed: int = 0) -> Monitor:
    """Evaluation environment."""
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=1.0,
        tau=0.3,
        noise_std=0.0,
        horizon_s=180.0,  # Run 1 used 180s episodes
        seed=seed,
    )
    return Monitor(env)


def evaluate(checkpoint_path: str, n_episodes: int = 200):
    """Evaluate checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = SAC.load(checkpoint_path)
    
    eval_env = make_eval_env(seed=999)
    
    total_rewards = []
    last_powers = []
    tail_powers = []
    
    print(f"\nEvaluating over {n_episodes} episodes...")
    for ep in range(n_episodes):
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
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
        
        # Tail power: average of last 100 steps
        tail = powers[-100:] if len(powers) >= 100 else powers
        tail_powers.append(sum(tail) / len(tail) if tail else 0.0)
    
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_last_power = sum(last_powers) / len(last_powers)
    avg_tail_power = sum(tail_powers) / len(tail_powers)
    
    success_count = sum(1 for tp in tail_powers if tp > 0.95)
    success_rate = success_count / len(tail_powers)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS - SAC Run 1 (600K steps)")
    print("="*60)
    print(f"Avg total reward            : {avg_total_reward:.2f}")
    print(f"Avg last power (single step): {avg_last_power:.4f}")
    print(f"Avg tail power (last 100 steps): {avg_tail_power:.4f}")
    print(f"Success (tail_avg > 0.95)   : {success_rate:.1%}")
    print("="*60)
    
    eval_env.close()


if __name__ == "__main__":
    evaluate("models/checkpoints/sac_gaussian_run1_600000_steps.zip")
