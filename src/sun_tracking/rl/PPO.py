import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sun_tracking.envs.dish_env import SunDishEnv

# Constants
SUCCESS_POWER_THRESHOLD = 0.95  # Power threshold for considering sun tracking successful

# ---------- ENV FACTORIES ----------

def make_train_env(seed: int = 0) -> Monitor:
    """Environment used for training (with sensor noise)."""
    env = SunDishEnv(
        dt=0.5,
        v_max_deg_s=10.0,
        tau=0.3,
        noise_std=0,   # keep noise during training for robustness
        horizon_s=90.0,   # 90 s time limit
        seed=seed,
    )
    return Monitor(env)


def make_eval_env(seed: int = 0, noise_std: int = 0.0) -> Monitor:
    """Environment used for evaluation (optionally noise-free)."""
    env = SunDishEnv(
        dt=0.5,
        v_max_deg_s=10.0,
        tau=0.3,
        noise_std=noise_std,  # usually 0.0 for clean eval
        horizon_s=90.0,
        seed=seed,
    )
    return Monitor(env)


# ---------- EVALUATION FUNCTION ----------

def evaluate_policy_multi_episodes(
    model: PPO,
    n_episodes: int = 50,
    noise_std: float = 0.0,
    seed: int = 123,
    tail_k: int = 100,
) -> tuple[float, float, float]:
    """
    Evaluate a policy over multiple episodes.

    - model: trained PPO model
    - n_episodes: how many episodes to average over
    - noise_std: sensor noise during eval
    - seed: seed for env (controls initial conditions)
    - tail_k: how many last steps to average for "tail power"
    """
    eval_env = make_eval_env(seed=seed, noise_std=noise_std)

    returns = []
    last_powers = []      # single final power (for debugging)
    tail_powers = []      # average over last tail_k rewards

    for _ in range(n_episodes):
        obs, info = eval_env.reset()
        total_r = 0.0
        rewards_hist = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = eval_env.step(action)
            r = float(r)
            total_r += r
            rewards_hist.append(r)

            if terminated or truncated:
                # single final power (if env still stores it in info)
                last_powers.append(float(info.get("power", 0.0)))

                # average reward over last tail_k steps
                k = min(tail_k, len(rewards_hist))
                tail_avg = sum(rewards_hist[-k:]) / k
                tail_powers.append(tail_avg)

                returns.append(total_r)
                break

    avg_return = sum(returns) / len(returns)
    avg_last_power = sum(last_powers) / len(last_powers)
    avg_tail_power = sum(tail_powers) / len(tail_powers)

    # success based on *tail average* instead of one noisy sample
    success_rate = sum(1 for p in tail_powers if p > SUCCESS_POWER_THRESHOLD) / len(tail_powers)

    print(f"Eval over {n_episodes} episodes (noise_std={noise_std}):")
    print(f"  Avg total reward            : {avg_return:.2f}")
    print(f"  Avg last power (single step): {avg_last_power:.4f}")
    print(f"  Avg tail power (last {tail_k} steps): {avg_tail_power:.4f}")
    print(f"  Success (tail_avg > {SUCCESS_POWER_THRESHOLD})   : {success_rate * 100:.1f}%")

    return avg_return, avg_tail_power, success_rate



# ---------- MAIN ----------

def main() -> None:
    # makes the gym env look like a vector of size 1 so SB3 can use it
    env = DummyVecEnv([lambda: make_train_env(seed=42)])

    # policy is a neural network with 2 layers and 128 neurons each
    policy_kwargs = {"net_arch": [128, 128]}

    # this is the model we train
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log="runs/sun_dish_tb",
        seed=42,
    )

    total_steps = 1000000
    model.learn(total_timesteps=total_steps)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_sun_dish_candidate_best")

    # --- NEW: multi-episode evaluation ---
    evaluate_policy_multi_episodes(model, n_episodes=200, noise_std=0.01, seed=999)
if __name__ == "__main__":
    main()
