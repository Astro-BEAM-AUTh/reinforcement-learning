import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sun_tracking.envs.dish_env import SunDishEnv

# ---------- ENV FACTORIES ----------

def make_train_env(seed: int = 0) -> Monitor:
    """Environment used for training (with sensor noise)."""
    env = SunDishEnv(
        dt=0.5,
        v_max_deg_s=10.0,
        tau=0.3,
        noise_std=0.01,   # noise during training for robustness
        horizon_s=90.0,   # 90 s time limit
        seed=seed,
    )
    return Monitor(env)


def make_eval_env(seed: int = 0, noise_std: float = 0.0) -> Monitor:
    """Environment used for evaluation (optionally noise-free)."""
    env = SunDishEnv(
        dt=0.5,
        v_max_deg_s=10.0,
        tau=0.3,
        noise_std=noise_std,  # 0.0 for clean eval, >0 for noisy eval
        horizon_s=90.0,
        seed=seed,
    )
    return Monitor(env)


# ---------- EVALUATION FUNCTION ----------

def evaluate_policy_multi_episodes(
    model,
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
    last_powers = []      # single final power (for info)
    tail_powers = []      # avg over last tail_k rewards

    for ep in range(n_episodes):
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
                # final power from info (single step)
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

    # success based on tail average
    success_rate = sum(1 for p in tail_powers if p > 0.95) / len(tail_powers)

    print(f"Eval over {n_episodes} episodes (noise_std={noise_std}):")
    print(f"  Avg total reward               : {avg_return:.2f}")
    print(f"  Avg last power (single step)   : {avg_last_power:.4f}")
    print(f"  Avg tail power (last {tail_k} steps): {avg_tail_power:.4f}")
    print(f"  Success (tail_avg > 0.95)      : {success_rate * 100:.1f}%")

    return avg_return, avg_tail_power, success_rate


# ---------- MULTI-SEED TRAIN + EVAL + PLOT ----------

def train_and_eval_one_seed(
    seed: int,
    total_steps: int = 1_000_000,
    tail_k: int = 100,
) -> dict:
    """
    Train PPO for one seed, then evaluate (clean + noisy).
    Returns a dict with metrics.
    """
    # --- TRAIN ---
    env = DummyVecEnv([lambda s=seed: make_train_env(seed=s)])
    policy_kwargs = {"net_arch": [128, 128]}

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
        seed=seed,
    )

    model.learn(total_timesteps=total_steps)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/ppo_sun_dish_seed{seed}.zip"
    model.save(model_path)

    # --- EVAL: clean (no noise) ---
    clean_ret, clean_tail_power, clean_succ = evaluate_policy_multi_episodes(
        model,
        n_episodes=200,
        noise_std=0.0,
        seed=1000 + seed,   # vary initial conditions
        tail_k=tail_k,
    )

    # --- EVAL: noisy (realistic sensor) ---
    noisy_ret, noisy_tail_power, noisy_succ = evaluate_policy_multi_episodes(
        model,
        n_episodes=200,
        noise_std=0.01,
        seed=2000 + seed,
        tail_k=tail_k,
    )

    return {
        "seed": seed,
        "model_path": model_path,
        "clean_return": clean_ret,
        "clean_tail_power": clean_tail_power,
        "clean_success": clean_succ,
        "noisy_return": noisy_ret,
        "noisy_tail_power": noisy_tail_power,
        "noisy_success": noisy_succ,
    }


def multi_seed_experiment():
    seeds = [0, 1, 2, 3, 4] # list of seeds to run
    all_results = []

    for seed in seeds:
        print(f"\n================ SEED {seed} ================\n")
        res = train_and_eval_one_seed(
            seed=seed,
            total_steps=1_000_000,
            tail_k=100,
        )
        all_results.append(res)

    # --- Make a simple plot of success rates per seed ---
    seed_vals = [r["seed"] for r in all_results]
    clean_succ = [r["clean_success"] * 100.0 for r in all_results]  # %
    noisy_succ = [r["noisy_success"] * 100.0 for r in all_results]  # %

    x = np.arange(len(seed_vals))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, clean_succ, width, label="Clean (noise_std=0.0)")
    plt.bar(x + width/2, noisy_succ, width, label="Noisy (noise_std=0.01)")

    plt.xticks(x, [str(s) for s in seed_vals])
    plt.ylabel("Success rate (%)")
    plt.xlabel("Seed")
    plt.title("Sun tracker success rate per seed")
    plt.legend()
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plot_path = "models/multi_seed_success_rates.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")

    # Also print a small summary table
    print("\n=== Summary ===")
    for r in all_results:
        print(
            f"Seed {r['seed']}: "
            f"clean_succ={r['clean_success']*100:.1f}% "
            f"noisy_succ={r['noisy_success']*100:.1f}% "
            f"(model={r['model_path']})"
        )


# ---------- ENTRY POINT ----------

if __name__ == "__main__":
    multi_seed_experiment()
