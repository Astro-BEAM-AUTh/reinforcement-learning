"""
RecurrentPPO (LSTM) training for Gaussian beam sun tracking environment.

Key fixes:
- Use N parallel training envs (n_envs) and build the model with THAT env.
- Use a separate eval_env for EvalCallback (do NOT use training env for eval).
- Keep your manual multi-episode evaluation function.
"""
import os
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv


# ----------------- CONFIG -----------------

SUCCESS_POWER_THRESHOLD = 0.95  # Tail-power threshold for "success"


# ----------------- ENV FACTORIES -----------------

def make_train_env(seed: int = 0) -> Monitor:
    """Environment used for training (noise-free curriculum)."""
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=2.0,
        tau=0.3,
        noise_std=0.0,
        horizon_s=180.0,
        seed=seed,
    )
    return Monitor(env)


def make_eval_env(seed: int = 0, noise_std: float = 0.0, mode: str = "mixed") -> Monitor:
    """Environment used for evaluation (separate from training).
    
    Args:
        mode: "easy" (0.2-1.0°), "hard" (1.0-3.0°), or "mixed" (70/30 split)
    """
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=2.0,
        tau=0.3,
        noise_std=noise_std,
        horizon_s=180.0,
        seed=seed,
    )
    # Store mode for use in reset
    env._eval_mode = mode
    return Monitor(env)


# ----------------- EVALUATION (MANUAL) -----------------

def evaluate_policy_multi_episodes(
    model: RecurrentPPO,
    n_episodes: int = 50,
    noise_std: float = 0.0,
    seed: int = 123,
    tail_k: int = 100,
    mode: str = "mixed",
) -> tuple[float, float, float]:
    """
    Evaluate a RecurrentPPO policy over multiple episodes.

    Args:
        mode: "easy" (0.2-1.0°), "hard" (1.0-3.0°), or "mixed" (70/30 split)

    Returns:
      (avg_return, avg_tail_power, success_rate)
    """
    eval_env = make_eval_env(seed=seed, noise_std=noise_std, mode=mode)

    returns = []
    last_powers = []
    tail_powers = []
    final_angular_distances = []
    tail_angular_distances = []

    for _ in range(n_episodes):
        obs, info = eval_env.reset()

        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        total_r = 0.0
        power_hist = []
        angular_dist_hist = []

        while True:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            obs, r, terminated, truncated, info = eval_env.step(action)

            r = float(r)
            total_r += r

            power_hist.append(float(info.get("power", 0.0)))
            angular_dist_hist.append(float(info.get("angular_distance", 0.0)))

            # next step: only True on the first step of the *next* episode
            episode_start = np.array([terminated or truncated], dtype=bool)

            if terminated or truncated:
                last_powers.append(float(info.get("power", 0.0)))

                k = min(tail_k, len(power_hist))
                tail_avg = sum(power_hist[-k:]) / k
                tail_powers.append(tail_avg)

                final_angular_distances.append(angular_dist_hist[-1] if angular_dist_hist else 0.0)
                k_ang = min(tail_k, len(angular_dist_hist))
                tail_ang_avg = sum(angular_dist_hist[-k_ang:]) / k_ang if k_ang > 0 else 0.0
                tail_angular_distances.append(tail_ang_avg)

                returns.append(total_r)
                break

    avg_return = sum(returns) / len(returns)
    avg_last_power = sum(last_powers) / len(last_powers)
    avg_tail_power = sum(tail_powers) / len(tail_powers)
    avg_final_ang = sum(final_angular_distances) / len(final_angular_distances)
    avg_tail_ang = sum(tail_angular_distances) / len(tail_angular_distances)

    success_rate = sum(1 for p in tail_powers if p > SUCCESS_POWER_THRESHOLD) / len(tail_powers)
    success_rate_05 = sum(1 for p in tail_powers if p > 0.5) / len(tail_powers)
    success_rate_08 = sum(1 for p in tail_powers if p > 0.8) / len(tail_powers)

    print(f"Eval over {n_episodes} episodes (noise_std={noise_std}):")
    print(f"  Avg total reward            : {avg_return:.2f}")
    print(f"  Avg last power (single step): {avg_last_power:.4f}")
    print(f"  Avg tail power (last {tail_k} steps): {avg_tail_power:.4f}")
    print(f"  Avg final angular distance  : {avg_final_ang:.4f} deg")
    print(f"  Avg tail angular distance (last {tail_k} steps): {avg_tail_ang:.4f} deg")
    print(f"  Success (tail_avg > 0.5)    : {success_rate_05 * 100:.1f}%")
    print(f"  Success (tail_avg > 0.8)    : {success_rate_08 * 100:.1f}%")
    print(f"  Success (tail_avg > {SUCCESS_POWER_THRESHOLD})   : {success_rate * 100:.1f}%")

    return avg_return, avg_tail_power, success_rate


# ----------------- MAIN TRAINING -----------------

def main() -> None:
    # --- Training settings ---
    n_envs = 8
    total_steps = 2_000_000  # 2M steps with 100% hard curriculum [1°, 3°]

    # 1) Build TRAIN envs FIRST (vectorized)
    train_env = DummyVecEnv([
        (lambda i=i: make_train_env(seed=42 + i))
        for i in range(n_envs)
    ])

    # 2) Build EVAL env separately (single env)
    # Explicitly evaluate on hard mode [1.0°, 3.0°] to match training difficulty
    eval_env = DummyVecEnv([lambda: make_eval_env(seed=999, noise_std=0.0, mode="hard")])

    # 3) Create model with the TRAIN env
    policy_kwargs = {"net_arch": [128, 128]}

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        verbose=1,
        tensorboard_log="runs/gaussian_beam_tb",
        seed=42,
    )

    # 4) Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models/checkpoints/",
        name_prefix="recurrent_ppo_run16",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/eval/",
        eval_freq=100_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # 5) Train
    model.learn(total_timesteps=total_steps, callback=callback)

    # 6) Save final
    os.makedirs("models", exist_ok=True)
    model.save("models/recurrent_ppo_run16_final")

    # 7) Manual evaluation (optional)
    print("\n" + "=" * 60)
    print("EVALUATION - EASY MODE (offset [0.2°, 1.0°])")
    print("=" * 60)
    eval_easy_env = make_eval_env(seed=999, noise_std=0.0)
    evaluate_policy_multi_episodes(model, n_episodes=200, noise_std=0.0, seed=999, mode="easy")
    
    print("\n" + "=" * 60)
    print("EVALUATION - HARD MODE (offset [1.0°, 3.0°])")
    print("=" * 60)
    eval_hard_env = make_eval_env(seed=888, noise_std=0.0)
    evaluate_policy_multi_episodes(model, n_episodes=200, noise_std=0.0, seed=888, mode="hard")


if __name__ == "__main__":
    main()
