"""RecurrentPPO (LSTM) training for Gaussian beam sun tracking environment."""
import os

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

# Constants
SUCCESS_POWER_THRESHOLD = 0.95  # Power threshold for considering sun tracking successful

# ---------- ENV FACTORIES ----------

def make_train_env(seed: int = 0) -> Monitor:
    """Environment used for training (no noise for curriculum learning)."""
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=2.0,
        tau=0.3,
        noise_std=0.0,   # no noise - learn from clean gradients first
        horizon_s=180.0,  # 180 s time limit (more time to explore)
        seed=seed,
    )
    return Monitor(env)


def make_eval_env(seed: int = 0, noise_std: float = 0.0) -> Monitor:
    """Environment used for evaluation (optionally noise-free)."""
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=2.0,
        tau=0.3,
        noise_std=noise_std,  # usually 0.0 for clean eval
        horizon_s=180.0,  # 180 s time limit (more time to explore)
        seed=seed,
    )
    return Monitor(env)


# ---------- EVALUATION FUNCTION ----------

def evaluate_policy_multi_episodes(
    model: RecurrentPPO,
    n_episodes: int = 50,
    noise_std: float = 0.0,
    seed: int = 123,
    tail_k: int = 100,
) -> tuple[float, float, float]:
    """
    Evaluate a RecurrentPPO policy over multiple episodes.

    - model: trained RecurrentPPO model
    - n_episodes: how many episodes to average over
    - noise_std: sensor noise during eval
    - seed: seed for env (controls initial conditions)
    - tail_k: how many last steps to average for "tail power"
    """
    eval_env = make_eval_env(seed=seed, noise_std=noise_std)

    returns = []
    last_powers = []      # single final power (for debugging)
    tail_powers = []      # average power over last tail_k steps

    for _ in range(n_episodes):
        obs, info = eval_env.reset()
        # FIX 1: Reset LSTM states at episode start
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        total_r = 0.0
        power_hist = []  # Track actual power, not reward

        while True:
            # FIX 1: Pass LSTM states to predict
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            obs, r, terminated, truncated, info = eval_env.step(action)
            
            r = float(r)
            total_r += r
            power_hist.append(float(info.get("power", 0.0)))  # Store actual power
            
            # Update episode_start for next iteration
            episode_start = np.array([terminated or truncated], dtype=bool)

            if terminated or truncated:
                # single final power (if env still stores it in info)
                last_powers.append(float(info.get("power", 0.0)))

                # average POWER (not reward) over last tail_k steps
                k = min(tail_k, len(power_hist))
                tail_avg = sum(power_hist[-k:]) / k
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
    """Train RecurrentPPO (LSTM) on Gaussian beam sun tracking environment."""
    # makes the gym env look like a vector of size 1 so SB3 can use it
    env = DummyVecEnv([lambda: make_train_env(seed=42)])

    # FIX 1: Use LSTM policy with recurrent architecture
    # LSTM allows agent to remember past observations (power trajectory)
    # This is critical since we only observe power, not sun position
    policy_kwargs = {"net_arch": [128, 128]}  # Hidden layers before LSTM

    # this is the model we train
    model = RecurrentPPO(
        "MlpLstmPolicy",  # FIX 1: LSTM policy for memory
        env,
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

    # Run 16: RecurrentPPO with LSTM + absolute power reward + smoother control
    # 3° init (exact ring), v_max=5°/s, 180s episodes, noise=0.0, save every 100K steps
    total_steps = 250_000
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_eval_env(seed=999, noise_std=0.0)])
    
    # Checkpoint callback - saves model every 100K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models/checkpoints/",
        name_prefix="recurrent_ppo_run16",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # EvalCallback - saves best model based on evaluation performance
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/eval/",
        eval_freq=25_000,  # Evaluate every 25K steps
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    model.learn(total_timesteps=total_steps, callback=callback)

    os.makedirs("models", exist_ok=True)
    model.save("models/recurrent_ppo_run16_final")

    # --- Multi-episode evaluation ---
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    evaluate_policy_multi_episodes(model, n_episodes=200, noise_std=0.0, seed=999)


if __name__ == "__main__":
    main()
