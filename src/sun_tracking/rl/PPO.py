import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sun_tracking.envs.dish_env import SunDishEnv


def make_env(seed=0):
    env = SunDishEnv(
        dt=0.5,
        v_max_deg_s=10.0,
        tau=0.3,
        noise_std=0.01,
        horizon_s=90.0,  # 90 s time limit
        seed=seed,
    )
    return Monitor(env) 

def main():

    # makes the gym env look like a vector of size 1 so SB3 can use it
    env = DummyVecEnv([lambda: make_env(seed=42)])

    # policy is a neural network with 2 layers and 128 neurons each
    policy_kwargs = dict(net_arch=[128, 128])


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

    total_steps = 63000
    model.learn(total_timesteps=total_steps)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_sun_dish")

    # quick deterministic eval rollout
    eval_env = make_env(seed=123)
    obs, info = eval_env.reset()
    total_r = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = eval_env.step(action)
        total_r += float(r)
        if terminated or truncated:
            break
    print("Eval → last power:", round(info.get("power", 0.0), 4), "| total reward:", round(total_r, 2))

if __name__ == "__main__":
    main()

