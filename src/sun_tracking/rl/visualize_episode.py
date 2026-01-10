"""Quick test to visualize power trajectory over one episode."""
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv


def visualize_episode(model_path: str):
    """Run one episode and plot power over time."""
    
    # Load model
    try:
        model = RecurrentPPO.load(model_path)
        print(f"Loaded model from: {model_path}")
    except:
        print(f"Could not load model from {model_path}")
        print("Creating random model for comparison...")
        env_tmp = GaussianBeamDishEnv()
        model = RecurrentPPO("MlpLstmPolicy", env_tmp, verbose=0)
        env_tmp.close()
    
    # Create environment
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,
        noise_std=0.0,
        horizon_s=180.0,
        seed=42
    )
    
    # Run one episode
    obs, info = env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    
    powers = [info["power"]]
    rewards = []
    actions_az = []
    actions_el = []
    angular_distances = []
    
    print(f"\nInitial power: {info['power']:.4f}")
    
    for step in range(360):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True
        )
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_start = np.zeros((1,), dtype=bool)
        
        powers.append(info["power"])
        rewards.append(reward)
        actions_az.append(action[0])
        actions_el.append(action[1])
        angular_distances.append(info["angular_distance"])
        
        if step < 10 or step % 50 == 0:
            print(f"Step {step+1}: P={info['power']:.4f}, dist={info['angular_distance']:.2f}°, action=[{action[0]:.2f}, {action[1]:.2f}]")
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Print summary
    print(f"\nEpisode summary:")
    print(f"  Total steps: {len(powers)}")
    print(f"  Power range: [{min(powers):.4f}, {max(powers):.4f}]")
    print(f"  Mean power: {np.mean(powers):.4f}")
    print(f"  Tail power (last 100): {np.mean(powers[-100:]):.4f}")
    print(f"  Final angular distance: {angular_distances[-1]:.2f}°")
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    time = np.arange(len(powers)) * 0.5  # dt=0.5
    
    # Plot 1: Power over time
    axes[0].plot(time, powers, 'b-', linewidth=2)
    axes[0].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
    axes[0].set_ylabel('Power')
    axes[0].set_title('Power Over Time')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot 2: Angular distance
    axes[1].plot(time[1:], angular_distances, 'r-', linewidth=2)
    axes[1].axhline(y=3.0, color='k', linestyle='--', label='Initial 3°')
    axes[1].set_ylabel('Distance (deg)')
    axes[1].set_title('Angular Distance from Sun')
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot 3: Actions
    axes[2].plot(time[1:], actions_az, 'g-', label='Azimuth', alpha=0.7)
    axes[2].plot(time[1:], actions_el, 'orange', label='Elevation', alpha=0.7)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('Action')
    axes[2].set_title('Control Actions')
    axes[2].grid(True)
    axes[2].legend()
    
    # Plot 4: Reward
    axes[3].plot(time[1:], rewards, 'm-', linewidth=2)
    axes[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylabel('Reward')
    axes[3].set_title('Reward Over Time')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('episode_visualization.png', dpi=150)
    print(f"\nPlot saved to: episode_visualization.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/recurrent_ppo_run16_final"
        print(f"No model path provided, using: {model_path}")
    
    visualize_episode(model_path)
