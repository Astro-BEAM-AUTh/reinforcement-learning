"""Evaluate the best model saved by EvalCallback."""
import sys
sys.path.insert(0, 'src')

from sb3_contrib import RecurrentPPO
from sun_tracking.rl.PPO_gaussian import evaluate_policy_multi_episodes

# Load the best model
print("Loading best model from models/best_model/best_model.zip...")
model = RecurrentPPO.load("models/best_model/best_model.zip")

# Evaluate with 200 episodes
print("\n" + "="*60)
print("EVALUATING BEST MODEL")
print("="*60)
avg_return, avg_tail_power, success_rate = evaluate_policy_multi_episodes(
    model, 
    n_episodes=200, 
    noise_std=0.0, 
    seed=999
)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Average Return: {avg_return:.2f}")
print(f"Average Tail Power (last 100 steps): {avg_tail_power:.4f} ({avg_tail_power*100:.2f}%)")
print(f"Success Rate (>95% power): {success_rate*100:.1f}%")
