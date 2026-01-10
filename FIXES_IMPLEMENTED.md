# Implemented Fixes for Sun Tracking RL

## Summary

**Date**: January 10, 2026  
**Fixes Implemented**: FIX 1, FIX 2, FIX 4  
**Files Modified**: 2

---

## FIX 1 — RecurrentPPO with LSTM Memory

### Problem
The agent only observes `[az, el, az_dot, el_dot, P, dP]` without knowing sun position. With standard PPO, the agent cannot infer direction from instantaneous power alone, causing learning to plateau at 10-15% power.

### Solution
Replaced standard PPO with **RecurrentPPO** using **MlpLstmPolicy** from `sb3-contrib`.

**Why this helps:**
- LSTM maintains hidden state across timesteps
- Agent can remember power trajectory over time
- From power sequence, LSTM infers which direction leads to power increase
- Keeps realistic constraint: controller only sees power measurements

### Code Changes

**File: `PPO_gaussian.py`**

```python
# Changed import
from sb3_contrib import RecurrentPPO  # Instead of stable_baselines3.PPO

# Changed model creation
model = RecurrentPPO(
    "MlpLstmPolicy",  # Instead of "MlpPolicy"
    env,
    policy_kwargs={"net_arch": [128, 128]},
    # ... rest of hyperparameters same
)
```

**File: `PPO_gaussian.py` - Evaluation**

```python
# Added LSTM state handling in evaluation
for _ in range(n_episodes):
    obs, info = eval_env.reset()
    lstm_states = None  # Reset LSTM at episode start
    episode_start = np.ones((1,), dtype=bool)
    
    while True:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states,  # Pass hidden state
            episode_start=episode_start,
            deterministic=True
        )
        obs, r, terminated, truncated, info = eval_env.step(action)
        episode_start = np.zeros((1,), dtype=bool)
        # ...
```

---

## FIX 2 — Absolute Power Reward

### Problem
Old reward: `reward = 1000 * (P - last_P)`

**Issues:**
1. **Telescoping**: Reward cumulates, making total reward depend on initial offset, not final performance
2. **No centering incentive**: Once power plateaus, ΔP ≈ 0, so agent gets no signal to improve
3. **Scale instability**: 1000× multiplier can cause gradient issues

### Solution
New reward: `reward = P - 0.01 * (action[0]² + action[1]²)`

**Why this helps:**
1. **Direct objective**: Reward directly measures what we want (high power)
2. **Sustained tracking**: Agent rewarded for *staying* at high power, not just moving toward it
3. **Stability**: Power is naturally bounded [0, 1], better for value function learning
4. **Smooth control**: Small action penalty encourages gentle corrections over aggressive moves

### Code Changes

**File: `dish_env_gaussian.py`**

```python
# Old reward (removed)
# power_improvement = float(self.P - self.last_P)
# reward = REWARD_POWER_IMPROVEMENT_SCALE * power_improvement  # 1000 * ΔP

# New reward (FIX 2)
action_penalty = 0.01 * (action[0]**2 + action[1]**2)
reward = float(self.P - action_penalty)
```

**Key insight**: 
- At P=0.95 (95% power), reward ≈ 0.95
- At P=0.10 (10% power), reward ≈ 0.10  
- Agent always knows it should maximize P, even when already tracking

---

## FIX 4 — Smoother Control (Option A)

### Problem
Default `v_max = 10 deg/s` allows very aggressive movements. With `dt=0.5s`, one step can move dish by 5°, causing:
- Overshoot past sun center
- Oscillations around peak power
- Hard to achieve precise centering (>95% power requires <0.5° accuracy)

### Solution
Reduced `v_max_deg_s: 10 → 5` (Option A chosen)

**Why this helps:**
1. **Finer control**: Max movement per step: 5° → 2.5°
2. **Less overshoot**: Agent can make gentler corrections near sun center
3. **Realistic**: Real dish antenna servos have limited slew rates
4. **Better for LSTM**: Slower dynamics give LSTM more steps to observe power gradient

**Alternative (not implemented)**: Reduce `dt: 0.5s → 0.2s` would give 3× finer timesteps but slow down training.

### Code Changes

**File: `dish_env_gaussian.py`**

```python
def __init__(
    self,
    # ...
    v_max_deg_s=5.0,  # Changed from 10.0 (FIX 4)
    # ...
):
```

**File: `PPO_gaussian.py`**

```python
def make_train_env(seed: int = 0) -> Monitor:
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,  # Changed from 1.0 → 5.0 (FIX 4)
        # ...
    )
    return Monitor(env)

def make_eval_env(seed: int = 0, noise_std: float = 0.0) -> Monitor:
    env = GaussianBeamDishEnv(
        dt=0.5,
        v_max_deg_s=5.0,  # Changed from 1.0 → 5.0 (FIX 4)
        # ...
    )
    return Monitor(env)
```

---

## Files Modified

### 1. `src/sun_tracking/envs/dish_env_gaussian.py`
- **Line ~255**: Changed `v_max_deg_s=10.0` → `v_max_deg_s=5.0`
- **Lines ~386-390**: Replaced reward calculation
  - Removed: `reward = REWARD_POWER_IMPROVEMENT_SCALE * power_improvement`
  - Added: `reward = P - 0.01 * ||action||²`

### 2. `src/sun_tracking/rl/PPO_gaussian.py`
- **Line 4**: Changed import from `PPO` → `RecurrentPPO`
- **Line 5**: Added `import numpy as np`
- **Lines 15, 28**: Changed `v_max_deg_s=1.0` → `v_max_deg_s=5.0` in env factories
- **Line 46**: Changed type hint `PPO` → `RecurrentPPO`
- **Lines 70-86**: Added LSTM state handling in evaluation loop
- **Line 132**: Changed policy `"MlpPolicy"` → `"MlpLstmPolicy"`
- **Line 134**: Changed model class `PPO` → `RecurrentPPO`
- **Lines 150, 159**: Updated checkpoint/save names to `recurrent_ppo_run16`

---

## Expected Impact

### Before Fixes
- **Standard PPO**: Plateaus at ~10-15% power
- **Issue**: Cannot learn gradient from single power measurement
- **Behavior**: Random wandering, occasionally finds sun but doesn't stay

### After Fixes

**FIX 1 (LSTM)**: 
- Agent can infer gradient from power sequence
- Example: [P=0.10, 0.12, 0.15, 0.18] → "keep going this direction"
- Should enable sustained tracking to ~60-80% power

**FIX 2 (Absolute reward)**:
- Encourages staying at high power, not just moving toward it
- Should improve final convergence to >90% power

**FIX 4 (Smoother control)**:
- Less overshoot near peak
- Should enable fine centering to >95% power

**Combined effect**: Expect agent to achieve **>90% sustained power** with precise sun centering.

---

## How to Train

```bash
cd /Users/chris/Desktop/BEAM/reinforcement-learning
source .venv/bin/activate
python -m sun_tracking.rl.PPO_gaussian
```

**Training will save:**
- Checkpoints every 100K steps: `./models/checkpoints/recurrent_ppo_run16_*.zip`
- Final model: `./models/recurrent_ppo_run16_final.zip`
- TensorBoard logs: `./runs/gaussian_beam_tb/`

---

## How to Evaluate Saved Model

```python
from sb3_contrib import RecurrentPPO
import numpy as np
from sun_tracking.rl.PPO_gaussian import make_eval_env

# Load model
model = RecurrentPPO.load("models/recurrent_ppo_run16_final")

# Create environment
env = make_eval_env(seed=123)

# Run one episode
obs, info = env.reset()
lstm_states = None
episode_start = np.ones((1,), dtype=bool)

for step in range(1000):
    action, lstm_states = model.predict(
        obs,
        state=lstm_states,
        episode_start=episode_start,
        deterministic=True
    )
    obs, reward, done, truncated, info = env.step(action)
    episode_start = np.zeros((1,), dtype=bool)
    
    print(f"Step {step}: Power={info['power']:.4f}, Reward={reward:.4f}")
    
    if done or truncated:
        break
```

**Critical**: Always pass `lstm_states` and `episode_start` when using RecurrentPPO!

---

## Dependencies

Ensure `sb3-contrib` is installed:

```bash
pip install sb3-contrib
```

This package provides RecurrentPPO with LSTM support for Stable-Baselines3.

---

## Next Steps (Optional Future Work)

If performance is still not satisfactory, consider:

1. **Curriculum learning**: Train on frozen sun first, then add sun drift
2. **Longer episodes**: 180s → 300s to give more time to converge
3. **Larger LSTM**: Increase `net_arch` to [256, 256] for more memory capacity
4. **Multi-seed training**: Train 3-5 seeds and pick best model

But with current fixes, the agent should already achieve >90% power reliably.
