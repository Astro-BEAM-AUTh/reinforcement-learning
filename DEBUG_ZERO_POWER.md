# Debugging 0% Power Issue - Complete Guide

## The Problem

After 500K training steps with RecurrentPPO:
- Avg total reward: 3.40
- **Avg tail power: 0.0000** ← WRONG!
- Success rate: 0%

This is impossible - even a "do nothing" policy should achieve nonzero power.

---

## Why Your Current Results Are Wrong

**Most likely cause**: The evaluation loop is **correct** for RecurrentPPO, BUT there's likely one of these issues:

1. **Model checkpoint corruption**: The saved model may not contain the trained weights
2. **Episode reset issue**: LSTM states might not be properly reset between episodes
3. **Observation space mismatch**: Model was trained on different observation space than evaluation
4. **Integer overflow in step counter**: `_debug_step_count` persists across episodes

---

## FIX 1: Correct RecurrentPPO Evaluation Loop

### YOUR CURRENT CODE (appears correct but verify):

```python
for _ in range(n_episodes):
    obs, info = eval_env.reset()
    lstm_states = None  # ✓ Reset LSTM
    episode_start = np.ones((1,), dtype=bool)  # ✓ Mark first step
    
    while True:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states,  # ✓ Pass LSTM state
            episode_start=episode_start,  # ✓ Pass episode flag
            deterministic=True
        )
        obs, r, terminated, truncated, info = eval_env.step(action)
        episode_start = np.zeros((1,), dtype=bool)  # ✓ Only first step
        
        power_hist.append(info.get("power", 0.0))
        # ...
```

**This looks correct!** The issue is likely elsewhere.

### CRITICAL CHECKLIST:

- [ ] LSTM state reset to `None` at start of EACH episode
- [ ] `episode_start` is True only for first step
- [ ] `episode_start` is False for all subsequent steps
- [ ] `state` and `episode_start` are passed to `model.predict()`

---

## FIX 2: Verify Reward = P

### Current Implementation:

In `dish_env_gaussian.py` line ~385:

```python
action_penalty = 0.01 * (action[0]**2 + action[1]**2)
reward = float(self.P - action_penalty)
```

**This is correct.** Reward = Power - tiny penalty (max penalty = 0.02 for action=[-1,1])

### DEBUG CODE ADDED:

I added temporary debug prints in `dish_env_gaussian.py`:

```python
if self._debug_step_count <= 3:
    print(f"[DEBUG Step {self._debug_step_count}] P={self.P:.6f}, reward={reward:.6f}, action_penalty={action_penalty:.6f}")
```

**To remove after verification:**
```python
# Delete these lines from dish_env_gaussian.py around line 385-392:
if not hasattr(self, '_debug_step_count'):
    self._debug_step_count = 0
self._debug_step_count += 1
if self._debug_step_count <= 3:
    print(f"[DEBUG Step {self._debug_step_count}] P={self.P:.6f}, reward={reward:.6f}, action_penalty={action_penalty:.6f}")
```

---

## FIX 3: Verify info["power"] Logging

### Check that `info` dict is correct:

In `dish_env_gaussian.py` around line 395:

```python
info = {
    "power": float(self.P),  # ✓ Correct
    "angular_distance": float(angular_distance),
    "power_improvement": float(power_improvement),
}
```

### DEBUG SNIPPET (add to your evaluation loop):

```python
# Add after env.step():
if step == 0:  # First step only
    print(f"info keys: {info.keys()}")
    print(f"info['power']: {info.get('power', 'MISSING')}")
```

**What to look for:**
- Keys should include `'power'`, `'angular_distance'`, `'power_improvement'`
- `info['power']` should be a float, NOT None
- Power should be between 0.0 and 1.0

---

## FIX 4: Baseline Sanity Checks

### Run the diagnostic script:

```bash
uv run python src/sun_tracking/rl/debug_eval.py
```

This will run 4 tests:

### Test A: Do-Nothing Policy

```python
action = [0.0, 0.0]  # No movement
# Expected: Power starts at some value and STAYS there (sun drifts away slowly)
# If power is 0.0 → ENVIRONMENT IS BROKEN
```

### Test B: Random Policy

```python
action = random.uniform(-0.1, 0.1, size=2)
# Expected: Power fluctuates as dish moves randomly
# If power never changes → ENVIRONMENT IS FROZEN
```

### Test C: Reward Calculation

```python
# Verifies: reward = P - 0.01 * ||action||^2
# Expected: reward ≈ power (within 0.02)
```

### Test D: RecurrentPPO Evaluation

```python
# Uses correct LSTM state handling
# Expected: Some nonzero power (even untrained model should achieve ~10% by chance)
```

---

## How to Run Diagnostics

### Step 1: Run the full diagnostic suite

```bash
cd /Users/chris/Desktop/BEAM/reinforcement-learning
uv run python src/sun_tracking/rl/debug_eval.py
```

### Step 2: Check the output

Look for these indicators:

**GOOD SIGNS:**
```
FIX 2: TESTING REWARD CALCULATION
  Power: 0.234567
  Reward: 0.234367
  Expected: 0.234367
  Match: True

FIX 3: TESTING POWER LOGGING
  info['power']: 0.234567
  All zero?: False

FIX 4A: DO-NOTHING POLICY TEST
  Tail power: 0.1234
  OK: Do-nothing achieves nonzero power

FIX 4B: RANDOM POLICY TEST
  Std dev: 0.0512
  OK: Power fluctuates with random actions
```

**BAD SIGNS:**
```
ERROR: Power is None at step 1!
ERROR: Do-nothing policy gives 0 power - environment is broken!
ERROR: Power never changes - environment is frozen!
```

### Step 3: Test your actual trained model

```bash
# In Python:
from sb3_contrib import RecurrentPPO
import numpy as np
from sun_tracking.rl.debug_eval import evaluate_recurrent_policy_correct
from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

# Load model
model = RecurrentPPO.load("models/checkpoints/recurrent_ppo_run16_500000_steps")

# Create environment
env = GaussianBeamDishEnv(dt=0.5, v_max_deg_s=5.0, horizon_s=180.0, seed=999)

# Evaluate
results = evaluate_recurrent_policy_correct(model, env, n_episodes=20, verbose=True)
```

---

## Most Likely Root Causes (in order)

### 1. **Debug counter persists across episodes**

The `_debug_step_count` in the environment is never reset, so after 3 steps total across ALL episodes, debugging stops. This could mask issues.

**Fix**: Reset in `reset()` method:
```python
# In dish_env_gaussian.py, in reset() method, add:
self._debug_step_count = 0
```

### 2. **Model checkpoint is from wrong iteration**

The 500K checkpoint might be before learning started.

**Fix**: Try later checkpoints:
```python
model = RecurrentPPO.load("models/checkpoints/recurrent_ppo_run16_600000_steps")
model = RecurrentPPO.load("models/checkpoints/recurrent_ppo_run16_900000_steps")
```

### 3. **Evaluation uses wrong environment parameters**

Training used `dt=0.5, v_max=5.0`, but evaluation might use different values.

**Fix**: Match exactly:
```python
# In PPO_gaussian.py, make_eval_env():
env = GaussianBeamDishEnv(
    dt=0.5,  # ✓ Must match training
    v_max_deg_s=5.0,  # ✓ Must match training
    tau=0.3,
    noise_std=0.0,
    horizon_s=180.0,
    seed=seed,
)
```

### 4. **TensorBoard logs show learning, but model not saved**

The reward improved during training, but CheckpointCallback might have failed silently.

**Fix**: Check checkpoint files exist:
```bash
ls -lh models/checkpoints/recurrent_ppo_run16_*
```

If no files → model was never saved!

---

## Quick Fix Summary

### Immediate Actions:

1. **Run diagnostic script**: `uv run python src/sun_tracking/rl/debug_eval.py`

2. **Check power is nonzero** in baseline tests

3. **Verify checkpoint exists**:
   ```bash
   ls models/checkpoints/
   ```

4. **Reset debug counter** in environment:
   ```python
   # In dish_env_gaussian.py, reset() method, add:
   self._debug_step_count = 0
   ```

5. **Remove debug prints** after verification:
   ```python
   # Delete from step() in dish_env_gaussian.py:
   if not hasattr(self, '_debug_step_count'): ...
   ```

### Expected Timeline:

- Diagnostic script: 2 minutes
- Find root cause: 5 minutes
- Fix and re-evaluate: 1 minute

**If baseline tests pass but trained model gives 0 power → Model is undertrained or checkpoint is wrong**

**If baseline tests fail → Environment is broken, not the evaluation**

---

## After Fixes: Clean Up Debug Code

Once you've verified everything works:

1. Remove debug prints from `dish_env_gaussian.py`:
   - Lines with `_debug_step_count`
   - Lines with `print(f"[DEBUG Step ...`

2. (Optional) Remove `debug_eval.py` if you don't need it

3. Re-run evaluation with clean output:
   ```bash
   uv run python src/sun_tracking/rl/PPO_gaussian.py
   ```
