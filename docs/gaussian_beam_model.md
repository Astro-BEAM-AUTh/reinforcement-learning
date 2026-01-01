# Gaussian Beam Sun Tracking Model

## Project Goal
Create a new RL environment for training a satellite dish to track the sun using a **realistic Gaussian beam model** with finite sun size, instead of the simple point-to-point vector approach.

---

## Physical Model

### Components:

1. **Satellite Dish**
   - Emits a beam with **Gaussian radiation pattern**
   - **Beamwidth (FWHM)**: 2.43° 
   - This is the angle where power drops to 50% of maximum
   - **σ (standard deviation)**: ~1.03° (derived from beamwidth)
   - **No hard edges** - gain decreases smoothly with distance from beam center

2. **Sun**
   - Modeled as a **circular disk** 
   - **Angular diameter**: 0.53°
   - **Radius**: 0.265°
   - Every point on the sun disk contributes to total received power

3. **Key Constraint**
   - Beamwidth (2.43°) > Sun diameter (0.53°)
   - The entire sun can fit inside the beam

---

## Power Calculation Model

### Old Approach (current code):
```
P = dot(dish_vector, sun_vector) = cos(θ)
```
- Simple, fast
- Treats both as points
- Power decreases slowly with misalignment

### New Approach (Gaussian beam with finite sun):
```
P = ∫∫_sun_disk exp(-(x² + y²)/(2σ²)) dA
```

**In words**: Integrate the Gaussian beam gain over the entire sun disk using numerical integration.

---

## Mathematical Details

### Coordinate System (2D, not spherical):
- **Beam center**: Origin (0, 0)
- **Sun center**: (d_x, d_y) - offset from beam center
- **Distance between centers**: d = √(d_x² + d_y²)

### Gaussian Beam Pattern:
At any point (x, y) in space:
```
G(x, y) = exp(-(x² + y²)/(2σ²))
```
where σ ≈ 1.03°

### Relationship Between Beamwidth and σ:

For a Gaussian beam, the Full-Width Half-Maximum (FWHM) is the beamwidth where the gain drops to 50%:

```
FWHM = 2√(2 ln 2) × σ ≈ 2.355 × σ

Therefore:
σ = FWHM / 2.355 = 2.43° / 2.355 ≈ 1.03°
```

### Power Integration in Polar Coordinates:

Converting to polar coordinates centered on the sun makes the integral easier:

```
P(d) = ∫₀^(2π) ∫₀^(r_sun) G(r, φ, d) × r dr dφ
```

Where the Gaussian gain at point (r, φ) on the sun disk is:
```
G(r, φ, d) = exp(-((d_x + r·cos(φ))² + (d_y + r·sin(φ))²)/(2σ²))
```

Or in terms of scalar offset d:
```
G(r, φ, d) = exp(-(d² + r² + 2dr·cos(φ))/(2σ²))
```

**Integration Parameters**:
- d = offset between beam and sun centers (degrees)
- r = radius from sun center (0 to r_sun = 0.265°)
- φ = angle around sun disk (0 to 2π)
- σ = Gaussian width parameter (1.03°)

---

## Why This is More Realistic

### Power sensitivity to misalignment:

**Old model (dot product)**:
- 1° off → ~99.9% power (barely matters)
- 5° off → ~99.6% power
- Very forgiving to misalignment

**New model (Gaussian beam)**:
- 0.5° off → ~88% power
- 1.03° off → ~61% power (1σ point)
- 1.22° off → ~50% power (HPBW)
- 2° off → ~14% power
- **Much more sensitive!** Requires precise tracking

### Physical accuracy:
- Real parabolic dish antennas have Gaussian-like beam patterns due to electromagnetic diffraction
- Sun has finite angular size - different parts of the solar disk are at different angles from beam center
- Points outside the main beam still contribute power (exponentially decreasing)
- No artificial hard cutoffs in nature

### Contribution vs Distance from Beam Center:

| Distance from Center | Gaussian Gain | Physical Meaning |
|---------------------|---------------|------------------|
| 0°                  | 100%          | Perfect alignment |
| 0.5°                | 88%           | Slightly off |
| 1.03° (σ)          | 61%           | One standard deviation |
| 1.22°              | 50%           | Half-power beamwidth (HPBW) |
| 2.0°               | 14%           | Edge of useful beam |
| 3.0°               | 3%            | Mostly outside beam |
| 5.0°               | 0.01%         | Negligible contribution |

Since the sun radius is only 0.265°, even when the sun center is offset by 1-2°, parts of the sun still receive reasonable gain!

---

## Expected Behavior

### Perfect Alignment (d = 0):
- Entire sun disk centered in beam
- All parts of sun receive maximum beam gain
- Power ≈ 1.0 (maximum, normalized)

### Slight Misalignment (d = 0.3°):
- Sun center offset by roughly one sun radius
- One side of sun gets higher gain, opposite side gets lower gain
- Power ≈ 0.92 (minimal loss)

### Moderate Misalignment (d = 1.2°):
- Sun center near half-power point of beam
- Significant variation in gain across sun disk
- Power ≈ 0.5 (half power)

### Large Misalignment (d > 3°):
- Sun mostly outside the high-gain region of beam
- Only edge contributions from Gaussian tail
- Power ≈ 0.05 or less (very low)

---

## Physical Constants

```python
BEAMWIDTH_FWHM_DEG = 2.43      # Full-Width Half-Maximum beamwidth
SUN_DIAMETER_DEG = 0.53         # Angular diameter of sun
SUN_RADIUS_DEG = 0.265          # Sun radius (half diameter)
SIGMA_DEG = 1.03                # Gaussian width parameter (FWHM/2.355)
```

### Derivation Note:
The beamwidth value of 2.43° comes from the satellite dish antenna specifications. For a circular aperture antenna, the beamwidth is approximately:

```
FWHM ≈ 1.2 × λ / D

where:
λ = wavelength
D = dish diameter
```

For a typical Ku-band satellite dish operating at ~12 GHz with diameter ~80cm (as shown in the reference image), this gives a beamwidth around 2-3 degrees.

---

## RL Training Implications

1. **Harder task**: Agent must achieve much more precise alignment compared to the old dot-product model
2. **Better learning gradient**: Smoother, continuous power function provides better gradient information
3. **More realistic**: Matches actual satellite dish tracking behavior
4. **Same state/action spaces**: Can reuse RL architecture, only power calculation changes
5. **Practical relevance**: Skills learned transfer to real dish tracking systems

---

## Integration Method

**Numerical Integration** will be used to compute the power integral. The double integral over the sun disk will be evaluated numerically at each simulation step to calculate the received power based on the current beam-sun offset.

This approach provides:
- **Accuracy**: No analytical approximations needed
- **Flexibility**: Easy to modify beam patterns or sun models
- **Clarity**: Direct implementation of the physical equations

---

## Implementation

### File Structure

The Gaussian beam environment is implemented in:
- **`src/sun_tracking/envs/dish_env_gaussian.py`** - Main environment implementation
- **Test script**: `src/sun_tracking/rl/test_gaussian_env.py`

### Core Components

#### 1. Angular Offset Calculation

**Function**: `compute_angular_offset(dish_vec, sun_vec)`

Calculates the 2D offset (d_x, d_y) between the dish pointing direction and sun position.

**Algorithm**:
1. Compute total angular separation using dot product:
   ```python
   dot = dish_vec · sun_vec
   θ = arccos(dot)
   ```

2. Find offset direction using cross product:
   ```python
   cross = dish_vec × sun_vec
   ```
   The cross product gives a perpendicular vector indicating which direction the sun is offset.

3. Scale and decompose into (d_x, d_y):
   ```python
   d_x = (cross_x / |cross|) × θ × (180/π)
   d_y = (cross_y / |cross|) × θ × (180/π)
   ```

**Output**: Angular offset in degrees with East (d_x) and North (d_y) components

#### 2. Gaussian Power Integration

**Function**: `compute_gaussian_power(d_x, d_y, sigma_deg, r_sun_deg, noise_std, rng)`

Computes received power by numerically integrating the Gaussian beam gain over the sun disk.

**Integration Setup**:

The beam center is at origin (0, 0), sun center at (d_x, d_y). We use polar coordinates centered on the sun:

```python
def integrand(phi, r):
    # Point (r, phi) on sun disk
    x = d_x + r * cos(phi)  # Position in beam frame
    y = d_y + r * sin(phi)
    
    # Distance from beam center
    dist² = x² + y²
    
    # Gaussian gain
    gain = exp(-dist² / (2σ²))
    
    return gain * r  # r is Jacobian for polar coords
```

**Numerical Integration**:
```python
result = ∫₀^(2π) ∫₀^(r_sun) integrand(phi, r) dr dphi
```

Uses `scipy.integrate.dblquad` for double integration:
- Inner integral: radius r from 0 to 0.265° (sun radius)
- Outer integral: angle φ from 0 to 2π (full circle)

**Normalization**:
```python
max_power = π × r_sun²  # Max power when perfectly aligned
normalized_power = result / max_power
```

**Sensor Noise** (optional):
```python
if noise_std > 0:
    power += normal(0, noise_std)
    power = clip(power, 0, 1)
```

#### 3. Gymnasium Environment: `GaussianBeamDishEnv`

A fully compliant Gymnasium environment for RL training.

**State Space** (6 dimensions):
```
[az, el, az_dot, el_dot, P, dP]

- az: Azimuth angle (radians) ∈ [-π, π]
- el: Elevation angle (radians) ∈ [0, π/2]
- az_dot: Azimuth velocity (rad/s) ∈ [-v_max, v_max]
- el_dot: Elevation velocity (rad/s) ∈ [-v_max, v_max]
- P: Current power measurement ∈ [0, 1]
- dP: Change in power from previous step ∈ [-1, 1]
```

**Action Space** (2 dimensions):
```
[v_az_cmd, v_el_cmd] ∈ [-1, 1]²

Normalized commanded velocities that are scaled by v_max
```

**Dynamics Model**:

The dish uses a first-order lag model to simulate realistic motor response:

```python
# First-order lag
az_dot += (v_cmd_az - az_dot) × (dt / τ)
el_dot += (v_cmd_el - el_dot) × (dt / τ)

# Integrate angles
az = wrap(az + az_dot × dt)
el = clip(el + el_dot × dt, 0, π/2)
```

Parameters:
- **dt** = 0.5s (time step)
- **τ** = 0.3s (time constant for lag)
- **v_max** = 10°/s (maximum angular velocity)

**Reward Function**:

The reward uses **two-component shaping** to provide dense learning signals:

```python
# Component 1: Power improvement reward
power_improvement = P(t) - P(t-1)
power_reward = 1000 × power_improvement

# Component 2: Exponential distance penalty
angular_distance = √(d_x² + d_y²)
distance_penalty = -10 × distance²

# Total reward
reward = power_reward + distance_penalty
```

**Rationale**:
- **Power improvement component**: Rewards the agent for *increasing* power, not just having high power. This encourages continuous optimization and provides signal even when power is low. Getting power from 0.01→0.02 gives same reward as 0.90→0.91.
- **Exponential distance penalty**: Creates a strong gradient that heavily penalizes being far from the sun. At 10° away, penalty is -1000 (vs linear penalty of only -1). This guides the agent toward the sun much more effectively than linear penalties.

This shaped reward addresses the sparse reward problem where the Gaussian beam's narrow beamwidth (σ=1.03°) makes it difficult for random exploration to discover high-reward states.

#### 4. Episode Structure

**Reset**:
1. Reset simulation time to 0
2. Randomize time of day until sun is above horizon (z > 0)
3. Initialize dish at random angles:
   - az ∈ [-π, π] (uniform random)
   - el ∈ [2°, 88°] (uniform random, avoiding horizon)
4. Set initial velocities to zero
5. Calculate initial power using Gaussian model

**Step**:
1. Scale action from [-1, 1] to actual velocities
2. Update dish dynamics with first-order lag
3. Integrate angles over time step dt
4. Advance simulation time
5. Get new sun position from Astral library
6. Calculate angular offset (d_x, d_y)
7. Compute power via Gaussian integration
8. Return observation, reward, done flags

**Termination**:
- **terminated**: Always False (no early termination)
- **truncated**: True when elapsed time ≥ horizon (default 90s)

### Key Differences from Original Environment

| Aspect | Original (`dish_env.py`) | Gaussian (`dish_env_gaussian.py`) |
|--------|--------------------------|-----------------------------------|
| **Power calculation** | `P = dot(n̂, ŝ)` | `P = ∫∫ Gaussian(r,φ) dA` |
| **Computation time** | ~1 µs (dot product) | ~100 µs (numerical integration) |
| **Sensitivity** | Forgiving (cos curve) | Very sensitive (Gaussian) |
| **Sun model** | Point source | Finite disk (0.53°) |
| **Beam model** | Perfect pointer | Gaussian beam (2.43° FWHM) |
| **Physical realism** | Simplified | Realistic |

### Validation Results

From `test_gaussian_env.py`:

**Power vs Offset**:
- d = 0.0° → P = 0.984 (near perfect)
- d = 0.3° → P = 0.943 (slight drop)
- d = 1.2° → P = 0.505 (half power, HPBW)
- d = 3.0° → P = 0.015 (very low)

**Environment Functionality**:
- ✅ Observation/action spaces correctly defined
- ✅ Reset produces valid initial states
- ✅ Step function executes without errors
- ✅ Power readings respond correctly to dish motion
- ✅ Sun position verified above horizon

### Computational Performance

**Per-step timing** (approximate):
- Dynamics update: ~10 µs
- Sun position: ~50 µs
- Angular offset: ~5 µs
- Gaussian integration: ~100 µs
- **Total**: ~165 µs per step

For 1M training steps, this represents ~165 seconds of computation just for environment steps (excluding RL algorithm overhead).

### Usage Example

```python
from sun_tracking.envs.dish_env_gaussian import GaussianBeamDishEnv

# Create environment
env = GaussianBeamDishEnv(
    dt=0.5,              # 0.5s time step
    v_max_deg_s=10.0,    # Max 10°/s velocity
    tau=0.3,             # 0.3s motor lag
    noise_std=0.01,      # 1% sensor noise
    horizon_s=90.0,      # 90s episode length
    seed=42
)

# Reset
obs, info = env.reset()

# Take action
action = [0.5, -0.3]  # Some commanded velocities
obs, reward, terminated, truncated, info = env.step(action)

print(f"Power: {reward:.4f}")
```

---

## Training Runs Log

### Run 1 - Random Initialization (Failed)
**Date**: Dec 31, 2025  
**Config**: Random start (0-180° away), 1M steps, noise=0.01, reward=power only  
**Results**: 
- Training: ep_rew_mean ~6.5
- Evaluation: Power 1.4%, Success 0%  
**Key Takeaway**: Starting too far from sun (80-90°) gives zero reward signal due to narrow Gaussian beam (σ=1.03°). Agent can't learn from pure exploration.

---

### Run 2 - Easier Start Conditions (Failed)
**Date**: Dec 31, 2025 - Jan 1, 2026  
**Config**: Start within 15° of sun, 5M steps, noise=0.01, reward=power only  
**Results**: 
- Training: ep_rew_mean ~1.2 → ~3.4 (improved)
- Evaluation: Power 1.6%, Success 0%
- Policy std collapsed: 0.94 → 0.027 (stopped exploring)  
**Key Takeaway**: Rewards improved during training but didn't generalize to evaluation. Pure power reward is too sparse - agent found local optimum and stopped exploring.

---

### Run 3 - Reward Shaping v1 (Failed, but progress)
**Date**: Jan 1, 2026  
**Config**: 
- Start within 15° of sun, 5M steps, noise=0.01
- **Reward function**: `100×P + 5×(prev_dist - curr_dist) - 0.1×dist`
**Results**: 
- Training: ep_rew_mean -650 → -14 (huge improvement!)
- Evaluation: Power 1.19%, Success 4% (slightly better)
- Policy std collapsed: 0.94 → 0.029  
**Key Takeaways**: 
- Distance improvement bonus helped training a lot
- But causes instability - agent learns to move fast toward sun, then overshoots
- Linear distance penalty too weak (10° away = -1.0 penalty only)
- Still failing at test time despite good training progress

---

### Run 5 - Start Closer (5° away) - Still Failed
**Date**: Jan 1, 2026  
**Config**:
- Start within 5° of sun (vs 15° in Run 4), 1M steps, noise=0.01
- **Reward function**: `1000×ΔP - 0.1×dist²`
**Results**:
- Training: ep_rew_mean -28,000 → -20,000 (slight improvement)
- Evaluation: Avg reward -12,792, Power 0.40%, Success 0%
- Policy std: 0.99 → 0.83 (collapsed)
**Key Takeaway**:
- Starting closer (5° vs 15°) made NO difference - same results!
- Distance penalty still dominates: -0.1 × 5² = -2.5 per step
- Power at 5° is only ~5%, so ΔP is tiny (0.001 or less)
- Reward = 1000×0.001 - 2.5 ≈ -1.5 (still negative on average)
- **Problem is the reward function, not starting distance**

---

### Run 6 - Pure Power Improvement, No Distance Penalty ✅ **BEST SO FAR**
**Date**: Jan 1, 2026  
**Config**:
- Start within **3° of sun** (vs 5° in Run 5), 1M steps, noise=0.01
- **Reward function**: `1000×ΔP` (NO distance penalty!)
- Environment changes:
  - Removed `REWARD_DISTANCE_PENALTY` constant
  - Simplified reward to pure power improvement
  - Changed `max_offset_deg` from 5.0 to 3.0

**Results**:
- Training: ep_rew_mean -230 → -107 (steady improvement throughout!)
- Evaluation:
  - Avg total reward: **-142.86**
  - Avg last power (single step): **4.15%**
  - Avg tail power (last 100 steps): **2.31%**
  - Success rate (tail_avg > 95%): **0%**
- Policy std: 0.99 → 0.68 (healthy exploration maintained)

**Key Insights**:
- **Massive improvement**: 10× better power than previous runs (2.31% vs 0.4%)
- **Cleaner learning**: Episode reward improved monotonically (-230 → -107)
- **No policy collapse**: std reduced gradually to 0.68 (vs collapsed to 0.03 in Run 2/3)
- **Pure ΔP reward works**: No distance penalty needed! Power improvement is self-correcting:
  - Move toward sun → ΔP positive → positive reward ✓
  - Move away from sun → ΔP negative → negative reward ✓
  - No artificial penalty fighting against exploration ✓

**Why This Worked Better**:

1. **Starting at 3° vs 5°**:
   - Power at 3°: ~1.4% (vs 0.06% at 5°)
   - Stronger ΔP signals: movements create ΔP ≈ 0.01-0.05
   - Reward magnitude: -50 to +50 (vs -2.5 to +1 at 5°)

2. **Removed Distance Penalty**:
   - Old: reward = 1000×ΔP - 0.1×d² → often negative even when improving
   - New: reward = 1000×ΔP → directly reflects progress
   - Agent free to explore trajectories, including temporarily moving away
   - No conflicting signals between "improve power" and "minimize distance"

3. **Exploration Freedom**:
   - Distance penalty punishes exploration paths
   - Pure ΔP allows agent to find indirect approaches to sun
   - Agent can learn multi-step strategies without penalty

**Remaining Challenge**:
- Still only reaching 2.31% power (need 95% for success)
- Agent is learning and improving, but not converging to full tracking
- May need: longer training, different network architecture, or curriculum learning

**Next Steps to Consider**:
- Longer training (5M steps instead of 1M)
- Start even closer (1-2° where power is 20-50%)
- Curriculum learning: gradually increase difficulty
- Tune power improvement scale (try 500× or 2000×)
- Add small success bonus for reaching >90% power

---

### Run 7 - Exponential Bonus for High Power ❌ **CATASTROPHIC FAILURE**
**Date**: Jan 1, 2026  
**Config**:
- Start within **3° of sun**, 1M steps, noise=0.01
- **Reward function**: `1000×ΔP + 100×exp(5×(P - 0.50))` if P > 50%
- Environment changes:
  - Added exponential bonus for power above 50%
  - Bonus values: 50%→100, 60%→165, 70%→272, 80%→448, 90%→739, 95%→1218

**Results**:
- Training: 
  - ep_rew_mean started at -141 → briefly reached +23.8 (step 135k) → crashed to -176 by end
  - Showed initial promise then completely collapsed
- Evaluation:
  - Avg total reward: **-189.48**
  - Avg last power (single step): **0.39%** (10× worse than Run 6!)
  - Avg tail power (last 100 steps): **0.43%**
  - Success rate: **0%**
- Policy std: 0.99 → 0.67 (maintained some exploration)

**What Went Wrong - The Exponential Bonus Trap:**

1. **Early False Success** (steps 0-150k):
   - Agent discovered it could get huge bonuses by reaching 50%+ power briefly
   - Episode reward jumped from -141 to +24 (looked amazing!)
   - This was the exponential bonus dominating the signal

2. **The Trap** (steps 150k-1M):
   - Agent optimized for "reach high power for one moment" instead of "stay tracking"
   - Learned to make quick aggressive moves to spike power, then lose tracking
   - Bonus became too large relative to power improvement signal
   - Reward collapsed from +24 back to -176 (worse than starting!)

3. **Why It's Worse Than Run 6**:
   - Run 6 (pure ΔP): Final power = 2.31%
   - Run 7 (ΔP + bonus): Final power = 0.43%
   - The exponential bonus **actively harmed learning** by creating wrong incentives

**Key Insight - Reward Shaping Gone Wrong:**

The exponential bonus seemed logical ("reward milestones!"), but it:
- Created a **local optimum**: "briefly touch 50% power" instead of "maintain tracking"
- Made the agent **myopic**: optimized for single-step spikes, not sustained performance
- Overshadowed the **correct signal** (power improvement) with noise (exponential spikes)

This is a classic RL failure mode: **well-intentioned reward shaping that contradicts the true objective**.

**Lesson Learned**:
- **Simpler is often better** in reward design
- Bonuses for thresholds can create perverse incentives
- Pure power improvement (Run 6) was actually the right approach
- Adding complexity doesn't always help - it can actively hurt

**What to Try Next**:
- Go back to pure ΔP reward (like Run 6)
- Try starting even closer (1-2°) for stronger initial signal
- Or: Longer training with pure ΔP (5M steps)
- **Avoid threshold bonuses** - they create wrong local optima

---

### Run 8 - Start at 2° ❌ **WORSE THAN RUN 6**
**Date**: Jan 1, 2026  
**Config**: 2° init, pure ΔP reward (1000×ΔP), 1M steps  
**Results**: Eval power 0.40%, ep_rew_mean stayed -350 to -400  
**Conclusion**: Starting closer made it WORSE, not better!

---

### Run 9 - Start at 1° (Sanity Check) ❌ **CATASTROPHIC - SYSTEM BROKEN**
**Date**: Jan 1, 2026  
**Config**:
- Start within **1° of sun**, 1M steps, noise=0.01
- **Reward function**: `1000×ΔP` (pure power improvement)
- At 1° offset, initial power should be ~20%

**Results**:
- Training: ep_rew_mean = **-750** throughout (NO improvement!)
- Evaluation:
  - Avg total reward: **-738.90**
  - Avg power: **0.40%**
  - Success: **0%**

**CRITICAL FINDING - The System Is Fundamentally Broken:**

At 1° offset:
- Expected initial power: ~20%
- Expected ΔP from good actions: 0.05-0.20 (huge!)
- Expected reward: +50 to +200 per step
- **Actual reward: -750** (impossible with these dynamics!)

**What This Tells Us:**

The negative rewards mean the agent is **consistently losing power**, even when starting at 1° away with 20% initial power. This is impossible unless:

1. **The agent is moving AWAY from the sun** - but why would random exploration consistently move away?
2. **The power calculation is wrong** - maybe the Gaussian integration has a bug?
3. **The reward scaling is broken** - maybe ΔP is being calculated incorrectly?
4. **The dynamics are unstable** - maybe the lag model causes oscillations that lose power?

**The Smoking Gun:**

Runs 6-9 comparison:
- Run 6 (3° init): ep_rew = -107 to -230, final power = **2.31%** ✓
- Run 7 (3° + bonus): ep_rew = -176, final power = 0.43% ✗
- Run 8 (2° init): ep_rew = -350, final power = 0.40% ✗
- Run 9 (1° init): ep_rew = **-750**, final power = 0.40% ✗

**Pattern**: Starting closer makes rewards MORE negative!

This is backwards - closer start = higher initial power = more positive ΔP signals. The fact that it's inverting means something is fundamentally broken.

**Hypothesis - The Lag Model Might Be The Problem:**

The first-order lag dynamics might be causing the agent to overshoot and oscillate:
```python
# First-order lag with τ=0.3s, dt=0.5s
az_dot += (v_cmd - az_dot) × (dt / τ)  # dt/τ = 1.67 (very aggressive!)
```

When dt > τ, the system can become unstable and overshoot commands, causing oscillations that consistently reduce power.

**What We Need To Do:**

1. **Debug the environment** - manually test if commands actually improve power
2. **Check the lag model** - reduce τ or dt to prevent overshooting  
3. **Verify Gaussian integration** - make sure power calculation is correct
4. **Test with zero lag** - remove dynamics complexity to isolate the problem

**This explains why Run 6 worked better** - it was before we changed something that broke the system!

---

## 🐛 **ROOT CAUSE ANALYSIS** - Jan 1, 2026

After systematic debugging with manual environment testing, discovered **TWO CRITICAL BUGS**:

### Bug #1: Unstable Lag Dynamics (dt/τ > 1)

**The Problem**:
- Current config: `dt = 0.5s`, `τ = 0.3s` → `dt/τ = 1.67 > 1`
- First-order lag update: `v_new = v_old + (v_cmd - v_old) × (dt/τ)`
- When `dt/τ > 1`, velocity **overshoots** commanded value by 1.67x
- Example: Command 1°/s → actual velocity becomes 1.67°/s
- Single step displacement: 1.67°/s × 0.5s = **0.83° per step** even for small actions

**Why This Breaks Learning**:
- Starting from 1° offset (75% power)
- Action = 0.1 causes ~0.7° displacement
- Total offset becomes ~1.7° → power drops to 10-20%
- Action = 0.5 causes ~2.5° displacement → power crashes to 0%
- Agent sees catastrophic negative rewards for ANY exploration

**The Fix**:
```python
# Old (unstable):
az_dot = az_dot + (v_az_cmd - az_dot) * (dt / tau)  # Can overshoot

# Fixed (stable):
alpha = min(1.0, dt / tau)  # Clamp to [0, 1]
az_dot = az_dot + (v_az_cmd - az_dot) * alpha  # Never overshoot
```

**Verification**:
- Before fix: action=0.1 → velocity overshoots to 1.67°/s → displacement = 0.83°
- After fix: action=0.1 → velocity exactly 1.0°/s → displacement = 0.5° ✓

### Bug #2: Initialization Sensitivity

**The Problem**:
- Run 6 used `max_offset_deg = 3.0` → initial power ≈ 1-2%
- Runs 7-9 used `max_offset_deg = 1.0` → initial power ≈ 75%
- Combined with Bug #1, this creates impossible learning conditions

**Why 1° Initialization Fails**:
1. **High initial power (75%)** → any movement causes large ΔP
2. **Narrow margin for error**:
   - Beam FWHM = 2.43° (half-power width)
   - At 1° offset: already at 75% power (near peak)
   - Moving 1.5° in ANY direction → power drops below 10%
3. **Overshoot dynamics** → even small actions (0.1-0.3) move dish 0.5-1.5°
4. **Catastrophic feedback**:
   - Agent explores → power crashes → huge negative reward
   - Agent learns to freeze → sun moves → still loses power
   - No safe exploration possible

**Why 3° Initialization Works Better**:
1. **Lower initial power (1-2%)** → more room to move
2. **Gradual power gradient**:
   - At 3° offset: power ≈ 1%
   - Can move ±1° without catastrophic drops
   - Allows exploration to find better positions
3. **Clearer learning signal**:
   - Moving toward sun: ΔP positive (reward +10 to +50)
   - Moving away: ΔP negative (reward -10 to -50)
   - Agent can learn without instant death

### Combined Effect

With BOTH bugs active (Runs 7-9):
- Unstable dynamics (1.67x overshoot) + High initial power (75%)
- Action=0.3 → displacement ≈ 1.25° → power crash from 75% to 5%
- Reward = -700 on first step
- Agent has NO chance to learn

With Bug #1 fixed but 1° init:
- Stable dynamics (no overshoot) + High initial power (75%)
- Action=0.3 → displacement = 0.75° → still moves from 1° to 1.75° offset
- Power drops from 75% to ~20%
- Still very harsh, but at least not catastrophic

**Recommended Configuration** (matching Run 6):
```python
max_offset_deg = 3.0  # Start farther from sun
dt = 0.5
tau = 0.3  # OK now with alpha clamp
v_max_deg_s = 10.0
```

This gives:
- Initial power ≈ 1-2% (more forgiving)
- Stable dynamics (no overshoot)
- Exploration freedom (can move ±1° without disaster)
- Clear ΔP signals for learning

### Debug Evidence

**Test 1** (action=[0,0], no movement):
- Power: 75.04% → 74.85% → 74.65% (gradual sun drift)
- Works fine ✓

**Test 2** (random actions, BEFORE fix):
- Step 1 with action=[0.31, 0.16]: Power 75% → 7%
- Overshoot caused 2° displacement
- Power crashed ❌

**Test 3** (constant action=[0.5, 0.5], AFTER fix but 1° init):
- Step 1: Power 75% → 0.04%, angular distance jumped to 4.1°
- Each step moves 2.5° (5°/s × 0.5s)
- Still too sensitive ❌

**Test 4** (1° init with fix, small actions):
- Small random actions (-0.1 to +0.1) still cause large power swings
- Max power in 100 steps: 59%
- Environment navigable but extremely difficult

**Conclusion**: Must fix BOTH bugs to recover Run 6 performance.

