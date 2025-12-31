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
   - Modeled as a **circular disk** (not a point!)
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

*This document describes the theoretical foundation. Implementation details will be added after code development.*
