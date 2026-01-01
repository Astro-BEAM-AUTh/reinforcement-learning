"""Test the fixed lag dynamics."""
import numpy as np
import math
from src.sun_tracking.envs.dish_env_gaussian import step_with_lag

dt = 0.5
tau = 0.3
v_max = math.radians(10.0)

print(f"Testing FIXED lag dynamics")
print(f"dt = {dt}s, tau = {tau}s, dt/tau = {dt/tau:.3f}")
print(f"v_max = {math.degrees(v_max):.2f}°/s")
print()

# Start from rest
az, el = 0.0, 0.0
az_dot, el_dot = 0.0, 0.0

# Command small velocity
action = np.array([0.1, 0.1])
v_az_cmd = action[0] * v_max
v_el_cmd = action[1] * v_max

print(f"Action: {action}")
print(f"Commanded: v_az={math.degrees(v_az_cmd):.2f}°/s, v_el={math.degrees(v_el_cmd):.2f}°/s")

# Apply one step
az_new, el_new, az_dot_new, el_dot_new, n = step_with_lag(
    az, el, az_dot, el_dot, v_az_cmd, v_el_cmd, dt, v_max, tau
)

print(f"\nAfter step:")
print(f"  Velocities: az_dot={math.degrees(az_dot_new):.2f}°/s, el_dot={math.degrees(el_dot_new):.2f}°/s")
print(f"  Angle change: Δaz={math.degrees(az_new-az):.4f}°, Δel={math.degrees(el_new-el):.4f}°")

overshoot_az = az_dot_new / v_az_cmd if v_az_cmd != 0 else 0
overshoot_el = el_dot_new / v_el_cmd if v_el_cmd != 0 else 0
print(f"  Overshoot factor: {overshoot_az:.2f}x")

if overshoot_az <= 1.01:
    print(f"\n✅ FIXED! No overshoot (alpha clamped to 1.0)")
else:
    print(f"\n❌ Still overshooting!")

total_displacement = math.sqrt((az_new - az)**2 + (el_new - el)**2)
print(f"\nTotal displacement: {math.degrees(total_displacement):.4f}°")
