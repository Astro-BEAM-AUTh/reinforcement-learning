"""Debug the lag dynamics to see the overshoot."""
import numpy as np
import math

dt = 0.5
tau = 0.3
v_max = math.radians(10.0)  # 10°/s

print(f"dt = {dt}s")
print(f"tau = {tau}s")
print(f"dt/tau = {dt/tau:.3f}")
print(f"v_max = {math.degrees(v_max):.2f}°/s = {v_max:.4f} rad/s")
print()

# Test case: start at az=0, el=0, velocities=0
# Command a small action like [0.1, 0.1]
az = 0.0
el = 0.0
az_dot = 0.0
el_dot = 0.0

action = np.array([0.1, 0.1])
v_az_cmd = action[0] * v_max
v_el_cmd = action[1] * v_max

print(f"Action = {action}")
print(f"Commanded velocities:")
print(f"  v_az_cmd = {math.degrees(v_az_cmd):.2f}°/s")
print(f"  v_el_cmd = {math.degrees(v_el_cmd):.2f}°/s")
print()

# Apply lag dynamics
az_dot_new = az_dot + (v_az_cmd - az_dot) * (dt / tau)
el_dot_new = el_dot + (v_el_cmd - el_dot) * (dt / tau)

print(f"New velocities after lag update:")
print(f"  az_dot = {math.degrees(az_dot_new):.2f}°/s (commanded: {math.degrees(v_az_cmd):.2f}°/s)")
print(f"  el_dot = {math.degrees(el_dot_new):.2f}°/s (commanded: {math.degrees(v_el_cmd):.2f}°/s)")
print()

overshoot_az = az_dot_new / v_az_cmd if v_az_cmd != 0 else 0
overshoot_el = el_dot_new / v_el_cmd if v_el_cmd != 0 else 0
print(f"Overshoot factor: {overshoot_az:.2f}x commanded velocity!")
print()

# Integrate angles
az_new = az + az_dot_new * dt
el_new = el + el_dot_new * dt

print(f"Angle changes in one step:")
print(f"  Δaz = {math.degrees(az_new - az):.4f}°")
print(f"  Δel = {math.degrees(el_new - el):.4f}°")
print()

# Total angular displacement
total_displacement = math.sqrt((az_new - az)**2 + (el_new - el)**2)
print(f"Total angular displacement: {math.degrees(total_displacement):.4f}°")
print()

print("🔴 PROBLEM: With dt/tau = 1.67, the system is unstable!")
print("🔴 Even a small action causes ~1.67x overshoot in velocity.")
print("🔴 Starting from 1° offset, a single step can move the dish 1-2°,")
print("🔴 which moves it OUTSIDE the Gaussian beam (FWHM=2.43°).")
