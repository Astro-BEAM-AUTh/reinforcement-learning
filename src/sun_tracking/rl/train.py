import math

from sun_tracking.envs.dish_env import compute_n_from_angles, step_with_lag


def main():
    # ----- params you can tweak -----
    dt   = 0.5                 # seconds per step
    tau  = 0.3                 # motor time constant (s)
    vmax = math.radians(10.0)  # max speed = 10°/s

    # initial dish state
    az, el       = 0.0, math.radians(10.0)  # az=0°, el=10°
    az_dot, el_dot = 0.0, 0.0               # start at rest

    # commanded velocities (what the agent would output)
    v_az_cmd = math.radians(5.0)    # +5°/s az
    v_el_cmd = math.radians(-3.0)   # -3°/s el

    print("k |  az(deg)  el(deg) | az_dot(deg/s) el_dot(deg/s) | n_hat (x,y,z)")
    for k in range(8):
        az, el, az_dot, el_dot, n = step_with_lag(
            az, el, az_dot, el_dot,
            v_az_cmd, v_el_cmd,
            dt, vmax, tau
        )
        print(f"{k:1d} | {math.degrees(az):8.3f} {math.degrees(el):8.3f} |"
              f" {math.degrees(az_dot):12.3f} {math.degrees(el_dot):13.3f} |"
              f" ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")

    # optional: one more check — compute n̂ directly from final angles
    n2 = compute_n_from_angles(az, el)
    print("n_hat from angles (final) =", tuple(round(c, 4) for c in n2))

if __name__ == "__main__":
    main()
