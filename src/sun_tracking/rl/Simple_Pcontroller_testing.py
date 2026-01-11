import math
from datetime import datetime, timezone

from sun_tracking.envs.dish_env import (
    DishSim, step_with_lag, compute_n_from_angles, compute_power
)

def az_el_from_vec(s):
    # ENU → az = atan2(y,x) in [-pi,pi], el = asin(z) in [-pi/2, pi/2]
    x, y, z = s
    az = math.atan2(y, x)
    el = math.asin(max(-1.0, min(1.0, z)))
    return az, el

def main():
    # ---- sim/site/time ----
    dt   = 0.5              # seconds per step
    tau  = 0.3              # motor lag (s)
    vmax = math.radians(10) # max speed (rad/s)

    # Start at a *daytime* UTC so we see non-zero power (Nov 16, 2025 ~ noon local → 10:00 UTC)
    start = datetime(2025, 11, 16, 10, 0, 0, tzinfo=timezone.utc)
    sim = DishSim(lat_deg=40.64, lon_deg=22.94, dt=dt, start_dt_utc=start)

    # ---- dish initial state ----
    az, el = math.radians(+90), math.radians(5)   # looking North, Sun is South
    az_dot, el_dot = 0.0, 0.0

    # simple P-controller gains (rad/s per rad of error), clipped by vmax inside step
    k_az = 1.0
    k_el = 1.0

    print("k |   t(s) |   az(deg)  el(deg) |   P (n·s)")
    for k in range(40):  # ~20 seconds simulated time
        # Sun direction and target angles
        s = sim.sun_vec()
        az_s, el_s = az_el_from_vec(s)

        # commanded velocities toward Sun
        v_az_cmd = k_az * (az_s - az)
        v_el_cmd = k_el * (el_s - el)

        # update dish with first-order lag
        az, el, az_dot, el_dot, n = step_with_lag(
            az, el, az_dot, el_dot,
            v_az_cmd, v_el_cmd,
            dt, vmax, tau
        )

        # power (clamped cosine with tiny noise for realism)
        P = compute_power(n, s, noise_std=0.0)

        # log
        print(f"{k:2d} | {sim.t:6.1f} | {math.degrees(az):9.3f} {math.degrees(el):8.3f} | {P:7.4f}")

        # advance sim time
        sim.tick()

if __name__ == "__main__":
    main()
