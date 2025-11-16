import math  # noqa: INP001
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sun_tracking.sky_objects.sun_model import sun_vector


def _wrap_pi(a):
    return ((a + math.pi) % (2.0 * math.pi)) - math.pi

def _clip_el(el) -> float:
    return float(np.clip(el, 0.0, math.pi / 2.0))


# change the coordinates (az, el) to (x, y, z)
def compute_n_from_angles(az, el):
    x = math.cos(el) * math.cos(az)  # East
    y = math.cos(el) * math.sin(az)  # North
    z = math.sin(el)                 # Up
    return (x, y, z)  # |n̂| = 1

def step_with_lag(az, el, az_dot, el_dot, v_az_cmd, v_el_cmd, dt, v_max, tau):
    """
    First-order motor lag simulation:
      - clamp commanded velocities v_cmd to ±v_max
      - update actual speeds toward v_cmd: x_dot += (v_cmd - x_dot) * dt/tau (x_dot = az_dot or el_dot, tau is the lag)
      - integrate angles with actual speeds
      - return (az, el, az_dot, el_dot, n̂)
    All angles in radians; velocities in rad/s; dt,tau in seconds.
    """
    # make sure the policy does not give speeds that are super fast and unrealistic
    v_az_cmd = float(np.clip(v_az_cmd, -v_max, v_max))
    v_el_cmd = float(np.clip(v_el_cmd, -v_max, v_max))

    # actuall speeds with first order lag
    az_dot = az_dot + (v_az_cmd - az_dot) * (dt / tau)
    el_dot = el_dot + (v_el_cmd - el_dot) * (dt / tau)

    # we wrap az around [-π, π]
    az = _wrap_pi(az + az_dot * dt)

    # clipping el to not allow it to go down from the horizon
    el = _clip_el(el + el_dot * dt)

    # optional: hard-limit actual speeds too
    az_dot = float(np.clip(az_dot, -v_max, v_max))
    el_dot = float(np.clip(el_dot, -v_max, v_max))

    # calculate the n vector of the dish from az and el
    n = compute_n_from_angles(az, el)

    return az, el, az_dot, el_dot, n

def compute_power(n, s, noise_std):
        """
        Power(P) we read from the sensor of the dish. In our model we calculate P as the inner product
        of the sun vector(s) and the dish vector(n), so the more alligned we are with the sun(so the sun vector)
        the more Power we get.
        """
        # calculate the inner product of n and s, P = n*s
        dot = n[0]*s[0] + n[1]*s[1] + n[2]*s[2]

        # make sure P is greater than 0
        P = max(0.0, float(dot))

        # take potential power sensor noise
        if noise_std and noise_std > 0.0:
            # P = P(without noise) + noise(follows N(0, noise_std))
            P += np.random.normal(0.0, float(noise_std))
            P = float(np.clip(P, 0.0, 1.0))
        return P

class DishSim:
    def __init__(self, lat_deg=40.64, lon_deg=22.94, dt=0.5, start_dt_utc=None):
        self.lat = float(lat_deg)
        self.lon = float(lon_deg)
        self.dt  = float(dt)
        self.start = start_dt_utc or datetime.now(timezone.utc)
        self.t = 0.0
    
    # the clock of the simulation
    def tick(self):
        self.t += self.dt
        return self.t
    
    # get the sun vector
    def sun_vec(self):
        sx, sy, sz = sun_vector(self.t, self.lat, self.lon, self.start)
        return (sx, sy, sz)



class SunDishEnv(gym.Env):
    """
    Creating a simple gym Env. 
    - actions: change the velocity's in azimouth and elavation normalized. So [v_az, v_el] in[-1,1]
    - obs: [az, el, az_dot, el_dot, P, dP]
    - reward: P (we want to find max P, aka center of the sun)
    """

    # initialize with arbitary values
    def __init__(self,
        lat_deg=40.64, lon_deg=22.94,
        dt=0.5,
        start_dt_utc=None,
        v_max_deg_s=10.0,
        tau=0.3,
        noise_std=0.01,
        horizon_s=90.0,
        seed=None):

        # initialize the parent class, in our case gym.Env
        super().__init__()

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # start the dish simulation(starts the clock)
        self.sim = DishSim(lat_deg=lat_deg, lon_deg=lon_deg, dt=dt, start_dt_utc=start_dt_utc)

        # initialize the dynamic parameters
        self.v_max = math.radians(v_max_deg_s)
        self.tau = float(tau)
        self.noise_std = float(noise_std)
        self.horizon_s = float(horizon_s)

        # state
        self.az = 0.0
        self.el = 0.0
        self.az_dot = 0.0
        self.el_dot = 0.0

        # P measurements
        self.P = 0.0
        self.last_P = 0.0

        # spaces determines a set of valid values for observations and actions
        # obs: [az, el, az_dot, el_dot, P, dP]  # noqa: ERA001
        # we say bound the observation elements by their respective bounds(ex. az is from -math.pi(π) up to math.pi(π))

        hi = np.array([ math.pi, math.pi/2, self.v_max, self.v_max, 1.0, 1.0 ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-hi, high=hi, dtype=np.float32)


        # action: normalized velocities in [-1,1] for v_az, v_el
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # elapsed episode time in seconds
        self.elapsed = 0.0

    # the observation(we also use dP as a secondary indicator to determine if we move to the center of the sun)
    def _obs(self):
        dP = self.P - self.last_P
        return np.array([self.az, self.el, self.az_dot, self.el_dot, self.P, dP], dtype=np.float32)
    
    # reset function to restart the simulation
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # reset the clock
        self.sim.start = self.sim.start 
        self.sim.t = 0.0
        self.elapsed = 0.0


        # initialize az, el with random values from a normal distribution to start(here we will put the coordinates from the science paper)
        # velocities are 0
        self.az = float(self.np_random.uniform(-math.pi, math.pi))
        self.el = float(self.np_random.uniform(math.radians(2.0), math.pi/2 - math.radians(2.0)))
        self.az_dot = 0.0
        self.el_dot = 0.0

        # get (x, y, z) coordinates from az, el
        n = compute_n_from_angles(self.az, self.el)

        # get sun vector from sim
        s = self.sim.sun_vec()

        self.P = self.sim.compute_power(n, s, noise_std = self.noise_std)
        self.last_P = self.P

        # return the observation with the values after reset
        return self._obs()

    def step(self, action):
        """
        Here is where our simulation actually takes our recomended action and runs the simulation and updates the state,
        gives the new power reading at the new position, advances time and gives ur our reward as feedback for the training.
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)             # enforce normalized range
        v_cmd_az = float(action[0]) * self.v_max        # [-1,1] -> [-v_max, +v_max]  # noqa: F841
        v_cmd_el = float(action[1]) * self.v_max  # noqa: F841

        self.az, self.el, self.az_dot, self.el_dot, self.n = step_with_lag(self.az, self.el, self.az_dot, 
        self.el_dot, self.v_az_cmd, self.v_el_cmd, self.dt, self.v_max, self.tau)


        # get sun vector, store previous P value and calculate new P value
        s = self.sim.sun_vec()
        self.last_P = self.P
        self.P = compute_power(self.n, s, self.noise_std)

        # advance simulation clock
        self.sim.tick()
        self.elapsed += self.sim.dt   # count how much time has passed

        # we use as reward P. So the rl algo is happy when P is bigger than last time
        reward = float(self.P)

        # give the agent horizon_s time (in seconds) to find the sun(currently set to 90s)
        terminated = False # needed for gymnasium
        truncated  = bool(self.elapsed >= self.horizon_s)  # cut off by time horizon

        return self._obs(), reward, terminated, truncated, {}
