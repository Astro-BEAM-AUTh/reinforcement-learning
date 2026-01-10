"""Gaussian beam sun tracking environment with finite sun disk model."""
import math
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy import integrate

from sun_tracking.sky_objects.sun_model import sun_vector

# ==================== PHYSICAL CONSTANTS ====================

BEAMWIDTH_FWHM_DEG = 2.43  # Full-Width Half-Maximum beamwidth of dish antenna
SUN_DIAMETER_DEG = 0.53    # Angular diameter of the sun
SUN_RADIUS_DEG = 0.265     # Sun radius (half diameter)
SIGMA_DEG = 1.03           # Gaussian width parameter (FWHM/2.355)

# Reward shaping parameters
REWARD_POWER_IMPROVEMENT_SCALE = 1000.0  # Scale for power improvement reward


# ==================== HELPER FUNCTIONS ====================

def _wrap_pi(a):
    """Wrap angle to [-π, π] range."""
    return ((a + math.pi) % (2.0 * math.pi)) - math.pi


def _clip_el(el) -> float:
    """Clip elevation angle to [0, π/2] range (horizon to zenith)."""
    return float(np.clip(el, 0.0, math.pi / 2.0))


def compute_n_from_angles(az, el):
    """Convert azimuth and elevation angles to unit vector in ENU coordinates."""
    x = math.cos(el) * math.cos(az)  # East
    y = math.cos(el) * math.sin(az)  # North
    z = math.sin(el)                 # Up
    return (x, y, z)  # |n̂| = 1


def compute_angular_offset(dish_vec, sun_vec):
    """
    Compute 2D angular offset (d_x, d_y) between dish pointing and sun position.
    
    Uses small angle approximation for angles < 5 degrees.
    Returns offset in degrees.
    
    Args:
        dish_vec: (x, y, z) unit vector of dish pointing direction
        sun_vec: (x, y, z) unit vector of sun position
        
    Returns:
        (d_x, d_y): Angular offset in degrees (East, North components)
    """
    # Calculate angular separation using dot product
    dot = dish_vec[0]*sun_vec[0] + dish_vec[1]*sun_vec[1] + dish_vec[2]*sun_vec[2]
    dot = np.clip(dot, -1.0, 1.0)
    theta_rad = math.acos(dot)
    
    # For small angles, compute 2D offset using cross product
    # This gives us the direction of the offset
    if theta_rad < 1e-6:  # Perfectly aligned
        return 0.0, 0.0
    
    # Cross product gives perpendicular direction
    cross_x = dish_vec[1]*sun_vec[2] - dish_vec[2]*sun_vec[1]
    cross_y = dish_vec[2]*sun_vec[0] - dish_vec[0]*sun_vec[2]
    
    # Normalize and scale by angular separation
    cross_norm = math.sqrt(cross_x**2 + cross_y**2)
    if cross_norm < 1e-9:
        # Vectors are aligned or anti-aligned, offset in arbitrary direction
        return theta_rad * 180.0 / math.pi, 0.0
    
    d_x = (cross_x / cross_norm) * theta_rad * 180.0 / math.pi
    d_y = (cross_y / cross_norm) * theta_rad * 180.0 / math.pi
    
    return d_x, d_y


def compute_gaussian_power(d_x, d_y, sigma_deg=SIGMA_DEG, r_sun_deg=SUN_RADIUS_DEG, noise_std=0.0, rng=None):
    """
    Compute power by integrating Gaussian beam gain over the sun disk.
    
    The beam center is at (0, 0) and sun center is at (d_x, d_y).
    We integrate the Gaussian pattern over the circular sun disk.
    
    Args:
        d_x, d_y: Offset of sun center from beam center (degrees)
        sigma_deg: Gaussian beam width parameter (degrees)
        r_sun_deg: Sun radius (degrees)
        noise_std: Sensor noise standard deviation
        rng: Random number generator for noise
        
    Returns:
        Power value normalized to [0, 1]
    """
    # Convert to radians for numerical stability
    sigma = math.radians(sigma_deg)
    r_sun = math.radians(r_sun_deg)
    d_x_rad = math.radians(d_x)
    d_y_rad = math.radians(d_y)
    
    # Define the Gaussian beam pattern in polar coordinates centered on sun
    # Point (r, phi) on sun disk in beam-centered frame is:
    # x = d_x + r*cos(phi), y = d_y + r*sin(phi)
    # Distance from beam center: sqrt(x^2 + y^2)
    # Gaussian gain: exp(-distance^2 / (2*sigma^2))
    
    def integrand(phi, r):
        """Gaussian gain at point (r, phi) on sun disk."""
        x = d_x_rad + r * math.cos(phi)
        y = d_y_rad + r * math.sin(phi)
        dist_sq = x**2 + y**2
        gain = math.exp(-dist_sq / (2.0 * sigma**2))
        return gain * r  # r is the Jacobian for polar coordinates
    
    # Integrate over sun disk: phi from 0 to 2π, r from 0 to r_sun
    try:
        result, error = integrate.dblquad(
            integrand,
            0.0,          # r min
            r_sun,        # r max
            0.0,          # phi min
            2.0 * math.pi # phi max
        )
    except Exception as e:
        # If integration fails, fall back to simple Gaussian
        print(f"Integration failed: {e}")
        d = math.sqrt(d_x_rad**2 + d_y_rad**2)
        result = math.pi * r_sun**2 * math.exp(-d**2 / (2.0 * sigma**2))
    
    # Normalize by the maximum possible power (perfect alignment)
    # Max power is when d_x = d_y = 0
    max_power = math.pi * r_sun**2  # Area of sun disk * max gain (1.0)
    
    power = result / max_power
    power = float(np.clip(power, 0.0, 1.0))
    
    # Add sensor noise if requested
    if noise_std > 0.0:
        if rng is None:
            noise = np.random.normal(0.0, float(noise_std))
        else:
            noise = rng.normal(0.0, float(noise_std))
        power += noise
        power = float(np.clip(power, 0.0, 1.0))
    
    return power


def step_with_lag(az, el, az_dot, el_dot, v_az_cmd, v_el_cmd, dt, v_max, tau):
    """
    First-order motor lag simulation.
    
    Args:
        az, el: Current angles (radians)
        az_dot, el_dot: Current angular velocities (rad/s)
        v_az_cmd, v_el_cmd: Commanded velocities (rad/s)
        dt: Time step (seconds)
        v_max: Maximum velocity (rad/s)
        tau: Time constant for first-order lag (seconds)
        
    Returns:
        Updated (az, el, az_dot, el_dot, n) where n is the dish unit vector
    """
    # Clamp commanded velocities
    v_az_cmd = float(np.clip(v_az_cmd, -v_max, v_max))
    v_el_cmd = float(np.clip(v_el_cmd, -v_max, v_max))
    
    # First-order lag: velocity moves toward commanded value
    # Clamp alpha to [0, 1] for numerical stability when dt/tau > 1
    alpha = min(1.0, dt / tau)
    az_dot = az_dot + (v_az_cmd - az_dot) * alpha
    el_dot = el_dot + (v_el_cmd - el_dot) * alpha
    
    # Integrate angles
    az = _wrap_pi(az + az_dot * dt)
    el = _clip_el(el + el_dot * dt)
    
    # Hard-limit actual speeds
    az_dot = float(np.clip(az_dot, -v_max, v_max))
    el_dot = float(np.clip(el_dot, -v_max, v_max))
    
    # Compute dish direction vector
    n = compute_n_from_angles(az, el)
    
    return az, el, az_dot, el_dot, n


# ==================== SIMULATION CLASS ====================

class DishSim:
    """
    Simulate sun position over time at a given geographic location.
    
    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        dt: Time step in seconds
        start_dt_utc: Starting datetime (UTC timezone-aware)
    """
    def __init__(self, lat_deg=40.64, lon_deg=22.94, dt=0.5, start_dt_utc=None):
        self.lat = float(lat_deg)
        self.lon = float(lon_deg)
        self.dt = float(dt)
        self.start = start_dt_utc or datetime.now(timezone.utc)
        self.t = 0.0
    
    def tick(self):
        """Advance simulation time by dt."""
        self.t += self.dt
        return self.t
    
    def sun_vec(self):
        """Get current sun position as ENU unit vector."""
        sx, sy, sz = sun_vector(self.t, self.lat, self.lon, self.start)
        return (sx, sy, sz)


# ==================== GYMNASIUM ENVIRONMENT ====================

class GaussianBeamDishEnv(gym.Env):
    """
    Gymnasium environment for sun tracking with Gaussian beam model.
    
    The dish emits a Gaussian beam pattern and the sun is modeled as a finite disk.
    Power is computed by integrating the beam gain over the sun disk.
    
    Observation space: [az, el, az_dot, el_dot, P, dP]
    - az: azimuth angle (radians)
    - el: elevation angle (radians)  
    - az_dot: azimuth angular velocity (rad/s)
    - el_dot: elevation angular velocity (rad/s)
    - P: current power measurement [0, 1]
    - dP: change in power from last step
    
    Action space: [v_az, v_el] in [-1, 1]
    - Normalized commanded angular velocities for azimuth and elevation
    
    Reward: Power (maximize alignment between beam and sun)
    """
    
    def __init__(
        self,
        lat_deg=40.64,
        lon_deg=22.94,
        dt=0.5,
        start_dt_utc=None,
        v_max_deg_s=2.0,  # FIX 4: Reduced from 10 for smoother control
        tau=0.3,
        noise_std=0.01,
        horizon_s=90.0,
        seed=None,
    ):
        super().__init__()
        
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Initialize simulation
        self.sim = DishSim(lat_deg=lat_deg, lon_deg=lon_deg, dt=dt, start_dt_utc=start_dt_utc)
        
        # Dynamic parameters
        self.v_max = math.radians(v_max_deg_s)
        self.tau = float(tau)
        self.noise_std = float(noise_std)
        self.horizon_s = float(horizon_s)
        
        # State variables
        self.az = 0.0
        self.el = 0.0
        self.az_dot = 0.0
        self.el_dot = 0.0
        
        # Power measurements
        self.P = 0.0
        self.last_P = 0.0
        
        # Observation space: [az, el, az_dot, el_dot, P, dP]
        hi = np.array([math.pi, math.pi/2, self.v_max, self.v_max, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-hi, high=hi, dtype=np.float32)
        
        # Action space: normalized velocities in [-1, 1] for v_az, v_el
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Episode time tracking
        self.elapsed = 0.0
    
    def _obs(self):
        """Construct observation vector."""
        dP = self.P - self.last_P
        return np.array([self.az, self.el, self.az_dot, self.el_dot, self.P, dP], dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Reset time
        self.sim.t = 0.0
        self.elapsed = 0.0
        
        # Randomize time within a day until sun is above horizon
        for _ in range(24):
            self.sim.t = float(self.np_random.uniform(0.0, 24.0 * 3600.0))
            s_try = self.sim.sun_vec()
            if s_try[2] > 0.0:  # Sun above horizon
                break
        
        # Get sun position
        s = self.sim.sun_vec()
        sun_az = math.atan2(s[1], s[0])
        sun_el = math.asin(s[2])
        
        # Start at exactly 3 degrees from sun in random direction
        # (ensures consistent difficulty and good gradients)
        target_offset_deg = 3.0
        angle = self.np_random.uniform(0, 2 * math.pi)  # Random direction
        offset_az = target_offset_deg * math.cos(angle)
        offset_el = target_offset_deg * math.sin(angle)
        
        self.az = sun_az + math.radians(offset_az)
        self.el = sun_el + math.radians(offset_el)
        
        # Clip elevation to valid range [0, π/2]
        self.el = _clip_el(self.el)
        # Wrap azimuth to [-π, π]
        self.az = _wrap_pi(self.az)
        
        self.az_dot = 0.0
        self.el_dot = 0.0
        
        # Get initial dish direction and sun position
        n = compute_n_from_angles(self.az, self.el)
        s = self.sim.sun_vec()
        
        # Calculate initial power using Gaussian model
        d_x, d_y = compute_angular_offset(n, s)
        self.P = compute_gaussian_power(d_x, d_y, noise_std=self.noise_std, rng=self.np_random)
        self.last_P = self.P
        
        return self._obs(), {"power": float(self.P)}
    
    def step(self, action):
        """
        Execute one simulation step.
        
        Args:
            action: [v_az, v_el] normalized commanded velocities
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action to commanded speeds
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        v_cmd_az = float(action[0]) * self.v_max
        v_cmd_el = float(action[1]) * self.v_max
        
        # Update dish dynamics
        self.az, self.el, self.az_dot, self.el_dot, n = step_with_lag(
            self.az, self.el, self.az_dot, self.el_dot,
            v_cmd_az, v_cmd_el,
            self.sim.dt, self.v_max, self.tau
        )
        
        # Advance simulation time
        self.sim.tick()
        self.elapsed += self.sim.dt
        
        # Get new sun position
        s = self.sim.sun_vec()
        
        # Calculate angular offset and power using Gaussian model
        d_x, d_y = compute_angular_offset(n, s)
        self.last_P = self.P
        self.P = compute_gaussian_power(d_x, d_y, noise_std=self.noise_std, rng=self.np_random)
        
        # FIX 2: Compute absolute power reward
        # Using P directly incentivizes staying at high power (sustainable tracking)
        angular_distance = math.sqrt(d_x**2 + d_y**2)
        power_improvement = float(self.P - self.last_P)
        
        reward = float(self.P)
        
        # DEBUG: Print first few steps (remove after verification)
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        if self._debug_step_count <= 3:
            print(f"[DEBUG Step {self._debug_step_count}] P={self.P:.6f}, reward={reward:.6f}")
        
        # Episode termination
        terminated = False
        truncated = bool(self.elapsed >= self.horizon_s)
        
        info = {
            "power": float(self.P),
            "angular_distance": float(angular_distance),
            "power_improvement": float(power_improvement),
        }
        
        return self._obs(), reward, terminated, truncated, info
