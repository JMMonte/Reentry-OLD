from astropy import units as u
from astropy.time import Time
from coordinate_converter import (enu_to_ecef, ecef_to_eci, geodetic_to_spheroid, eci_to_ecef)
import numpy as np
from scipy.integrate import solve_ivp


import time
from numba import jit, njit
# Constants
# ---------
# Operations
PI = np.pi
DEG_TO_RAD = float(PI / 180.0) # degrees to radians
RAD_TO_DEG = float(180.0 / PI) # radians to degrees
JD_AT_0 = 2451545.0 # Julian date at 0 Jan 2000

#Earth physical constants
EARTH_MU = 3.986004418e14  # gravitational parameter of Earth in m^3/s^2
EARTH_MU_KM = 3.986004418e5  # gravitational parameter of Earth in km^3/s^2
EARTH_R = 6378137.0  # radius of Earth in m
EARTH_R_KM = 6378.137  # radius of Earth in m
EARTH_R_POLAR = 6356752.3142  # polar radius of Earth in m
EARTH_OMEGA = 7.292114146686322e-05  # Earth rotation speed in rad/s
EARTH_J2 = 0.00108263 # J2 of Earth (dimensionless)
EARTH_MASS = 5.972e24  # Mass (kg)
YEAR_D = 365.25636 # days in a year
YEAR_S = 365.25636 * 24 * 60 * 60  # seconds in a year
EARTH_ROT_S = 86164.0905  # Earth rotation period in seconds
EARTH_GRAVITY = 9.80665  # Gravity (m/s^2)
# Derived physical constants
EARTH_ROTATION_RATE_DEG_PER_SEC = 360 / EARTH_ROT_S  # Earth rotation rate in degrees per second
EARTH_RORATION_RATE_RAD_PER_SEC = EARTH_ROTATION_RATE_DEG_PER_SEC * DEG_TO_RAD  # Earth rotation rate in radians per second
EARTH_PERIMETER = EARTH_R * 2 * PI  # Earth perimeter in meters
EARTH_ROTATION_RATE_M_S = EARTH_PERIMETER / EARTH_ROT_S  # Earth rotation rate in meters per second

# Moon
MOON_A = 384400000.0  # Semi-major axis (meters)
MOON_E = 0.0549  # Eccentricity
MOON_I = 5.145 * DEG_TO_RAD  # Inclination (radians)
MOON_OMEGA = 125.045 * DEG_TO_RAD  # Longitude of ascending node (radians)
MOON_W = 318.0634 * DEG_TO_RAD  # Argument of perigee (radians)
MOON_M0 = 115.3654 * DEG_TO_RAD  # Mean anomaly at epoch J2000 (radians)
MOON_MASS = 7.34767309e22  # Mass (kg)
MOON_ROT_D = 27.321661  # Rotation period in days
MOON_ROT_S = MOON_ROT_D * 24 * 3600  # Rotation period in seconds
MOON_MMOTION_DEG =  360 / MOON_ROT_D # Mean motion (degrees/day)
MOON_MMOTION_RAD = MOON_MMOTION_DEG * DEG_TO_RAD  # Mean motion (radians/day)

#Sun
SUN_MU = 132712442099.00002 # gravitational parameter of Sun in km^3/s^2
SUN_A = 149598022990.63  # Semi-major axis (meters)
SUN_E = 0.01670862  # Eccentricity
SUN_I = 0.00005 * DEG_TO_RAD  # Inclination (radians)
SUN_OMEGA = -11.26064 * DEG_TO_RAD  # Longitude of ascending node (radians)
SUN_W = 102.94719 * DEG_TO_RAD  # Argument of perigee (radians)
SUN_L0 = 100.46435 * DEG_TO_RAD  # Mean longitude at epoch J2000 (radians)
SUN_MASS = 1.988544e30  # Mass (kg)

# Atmosphere
# U.S. Standard Atmosphere altitude breakpoints and temperature gradients (m, K/m)
ALTITUDE_BREAKPOINTS = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
TEMPERATURE_GRADIENTS = np.array([-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002])
# U.S. Standard Atmosphere base temperatures and pressures at altitude breakpoints (K, Pa)
BASE_TEMPERATURES = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
BASE_PRESSURES = np.array([101325, 22632.1, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734])
EARTH_AIR_MOLAR_MASS = 0.0289644 # molar mass of Earth's air (kg/mol)
EARTH_GAS_CONSTANT = 8.31447 # Gas Constant Values based on Energy Units ; J · 8.31447, 3771.38
R_GAS = 287.0  # J/kgK for air; This value is appropriate for air if Joule is chosen for the unit of energy, kg as unit of mass and K as unit of temperature, i.e. $ R = 287 \;$   J$ /($kg$ \;$K$ )$
SCALE_HEIGHT = 7500.0  # Scale height (m)

# numba functions
# ----------------
@njit
def euclidean_norm(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

@jit(nopython=True)
def earth_rotational_velocity(omega, r):
    omega_cross_r = np.cross(np.array([0, 0, omega]), r)
    return omega_cross_r

@jit(nopython=True)
def atmosphere_model(altitude):
    if altitude > 1e6:
        return 0, 0
    elif altitude <= 0:
        return 1.225, 288.15
    else:
        i = np.searchsorted(ALTITUDE_BREAKPOINTS, altitude, side='right') - 1
        delta_altitude = altitude - ALTITUDE_BREAKPOINTS[i]
        T = BASE_TEMPERATURES[i] + TEMPERATURE_GRADIENTS[i] * delta_altitude

        if TEMPERATURE_GRADIENTS[i] == 0:
            P = BASE_PRESSURES[i] * np.exp(-EARTH_GRAVITY * EARTH_AIR_MOLAR_MASS * delta_altitude / (EARTH_GAS_CONSTANT * BASE_TEMPERATURES[i]))
        else:
            P = BASE_PRESSURES[i] * (T / BASE_TEMPERATURES[i]) ** (-EARTH_GRAVITY * EARTH_AIR_MOLAR_MASS / (EARTH_GAS_CONSTANT * TEMPERATURE_GRADIENTS[i]))

        if i == len(ALTITUDE_BREAKPOINTS) - 1:
            P *= np.exp(-(altitude - ALTITUDE_BREAKPOINTS[-1]) / SCALE_HEIGHT)

        rho = P / (R_GAS * T)

        return rho, T

@jit(nopython=True)
def atmospheric_drag(Cd, A, r_ecef, v_ecef, x0=100000, k=0.0001):
    r_norm = euclidean_norm(r_ecef)
    v_norm = euclidean_norm(v_ecef)
    v_unit = np.zeros(3)
    if v_norm != 0:
        v_unit = v_ecef / v_norm

    altitude = r_norm - EARTH_R
    rho, _ = atmosphere_model(altitude)

    # Apply logistic function to altitude
    smooth_factor = 1 / (1 + np.exp(-k * (altitude - x0)))
    a_drag_ecef = -0.5 * Cd * A * rho * v_norm ** 2 * v_unit * smooth_factor

    return a_drag_ecef

@jit(nopython=True)
def thermal_power(altitude, velocity_vec, A=10.5, C_D=1.5):
    rho, _ = atmosphere_model(altitude)
    v = euclidean_norm(velocity_vec)
    Q_dot = 0.5 * C_D * rho * v**3 * A
    return Q_dot

@jit(nopython=True)
def moon_position_vector(jd):
    # Time since J2000 (in days)
    t = jd - JD_AT_0 # 2451545.0 is the Julian date for J2000

    # Compute mean anomaly
    M = MOON_M0 + MOON_MMOTION_RAD * t

    # Solve Kepler's equation for eccentric anomaly (E) using Newton-Raphson method
    E = M
    for _ in range(10):  # Iterate a few times to get an accurate solution
        E = E - (E - MOON_E * np.sin(E) - M) / (1 - MOON_E * np.cos(E))

    # Compute true anomaly (f)
    f = 2 * np.arctan2(np.sqrt(1 + MOON_E) * np.sin(E / 2), np.sqrt(1 - MOON_E) * np.cos(E / 2))

    # Compute heliocentric distance (r)
    r = MOON_A * (1 - MOON_E * np.cos(E))

    # Compute position vector in the orbital plane
    x_prime = r * np.cos(f)
    y_prime = r * np.sin(f)

    # Rotate position vector to the ecliptic coordinate system
    x = (x_prime * (np.cos(MOON_OMEGA) * np.cos(MOON_W) - np.sin(MOON_OMEGA) * np.sin(MOON_W) * np.cos(MOON_I))
         - y_prime * (np.sin(MOON_OMEGA) * np.cos(MOON_W) + np.cos(MOON_OMEGA) * np.sin(MOON_W) * np.cos(MOON_I)))
    y = (x_prime * (np.sin(MOON_OMEGA) * np.cos(MOON_W) + np.cos(MOON_OMEGA) * np.sin(MOON_W) * np.cos(MOON_I))
         + y_prime * (np.cos(MOON_OMEGA) * np.cos(MOON_W) - np.sin(MOON_OMEGA) * np.sin(MOON_W) * np.cos(MOON_I)))
    z = x_prime * np.sin(MOON_W) * np.sin(MOON_I) + y_prime * np.cos(MOON_W) * np.sin(MOON_I)

    return np.array([x, y, z])

@jit(nopython=True)
def sun_position_vector(jd):
    # Time since J2000 (in days)
    t = jd - JD_AT_0

    # Compute mean anomaly
    n = 2 * PI / YEAR_D  # Mean motion (radians per day)
    L = SUN_L0 + n * t
    M = L - SUN_OMEGA - SUN_W

    # Solve Kepler's equation for eccentric anomaly (E) using Newton-Raphson method
    E = M
    for _ in range(10):  # Iterate a few times to get an accurate solution
        E = E - (E - SUN_E * np.sin(E) - M) / (1 - SUN_E * np.cos(E))

    # Compute true anomaly (f)
    f = 2 * np.arctan2(np.sqrt(1 + SUN_E) * np.sin(E / 2), np.sqrt(1 - SUN_E) * np.cos(E / 2))

    # Compute heliocentric distance (r)
    r = SUN_A * (1 - SUN_E * np.cos(E))

    # Compute position vector in the orbital plane
    x_prime = r * np.cos(f)
    y_prime = r * np.sin(f)

    # Rotate position vector to the ecliptic coordinate system
    x = (x_prime * (np.cos(SUN_OMEGA) * np.cos(SUN_W) - np.sin(SUN_OMEGA) * np.sin(SUN_W) * np.cos(SUN_I))
         - y_prime * (np.sin(SUN_OMEGA) * np.cos(SUN_W) + np.cos(SUN_OMEGA) * np.sin(SUN_W) * np.cos(SUN_I)))
    y = (x_prime * (np.sin(SUN_OMEGA) * np.cos(SUN_W) + np.cos(SUN_OMEGA) * np.sin(SUN_W) * np.cos(SUN_I))
         + y_prime * (np.cos(SUN_OMEGA) * np.cos(SUN_W) - np.sin(SUN_OMEGA) * np.sin(SUN_W) * np.cos(SUN_I)))
    z = x_prime * np.sin(SUN_W) * np.sin(SUN_I) + y_prime * np.cos(SUN_W) * np.sin(SUN_I)

    return np.array([x, y, z])

@jit(nopython=True)
def third_body_acceleration(satellite_position, third_body_position, satellite_mass, third_body_mass):
    # Vector from the satellite to the third body
    r_satellite_to_third_body = third_body_position - satellite_position

    # Gravitational constant
    G = 6.67430e-11  # m^3 kg^-1 s^-2

    # Acceleration due to the third body
    a_third_body = G * third_body_mass * r_satellite_to_third_body / euclidean_norm(r_satellite_to_third_body)**3

    # Acceleration that Earth would experience due to the third body
    a_earth = G * third_body_mass * third_body_position / euclidean_norm(third_body_position)**3

    # Perturbation acceleration acting on the satellite
    a_perturbation = a_third_body - a_earth

    return a_perturbation / satellite_mass

@jit(nopython=True)
def J2_perturbation_numba(r, k, J2, R):
    x, y, z = r[0], r[1], r[2]
    r_vec = np.array([x, y, z])
    r = euclidean_norm(r_vec)

    factor = (3.0 / 2.0) * k * J2 * (R**2) / (r**5)

    a_x = 5.0 * z ** 2 / r**2 - 1
    a_y = 5.0 * z ** 2 / r**2 - 1
    a_z = 5.0 * z ** 2 / r**2 - 3
    return np.array([a_x, a_y, a_z]) * r_vec * factor

# ----------------

class SpacecraftModel:
    def __init__(self, Cd=2.2, Cp=500.0, A=20.0, m=500.0, epoch=Time('2024-01-01 00:00:00'), gmst0=0.0, sim_type='RK45'):
        self.Cd = Cd  # drag coefficient
        self.A = A  # cross-sectional area of spacecraft in m^2
        self.m = m  # mass of spacecraft in kg
        self.epoch = epoch # 
        self.start_time = time.time() # start time of simulation
        self.A_over_m = (self.A) / self.m # A/m
        self.Cp = Cp  # Example heat transfer coefficient (W/m²·K)
        self.gmst0 = gmst0 # Greenwich Mean Sidereal Time at epoch (degrees)
        self.sim_type = sim_type

    def get_initial_state(self, v, lat, lon, alt, azimuth, gamma, gmst=0.0):
        # Convert geodetic to ECEF
        x_ecef, y_ecef, z_ecef = geodetic_to_spheroid(lat, lon, alt)
        r_ecef = np.array([x_ecef, y_ecef, z_ecef])

        # Convert velocity from polar to horizontal ENU coordinates
        azimuth_rad = DEG_TO_RAD * azimuth
        gamma_rad = DEG_TO_RAD * gamma
        v_east = v * np.sin(azimuth_rad) * np.cos(gamma_rad)
        v_north = v * np.cos(azimuth_rad) * np.cos(gamma_rad)
        v_up = v * np.sin(gamma_rad)
        v_enu = np.array([v_east, v_north, v_up])

        # Convert ENU to ECEF
        v_ecef = enu_to_ecef(v_enu, lat, lon)

        # Convert position and velocity from ECEF to ECI
        r_eci = ecef_to_eci(r_ecef, gmst)
        v_eci = ecef_to_eci(v_ecef, gmst)

        # Combine position and velocity to create the initial state vector
        y0 = np.concatenate((r_eci, v_eci))

        return y0

    def equations_of_motion(self, t, y):
        r_eci, v_eci = y[0:3], y[3:6]
        r_norm = euclidean_norm(r_eci)

        # Calculate Greenwich Mean Sidereal Time at every time step
        epoch = self.epoch + t * u.s
        gmst = self.gmst0 + EARTH_OMEGA * t

        # Calculate ECEF position and ground velocity
        r_ecef = eci_to_ecef(r_eci, gmst)
        v_ground = eci_to_ecef(v_eci, gmst)
        v_rel = v_ground - np.array([-EARTH_OMEGA * r_ecef[1], EARTH_OMEGA * r_ecef[0], 0])

        # Calculate accelerations
        a_grav = -EARTH_MU * r_eci / (r_norm ** 3)
        a_J2 = J2_perturbation_numba(r_eci, k=EARTH_MU, J2=EARTH_J2, R=EARTH_R)
        moon_r = moon_position_vector(epoch.jd)
        sun_r = sun_position_vector(epoch.jd)
        a_moon = third_body_acceleration(r_eci, moon_r, self.m, MOON_MASS)
        a_sun = third_body_acceleration(r_eci, sun_r, self.m, SUN_MASS)

        # Calculate drag acceleration
        a_drag_ecef = atmospheric_drag(Cd=self.Cd, A=self.A, r_ecef=r_ecef, v_ecef=v_rel)
        a_drag = ecef_to_eci(a_drag_ecef, gmst)

        # Calculate heat rate
        altitude = r_norm - EARTH_R
        q = thermal_power(altitude=altitude, velocity_vec=v_rel, A=self.A, C_D=self.Cp)

        # Calculate total acceleration
        a_total = a_grav + a_J2 + a_moon + a_sun + a_drag

        return {
            'velocity': v_eci,
            'acceleration': a_total,
            'gravitational_acceleration': a_grav,
            'J2_acceleration': a_J2,
            'moon_acceleration': a_moon,
            'drag_acceleration': a_drag,
            'altitude': altitude,
            'sun_acceleration': a_sun,
            'heat_rate': q,
        }

    
    def run_simulation(self, t_span, y0, t_eval, progress_callback=None):
        def rhs(t, y):
            dy = self.equations_of_motion(t, y)
            return np.concatenate((dy['velocity'], dy['acceleration']))
        
        def altitude_event(t, y):
            r = y[0:3]
            r_norm = euclidean_norm(r)
            altitude = r_norm - EARTH_R
            # Add a tolerance value to avoid triggering the event too close to the atmospheric boundary
            return altitude - 1000.0
        
        altitude_event.terminal = True
        altitude_event.direction = -1

        def progress_event(t, y):
            if progress_callback is not None:
                progress = min((t - t_span[0]) / (t_span[1] - t_span[0]), 1.0)  # Make sure progress doesn't exceed 1.0
                elapsed_time = time.time() - self.start_time
                progress_callback(progress, elapsed_time)
            return 0

        progress_event.terminal = False
        progress_event.direction = 0

        sol = solve_ivp(rhs, t_span, y0, method=self.sim_type, t_eval=t_eval, rtol=1e-8, atol=1e-10, events=[altitude_event, progress_event])
        additional_data_list = [self.equations_of_motion(t, y) for t, y in zip(sol.t, sol.y.T)]
        sol.additional_data = {key: np.array([d[key] for d in additional_data_list]) for key in additional_data_list[0].keys()}
        return sol
    
