from astropy import units as u
from astropy.time import Time
from coordinate_converter import CoordinateConverter
import numpy as np
from poliastro.core.perturbations import J2_perturbation
from poliastro.ephem import Ephem
from scipy.integrate import solve_ivp
import time
import numba
from numba import jit, njit
import math
# Constants
# ---------
# Operations
DEG_TO_RAD = np.pi / 180.0 # degrees to radians
UA_TO_M = 149597870700 # 1 astronomical unit in meters

#Earth
EARTH_MU = 3.986004418e14  # gravitational parameter of Earth in m^3/s^2
EARTH_MU_KM = 3.986004418e5  # gravitational parameter of Earth in km^3/s^2
EARTH_R = 6378137.0  # radius of Earth in m
EARTH_R_KM = 6378.137  # radius of Earth in m
EARTH_R_POLAR = 6356752.3142  # polar radius of Earth in m
EARTH_OMEGA = 7.292114146686322e-05  # Earth rotation speed in rad/s
EARTH_J2 = 0.00108263 # J2 of Earth
EARTH_MASS = 5.972e24  # Mass (kg)

# Moon
MOON_A = 384400000.0  # Semi-major axis (meters)
MOON_E = 0.0549  # Eccentricity
MOON_I = 5.145 * DEG_TO_RAD  # Inclination (radians)
MOON_OMEGA = 125.045 * DEG_TO_RAD  # Longitude of ascending node (radians)
MOON_W = 318.0634 * DEG_TO_RAD  # Argument of perigee (radians)
MOON_M0 = 115.3654 * DEG_TO_RAD  # Mean anomaly at epoch J2000 (radians)
MOON_MASS = 7.34767309e22  # Mass (kg)

#Sun
SUN_MU = 132712442099.00002 # gravitational parameter of Sun in km^3/s^2
SUN_A = 149598022990.63  # Semi-major axis (meters)
SUN_E = 0.01670862  # Eccentricity
SUN_I = 0.00005 * DEG_TO_RAD  # Inclination (radians)
SUN_OMEGA = -11.26064 * DEG_TO_RAD  # Longitude of ascending node (radians)
SUN_W = 102.94719 * DEG_TO_RAD  # Argument of perigee (radians)
SUN_L0 = 100.46435 * DEG_TO_RAD  # Mean longitude at epoch J2000 (radians)
SUN_MASS = 1.988544e30  # Mass (kg)


# numba functions
# ----------------
@jit(nopython=True)
def earth_rotational_velocity(omega, r):
    omega_cross_r = np.cross(np.array([0, 0, omega]), r)
    return omega_cross_r


@jit(nopython=True)
def atmospheric_drag(Cd, A_over_m, rho0, H0, state, omega):
    r, v = state[0:3], state[3:6]
    r_norm = np.linalg.norm(r)
    v_relative = v - earth_rotational_velocity(omega, r)
    v_norm = np.linalg.norm(v_relative)
    v_unit = v_relative / v_norm
    altitude = r_norm - 6378137.0
    rho = rho0 * np.exp(-altitude / H0)
    a_drag = -0.5 * Cd * A_over_m * rho * v_norm ** 2 * v_unit

    return a_drag, rho

@njit
def euclidean_norm(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


@jit(nopython=True)
def estimate_reentry_heating(velocity_eci, rho, Cp, A):
    V = euclidean_norm(velocity_eci) # m/s
    q = 0.5 * rho * V**3 * Cp * A # W/m^2
    return q

@jit(nopython=True)
def moon_position_vector(jd):
    # Time since J2000 (in days)
    t = jd - 2451545.0

    # Compute mean anomaly
    n = 2 * np.pi / 27.321582  # Mean motion (radians per day)
    M = MOON_M0 + n * t

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
    t = jd - 2451545.0

    # Compute mean anomaly
    n = 2 * np.pi / 365.25636  # Mean motion (radians per day)
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

# ----------------

class SpacecraftModel:
    def __init__(self, Cd=2.2, Cp=500.0, A=20.0, m=500.0, epoch=Time('2024-01-01 00:00:00')):
        self.Cd = Cd  # drag coefficient
        self.A = A  # cross-sectional area of spacecraft in m^2
        self.m = m  # mass of spacecraft in kg
        self.epoch = epoch # 
        self.start_time = time.time() # start time of simulation
        self.A_over_m = (self.A) / self.m # A/m
        self.rho0 = 1.3 # Example density at sea level (kg/m³)
        self.H0 = 8500 # Example scale height (m)
        self.Cp = Cp  # Example heat transfer coefficient (W/m²·K)

    def get_initial_state(self, v, lat, lon, alt, azimuth, gamma, attractor_R, attractor_R_Polar, gmst=0):
        lat_rad, lon_rad, azimuth_rad, gamma_rad = np.radians([lat, lon, azimuth, gamma])
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
        sin_azimuth, cos_azimuth = np.sin(azimuth_rad), np.cos(azimuth_rad)
        sin_gamma, cos_gamma = np.sin(gamma_rad), np.cos(gamma_rad)

        x_ecef, y_ecef, z_ecef = CoordinateConverter.geo_to_spheroid(lat, lon, alt,attractor_R, attractor_R_Polar)

        r_eci = CoordinateConverter.ecef_to_eci(x_ecef, y_ecef, z_ecef, gmst)

        sin_azimuth, cos_azimuth = np.sin(azimuth_rad), np.cos(azimuth_rad)
        sin_gamma, cos_gamma = np.sin(gamma_rad), np.cos(gamma_rad)

        v_east = v * sin_azimuth * cos_gamma
        v_north = v * cos_azimuth * cos_gamma
        v_up = v * sin_gamma

        v_x_ecef = -v_east * sin_lon - v_north * sin_lat * cos_lon + v_up * cos_lat * cos_lon
        v_y_ecef = v_east * cos_lon - v_north * sin_lat * sin_lon + v_up * cos_lat * sin_lon
        v_z_ecef = v_north * cos_lat + v_up * sin_lat

        v_x_rot = -EARTH_OMEGA * y_ecef
        v_y_rot = EARTH_OMEGA * x_ecef

        v_x_ecef_total = v_x_ecef + v_x_rot
        v_y_ecef_total = v_y_ecef + v_y_rot
        v_z_ecef_total = v_z_ecef

        v_eci = CoordinateConverter.ecef_to_eci(v_x_ecef_total, v_y_ecef_total, v_z_ecef_total, gmst)
        v_eci -= earth_rotational_velocity(7.292114146686322e-05,r_eci)

        y0 = np.concatenate((r_eci, v_eci))

        return y0

    def equations_of_motion(self, t, y):
        r, v = y[0:3], y[3:6]

        # Precompute r_norm and other reused values
        r_norm = np.linalg.norm(r)

        a_grav = -EARTH_MU * r / (r_norm ** 3)

        y_km_s = y * 1e-3

        # Compute J2 acceleration
        a_J2 = J2_perturbation(t, y_km_s, EARTH_MU_KM, J2=EARTH_J2, R=EARTH_R_KM)
        a_J2 = (np.array(a_J2) * u.km / u.s ** 2).to(u.m / u.s ** 2).value

        # Get sun and moon position
        epoch = self.epoch + t * u.s
        # moon
        moon_r = moon_position_vector(epoch.jd)
        # sun
        sun_r = sun_position_vector(epoch.jd)

        # Compute moon acceleration
        a_moon = third_body_acceleration(r, moon_r, self.m, MOON_MASS)
        # Compute sun acceleration        
        a_sun = third_body_acceleration(r, sun_r, self.m, SUN_MASS)

        # Compute drag acceleration
        altitude = r_norm - EARTH_R
        if 0 <= altitude <= 1e6:
            a_drag = atmospheric_drag(self.Cd, self.A_over_m, self.rho0, self.H0, y, EARTH_OMEGA)[0]
            rho = atmospheric_drag(self.Cd, self.A_over_m, self.rho0, self.H0, y, EARTH_OMEGA)[1]

        else:
            a_drag = np.zeros(3)
            rho = 0

        # In-place addition of accelerations
        a_total = a_grav + a_J2 + a_moon + a_sun + a_drag

        #compute heating
        q = estimate_reentry_heating(v, rho, self.Cp, self.A)

        return {
            'velocity': v,
            'acceleration': a_total,
            'gravitational_acceleration': a_grav,
            'J2_acceleration': a_J2,
            'moon_acceleration': a_moon,
            'drag_acceleration': a_drag,
            'altitude': altitude,
            'sun_acceleration': a_sun,
            'heat_rate': q
        }
    
    def run_simulation(self, t_span, y0, t_eval, progress_callback=None):
        def rhs(t, y):
            dy = self.equations_of_motion(t, y)
            return np.concatenate((dy['velocity'], dy['acceleration']))
        
        def altitude_event(t, y):
            r = y[0:3]
            r_norm = np.linalg.norm(r)
            altitude = r_norm - EARTH_R
            return altitude
        
        altitude_event.terminal = True
        altitude_event.direction = -1

        def progress_event(t, y):
            if progress_callback is not None:
                progress = min((t - t_span[0]) / (t_span[1] - t_span[0]) * 2, 1.0)  # Make sure progress doesn't exceed 1.0
                elapsed_time = time.time() - self.start_time
                progress_callback(progress, elapsed_time)
            return 0

        progress_event.terminal = False
        progress_event.direction = 0

        sol = solve_ivp(rhs, t_span, y0, method='BDF', t_eval=t_eval, rtol=1e-8, atol=1e-10, events=[altitude_event, progress_event])
        additional_data_list = [self.equations_of_motion(t, y) for t, y in zip(sol.t, sol.y.T)]
        sol.additional_data = {key: np.array([d[key] for d in additional_data_list]) for key in additional_data_list[0].keys()}
        return sol
    
