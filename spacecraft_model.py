import math
from astropy import units as u
from astropy.time import Time
from coordinate_converter import (enu_to_ecef, ecef_to_eci, geodetic_to_spheroid, eci_to_ecef, ecef_to_geodetic)
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
G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant

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
MOON_K = 4.9048695e12  # Surface gravity (m/s^2)

#Sun
SUN_MU = 132712442099.00002 # gravitational parameter of Sun in km^3/s^2
SUN_A = 149598022990.63  # Semi-major axis (meters)
SUN_E = 0.01670862  # Eccentricity
SUN_I = 0.00005 * DEG_TO_RAD  # Inclination (radians)
SUN_OMEGA = -11.26064 * DEG_TO_RAD  # Longitude of ascending node (radians)
SUN_W = 102.94719 * DEG_TO_RAD  # Argument of perigee (radians)
SUN_L0 = 100.46435 * DEG_TO_RAD  # Mean longitude at epoch J2000 (radians)
SUN_MASS = 1.988544e30  # Mass (kg)
SUN_K = 1.32712440042e20  # Surface gravity (m/s^2)
SOLAR_CONSTANT = 1361  # W/m^2

# Atmosphere
# U.S. Standard Atmosphere altitude breakpoints and temperature gradients (m, K/m)
ALTITUDE_BREAKPOINTS = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
TEMPERATURE_GRADIENTS = np.array([-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002])
BASE_TEMPERATURES = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
BASE_PRESSURES = np.array([101325, 22632.1, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734])
# Earth atmosphere constants
EARTH_AIR_MOLAR_MASS = 0.0289644 # molar mass of Earth's air (kg/mol)
EARTH_GAS_CONSTANT = 8.31447 # Gas Constant Values based on Energy Units ; J · 8.31447, 3771.38
R_GAS = 287.0  # J/kgK for air; This value is appropriate for air if Joule is chosen for the unit of energy, kg as unit of mass and K as unit of temperature, i.e. $ R = 287 \;$   J$ /($kg$ \;$K$ )$
SCALE_HEIGHT = 7500.0  # Scale height (m)
#Solar weather constants
F107_MIN = 70.0
F107_MAX = 230.0
F107_AMPLITUDE = (F107_MAX - F107_MIN) / 2.0
DAYS_PER_MONTH = 30.44  # Average number of days per month

# Themodynamic constants
GAMMA_AIR = 1.4  # Ratio of specific heats for air
RECOVERY_FACTOR_AIR = 0.9  # Assuming laminar flow over a flat plate
CP_AIR = 1005 # Specific heat of air at constant pressure (J/kg-K)
STAG_K = 1.83e-4 # Stagnation point heat transfer coefficient (W/m^2-K^0.5)
FLOW_TYPE_EXP = 0.5 # Flow type exponent (0.5 for laminar, 0.8 for turbulent)
EMISSIVITY_SURF = 0.8 # Emissivity of surface
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8 # Stefan-Boltzmann constant (W/m^2-K^4)
T_REF = 273.15  # Reference temperature in Kelvin
MU_REF = 1.716e-5  # Reference dynamic viscosity in Pa.s
SUTHERLAND_CONSTANT = 110.4  # Sutherland's constant in Kelvin
CP_BASE = 1000  # Base specific heat at constant pressure at 298 K
CP_RATE = 0.5  # Rate of change of specific heat with temperature
K_AIR_COEFFICIENT = 2.64638e-3  # Coefficient for thermal conductivity of air


# numba functions
# ----------------
@njit
def euclidean_norm(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

@jit(nopython=True)
def simplified_nrlmsise_00(altitude, latitude):
    
    # Latitude factor (simplified)
    latitude_factor = 1 + 0.01 * np.abs(latitude) / 90.0

    # Find the appropriate altitude layer
    i = np.searchsorted(ALTITUDE_BREAKPOINTS, altitude, side='right') - 1
    delta_altitude = altitude - ALTITUDE_BREAKPOINTS[i]
    T = BASE_TEMPERATURES[i] + TEMPERATURE_GRADIENTS[i] * delta_altitude

    if TEMPERATURE_GRADIENTS[i] == 0:
        P = BASE_PRESSURES[i] * np.exp(-EARTH_GRAVITY * EARTH_AIR_MOLAR_MASS * delta_altitude / (EARTH_GAS_CONSTANT * BASE_TEMPERATURES[i]))
    else:
        P = BASE_PRESSURES[i] * (T / BASE_TEMPERATURES[i]) ** (-EARTH_GRAVITY * EARTH_AIR_MOLAR_MASS / (EARTH_GAS_CONSTANT * TEMPERATURE_GRADIENTS[i]))

    if i == len(ALTITUDE_BREAKPOINTS) - 1:
        P *= np.exp(-(altitude - ALTITUDE_BREAKPOINTS[-1]) / SCALE_HEIGHT)

    # Apply the latitude factor
    P *= latitude_factor

    rho = P / (R_GAS * T)

    return rho, T

# test simplified_nrlmsise_00
# ---------------------------
# altitude = 600000
# latitude = 0
# rho, T = simplified_nrlmsise_00(altitude, latitude)
# print(rho, T)

@jit(nopython=True)
def solar_activity_factor(jd_epoch, jd_solar_min=2454833.0, f107_average=150.0, solar_cycle_months=132):
    # Calculate the time since the last solar minimum in months
    days_since_min = jd_epoch - jd_solar_min
    months_since_min = days_since_min / DAYS_PER_MONTH
    months_since_cycle_start = (months_since_min % solar_cycle_months) / solar_cycle_months * 2 * PI

    # Estimate the F10.7 value based on a simple sinusoidal model
    f107 = f107_average + F107_AMPLITUDE * np.sin(months_since_cycle_start)
    
    # Calculate the solar activity factor
    factor = 1 + (f107 - f107_average) / f107_average
    
    return factor

# test solar_activity_factor
# ---------------------------
# epoch=Time('2024-01-01 00:00:00').jd
# jd_epoch = epoch + 1 / 86400
# jd_solar_min = 2454833.0
# f107_average = 150.0
# solar_cycle_months = 132
# factor = solar_activity_factor(jd_epoch, jd_solar_min, f107_average, solar_cycle_months)
# print(factor)

@jit(nopython=True)
def atmosphere_model(altitude, latitude, jd_epoch, jd_solar_min=2454833.0, f107_average=150.0, solar_cycle_months=132):
    if altitude <= 0:
        return 1.225, 288.15
    else:
        # Calculate the solar activity factor
        factor = solar_activity_factor(jd_epoch, jd_solar_min, f107_average, solar_cycle_months)

        # Calculate the density and temperature
        rho, T = simplified_nrlmsise_00(altitude, latitude)
        # make solar factor change with altitude (0 at surface, 1 at 50km and stay 1 above 50km)
        if altitude < 50000:
            factor = 1 - altitude / 50000 * np.exp(- factor)
        elif altitude >= 50000:
            factor = 1
            if rho < 1e-20:
                rho = 1e-20
        rho *= factor

        return rho, T
    
# test atmosphere_model
# ---------------------------
# altitude = 102010.012
# latitude = 0
# epoch=Time('2024-01-01 00:00:00').jd
# jd_epoch = epoch + 1 / 86400
# jd_solar_min = 2454833.0
# f107_average = 150.0
# solar_cycle_months = 132
# rho, T = atmosphere_model(altitude, latitude, jd_epoch, jd_solar_min, f107_average, solar_cycle_months)
# print(rho, T)

@jit(nopython=True)
def atmospheric_drag(Cd, A, atmospheric_rho, v, mass):
    F_d = 0.5 * atmospheric_rho * Cd * A * euclidean_norm(v)**2
    drag_force_vector = -(F_d / euclidean_norm(v)) * v
    drag_accelleration_vector = drag_force_vector / mass
    return drag_accelleration_vector

# test atmospheric_drag
# ---------------------------
# Cd = 2.2
# A = 0.1
# altitude = 8010.012
# latitude = 0
# epoch=Time('2024-01-01 00:00:00').jd
# jd_epoch = epoch + 1 / 86400
# atmospheric_rho, _ = atmosphere_model(altitude, latitude, jd_epoch)
# v = np.array([7500, 0, 0])
# drag_force_vector = atmospheric_drag(Cd, A, atmospheric_rho, v)
# print(drag_force_vector)


@njit
def heat_transfer(altitude, v,ablation_efficiency, T_s, atmo_T, thermal_conductivity, capsule_length, emissivity,spacecraft_m, a_drag, specific_heat_capacity, dt):
    v_norm = euclidean_norm(v)
    a_drag = euclidean_norm(a_drag)

    drag_force = spacecraft_m * a_drag

    # Calculate work done (W) using the drag force and change in velocity (dv)
    W = drag_force * v_norm

    # Calculate the heat generated (Q) from the work done (W) and the ablation efficiency factor
    Q = ablation_efficiency * W
    Qc = thermal_conductivity * (T_s - atmo_T) / capsule_length
    Qr = emissivity * STEFAN_BOLTZMANN_CONSTANT * (T_s**4 - atmo_T**4)
    Q_net = Q - Qc - Qr
    dT = Q_net / (spacecraft_m * specific_heat_capacity) * dt

    return Qc, Qr, Q_net, Q, T_s, dT

# @jit(nopython=True)
def spacecraft_temperature(altitude, v, atmo_T, a_drag, capsule_length, dt, thermal_conductivity ,specific_heat_capacity, emissivity, ablation_efficiency, iter_fact=2, spacecraft_m=500):
    # Initialize the spacecraft temperature to the atmospheric temperature
    T_s = atmo_T
    dt = int(dt / iter_fact)
    iterations = dt

    for _ in range(iterations):
        # Calculate radiative heat transfer (Qr) using the emissivity of the heat shield material and the Stefan-Boltzmann constant (sigma)
        Qc, Qr, Q_net, Q, T_s, dT = heat_transfer(altitude, v,ablation_efficiency, T_s, atmo_T, thermal_conductivity, capsule_length, emissivity,spacecraft_m, a_drag, specific_heat_capacity, dt)
        # Update the spacecraft temperature (T_s) by adding the temperature change (dT) to the current temperature
        T_s += dT

    return Qc, Qr, Q_net, Q, T_s, dT

# test spacecraft_temperature
# ---------------------------
# V = np.array([7500, 0, 0])
# atmo_rho = 0.0001
# material = ['Aluminum', 0.0001, 2700, 0.3, 0.3, 0.3, 0.3]
# atmo_T = 185
# thickness = 0.1
# altitude = 100000
# characteristic_length = 0.3
# T_surface = spacecraft_temperature(V, atmo_rho, material, atmo_T, thickness, altitude, characteristic_length)
# print(T_surface)

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

# Test moon_position_vector
# -------------------------
# jd = 2451545.0
# moon_pos = moon_position_vector(jd)
# norm_moon_pos = euclidean_norm(moon_pos)
# norm_moon_pos_km = norm_moon_pos / 1000
# print("Moon position vector (m):", moon_pos)
# print("Moon position vector magnitude (m):", norm_moon_pos)
# print("Moon position vector magnitude (km):", norm_moon_pos_km)

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

# Test sun_position_vector
# -------------------------
# jd = 2451545.0
# sun_pos = sun_position_vector(jd)
# norm_sun_pos = euclidean_norm(sun_pos)
# norm_sun_pos_km = norm_sun_pos / 1000
# print("Sun position vector (m):", sun_pos)
# print("Sun position vector magnitude (m):", norm_sun_pos)

@jit(nopython=True)
def third_body_acceleration(satellite_position, third_body_position, k_third):
    # Calculate the vector from the satellite to the third body
    r_satellite_to_third_body = third_body_position - satellite_position

    return (
        k_third * r_satellite_to_third_body / euclidean_norm(r_satellite_to_third_body) ** 3
        - k_third * third_body_position / euclidean_norm(third_body_position) ** 3
    )

# Test third_body_acceleration
# ----------------------------
# jd = 2451545.0
# satellite_position = np.array([8000000.0, 0.0, 0.0])
# sun_position = sun_position_vector(jd)
# k_third = SUN_K
# a_third = third_body_acceleration(satellite_position, sun_position, k_third)
# a_third_norm = euclidean_norm(a_third)
# print("Third body acceleration (m/s^2):", a_third)
# print("Third body acceleration magnitude (m/s^2):", a_third_norm)


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
    def __init__(self, Cd=2.2, A=20.0, m=500.0, epoch=Time('2024-01-01 00:00:00'), gmst0=0.0, sim_type='RK45', material=[233, 1, 1, 0.1], dt=10, iter_fact=2):
        self.Cd = Cd  # drag coefficient
        self.A = A  # cross-sectional area of spacecraft in m^2
        self.height = np.sqrt(self.A / PI) * 1.315 # height of spacecraft in m, assuming orion capsule design
        self.m = m  # mass of spacecraft in kg
        self.epoch = epoch.jd # 
        self.start_time = time.time() # start time of simulation
        self.A_over_m = (self.A) / self.m # A/m
        self.gmst0 = gmst0 # Greenwich Mean Sidereal Time at epoch (degrees)
        self.thickness = 0.1 # thickness of spacecraft's heatshield in m
        self.sim_type = sim_type
        self.mat = material
        self.thermal_conductivity = material[0]
        self.specific_heat_capacity = material[1]
        self.emissivity = material[2]
        self.ablation_efficiency = material[3]
        self.dt = dt
        self.iter_fact = iter_fact

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
        epoch = self.epoch + t / 86400.0 # convert seconds to days
        gmst = self.gmst0 + EARTH_OMEGA * t

        # Calculate ECEF position and ground velocity
        r_ecef = eci_to_ecef(r_eci, gmst)
        v_ground = eci_to_ecef(v_eci, gmst)
        v_rel = v_ground - np.array([-EARTH_OMEGA * r_ecef[1], EARTH_OMEGA * r_ecef[0], 0])

        # Calculate accelerations
        a_grav = -EARTH_MU * r_eci / (r_norm ** 3)
        a_J2 = J2_perturbation_numba(r_eci, k=EARTH_MU, J2=EARTH_J2, R=EARTH_R)
        moon_r = moon_position_vector(epoch)
        sun_r = sun_position_vector(epoch)
        a_moon = third_body_acceleration(r_eci, moon_r, MOON_K)
        a_sun = third_body_acceleration(r_eci, sun_r, SUN_K)

        altitude = r_norm - EARTH_R
        x_ecef, y_ecef, z_ecef = r_ecef
        latitude, _, _ = ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
        rho, T = atmosphere_model(altitude, latitude, epoch)

        # Calculate drag acceleration
        a_drag_ecef = atmospheric_drag(Cd=self.Cd, A=self.A, atmospheric_rho=rho, v=v_rel, mass=self.m)
        a_drag = ecef_to_eci(a_drag_ecef, gmst)

        # Calculate surface temperature
        q_gen, q_c, q_r, q_net, T_s, dT = spacecraft_temperature(altitude, v_rel, T, a_drag, self.height, self.dt, self.thermal_conductivity ,self.specific_heat_capacity, self.emissivity, self.ablation_efficiency, self.iter_fact, self.m)
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
            'spacecraft_temperature': T_s,
            'spacecraft_heat_flux': q_net,
            'spacecraft_heat_flux_conduction': q_c,
            'spacecraft_heat_flux_radiation': q_r,
            'spacecraft_heat_flux_total': q_gen,
            'spacecraft_temperature_change': dT,
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
    
