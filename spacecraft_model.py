import math
from astropy import units as u
from poliastro.core.elements import coe2rv
from astropy.time import Time
from coordinate_converter import *
import numpy as np
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp
import time
from numba import jit, njit
from copy import deepcopy
from poliastro.twobody import Orbit
import base64
from constants import *

def match_array_length(array, target_length):
    if len(array) > target_length:
        return array[:target_length]
    elif len(array) < target_length:
        return np.pad(array, (0, target_length - len(array)), mode='constant')
    else:
        return array

#special functions
@jit(nopython=True)
def filter_results_by_altitude(sol, altitude):
    valid_indices = [i for i, alt in enumerate(altitude) if alt >= 0]
    sol = deepcopy(sol)
    sol.y = sol.y[:, valid_indices]
    sol.t = sol.t[valid_indices]
    sol.additional_data = [sol.additional_data[i] for i in valid_indices]
    return sol

def make_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def mpl_to_plotly_colormap(cmap, num_colors=256):
    colors = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(num_colors)]
    scale = np.linspace(0, 1, num=num_colors)
    return [list(a) for a in zip(scale, colors)]

def get_color(normalized_value, colormap):
    rgba = colormap(normalized_value)
    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"

def periapsis_apoapsis_points(orbit: Orbit):
    k = orbit.attractor.k.to_value("km^3 / s^2")

    # Calculate the true anomalies for periapsis and apoapsis
    nu_periapsis = 0
    nu_apoapsis = np.pi

    # Calculate the position and velocity vectors for periapsis and apoapsis in ECI frame
    r_periapsis_ECI, _ = coe2rv(k, orbit.p.to_value('km'), orbit.ecc.value, orbit.inc.to_value('rad'), orbit.raan.to_value('rad'), orbit.argp.to_value('rad'), nu_periapsis)
    r_apoapsis_ECI, _ = coe2rv(k, orbit.p.to_value('km'), orbit.ecc.value, orbit.inc.to_value('rad'), orbit.raan.to_value('rad'), orbit.argp.to_value('rad'), nu_apoapsis)

    # Convert the position vectors from km to m
    r_periapsis_ECI = r_periapsis_ECI * 1000
    r_apoapsis_ECI = r_apoapsis_ECI * 1000

    return r_periapsis_ECI, r_apoapsis_ECI

# numba functions
# ----------------
@njit
def euclidean_norm(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

@jit(nopython=True)
def simplified_nrlmsise_00(altitude, latitude, jd_epoch):
    factor = solar_activity_factor(jd_epoch, altitude)
    
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
    # make solar activity factor decrease exponentially bellow 20km

    rho *= factor

    return rho, T

# test simplified_nrlmsise_00
# ---------------------------
# altitude = 600000
# latitude = 0
# rho, T = simplified_nrlmsise_00(altitude, latitude)
# print(rho, T)

from scipy.optimize import curve_fit

@jit(nopython=True)
# normalized sigmoid function (y1 = 0, y2 = 1)
def normalized_sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

# find k and x0 for normalized sigmoids
def fit_normalized_sigmoid(x1, y1, x2, y2):
    x_data = np.array([x1, x2])
    y_data = np.array([(y1 - y1) / (y2 - y1), (y2 - y1) / (y2 - y1)])
    params, _ = curve_fit(normalized_sigmoid, x_data, y_data, p0=[0.0001, x1 + (x2 - x1) / 2], maxfev=10000)

    k, x0 = params
    return k, x0

@jit(nopython=True)
def sigmoid(x, y1, y2, k, x0, smoothness=0.0010):
    normalized_output = normalized_sigmoid(x, k * smoothness, x0)
    return y1 + (y2 - y1) * normalized_output

# discover k and x0 for normalized sigmoid with x1 = 0, y1 = 0, x2 = 40000, y2 = 150
# Points (x1, y1) and (x2, y2)
x1, y1 = 0, 100
x2 = 40000
y2 = 150  # This can be any value you want

k, x0 = fit_normalized_sigmoid(x1, y1, x2, y2)
print(f"Best-fit values: k = {k}, x0 = {x0}")

k = 0.036032536225379
x0 = 19709.47069118867

# Example usage with variable y2
y2_new = 180
altitude = 40000
factor = sigmoid(altitude, y1, y2_new, k, x0)
print(f"Factor at altitude {altitude} with y2 = {y2_new}: {factor}")


@jit(nopython=True)
def solar_activity_factor(jd_epoch, altitude, jd_solar_min=2454833.0, f107_average=150.0, solar_cycle_months=132):
    # Calculate the time since the last solar minimum in months
    days_since_min = jd_epoch - jd_solar_min
    months_since_min = days_since_min / DAYS_PER_MONTH
    months_since_cycle_start = (months_since_min % solar_cycle_months) / solar_cycle_months * 2 * PI

    # Estimate the F10.7 value based on a simple sinusoidal model
    f107 = f107_average + F107_AMPLITUDE * np.sin(months_since_cycle_start)
    
    # Calculate the solar activity factor
    factor = 1 + (f107 - f107_average) / f107_average

    # make solar activity factor decrease exponentially bellow 20km
    if altitude < 90000:
        # use sigmoid curve here to make the factor decrease exponentially between 0 and 40000m
        k = 0.036032536225379 # from fit_normalized_sigmoid
        x0 = 19709.47069118867
        factor = sigmoid(altitude, 1, factor, k, x0)
    
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

#@jit(nopython=True)
def atmosphere_model(altitude, latitude, jd_epoch):
    if altitude <= 0:
        return 1.225, 288.15
    else:
        # Calculate the density and temperature using the simplified NRLMSISE-00 model
        rho, T = simplified_nrlmsise_00(altitude, latitude, jd_epoch)

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
# rho, T, solar_factor = atmosphere_model(altitude, latitude, jd_epoch, jd_solar_min, f107_average, solar_cycle_months)
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
def heat_transfer(v,ablation_efficiency, T_s, atmo_T, thermal_conductivity, capsule_length, emissivity,spacecraft_m, a_drag, specific_heat_capacity, dt):
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
def spacecraft_temperature(v, atmo_T, a_drag, capsule_length, dt, thermal_conductivity ,specific_heat_capacity, emissivity, ablation_efficiency, iter_fact=2, spacecraft_m=500):
    # Initialize the spacecraft temperature to the atmospheric temperature
    T_s = atmo_T
    dt = int(dt / iter_fact)
    iterations = dt

    for _ in range(iterations):
        # Calculate radiative heat transfer (Qr) using the emissivity of the heat shield material and the Stefan-Boltzmann constant (sigma)
        Qc, Qr, Q_net, Q, T_s, dT = heat_transfer(v,ablation_efficiency, T_s, atmo_T, thermal_conductivity, capsule_length, emissivity,spacecraft_m, a_drag, specific_heat_capacity, dt)
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
        q_gen, q_c, q_r, q_net, T_s, dT = spacecraft_temperature(v_rel, T, a_drag, self.height, self.dt, self.thermal_conductivity ,self.specific_heat_capacity, self.emissivity, self.ablation_efficiency, self.iter_fact, self.m)
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
    
