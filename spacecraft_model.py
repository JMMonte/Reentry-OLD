from astropy import units as u
from astropy.time import Time
from coordinate_converter import CoordinateConverter
import numpy as np
from poliastro.bodies import Moon, Earth, Sun
from poliastro.earth.atmosphere import COESA76
from poliastro.core.perturbations import J2_perturbation, third_body, atmospheric_drag_exponential
from poliastro.ephem import Ephem
from scipy.integrate import solve_ivp
import time
from poliastro.constants import rho0_earth, H0_earth

class SpacecraftModel:
    def __init__(self, Cd=2.2, Cp=500.0, A=20.0, m=500.0, epoch=Time('2024-01-01 00:00:00')):
        self.mu = Earth.k.to(u.m**3 / u.s**2).value  # gravitational parameter of Earth in m^3/s^2
        self.R = Earth.R.to(u.m).value  # radius of Earth in m
        self.R_km = Earth.R.to(u.km).value  # radius of Earth in m
        self.omega = (2 * np.pi / (Earth.rotational_period * 3600 * 24)).value  # Earth rotation speed in rad/s
        self.Cd = Cd  # drag coefficient
        self.A = A  # cross-sectional area of spacecraft in m^2
        self.m = m  # mass of spacecraft in kg
        self.epoch = epoch
        self.start_time = time.time()
        self.attractor = Earth
        self.mu_sun = Sun.k.to_value(u.km ** 3 / u.s ** 2)
        self.mu_moon = Moon.k.to_value(u.km ** 3 / u.s ** 2)
        self.earth_j2 = Earth.J2.value
        self.ua_to_m = 149597870700 # 1 astronomical unit in meters
        self.k_km = Earth.k.to_value(u.km ** 3 / u.s ** 2) # gravitational parameter of Earth in km^3/s^2
        self.A_over_m = (self.A) / self.m
        self.rho0 = rho0_earth.to_value(u.kg / u.m ** 3)
        self.H0 = H0_earth.to_value(u.m)
        self.Cp = Cp  # Example heat transfer coefficient (W/m²·K)


    def earth_rotational_velocity(self, r):
        omega_cross_r = np.cross([0, 0, self.omega], r)
        return omega_cross_r

    def get_initial_state(self, v, lat, lon, alt, azimuth, gamma, attractor=Earth, gmst=0):
        lat_rad, lon_rad, azimuth_rad, gamma_rad = np.radians([lat, lon, azimuth, gamma])
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
        sin_azimuth, cos_azimuth = np.sin(azimuth_rad), np.cos(azimuth_rad)
        sin_gamma, cos_gamma = np.sin(gamma_rad), np.cos(gamma_rad)
        x_ecef, y_ecef, z_ecef = CoordinateConverter.geo_to_spheroid(lat, lon, alt, attractor.R.to(u.m).value, attractor.R_polar.to(u.m).value)

        r_eci = CoordinateConverter.ecef_to_eci(x_ecef, y_ecef, z_ecef, gmst)

        sin_azimuth, cos_azimuth = np.sin(azimuth_rad), np.cos(azimuth_rad)
        sin_gamma, cos_gamma = np.sin(gamma_rad), np.cos(gamma_rad)

        v_east = v * sin_azimuth * cos_gamma
        v_north = v * cos_azimuth * cos_gamma
        v_up = v * sin_gamma

        v_x_ecef = -v_east * sin_lon - v_north * sin_lat * cos_lon + v_up * cos_lat * cos_lon
        v_y_ecef = v_east * cos_lon - v_north * sin_lat * sin_lon + v_up * cos_lat * sin_lon
        v_z_ecef = v_north * cos_lat + v_up * sin_lat

        v_x_rot = -self.omega * y_ecef
        v_y_rot = self.omega * x_ecef

        v_x_ecef_total = v_x_ecef + v_x_rot
        v_y_ecef_total = v_y_ecef + v_y_rot
        v_z_ecef_total = v_z_ecef

        v_eci = CoordinateConverter.ecef_to_eci(v_x_ecef_total, v_y_ecef_total, v_z_ecef_total, gmst)
        v_eci -= self.earth_rotational_velocity(r_eci)

        y0 = np.concatenate((r_eci, v_eci))

        return y0
    
    def atmospheric_drag(self, state, attractor=Earth, method='drag_exponential'):
        r, v = state[0:3], state[3:6]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        v_relative = v - self.earth_rotational_velocity(r)
        v_unit = v_relative / v_norm
        altitude = r_norm - attractor.R.to(u.m).value
        if method == 'drag_exponential':
            rho = self.rho0 * np.exp(-altitude / self.H0)
        elif method == 'drag_table':
            rho = COESA76().density(altitude * u.m).to(u.kg / u.m ** 3).value
        a_drag = -0.5 * self.Cd * self.A_over_m * rho * v_norm ** 2 * v_unit

        return a_drag,rho
    
    def estimate_reentry_heating(self, position_eci, velocity_eci, rho, Cp, A):
        V = np.linalg.norm(velocity_eci)
        q = 0.5 * rho * V**3 * Cp * A
        return q

    def equations_of_motion(self, t, y):
        r, v = y[0:3], y[3:6]

        # Precompute r_norm and other reused values
        r_norm = np.linalg.norm(r)

        a_grav = -self.mu * r / (r_norm ** 3)

        # Compute J2 acceleration
        a_J2 = J2_perturbation(t, y, self.mu, J2=self.earth_j2, R=self.R_km)
        a_J2 = (np.array(a_J2) * u.km / u.s ** 2).to(u.m / u.s ** 2).value

        # Get sun and moon position
        epoch = self.epoch + t * u.s
        # moon
        moon_ephem = Ephem.from_body(Moon, epoch, attractor=Earth)
        moon_r = np.array(moon_ephem.rv()[0]) * u.au.to(u.m)
        # sun
        sun_ephem = Ephem.from_body(Sun, epoch, attractor=Earth)
        sun_r = np.array(sun_ephem.rv()[0]) * u.au.to(u.m)

        # Compute moon acceleration
        a_moon = third_body(t, (y[:3] * u.m.to(u.km)).reshape(3,), self.mu * u.km ** 3 / u.s ** 2 * u.km ** -3, self.mu_moon, lambda t0: moon_r.reshape(3,))
        a_moon = ((np.array(a_moon) * u.km / u.s**2).value * u.m/u.s**2 / self.m).value
        # Compute sun acceleration        
        a_sun = third_body(t, (y[:3] * u.m.to(u.km)).reshape(3,), self.mu * u.km ** 3 / u.s ** 2 * u.km ** -3, self.mu_sun, lambda t0: sun_r.reshape(3,))
        a_sun = ((np.array(a_sun) * u.km / u.s**2).value * u.m/u.s**2 / self.m).value

        # Compute drag acceleration
        altitude = r_norm - self.R
        if 0 <= altitude <= 1e6:
            a_drag,rho = self.atmospheric_drag(y, method='drag_exponential')
        else:
            a_drag,rho = np.zeros(3)

        # In-place addition of accelerations
        a_total = a_grav + a_J2 + a_moon + a_sun + a_drag

        #compute heating
        q = self.estimate_reentry_heating(r, v, rho, self.Cp, self.A)

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
            altitude = r_norm - self.R
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
    
