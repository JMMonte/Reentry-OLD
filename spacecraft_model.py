from astropy import units as u
from astropy.time import Time
from coordinate_converter import CoordinateConverter
import numpy as np
from poliastro.bodies import Moon, Earth
from poliastro.constants import J2_earth
from poliastro.core.perturbations import third_body
from poliastro.earth.atmosphere import COESA76
from poliastro.ephem import Ephem
from scipy.integrate import solve_ivp

class SpacecraftModel:
    def __init__(self, Cd=2.2, A=20.0, m=500.0):
        self.mu = Earth.k.to(u.m**3 / u.s**2).value  # gravitational parameter of Earth in m^3/s^2
        self.R = Earth.R.to(u.m).value  # radius of Earth in m
        self.omega = (2 * np.pi / (Earth.rotational_period * 3600 * 24)).value  # Earth rotation speed in rad/s
        self.Cd = Cd  # drag coefficient
        self.A = A  # cross-sectional area of spacecraft in m^2
        self.m = m  # mass of spacecraft in kg
    
    def earth_rotational_velocity(self, r):
        '''
        Calculates the Earth's rotational velocity vector in ECI frame
        :param r: position vector in ECI frame
        :return: Earth's rotational velocity vector in ECI frame
        '''
        omega_cross_r = np.cross([0, 0, self.omega], r)
        return omega_cross_r

    def get_initial_state(self, v, lat, lon, alt, azimuth, attractor=Earth, gmst=0):
        '''
        Calculates the initial state vector of the spacecraft
        :param v: velocity of spacecraft in m/s
        :param lat: latitude of spacecraft in degrees
        :param lon: longitude of spacecraft in degrees
        :param alt: altitude of spacecraft in meters
        :param azimuth: azimuth of spacecraft's velocity in degrees
        :param attractor: attractor object
        :param gmst: Greenwich mean sidereal time in degrees
        :return: initial state vector of spacecraft
        '''
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        x_ecef, y_ecef, z_ecef = CoordinateConverter.geo_to_spheroid(lat, lon, alt, attractor.R.to(u.m).value, attractor.R_polar.to(u.m).value)

        r_eci = CoordinateConverter.ecef_to_eci(x_ecef, y_ecef, z_ecef, gmst)

        # Calculate local tangent plane (ENU) velocity vector
        azimuth_rad = np.deg2rad(azimuth)
        v_east = v * np.sin(azimuth_rad)
        v_north = v * np.cos(azimuth_rad)
        v_up = 0

        # Convert ENU velocity to ECEF velocity
        v_x_ecef = -v_east * np.sin(lon_rad) - v_north * np.sin(lat_rad) * np.cos(lon_rad) + v_up * np.cos(lat_rad) * np.cos(lon_rad)
        v_y_ecef = v_east * np.cos(lon_rad) - v_north * np.sin(lat_rad) * np.sin(lon_rad) + v_up * np.cos(lat_rad) * np.sin(lon_rad)
        v_z_ecef = v_north * np.cos(lat_rad) + v_up * np.sin(lat_rad)

        # get omega from attractor's poliastro rotation period
        v_x_rot = -self.omega * y_ecef
        v_y_rot = self.omega * x_ecef

        v_x_ecef_total = v_x_ecef + v_x_rot
        v_y_ecef_total = v_y_ecef + v_y_rot
        v_z_ecef_total = v_z_ecef

        v_eci = CoordinateConverter.ecef_to_eci(v_x_ecef_total, v_y_ecef_total, v_z_ecef_total, gmst)
        # Subtract Earth's rotational velocity from the initial ECI velocity vector
        v_earth_rotation = self.earth_rotational_velocity(r_eci)
        v_eci -= v_earth_rotation

        y0 = np.concatenate((r_eci, v_eci))

        return y0
    
    def compute_J2_acceleration(self, r):
        '''
        Calculates the J2 perturbation acceleration vector
        :param r: position vector in ECI frame
        :return: J2 perturbation acceleration vector in ECI frame
        '''
        r_norm = np.linalg.norm(r)
        z2 = r[2] ** 2
        rxy2 = r[0] ** 2 + r[1] ** 2
        tmp = 1 - 5 * z2 / rxy2
        a_J2 = (
            (3 * J2_earth * self.mu * self.R ** 2 / 2)
            * r
            / r_norm ** 5
            * np.array([1 - 5 * z2 / rxy2, 1 - 5 * z2 / rxy2, 3 - 5 * z2 / rxy2])) * u.m / u.s**2  # Make sure the unit is m/s^2
        
        return a_J2.value

    def get_moon_position(self, t):
        '''
        Calculates the position vector of the Moon in ECI frame
        :param t: time in seconds
        :return: position vector of the Moon in ECI frame
        '''
        t_astropy = Time(t, format="jd", scale="tdb")
        moon_ephem = Ephem.from_body(Moon, t_astropy)
        moon_r = np.array(moon_ephem.rv()[0]) * u.km.to(u.m)  # Convert to meters
        return moon_r
    
    def equations_of_motion(self, t, y):
        '''
        Calculates the derivatives of the state vector
        :param t: time in seconds
        :param y: state vector
        :return: derivatives of the state vector
        '''
        r = y[0:3]  # position vector
        v = y[3:6]  # velocity vector

        # Compute Earth's rotational velocity at the spacecraft's position
        v_earth_rotation = self.earth_rotational_velocity(r)

        # Compute the velocity of the spacecraft relative to the Earth's atmosphere
        v_relative = v - v_earth_rotation

        # Compute gravitational acceleration
        a_grav = -self.mu * r / np.linalg.norm(r)**3

        # Compute J2 perturbation
        a_J2 = self.compute_J2_acceleration(r)

        # Compute third body perturbation (moon)
        moon_r = self.get_moon_position(t).flatten()  # Add .flatten() here
        mu_moon = Moon.k.to_value(u.km ** 3 / u.s ** 2)  # Convert to km^3/s^2
        a_moon = third_body(t, y * u.m.to(u.km), self.mu * u.km ** 3 / u.s ** 2 * u.km ** -3, mu_moon, lambda t0: moon_r)  # Convert y from meters to kilometers
        a_moon = (np.array(a_moon) * u.m / u.s**2 * u.m / u.s**2).value * u.m/u.s**2 / self.m  # Convert the acceleration from km/s^2 to m/s^2 and account for the mass of the spacecraft

        # Compute atmospheric density and drag acceleration
        altitude = np.linalg.norm(r) - self.R
        if 0 <= altitude <= 1e6:  # Check if altitude is within the valid range of the atmospheric model
            rho = COESA76().density(altitude*u.m).to(u.kg / u.m**3).value
            v_norm = np.linalg.norm(v_relative)
            a_drag = -0.5 * self.Cd * self.A * rho * v_norm * v_relative / self.m
        else:
            a_drag = np.zeros(3)  # Set drag acceleration to zero if altitude is outside the valid range

        # Compute total acceleration vector
        a = (a_grav + a_J2 + a_moon.value + a_drag)

        return {
            'velocity': v,
            'acceleration': a,
            'gravitational_acceleration': a_grav,
            'J2_acceleration': a_J2,
            'moon_acceleration': a_moon,
            'drag_acceleration': a_drag,
            'altitude': altitude
        }
    
    def run_simulation(self, t_span, y0, t_eval, previous_sol=None):
        '''
        Runs the simulation
        :param t_span: time span in seconds
        :param y0: initial state vector
        :param t_eval: time points to evaluate the solution
        :param previous_sol: previous solution to concatenate with
        :return: solution object
        '''
        def rhs(t, y):
            '''
            Right-hand side of the differential equation
            :param t: time in seconds
            :param y: state vector
            :return: derivatives of the state vector
            '''
            dy = self.equations_of_motion(t, y)
            return np.concatenate((dy['velocity'], dy['acceleration']))

        sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
        
        # Store additional data
        sol.additional_data = [self.equations_of_motion(t, y) for t, y in zip(sol.t, sol.y.T)]

        # Concatenate previous solution and current solution
        if previous_sol is not None:
            sol.y = np.concatenate((previous_sol.y, sol.y), axis=1)
            sol.t = np.concatenate((previous_sol.t, sol.t))
            sol.additional_data.extend(previous_sol.additional_data)

        return sol