import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from astropy.coordinates import get_sun
from astropy.coordinates import EarthLocation, AltAz

class CoordinateConverter:
    @staticmethod
    def ecef_to_eci(x, y, z, gmst):
        '''
        Converts ECEF coordinates to ECI coordinates
        :param x: x-coordinate in ECEF frame
        :param y: y-coordinate in ECEF frame
        :param z: z-coordinate in ECEF frame
        :param gmst: Greenwich Mean Sidereal Time in degrees
        :return: x, y, z coordinates in ECI frame
        '''
        theta = (gmst * u.deg).to_value(u.rad)  # Convert GMST to radians and extract the value
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        eci_coords = np.dot(rotation_matrix, np.array([x, y, z]))
        return eci_coords

    
    @staticmethod
    def eci_to_ecef(x_eci, y_eci, z_eci, gmst):
        '''
        Converts ECI coordinates to ECEF coordinates
        :param x_eci: x-coordinate in ECI frame
        :param y_eci: y-coordinate in ECI frame
        :param z_eci: z-coordinate in ECI frame
        :param gmst: Greenwich Mean Sidereal Time in degrees
        :return: x, y, z coordinates in ECEF frame
        '''
        R = np.array(
            [
                [np.cos(-gmst), -np.sin(-gmst), 0],
                [np.sin(-gmst), np.cos(-gmst), 0],
                [0, 0, 1]
            ]
        )
        
        ecef_coords = R @ np.array([x_eci, y_eci, z_eci])
        return ecef_coords[0], ecef_coords[1], ecef_coords[2]

    
    @staticmethod
    def geo_to_spheroid(lat, lon, alt=0, R_equatorial=Earth.R.to(u.m).value, R_polar=Earth.R_polar.to(u.m).value):
        '''
        Converts geodetic coordinates to spheroid coordinates
        :param lat: latitude in degrees
        :param lon: longitude in degrees
        :param alt: altitude in meters
        :param R_equatorial: equatorial radius of the Earth in meters
        :param R_polar: polar radius of the Earth in meters
        '''
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        R_lat = (R_equatorial * R_polar) / np.sqrt((R_equatorial * np.cos(lat_rad))**2 + (R_polar * np.sin(lat_rad))**2)
        
        x = (R_lat + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (R_lat + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (R_lat + alt) * np.sin(lat_rad)
        
        return x, y, z
    
    @staticmethod
    def geo_to_ecef(lon_rad, lat_rad, alt=0):
        '''
        Converts geodetic coordinates to ECEF coordinates
        :param lat_rad: latitude in radians
        :param lon_rad: longitude in radians
        :param alt: altitude in meters
        :return: x, y, z coordinates in ECEF frame
        '''
        a = Earth.R.to(u.m).value  # Earth's equatorial radius in meters (WGS-84)
        f = 1 / 298.257223563  # Earth's flattening factor (WGS-84)
        e2 = 2 * f - f**2  # Earth's eccentricity squared (WGS-84)

        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)  # Prime vertical radius of curvature
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = ((1 - e2) * N + alt) * np.sin(lat_rad)

        return x, y, z # ECEF coordinates in meters
    
    @staticmethod
    def ecef_to_geo(x, y, z):
        '''
        Converts ECEF coordinates to geodetic coordinates
        :param x: x-coordinate in ECEF frame
        :param y: y-coordinate in ECEF frame
        :param z: z-coordinate in ECEF frame
        :return: latitude, longitude, and altitude in that order
        '''
        a = Earth.R.to(u.m).value  # Earth's equatorial radius in meters (WGS-84)
        f = 1 / 298.257223563  # Earth's flattening factor (WGS-84)
        e2 = 2 * f - f**2  # Earth's eccentricity squared (WGS-84)

        lon_rad = np.arctan2(y, x) # Longitude is easy
        p = np.sqrt(x**2 + y**2) # Distance from z-axis
        lat_rad = np.arctan2(z, p * (1 - e2)) # Numerator is z-coordinate, denominator corrected for flattening
        alt = p / np.cos(lat_rad) - a / np.sqrt(1 - e2 * np.sin(lat_rad)**2) # Altitude in meters

        return lat_rad, lon_rad, alt # Return latitude, longitude, and altitude in that order
    
    @staticmethod
    def eci_velocity_to_ground_velocity(v_eci, lat, lon, gmst):
        '''
        Converts ECI velocity to ground velocity
        :param v_eci: velocity vector in ECI frame
        :param lat: latitude in degrees
        :param lon: longitude in degrees
        :param gmst: Greenwich Mean Sidereal Time in degrees
        :return: velocity vector in ENU frame
        '''
        v_ecef = CoordinateConverter.eci_to_ecef(v_eci[0], v_eci[1], v_eci[2], gmst)
        v_enu = CoordinateConverter.ecef_to_enu(v_ecef[0], v_ecef[1], v_ecef[2], lat, lon)
        return v_enu
    
    @staticmethod
    def ecef_to_enu(v_x_ecef, v_y_ecef, v_z_ecef, lat, lon):
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        v_east = (-v_x_ecef * np.sin(lon_rad)) + (v_y_ecef * np.cos(lon_rad))
        v_north = (-v_x_ecef * np.sin(lat_rad) * np.cos(lon_rad)) - (v_y_ecef * np.sin(lat_rad) * np.sin(lon_rad)) + (v_z_ecef * np.cos(lat_rad))
        v_up = (v_x_ecef * np.cos(lat_rad) * np.cos(lon_rad)) + (v_y_ecef * np.cos(lat_rad) * np.sin(lon_rad)) + (v_z_ecef * np.sin(lat_rad))

        return np.array([v_east, v_north, v_up])
    
    @staticmethod
    def body_rotation_speed(attractor,output='deg/s'):
        '''
        Returns the rotation speed of a body as float, given an output unit.
        :param attractor: name of the body
        :param output: output units
        :return: rotation period of the body in the given output units
        '''
        output_metrics = {
            'deg/s': 360/attractor.rotational_period.to(u.s).value,
            'rad/s': 2*np.pi/attractor.rotational_period.to(u.s).value,
            'rpm': 60/attractor.rotational_period.to(u.s).value,
            'm/s': attractor.R.to(u.m).value * np.pi * 2 / attractor.rotational_period.to(u.s).value,
            }
        return output_metrics[output]
    
    @staticmethod
    def ecef_distance(x1, y1, z1, x2, y2, z2):
        '''
        Returns the distance between two points in ECEF coordinates
        :param x1: x-coordinate of first point
        :param y1: y-coordinate of first point
        :param z1: z-coordinate of first point
        :param x2: x-coordinate of second point
        :param y2: y-coordinate of second point
        :param z2: z-coordinate of second point
        :return: distance between the two points
        '''
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    @staticmethod
    def solar_zenith_angle(final_time):
        sun_coord = get_sun(final_time)
        
        lats = np.linspace(-90, 90, num=91)  # Reduced number of points
        lons = np.linspace(-180, 180, num=181)  # Reduced number of points
        
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        
        location = EarthLocation.from_geodetic(lon_grid, lat_grid, height=0)
        altaz_frame = AltAz(obstime=final_time, location=location)
        altaz_sun = sun_coord.transform_to(altaz_frame)

        sza = 90 - altaz_sun.alt.deg
        
        return sza, lat_grid, lon_grid
    
    @staticmethod
    def night_side_coordinates(sza, lat_grid, lon_grid):
        night_side_mask = sza > 90

        night_side_lats = lat_grid[night_side_mask].flatten().tolist()
        night_side_lons = lon_grid[night_side_mask].flatten().tolist()

        return night_side_lons, night_side_lats
