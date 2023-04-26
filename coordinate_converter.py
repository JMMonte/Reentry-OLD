import numpy as np
from astropy import units as u
from poliastro.bodies import Earth

class CoordinateConverter:
    @staticmethod
    def ecef_to_eci(x, y, z, gmst):
        theta = (gmst * u.deg).to_value(u.rad)  # Convert GMST to radians and extract the value
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        eci_coords = np.dot(rotation_matrix, np.array([x, y, z]))
        return eci_coords

    
    @staticmethod
    def eci_to_ecef(x_eci, y_eci, z_eci, gmst):
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
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        R_lat = (R_equatorial * R_polar) / np.sqrt((R_equatorial * np.cos(lat_rad))**2 + (R_polar * np.sin(lat_rad))**2)
        
        x = (R_lat + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (R_lat + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (R_lat + alt) * np.sin(lat_rad)
        
        return x, y, z
    
    @staticmethod
    def geo_to_ecef(lon_rad, lat_rad, alt=0):
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

