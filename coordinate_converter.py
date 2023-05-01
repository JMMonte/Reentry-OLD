import math
import numpy as np
from astropy import units as u
from numba import jit

#constants
# Operations
PI = np.pi
DEG_TO_RAD = float(PI / 180.0) # degrees to radians
RAD_TO_DEG = float(180.0 / PI) # radians to degrees
JD_AT_0 = 2451545.0 # Julian date at 0 Jan 2000
#Earth
# Constants
EARTH_R = 6378137.0  # Earth's mean radius in meters
EARTH_FLATTENING = 1.0 / 298.257223563  # Earth's flattening factor (WGS84)
EARTH_E2 = 2.0 * EARTH_FLATTENING - EARTH_FLATTENING**2  # Earth's second eccentricity squared
EARTH_MU = 398600441800000.0  # gravitational parameter of Earth in m^3/s^2
EARTH_R_KM = 6378.137  # radius of Earth in m
EARTH_R_POLAR = 6356752.3142  # polar radius of Earth in m
EARTH_OMEGA = 7.292114146686322e-05  # Earth rotation speed in rad/s
EARTH_J2 = 0.00108263 # J2 of Earth
EARTH_MASS = 5.972e24  # Mass (kg)
EARTH_ROT_S = 86164.0905  # Earth rotation period in seconds
EARTH_ROTATION_RATE_DEG_PER_SEC = 360.0 / EARTH_ROT_S  # Earth rotation rate in degrees per second

# conversions ---------------------------------------------------------------

def eci_to_ecef(vector_eci, gmst):
    # Calculate the rotation matrix
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    rotation_matrix = np.array([[cos_gmst, sin_gmst, 0],
                                [-sin_gmst, cos_gmst, 0],
                                [0, 0, 1]])

    # Convert the ECI vector to ECEF
    vector_ecef = rotation_matrix @ vector_eci
    return vector_ecef

@jit
def ecef_to_eci(vector_ecef, gmst):
    # Calculate the rotation matrix (transpose of the ECI to ECEF rotation matrix)
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    rotation_matrix = np.array([[cos_gmst, -sin_gmst, 0],
                                [sin_gmst, cos_gmst, 0],
                                [0, 0, 1]])

    # Convert the ECEF vector to ECI
    vector_eci = rotation_matrix @ vector_ecef
    return vector_eci

@jit(nopython=True)
def geodetic_to_spheroid(lat, lon, alt):
    lat = DEG_TO_RAD * lat
    lon = DEG_TO_RAD * lon
    N = EARTH_R / np.sqrt(1 - EARTH_E2 * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - EARTH_E2) * N + alt) * np.sin(lat)
    return x, y, z

@jit(nopython=True)
def ecef_to_geodetic(x, y, z, max_iter=100, tol=1e-6):
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * EARTH_R, p * (1 - EARTH_E2 * EARTH_R))
    lat = np.arctan2(z + EARTH_E2 * EARTH_R * np.sin(theta)**3, p - EARTH_E2 * EARTH_R * np.cos(theta)**3)

    for _ in range(max_iter):
        N = EARTH_R / np.sqrt(1 - EARTH_E2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat_new = np.arctan2(z / p, 1 - EARTH_E2 * N / (N + h))
        if np.abs(lat - lat_new) < tol:
            break
        lat = lat_new

    lon = np.arctan2(y, x)
    return np.degrees(lat), np.degrees(lon), h

@jit(nopython=True)
def ecef_to_enu(x_ecef, y_ecef, z_ecef, lat, lon):
    x = -np.cos(DEG_TO_RAD*lon)*np.sin(DEG_TO_RAD*lat)*x_ecef - np.sin(DEG_TO_RAD*lon)*np.sin(DEG_TO_RAD*lat)*y_ecef + np.cos(DEG_TO_RAD*lat)*z_ecef
    y = -np.sin(DEG_TO_RAD*lon)*x_ecef + np.cos(DEG_TO_RAD*lon)*y_ecef
    z = np.cos(DEG_TO_RAD*lon)*np.cos(DEG_TO_RAD*lat)*x_ecef + np.sin(DEG_TO_RAD*lon)*np.cos(DEG_TO_RAD*lat)*y_ecef + np.sin(DEG_TO_RAD*lat)*z_ecef
    return x, y, z

@jit(nopython=True)
def enu_to_ecef(v_enu, lat, lon):
    lat_rad = DEG_TO_RAD * lat
    lon_rad = DEG_TO_RAD * lon

    R = np.array([[-np.sin(lon_rad), -np.sin(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.cos(lon_rad)],
                  [np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad) * np.sin(lon_rad)],
                  [0, np.cos(lat_rad), np.sin(lat_rad)]])

    v_ecef = R @ v_enu

    return v_ecef

@jit(nopython=True)
def ecef_distance(x1, y1, z1, x2, y2, z2):
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return np.sqrt(dx**2 + dy**2 + dz**2)

@jit(nopython=True)
def haversine_distance(lat1, lon1, lat2, lon2, R=EARTH_R):
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
