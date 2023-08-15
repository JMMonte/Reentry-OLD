import numpy as np

# -----------------
# GENERAL CONSTANTS
# -----------------

PI = np.pi
DEG_TO_RAD = float(PI / 180.0) # degrees to radians
RAD_TO_DEG = float(180.0 / PI) # radians to degrees
JD_AT_0 = 2451545.0 # Julian date at 0 Jan 2000
G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant

# -----------------
#EARTH CONSTANTS
# -----------------

EARTH_MU = 398600441800000.0  # gravitational parameter of Earth in m^3/s^2
EARTH_R = 6378137.0  # radius of Earth in m
EARTH_R_KM = 6378.137  # radius of Earth in m
EARTH_R_POLAR = 6356752.3142  # polar radius of Earth in m
EARTH_OMEGA = 7.292114146686322e-05  # Earth rotation speed in rad/s
EARTH_J2 = 0.00108263 # J2 of Earth
EARTH_MASS = 5.972e24  # Mass (kg)
EARTH_ROT_PERIOD_S = 23.9344696 * 3600 # Earth rotation period in seconds
EARTH_ROT_SPEED_RAD_S = 2 * np.pi / EARTH_ROT_PERIOD_S # Earth rotation speed in rad/s
EARTH_ROT_SPEED_DEG_S = 360 / EARTH_ROT_PERIOD_S # Earth rotation speed in deg/s
EARTH_ROT_SPEED_M_S = EARTH_R * EARTH_ROT_SPEED_RAD_S # Earth rotation speed in m/s
EARTH_MU_KM = 3.986004418e5  # gravitational parameter of Earth in km^3/s^2
YEAR_D = 365.25636 # days in a year
YEAR_S = 365.25636 * 24 * 60 * 60  # seconds in a year
EARTH_ROT_S = 86164.0905  # Earth rotation period in seconds
EARTH_GRAVITY = 9.80665  # Gravity (m/s^2)
EARTH_ROTATION_RATE_DEG_PER_SEC = 360 / EARTH_ROT_S  # Earth rotation rate in degrees per second
EARTH_RORATION_RATE_RAD_PER_SEC = EARTH_ROTATION_RATE_DEG_PER_SEC * DEG_TO_RAD  # Earth rotation rate in radians per second
EARTH_PERIMETER = EARTH_R * 2 * PI  # Earth perimeter in meters
EARTH_ROTATION_RATE_M_S = EARTH_PERIMETER / EARTH_ROT_S  # Earth rotation rate in meters per second

# ------------------
# MATERIAL CONSTANTS
# ------------------

MATERIALS = {
    "Aluminum": {
        'thermal_conductivity': 237, # W/m*K
        'specific_heat_capacity': 903, # 0.94 J/kg*K
        'emissivity': 0.1,
        'ablation_efficiency': 0.1 # (assumed)
    },
    "Copper": {
        'thermal_conductivity': 401, # W/m*K
        'specific_heat_capacity': 385, # 0.39 J/kg*K
        'emissivity': 0.03,
        'ablation_efficiency': 0.1 # (assumed)
    },
    'PICA': {
        'thermal_conductivity': 0.167, # W/m*K
        'specific_heat_capacity': 1260, # 0.094 J/kg*K
        'emissivity': 0.9,
        'ablation_efficiency': 0.7 # (assumed)
    },
    "RCC": {
        'thermal_conductivity': 7.64, # W/m*K
        'specific_heat_capacity': 1670, # 1.67 J/kg*K
        'emissivity': 0.5,
        'ablation_efficiency': 0.99 # completely ablates (assumed)
    },
    "Cork": {
        'thermal_conductivity': 0.043, # W/m*K
        'specific_heat_capacity': 2100, # 2.01 J/kg*K
        'emissivity': 0.7,
        'ablation_efficiency': 0.3 # (assumed)
    },
    "InconelX": {
        'thermal_conductivity': 35.3, # W/m*K
        'specific_heat_capacity': 540, # 0.54 kJ/kg*K
        'emissivity': 0.2,
        'ablation_efficiency': 0.1 # (assumed)
    },
    "Alumina enhanced thermal barrier rigid tile": {
        'thermal_conductivity': 0.064, # W/m*K
        'specific_heat_capacity': 630, # 0.63 kJ/kg*K
        'emissivity': 0.9,
        'ablation_efficiency': 0.7 # (assumed)
    },
}

# ------------------
# MOON CONSTANTS
# ------------------

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

# ------------------
# SUN CONSTANTS
# ------------------

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

# ------------------
# ATMOSPHERE CONSTANTS
# ------------------

# U.S. Standard Atmosphere altitude breakpoints and temperature gradients (m, K/m)
ALTITUDE_BREAKPOINTS = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
TEMPERATURE_GRADIENTS = np.array([-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002, 0])
BASE_TEMPERATURES = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
BASE_PRESSURES = np.array([101325, 22632.1, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734])
BASE_DENSITIES = np.array([1.225, 0.36391, 0.08803, 0.01322, 0.00143, 0.00086, 0.000064, 0.000006])
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
ATMO_LAYERS = [
    (0, 11000, 'rgba(196, 245, 255, 1)', 'Troposphere'),
    (11000, 47000, 'rgba(0, 212, 255, 1)', 'Stratosphere'),
    (47000, 86000, 'rgba(4, 132, 202, 1)', 'Mesosphere'),
    (86000, 690000, 'rgba(9, 9, 121, 1)', 'Thermosphere'),
    (690000, 1000000, 'rgba(2, 1, 42, 1)', 'Exosphere'),
]

# ------------------
# THERMODYNAMICS CONSTANTS
# ------------------

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