from poliastro.bodies import Earth
from astropy import units as u
from astropy.time import Time, TimeDelta
import base64
import cartopy.feature as cfeature
from copy import deepcopy
from coordinate_converter import (eci_to_ecef, ecef_to_geodetic, haversine_distance)
import datetime
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from poliastro.twobody import Orbit
from poliastro.core.elements import coe2rv
from spacecraft_model import SpacecraftModel, solar_activity_factor
from spacecraft_visualization import SpacecraftVisualization, create_capsule
import streamlit as st
import matplotlib.colors as mcolors
import matplotlib as mpl
from numba import jit
from spacecraft_model import atmosphere_model
import plotly.subplots as subplots
from plotly.subplots import make_subplots

#Constants
#Earth
EARTH_MU = 398600441800000.0  # gravitational parameter of Earth in m^3/s^2
EARTH_R = 6378137.0  # radius of Earth in m
EARTH_R_KM = 6378.137  # radius of Earth in m
EARTH_R_POLAR = 6356752.3142  # polar radius of Earth in m
EARTH_OMEGA = 7.292114146686322e-05  # Earth rotation speed in rad/s
EARTH_J2 = 0.00108263 # J2 of Earth
EARTH_MASS = 5.972e24  # Mass (kg)
#attractor_R, attractor_Rot_period
EARTH_ROT_PERIOD_S = 23.9344696 * 3600 # Earth rotation period in seconds
EARTH_ROT_SPEED_RAD_S = 2 * np.pi / EARTH_ROT_PERIOD_S # Earth rotation speed in rad/s
EARTH_ROT_SPEED_DEG_S = 360 / EARTH_ROT_PERIOD_S # Earth rotation speed in deg/s
EARTH_ROT_SPEED_M_S = EARTH_R * EARTH_ROT_SPEED_RAD_S # Earth rotation speed in m/s
# materials with name, lorentz number, density, specific heat, melting point, and emissivity
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


# Special functions
#--------------------------------------------
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

def update_progress(progress, elapsed_time):
    progress_bar.progress(progress,f"🔥 Cooking your TPS... {elapsed_time:.2f} seconds elapsed")

def periapsis_apoapsis_points(orbit: Orbit):
    k = orbit.attractor.k.to_value("km^3 / s^2")

    # Calculate the true anomalies for periapsis and apoapsis
    nu_periapsis = 0
    nu_apoapsis = np.pi

    # Calculate the position and velocity vectors for periapsis and apoapsis in ECI frame
    r_periapsis_ECI, v_periapsis_ECI = coe2rv(k, orbit.p.to_value('km'), orbit.ecc.value, orbit.inc.to_value('rad'), orbit.raan.to_value('rad'), orbit.argp.to_value('rad'), nu_periapsis)
    r_apoapsis_ECI, v_apoapsis_ECI = coe2rv(k, orbit.p.to_value('km'), orbit.ecc.value, orbit.inc.to_value('rad'), orbit.raan.to_value('rad'), orbit.argp.to_value('rad'), nu_apoapsis)

    # Convert the position vectors from km to m
    r_periapsis_ECI = r_periapsis_ECI * 1000
    r_apoapsis_ECI = r_apoapsis_ECI * 1000

    return r_periapsis_ECI, r_apoapsis_ECI

# Initialize the spacecraft model and visualization
spacecraft = SpacecraftModel()
visualization = SpacecraftVisualization()
data_directory = "data"
coastline_feature = cfeature.COASTLINE
country_feature = cfeature.BORDERS

layout = go.Layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    margin=dict(r=0, l=0, t=0, b=0),
    height=600,
)
spheroid_mesh = visualization.create_spheroid_mesh() # Add the new trajectory trace with altitude-based coloring




st.set_page_config(
    page_title="Spacecraft Reentry Simulator",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Begin the app
#--------------------------------------------
# main variables

st.title("Spacecraft Reentry Simulator")
# Sidebar user inputs
#--------------------------------------------
sidebar = st.sidebar
with st.sidebar:
    st.title("Mission Parameters")
    # Define initial state (position and velocity vectors)
    with st.expander("ℹ Help"):
        r'''
        This app aims to simulate the complex dynamics of a spacecraft orbits around the Earth taking into account the Earth's rotation, J2 perturbations, atmospheric drag, the sun's gravity and the Moon's gravity while predicting the spacecraft's trajectory.:s
        Before running your simulation, you can edit the spacecraft's initial state and the simulation parameters.👇 :s
        The simulation uses the amazing [poliastro](https://docs.poliastro.space/en/stable/) library, as well as [astropy](https://www.astropy.org/).:s
        To learn more about reentry astrodynamics I really reccommend this summary from the Aerostudents website: [Reentry](https://www.aerostudents.com/courses/rocket-motion-and-reentry-systems/ReEntrySummary.pdf).:s
        Made with ❤️ by [João Montenegro](https://monte-negro.space/).
        '''
    run_simulation = st.button("Run Simulation")
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(204, 49, 49);
    }
    </style>""", unsafe_allow_html=True)
    with st.expander("Edit spacecraft"):
        mass = st.number_input("Spacecraft mass (kg)", value=5000.0, step=10.0, min_value=0.1, key="mass", help="Spacecraft mass (kg) denotes the total weight of a spacecraft, including its structure, fuel, and payload. It plays a vital role in orbital and reentry trajectory planning, as it influences propulsion requirements, momentum, and heating rates. A lower mass can ease maneuvering and reduce fuel consumption (e.g., Sputnik 1), while a higher mass can pose challenges for propulsion and deceleration (e.g., International Space Station). Accurate knowledge of spacecraft mass is essential for efficient trajectory planning and mission success.")
        area = st.number_input("Cross section area (m^2)", value=14, help="The cross-sectional area (A) refers to a spacecraft's projected area perpendicular to the direction of motion during orbital and reentry trajectories. By adjusting A in the number input, you can evaluate its influence on drag forces, deceleration, heating, and trajectory accuracy.:s Smaller cross-sectional areas lead to reduced drag forces (e.g., Mercury capsule, A ~ 1.2 m²), promoting stability and requiring less deceleration. Larger cross-sectional areas increase drag (e.g., SpaceX's Starship, A ~ 354.3 m²), aiding in deceleration but potentially increasing heating rates.:sProperly managing cross-sectional area based on the spacecraft's design ensures optimized flight paths and successful mission outcomes.")
        codrag = st.number_input("Drag coefficient", value=1.3, min_value=0.0, help="The drag coefficient (Cd) quantifies a spacecraft's aerodynamic resistance during orbital and reentry trajectories. By adjusting Cd in the number input, you can explore its impact on the spacecraft's deceleration, heating, and trajectory accuracy.:s Lower Cd values indicate reduced aerodynamic drag (e.g., Mars Science Laboratory, Cd ~ 0.9), leading to smoother reentry and longer deceleration times. Higher Cd values result in increased drag (e.g., Apollo Command Module, Cd ~ 1.3), causing more rapid deceleration and potentially higher heating rates.:s Optimizing Cd based on the spacecraft's shape and design helps ensure efficient trajectory planning and mission success.")
        material_names = [*MATERIALS]
        selected_material = st.selectbox("Heat shield material", material_names, index=2, help="The heat shield material (M) refers to the material used to protect a spacecraft from the intense heat generated during reentry. By adjusting M in the dropdown menu, you can analyze its impact on the spacecraft's heating rate, deceleration, and trajectory accuracy.:s For example, the Apollo Command Module used an ablative heat shield (M ~ Avcoat), which gradually burned away during reentry to dissipate heat. The Space Shuttle, on the other hand, used a reusable heat shield (M ~ Reinforced Carbon-Carbon), which could withstand multiple reentries.:s Selecting an appropriate heat shield material based on the spacecraft's design and mission requirements ensures efficient trajectory planning and mission success.")
        selected_material_obj = MATERIALS[selected_material]
        material_properties = list(selected_material_obj.values())
        st.write(f"Thermal conductivity (k): {MATERIALS[selected_material]['thermal_conductivity']} W/m*K")
        st.write(f"Specific heat capacity (c): {MATERIALS[selected_material]['specific_heat_capacity']} J/kg*K")
        st.write(f"Emissivity (e): {MATERIALS[selected_material]['emissivity']}")
        st.write(f"Ablation_efficiency: {MATERIALS[selected_material]['ablation_efficiency']}")
    with st.expander("Edit initial state", expanded=True):
        v = st.number_input("Orbital Velocity (m/s)", value=7540.0, step=1e2, key="velocity", help="Orbital velocity (V) is the speed required for a spacecraft to maintain a stable orbit around a celestial body. By adjusting V in the number input, you can analyze its impact on orbit design, altitude, period, and mission objectives.:s For example, geostationary satellites orbit at a higher altitude than low Earth orbit spacecraft, but they have a lower orbital velocity (e.g., V ~ 3.07 km/s). The geostationary transfer orbit, on the other hand, is a high-velocity maneuver orbit used to transfer a spacecraft from low Earth orbit to geostationary orbit. This transfer orbit has a higher velocity than geostationary orbit (e.g., V ~ 10.3 km/s at perigee).:s Selecting an appropriate orbital velocity based on mission requirements and spacecraft capabilities ensures efficient orbit design and mission success.")
        azimuth = st.number_input("Azimuth (degrees)", value=90.0, min_value=0.0, max_value=360.0, step=1.0, key="azimuth", help="Azimuth represents the spacecraft's angle relative to a reference direction during orbital and reentry trajectories. By adjusting the azimuth in the number input, you can simulate how the spacecraft's orientation affects its mission outcome.:s Properly managing the spacecraft's azimuth is crucial for achieving optimal trajectory accuracy and minimizing aerodynamic drag. For example, during reentry, a steeper azimuth angle can result in higher heating rates due to increased deceleration, while a shallower angle can lead to a longer reentry duration.:s Historic missions such as Apollo 11 and the Space Shuttle program used specific azimuth angles to achieve their mission objectives. Apollo 11 had a roll angle of 69.5 degrees during reentry, while the Space Shuttle typically used an azimuth angle of around 40 degrees for its deorbit burn.:s Selecting the appropriate azimuth angle depends on the spacecraft's objectives and design. Properly managing the azimuth angle can help ensure safe and accurate trajectory planning for successful missions.")
        gamma = st.number_input("Flight path angle or gamma (degrees)", value=-2.0, min_value=-90.0, max_value=90.0, step=1.0, key="gamma", help="The flight path angle or gamma (degrees) represents the angle between the spacecraft's velocity vector and the local horizontal during orbital and reentry trajectories. By adjusting the flight path angle in the number input, you can simulate how the spacecraft's angle affects its mission outcome.:s Lower flight path angles (e.g., SpaceX's Dragon spacecraft, gamma ~ -12 degrees) result in steeper trajectories, leading to higher deceleration and increased heating rates during reentry. Higher flight path angles (e.g., Apollo Command Module, gamma ~ -6 degrees) result in shallower trajectories, leading to lower deceleration and reduced heating rates during reentry.:s Properly managing the flight path angle ensures optimized trajectory planning for successful missions, balancing the need for deceleration and minimizing heating effects.")
        lat = st.number_input("Latitude (deg)", value=45.0, min_value=-90.0, max_value=90.0, step=1.0, key="latitude")
        lon = st.number_input("Longitude (deg)", value=-75.0, min_value=-180.0, max_value=180.0, step=1.0, key="longitude")
        alt = st.number_input("Altitude (km)", value=500.0, step=100.0, key="altitude", help="Orbital altitude refers to the distance between the spacecraft and the Earth's surface during orbital trajectory planning. By changing the orbital altitude in the number input, you can simulate how it affects the spacecraft's orbital period, velocity, and energy requirements.:s Lower orbital altitudes (e.g., Low Earth Orbit, ~400 km) result in shorter orbital periods and higher spacecraft velocities. Higher orbital altitudes (e.g., Geostationary Orbit, ~36,000 km) lead to longer orbital periods and lower spacecraft velocities.:s The selected orbital altitude must consider the mission objectives, such as Earth observation, communication, or space exploration, and the spacecraft's capabilities, such as propulsion and power requirements. Careful planning of the orbital altitude ensures the successful accomplishment of the mission goals.")
        clock = st.time_input("Spacecraft Clock", value=datetime.time(20, 00), key="clock", help="The start time of the mission simulation.")
        calendar = st.date_input("Spacecraft Calendar", value=datetime.date.today(), key="calendar", help="The start date of the mission simulation.")

    # convert datetime to astropy time
    spacecraft_datetime_string = f"{calendar} {clock.hour}:{clock.minute}:{clock.second}"
    
    epoch = Time(spacecraft_datetime_string, format="iso", scale='tdb')
    
    gmst0 = epoch.sidereal_time('mean', 'greenwich').to_value(u.rad) # get the greenwich mean sidereal time
    
    with st.spinner("Loading spacecraft model..."):
        y0 = spacecraft.get_initial_state(v=v, lat=lat, lon=lon, alt=alt * 1000, azimuth=azimuth,gamma=gamma, gmst=gmst0)
    with st.expander("Simulation Parameters"):
        ts = 0 # initial time in seconds
        tf = st.number_input("Simulation duration (s)", min_value=0 , value=3700, step=1, key="tf", help="How long do you want to simulate the spacecraft?")  # final time in seconds
        dt = st.number_input("Time step (s)", min_value=0 , value=10, step=1, key="dt", help="The simulation will be broken down into a time step. Shorter timesteps give more precision but will increase the processing time.")  # time step in seconds
        sim_type = st.selectbox("Solver method", ["RK45", "RK23","DOP853","Radau","BDF","LSODA"], key="sim_type", help="The integration method to be used by the simulation physics solver. Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used for non-stiff problems and implicit methods ('Radau', 'BDF') for stiff problems. Among Runge-Kutta methods, 'DOP853' is recommended for solving with high precision (low values of `rtol` and `atol`).:s If not sure, first try to run 'RK45'. If it makes unusually many iterations, diverges, or fails, your problem is likely to be stiff and you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal choice, but it might be somewhat less convenient to work with as it wraps old Fortran code.:s You can also pass an arbitrary class derived from `OdeSolver` which implements the solver.")
        iter_fact = st.number_input("Iteration slowdown", value=3.0, min_value=0.0, help="Advanced: The iteration slowdown factor is used to slow down the temperature algorithm iterator. It has the purpose of fine tunning experimental data with simulation results. The default value is 2.0. If you are not sure, leave it as is.", key="iter_fact")
        max_points = st.number_input("Maximum number of points", value=10000, min_value=0, help="Advanced: The maximum number of points to use for the simulation. If the solver needs to take more points, it will increase the number of points in a logarithmic fashion. The default value is 10000. If you are not sure, leave it as is.", key="max_points")
    spacecraft = SpacecraftModel(Cd=codrag, A=area,m=mass, epoch=epoch, gmst0=gmst0, material=material_properties, dt=dt, iter_fact=iter_fact)
    # Define integration parameters
    orbit = Orbit.from_vectors(Earth, y0[0:3] * u.m, y0[3:6] * u.m / u.s, epoch)
    with sidebar.expander("Initial Orbit parameters", expanded=False):
        f'''
        Semimajor axis:s
        ${orbit.a}$

        Eccentricity:s
        ${orbit.ecc}$

        Inclination:s
        ${orbit.inc}$

        RAAN:s
        ${orbit.raan}$

        Argument of perigee:s
        ${orbit.argp}$

        True anomaly:s
        ${orbit.nu}$
        '''
    # Extract vectors for charts
    x_pos, y_pos, z_pos = y0[0:3] # Extract the position components
    x_vel, y_vel, z_vel = y0[3:6] # Extract the velocity components

    # Scale factor for the velocity vector
    scale_factor = 200  # Adjust this value to scale the velocity vector
    vel_arrow = visualization.create_3d_arrow(x_pos, y_pos, z_pos, x_pos + x_vel * scale_factor, y_pos + y_vel * scale_factor, z_pos + z_vel * scale_factor, 'green', 'Velocity vector') # Velocity vector scaled
    pos_arrow = visualization.create_3d_arrow(0, 0, 0, x_pos, y_pos, z_pos, 'red', 'Position vector') # Position vector

    # Update time span and t_eval based on ts
    t_span = (ts, tf)  # time span tuple
    t_eval = np.arange(ts, tf, dt)  # time array for output
    # get orbit from vectors


# Simulation
#--------------------------------------------
# Run the simulation
progress_bar = st.empty()
if run_simulation:
    progress_bar = st.progress(0)
    image = st.image("https://freight.cargo.site/t/original/i/9f858547e3d3296fcbdbb516f7788fa4f0f7413b754ffd259c5f954726f380ce/reentry.gif", use_column_width=True)
    sol = spacecraft.run_simulation(t_span, y0, t_eval, progress_callback=update_progress) # Run the simulation
    #add a data filter to limit the number of points displayed to a maximum of 150000
    #--------------------------------------------
    # Filter the data to a maximum of 150000 points
    
    if len(sol.t) > max_points:
        np.random.seed(0)
        indices = np.random.choice(len(sol.t), size=abs(len(sol.t) - max_points), replace=False)
        sol.t = np.delete(sol.t, indices)
        sol.y = np.delete(sol.y, indices, axis=1)
        if sol.additional_data:
            for key, value in sol.additional_data.items():
                if value.ndim == 1:
                    if np.max(indices) < value.shape[0]:
                        sol.additional_data[key] = np.delete(value, indices)
                    else:
                        print(f"Index {np.max(indices)} is out of bounds for axis 0 with size {value.shape[0]} for key {key}")
                elif value.ndim == 2:
                    if np.max(indices) < value.shape[1]:
                        sol.additional_data[key] = np.delete(value, indices, axis=1)
                    else:
                        print(f"Index {np.max(indices)} is out of bounds for axis 1 with size {value.shape[1]} for key {key}")
                else:
                    print(f"Value array for key {key} has an unexpected dimension")
        #--------------------------------------------
    progress_bar.empty()
    image.empty()

if not run_simulation:
    # 3D Earth figure
    with st.spinner("Loading 3D Earth figure..."):
        st.info('''Welcome to the your heatshield's worst nightmare!:s
        👈 To get started, open the sidebar and choose your spacecraft's characteristics and initial conditions. You can see your projected orbit below.:s Then choose the amount of time you want to simulate and press the big red flamy button.''')
        fig2 = go.Figure(layout=layout)

        # Add orbit trace
        fig2.add_trace(spheroid_mesh)
        orbit_trace = visualization.plot_orbit_3d(orbit, color='green', name='Classical orbit', dash='dot')
        fig2.add_trace(orbit_trace)

        # Add position and velocity arrows
        for trace in pos_arrow + vel_arrow:
            fig2.add_trace(trace)

        # Add coastline traces
        for feature in [country_feature, coastline_feature]:
            for trace in visualization.get_geo_traces(feature, gmst0):
                trace.showlegend = False
                fig2.add_trace(trace)

        # Add latitude and longitude lines
        for lines in [visualization.create_latitude_lines(gmst=gmst0), visualization.create_longitude_lines(gmst=gmst0)]:
            for line in lines:
                line.showlegend = False
                fig2.add_trace(line)

        # Add periapsis and apoapsis points
        periapsis_ECI, apoapsis_ECI = periapsis_apoapsis_points(orbit)
        # Calculate the altitude of periapsis and apoapsis points
        periapsis_altitude = (orbit.r_p.to_value('m') - EARTH_R) / 1000
        apoapsis_altitude = (orbit.r_a.to_value('m') - EARTH_R) / 1000

        # Add the periapsis marker
        fig2.add_trace(go.Scatter3d(x=[periapsis_ECI[0]],
                                    y=[periapsis_ECI[1]],
                                    z=[periapsis_ECI[2]],
                                    mode='markers',
                                    marker=dict(size=5, color='red', symbol='circle'),
                                    name='Periapsis'))

        # Add the apoapsis marker
        fig2.add_trace(go.Scatter3d(x=[apoapsis_ECI[0]],
                                    y=[apoapsis_ECI[1]],
                                    z=[apoapsis_ECI[2]],
                                    mode='markers',
                                    marker=dict(size=5, color='green', symbol='circle'),
                                    name='Apoapsis'))
        

        # Add the starting point marker
        fig2.add_trace(go.Scatter3d(x=[x_pos],
                                    y=[y_pos],
                                    z=[z_pos],
                                    mode='markers',
                                    marker=dict(size=5, color='yellow', symbol='circle'),
                                    name='Start'))
        
        
        # Add annotations for periapsis and apoapsis
        annotations_trace = go.Scatter3d(
                        x=[periapsis_ECI[0], apoapsis_ECI[0], x_pos],
                        y=[periapsis_ECI[1], apoapsis_ECI[1], y_pos],
                        z=[periapsis_ECI[2], apoapsis_ECI[2], z_pos],
                        mode='text',
                        text=[f"Periapsis<br>Altitude:<br>{periapsis_altitude:.2f} km",
                            f"Apoapsis<br>Altitude:<br>{apoapsis_altitude:.2f} km",
                            f"Start<br>Position<br>Altitude:<br>{alt:.2f} km"],
                        textfont=dict(color=["red", "green", "yellow"], size=12),
                        textposition="bottom center",
                        hoverinfo="none",
                        showlegend=False
        )
        # Add annotations to figure
        fig2.add_trace(annotations_trace)

        # Update layout
        fig2.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),
                        scene_camera=dict(eye=dict(x=-0.5, y=0.6, z=1)),
                        scene_aspectmode='data')

        st.plotly_chart(fig2, use_container_width=True, equal_axes=True)
        st.stop()




#--------------------------------------------
# unpack the solution
t_sol = sol.t # Extract the time array
r_eci = sol.y[0:3] # Extract the ECI coordinates
v_eci = sol.y[3:6] # Extract the ECI velocity components
additional_data = sol.additional_data # Compute additional parameters

# Calculate GMST for each time step
gmst_vals = gmst0 + EARTH_OMEGA * t_sol

# Convert ECI to ECEF and then to geodetic coordinates
r_ecef_vals = []
v_ecef_vals = []
geodetic_coords = []

for r, v, gmst in zip(r_eci.T, v_eci.T, gmst_vals):
    r_ecef = eci_to_ecef(r, gmst)
    v_ecef = eci_to_ecef(v, gmst)
    lat, lon, alt = ecef_to_geodetic(r_ecef[0], r_ecef[1], r_ecef[2])

    r_ecef_vals.append(r_ecef)
    v_ecef_vals.append(v_ecef)
    geodetic_coords.append([lat, lon, alt])

r_ecef_vals = np.array(r_ecef_vals)
v_ecef_vals = np.array(v_ecef_vals)
geodetic_coords = np.array(geodetic_coords)


# Unpack additional data
velocity = additional_data['velocity']
total_acceleration = additional_data['acceleration']
earth_grav_acceleration = additional_data['gravitational_acceleration']
j2_acceleration = additional_data['J2_acceleration']
moon_acceleration = additional_data['moon_acceleration']
drag_acceleration = additional_data['drag_acceleration']
altitude = additional_data['altitude']
sun_acceleration = additional_data['sun_acceleration']
T_aw_data = additional_data['spacecraft_temperature']
q_net_data = additional_data['spacecraft_heat_flux']
q_c_data = additional_data['spacecraft_heat_flux_conduction']
q_r_data = additional_data['spacecraft_heat_flux_radiation']
q_gen_data = additional_data['spacecraft_heat_flux_total']
dT_data = additional_data['spacecraft_temperature_change']


# normalize each acceleration vector
velocity_norm = np.linalg.norm(velocity, axis=1)
total_acceleration_norm = np.linalg.norm(total_acceleration, axis=1)
earth_grav_acceleration_norm = np.linalg.norm(earth_grav_acceleration, axis=1)
j2_acceleration_norm = np.linalg.norm(j2_acceleration, axis=1)
moon_acceleration_norm = np.linalg.norm(moon_acceleration, axis=1)
drag_acceleration_norm = np.linalg.norm(drag_acceleration, axis=1)
sun_acceleration_norm = np.linalg.norm(sun_acceleration, axis=1)

# convert to g's
gs_acceleration = total_acceleration_norm / 9.80665 # convert to g's

# compute downrange distance
downrange_distances = [0]

for i in range(1, len(geodetic_coords)):
    lat1, lon1, _ = geodetic_coords[i - 1]
    lat2, lon2, _ = geodetic_coords[i]
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    downrange_distances.append(downrange_distances[-1] + distance)
crossing_points_downrange, crossing_points = visualization.find_crossing_points(t_sol, downrange_distances, altitude, threshold=100000)

closest_indices = np.abs(np.subtract.outer(t_sol, crossing_points)).argmin(axis=0)
crossing_points_r_v = sol.y[:, closest_indices]
crossing_points_r = crossing_points_r_v[:3]

# Get touchdown array
altitude_event_times = sol.t_events[0]

# flatten the array
touchdown_time = np.int16(t_sol[-1])


# Upload/download simulation data
#--------------------------------------------
def match_array_length(array, target_length):
    if len(array) > target_length:
        return array[:target_length]
    elif len(array) < target_length:
        return np.pad(array, (0, target_length - len(array)), mode='constant')
    else:
        return array
    
arrays_to_match = [
    velocity_norm, altitude, drag_acceleration_norm, T_aw_data,
    r_eci[0], r_eci[1], r_eci[2], sol.y[3], sol.y[4], sol.y[5]
] + [
    [vec[i] for i in range(len(vec))]
    for vec in [velocity, total_acceleration, earth_grav_acceleration,
                j2_acceleration, moon_acceleration, drag_acceleration]
    for dim in range(3)
]

matched_arrays = [match_array_length(arr, len(sol.t)) for arr in arrays_to_match]


data = {
    't': t_sol,
    'velocity': matched_arrays[0],
    'altitude': matched_arrays[1],
    'drag_acceleration': matched_arrays[2],
    'spacecraft_temperature': matched_arrays[3],
    'x': matched_arrays[4],
    'y': matched_arrays[5],
    'z': matched_arrays[6],
    'vx': matched_arrays[7],
    'vy': matched_arrays[8],
    'vz': matched_arrays[9],
    'velocity_x': matched_arrays[10],
    'velocity_y': matched_arrays[11],
    'velocity_z': matched_arrays[12],
    'total_acceleration_x': matched_arrays[13],
    'total_acceleration_y': matched_arrays[14],
    'total_acceleration_z': matched_arrays[15],
    'earth_grav_acceleration_x': matched_arrays[16],
    'earth_grav_acceleration_y': matched_arrays[17],
    'earth_grav_acceleration_z': matched_arrays[18],
    'j2_acceleration_x': matched_arrays[19],
    'j2_acceleration_y': matched_arrays[20],
    'j2_acceleration_z': matched_arrays[21],
    'moon_acceleration_x': matched_arrays[22],
    'moon_acceleration_y': matched_arrays[23],
    'moon_acceleration_z': matched_arrays[24],
    'drag_acceleration_x': matched_arrays[25],
    'drag_acceleration_y': matched_arrays[26],
    'drag_acceleration_z': matched_arrays[27],
}

df = pd.DataFrame(data)
# Display the download link in the Streamlit app
st.sidebar.markdown(make_download_link(df, 'simulated_data.csv', 'Download simulated data'), unsafe_allow_html=True)



# Plots
#--------------------------------------------
# Create the 3d header plot

with st.spinner("Generating trajectory 3d plot..."):
    fig3 = go.Figure(layout=layout)
    orbit_trace = visualization.plot_orbit_3d(orbit, color='green', name='Classical orbit', dash='dot')
    fig3.add_trace(orbit_trace)

    fig3.add_traces(pos_arrow + vel_arrow)

    trajectory_trace = SpacecraftVisualization.create_3d_scatter(r_eci[0], r_eci[1], r_eci[2], T_aw_data, name='Simulated trajectory', colorscale='Agsunset')
    fig3.add_trace(trajectory_trace)
    # Add the starting point marker and annotation
    fig3.add_trace(go.Scatter3d(x=[r_eci[0, 0]],
                                y=[r_eci[1, 0]],
                                z=[r_eci[2, 0]],
                                mode='markers+text',
                                marker=dict(size=5, color='yellow', symbol='circle'),
                                text=["Starting point"],
                                textposition="bottom center",
                                showlegend=False))

    marker_color = 'purple' if altitude_event_times.size > 0 else 'red'
    marker_name = 'Touchdown' if altitude_event_times.size > 0 else 'Final position'
    fig3.add_trace(go.Scatter3d(x=[r_eci[0, -1]], y=[r_eci[1, -1]], z=[r_eci[2, -1]], mode='markers', marker=dict(size=6, color=marker_color), name=marker_name))

    # Add the final position or touchdown marker and annotation
    if altitude_event_times.size > 0:
        text_label = "Touchdown"
    else:
        text_label = "Final position"

    fig3.add_trace(go.Scatter3d(x=[r_eci[0, -1]],
                                y=[r_eci[1, -1]],
                                z=[r_eci[2, -1]],
                                mode='markers+text',
                                marker=dict(size=6, color=marker_color),
                                text=[text_label],
                                textposition="bottom center",
                                showlegend=False))

    impact_time = t_sol[-1]

    if crossing_points is not None:
        crossing_points_r_x = np.float64(crossing_points_r[0])
        crossing_points_r_y = np.float64(crossing_points_r[1])
        crossing_points_r_z = np.float64(crossing_points_r[2])
        fig3.add_trace(go.Scatter3d(x=[crossing_points_r_x],
                            y=[crossing_points_r_y],
                            z=[crossing_points_r_z],
                            mode='markers+text',
                            marker=dict(size=6, color='orange'),
                            text=["Karman line crossing"],
                            textposition="bottom center",
                            showlegend=False))

    gmst = gmst0 + EARTH_OMEGA * impact_time
    spheroid_mesh = visualization.create_spheroid_mesh()
    fig3.add_trace(spheroid_mesh)

    country_traces = visualization.get_geo_traces(country_feature, gmst)
    coastline_traces = visualization.get_geo_traces(coastline_feature, gmst)
    lat_lines = visualization.create_latitude_lines(gmst=gmst)
    lon_lines = visualization.create_longitude_lines(gmst=gmst)

    for trace in country_traces + coastline_traces + lat_lines + lon_lines:
        trace.showlegend = False
        fig3.add_trace(trace)

    fig3.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))
    fig3.update_layout(scene_camera=dict(eye=dict(x=-0.5, y=0.6, z=1)))
    fig3.update_layout(scene_aspectmode='data')
    st.plotly_chart(fig3, use_container_width=True, equal_axes=True)
    
    # Show heat rate by time
    # Normalize spacecraft_temperature data
    vmin, vmax = np.min(T_aw_data), np.max(T_aw_data)
    normalized_spacecraft_temperature = (T_aw_data - vmin) / (vmax - vmin)

    # Calculate tick values and tick text for the subdivisions
    num_subdivisions = 10
    spacecraft_temperature_tickvals = np.linspace(0, 1, num_subdivisions)
    spacecraft_temperature_ticktext = [f"{vmin + tick * (vmax - vmin):.3E}" for tick in spacecraft_temperature_tickvals]

    colormap = mpl.colormaps.get_cmap('plasma')
    custom_colorscale = mpl_to_plotly_colormap(colormap)
    fig_colorscale = go.Figure()
    fig_colorscale.add_trace(go.Heatmap(
        x=t_sol,  # Add this line to set x values to sol.t
        z=[normalized_spacecraft_temperature],
        text=[[f"{value:.3E} K" for value in T_aw_data]],  # Update text values to include units
        hoverinfo='x+y+text',
        colorscale=custom_colorscale,
        showscale=True,
        colorbar=dict(
            title="Temperature at Stagnation Point [K]",
            titleside="bottom",
            x=0.5,
            lenmode="fraction",
            len=1,
            yanchor="top",
            y=-1.1,
            thicknessmode="pixels",
            thickness=20,
            orientation="h",
            tickvals=spacecraft_temperature_tickvals,
            ticktext=spacecraft_temperature_ticktext,
        ),
    ))

    fig_colorscale.update_layout(
        autosize=True,
        width=800,
        height=200,
        margin=dict(l=0, r=0, t=0, b=100),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        xaxis=dict(
            showticklabels=True,
            showgrid=True,
            zeroline=False,
            title="Time [s]",
        ),
        showlegend=True,
    )

    st.plotly_chart(fig_colorscale, use_container_width=True)




# Calculate impact time
#--------------------------------------------
st.subheader("Crash Detection")
col2, col3 = st.columns(2)
duration = datetime.timedelta(seconds=impact_time.astype(float))


if altitude_event_times > 0:
    # get location of impact in ECEF
    last_r_ecef = np.array([r_eci[0, -1], r_eci[1, -1], r_eci[2, -1]])
    # get location of impact in lat, lon, alt
    last_r_geo = ecef_to_geodetic(last_r_ecef[0], last_r_ecef[1], last_r_ecef[2])
    # break down the lat, lon
    last_r_lat = last_r_geo[0]
    last_r_lon = last_r_geo[1]
    col2.info(f"📍 Touchdown detected at {last_r_lat}ºN, {last_r_lon}ºE")
    col2.error(f"⚠️ Touchdown detected {duration} (hh,mm,ss) after start intial time.")
if len(crossing_points) > 0:
    col2.warning(f"⚠️ You're a fireball! Crossing the Karman line at {', '.join([str(item) for item in crossing_points])} seconds after start intial time, experiencing a maximum deceleration of {max(gs_acceleration):.2f} G")
    st.plotly_chart(create_capsule(), use_container_width=True)

else:
    col2.success("Still flying high")
    # calculate final time of simulation using astropy


final_time = epoch + TimeDelta(impact_time, format='sec')
col2.warning(f"🌡️ The spacecraft reached a temperature of {max(T_aw_data):.3E} K during simulation. You can see what parts of the orbit were the hottest in the 3d plot above👆.")
col3.info(f"⏰ The simulation start time was {epoch} and ended on: {final_time}, with a total time simulated of: {duration} (hh,mm,ss)")
col3.info(f"🛰️ The spacecraft was at a ground speed of {np.around(np.linalg.norm(v_ecef[-1]),2)}m/s and at an altitude of {altitude[-1]:.2f}m at the end of the simulation")



# Begin charts
#--------------------------------------------
# Plot the altitude vs time
#--------------------------------------------
st.subheader("Altitude vs Time")
'''
Here you can see the altitude of the spacecraft over time. The red line is the trendline of the altitude over time. The blue line is the altitude over time. The Karman line is the unnofficial altitude that some consider to be the begining of space. The Karman line is at 100km.

In this simulation we are using the COESA76 atmospheric model that considers the Earth's atmosphere to be composed of 6 layers. The first layer is the troposphere, the second layer is the stratosphere, the third layer is the mesosphere, the fourth layer is the thermosphere, the fifth layer is the exosphere, and the sixth layer is the ionosphere. The Karman line is located in the thermosphere layer.
The model considers the atmosphere from 0 to 1000km, after that the atmosphere is considered to be a vacuum.
'''
with st.spinner("Generating altitude vs time graph..."):
    fig4 = go.Figure()

    z = np.polyfit(t_sol, altitude, 1)
    p = np.poly1d(z)

    atmosphere_layers = [
        (0, 11000, 'rgba(196, 245, 255, 1)', 'Troposphere'),
        (11000, 47000, 'rgba(0, 212, 255, 1)', 'Stratosphere'),
        (47000, 86000, 'rgba(4, 132, 202, 1)', 'Mesosphere'),
        (86000, 690000, 'rgba(9, 9, 121, 1)', 'Thermosphere'),
        (690000, 1000000, 'rgba(2, 1, 42, 1)', 'Exosphere'),
    ]

    for layer_y0, layer_y1, layer_color, layer_name in atmosphere_layers:
        fig4.add_shape(type='rect', x0=0, x1=max(t_sol), y0=layer_y0, y1=layer_y1, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor=layer_color, opacity=0.3)
        fig4.add_annotation(x=0, y=layer_y1, text=layer_name, xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)

    fig4.add_trace(go.Scatter(x=t_sol, y=[100000]*len(t_sol), mode='lines', line=dict(color='rgba(255,255,255,0.5)', width=2, dash='dot'), name='Karman Line'))
    fig4.add_trace(go.Scatter(x=t_sol, y=altitude, mode='lines', line=dict(color='#ff00f7', width=2), name='Altitude (m)'))
    fig4.add_trace(go.Scatter(x=t_sol, y=p(t_sol), mode='lines', line=dict(color='cyan', width=2, dash='dot'), name=f'Trendline{p}'))

    if altitude_event_times.size > 0:
        fig4.add_trace(go.Scatter(x=[touchdown_time], y=np.linspace(0, max(altitude), len(t_sol)), mode='lines', line=dict(color='rgba(0, 255, 0, 0.5)', width=2, dash='dot'), name='Touchdown'))
        fig4.add_annotation(x=[touchdown_time], y=max(altitude), text='Touchdown', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

    if crossing_points is not None:
        for idx, crossing_point in enumerate(crossing_points):
            fig4.add_shape(type='line', x0=crossing_point, x1=crossing_point, y0=0, y1=max(altitude), yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot'))
            fig4.add_annotation(x=crossing_point, y=max(altitude), text=f'Crossing Karman line {idx+1}', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

    fig4.update_yaxes(range=[0, max(altitude)])
    fig4.update_layout(xaxis_title='Time (s)', yaxis_title='Altitude (m)', legend=dict(y=1.3, yanchor="top", xanchor="left", x=0, orientation="h"), hovermode="x unified",xaxis= {"range": [0, max(t_sol)]})
    st.plotly_chart(fig4, use_container_width=True)



# Plot the downrange vs altitude
#--------------------------------------------
st.subheader("Downrange vs Altitude")
'''
Here you can see the downrange distance of the spacecraft from the launch site as a function of altitude.
Downrange is being measured in absolute distance from the starting point.
'''
# Calculate downrange using ecef_distance function
# Begin plotting
with st.spinner("Generating Downrange vs Altitude graph..."):
    # downrange to km
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=downrange_distances, y=altitude, mode='lines', line=dict(color='purple', width=2), name='Altitude'))
    fig6.add_trace(go.Scatter(x=[0, max(downrange_distances)], y=[100000]*2, mode='lines', line=dict(color='rgba(255,255,255,0.5)', width=2, dash= 'dot'), name='Karman Line'))

    for layer in atmosphere_layers:
        fig6.add_shape(type='rect', x0=0, x1=max(downrange_distances), y0=layer[0], y1=layer[1], yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor=layer[2], opacity=0.3, name=layer[3])
        fig6.add_annotation(x=0, y=layer[1], text=layer[3], xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)

    if crossing_points_downrange is not None:
        for idx, crossing_point in enumerate(crossing_points_downrange):
            fig6.add_shape(type='line', x0=crossing_point, x1=crossing_point, y0=0, y1=max(altitude), yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot'))
            fig6.add_annotation(x=crossing_point, y=max(altitude), text=f'Crossing Karman line {idx+1}', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

    fig6.update_yaxes(range=[0, max(altitude)])
    fig6.update_layout(legend=dict(y=1.2, yanchor="top", xanchor="left", x=0, orientation="h"))
    fig6.update_layout(xaxis_title='Downrange (m)', yaxis_title='Altitude (m)', hovermode="x unified")
    st.plotly_chart(fig6, use_container_width=True)



#Plot a groundtrack
#--------------------------------------------
# add a streamlit selectbox to select the map projection
# complete list of map projections: https://plotly.com/python/map-projections/
st.subheader('Groundtrack Projection')
'''
Here you can see the groundtrack of the spacecraft as a function of time.
Groundtrack's are a way to visualize the path of a spacecraft on a map in reference to the Earth's surface.
To do this we need to adjust our original frame of reference (Earth-Centered Inertial) to a new frame of reference (Earth-Centered Earth-Fixed).
'''
#'equirectangular', 'mercator', 'orthographic', 'natural earth', 'kavrayskiy7', 'miller', 'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant', 'conic equal area', 'conic conformal', 'conic equidistant', 'gnomonic', 'stereographic', 'mollweide', 'hammer', 'transverse mercator', 'albers usa', 'winkel tripel', 'aitoff', 'sinusoidal']

fig7 = go.Figure()
# Convert ECEF to geodetic coordinates
latitudes, longitudes = geodetic_coords[:, 0], geodetic_coords[:, 1]
altitudes = geodetic_coords[:, 2]
# limit data to 25000 points

# Custom function to convert matplotlib colormap to Plotly colorscale
altitudes = altitudes / 1000 # convert to km
colormap = mpl.colormaps.get_cmap('viridis')
custom_colorscale = mpl_to_plotly_colormap(colormap)
vmin, vmax = np.min(altitudes), np.max(altitudes)
normalized_altitude = (altitudes - vmin) / (vmax - vmin)

# Number of subdivisions in the color scale
num_subdivisions = 10

# Calculate tick values and tick text for the subdivisions
tickvals = np.linspace(0, 1, num_subdivisions)
ticktext = [f"{vmin + tick * (vmax - vmin):.2f}" for tick in tickvals]

# Add the final position label
final_lat_str = f"{latitudes[-1]:.5f}"
final_lon_str = f"{longitudes[-1]:.5f}"
final_position_label = f"Final position<br>Lat: {final_lat_str}º North,<br>Lon: {final_lon_str}º East,<br>{altitudes[-1]:.2f} km Altitude"

with st.spinner("Generating Groundtrack map..."):
    # Add a single trace for the ground track
    fig7.add_trace(go.Scattergeo(
        lon=longitudes,
        lat=latitudes,
        mode='markers',
        marker=dict(
            color=normalized_altitude,
            size=2,
            colorscale=custom_colorscale,
            showscale=True,
            colorbar=dict(
                title="Altitude (km)",
                tickvals=tickvals,
                ticktext=ticktext,
            ),
        ),
        showlegend=True,
        name='Groundtrack',
    ))
    # Add lines with colors from the color scale
    for i in range(len(longitudes) - 1):
        start_lat, start_lon = latitudes[i], longitudes[i]
        end_lat, end_lon = latitudes[i + 1], longitudes[i + 1]

        line_color = get_color(normalized_altitude[i], colormap)
        fig7.add_trace(go.Scattergeo(
            lon=[start_lon, end_lon],
            lat=[start_lat, end_lat],
            mode='lines',
            line=dict(color=line_color, width=2),
        showlegend=False,
        name='Groundtrack',
        ))

    # add point for starting point and another for final position
    fig7.add_trace(go.Scattergeo(
        lat=[latitudes[-1]],
        lon=[longitudes[-1]],
        marker={
            "color": "Red",
            "line": {
                "width": 1
            },
            "size": 10
        },
        mode="markers+text",
        name="Final position",
        text=[final_position_label],
        textfont={
            "color": "White",
            "size": 16
        },
        textposition="top right",
        showlegend=True,
    ))
    fig7.add_trace(go.Scattergeo(
        lon=[longitudes[0]],
        lat=[latitudes[0]],
        mode='markers',
        marker=dict(color='green', size=10),
        showlegend=True,
        name='Initial position'
    ))

    fig7.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, t=50, b=0),
        height=800,
        geo=dict(
            showland=True,
            showcountries=True,
            showocean=True,
            showlakes=True,
            showrivers=True,
            countrywidth=0.5,
            landcolor='rgba(0, 110, 243, 0.2)',
            oceancolor='rgba(0, 0, 255, 0.1)',
            bgcolor="rgba(0, 0, 0, 0)",
            coastlinecolor='blue',
            projection=dict(type='equirectangular'),
            lonaxis=dict(range=[-180, 180], showgrid=True, gridwidth=0.5, gridcolor='rgba(0, 0, 255, 0.5)'),
            lataxis=dict(range=[-90, 90], showgrid=True, gridwidth=0.5, gridcolor='rgba(0, 0, 255, 0.5)'),
        ),
    )
    fig7.update_geos(resolution=110)
    fig7.update_layout(legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"))
    st.plotly_chart(fig7, use_container_width=True)
#--------------------------------------------------------------------------------

# Plot the velocity vs time
#--------------------------------------------
st.subheader("Velocity vs Time")
''' 
Here you can see the velocity of the spacecraft both in absolute magnitude as well as in the x, y, and z directions.
One particularly useful measure is the ground velocity vs the orbital velocity.
'''
with st.spinner("Generating Velocity vs time graph..."):
    fig5 = go.Figure()
    
    ground_velocity_ecef = np.linalg.norm(v_ecef_vals[:, :2], axis=1)  # Magnitude of ground velocity in ECEF frame
    # calculate vertical velocity based on rate of change of altitude
    vertical_velocity_ecef = np.gradient(altitudes, t_sol)

    ground_velocity_geodetic = np.zeros(len(t_sol))
    vertical_velocity_geodetic = np.zeros(len(t_sol))

    for i, (lat, lon, alt) in enumerate(geodetic_coords):
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        rotation_matrix = np.array([[-np.sin(lon_rad), -np.cos(lon_rad) * np.sin(lat_rad), np.cos(lon_rad) * np.cos(lat_rad)],
                                    [np.cos(lon_rad), -np.sin(lon_rad) * np.sin(lat_rad), np.sin(lon_rad) * np.cos(lat_rad)],
                                    [0, np.cos(lat_rad), np.sin(lat_rad)]])

        v_geodetic = np.dot(rotation_matrix, v_ecef_vals[i])
        ground_velocity_geodetic[i] = np.linalg.norm(v_geodetic[:2])
        vertical_velocity_geodetic[i] = v_geodetic[2]

    velocities = {
        'Orbital Velocity': velocity_norm,
        'Ground Velocity (ECEF)': ground_velocity_ecef,
        'Vertical Velocity (ECEF)': vertical_velocity_ecef,
        'Ground Velocity (Geodetic)': ground_velocity_geodetic,
        'Vertical Velocity (Geodetic)': vertical_velocity_geodetic,
        'X Velocity': sol.y[3, :],
        'Y Velocity': sol.y[4, :],
        'Z Velocity': sol.y[5, :]
    }

    fig5 = go.Figure([go.Scatter(x=t_sol, y=velocities[vel], mode='lines', name=vel) for vel in velocities])

    # calculate highest value and lowest in the chart given allthe data
    max_velocity = max(np.max(velocities[vel]) for vel in velocities)
    min_velocity = min(np.min(velocities[vel]) for vel in velocities)
    
    if crossing_points is not None:
        for idx, crossing_point in enumerate(crossing_points):
            fig5.add_shape(type='line', x0=crossing_point, x1=crossing_point, y0=min_velocity, y1=max_velocity, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot'))
            fig5.add_annotation(x=crossing_point, y=max_velocity, text=f'Crossing Karman line {idx+1}', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

    if altitude_event_times.size > 0:
        fig5.add_shape(type='line', x0=[touchdown_time], x1=[touchdown_time], y0=min_velocity, y1=max_velocity, yref='y', xref='x', line=dict(color='rgba(0, 255, 0, 0.5)', width=2, dash='dot'))
        fig5.add_annotation(x=[touchdown_time], y=max_velocity, text='Touchdown', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

    fig5.update_layout(xaxis_title='Time (s)', yaxis_title='Velocity (m/s)',legend=dict(y=1.2, yanchor="top", xanchor="left", x=0, orientation="h"),hovermode="x unified")
    st.plotly_chart(fig5, use_container_width=True)

# Plot an acceleration vs time graph
#--------------------------------------------------------------------------------
# include acceleration (final) gravitational_acceleration, drag_acceleration,moon_acceleration, and J2_acceleration.
st.subheader('Perturbations over time')
'''
Last but not least, we can plot the acceleration vs time graph. This will show us how the acceleration changes over time. We can see that the acceleration is initially very high, but then decreases as the spacecraft gets further away from the Earth.
In this simulation we are taking into account:

- Earth's gravitational acceleration
- Drag acceleration
- Moon's gravitational acceleration
- Sun's gravitational acceleration
- J2 acceleration

In our starting scenario (in Low Earth Orbit), you can see that the total acceleration is mainly affected by the Earth's gravitational acceleration. However, you can click on the legend to hide the total acceleration to adjust the graphs y axis so the other accelerations are visible.
'''
with st.spinner('Loading accelerations graph...'):
    accelerations = {
        'Total acceleration': total_acceleration_norm,
        'Earth grav. acceleration': earth_grav_acceleration_norm,
        'J2 acceleration': j2_acceleration_norm,
        'Moon grav. acceleration': moon_acceleration_norm,
        'Drag acceleration': drag_acceleration_norm,
        'Sun grav. acceleration': sun_acceleration_norm
    }

    fig8 = go.Figure([go.Scatter(x=t_sol, y=accelerations[acc], name=acc) for acc in accelerations])
    max_accel = max(np.max(accelerations[acc]) for acc in accelerations)
    min_accel = min(np.min(accelerations[acc]) for acc in accelerations)

    if crossing_points is not None:
        for idx, crossing_point in enumerate(crossing_points):
            fig8.add_shape(type='line', x0=crossing_point, x1=crossing_point, y0=min_accel, y1=max_accel, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot'))
            fig8.add_annotation(x=crossing_point, y=max_accel, text=f'Crossing Karman line {idx+1}', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

    if altitude_event_times.size > 0:
        # Add touchdown line
        fig8.add_shape(type='line', x0=[touchdown_time], x1=[touchdown_time], y0=min_accel, y1=max_accel, yref='y', xref='x', line=dict(color='rgba(0, 255, 0, 0.5)', width=2, dash='dot'))
        # Add annotation for touchdown
        fig8.add_annotation(x=[touchdown_time], y=max_accel, text='Touchdown', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

    fig8.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Acceleration (m/s^2)',
        autosize=True,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"),
        hovermode="x unified"
    )
    st.plotly_chart(fig8, use_container_width=True)

#--------------------------------------------------------------------------------
# plot heat rate by time
#--------------------------------------------------------------------------------
st.subheader('Spacecraft Temperature over time')
with st.expander("Click here to learn more about this simulator's temperature model"):
    r'''
    The spacecraft temperature model is a simplified model that calculates the temperature change of the spacecraft during its trajectory through the atmosphere. This model was selected because it approximates the heat generated by the spacecraft performing work against Earth's atmosphere. The model considers the following heat transfer mechanisms:

    1. Heat generated (Q) due to the work done (W) by the drag force on the spacecraft: :s
    $$Q = ablation\_efficiency \times W$$
    where $ablation\_efficiency$ is the ablation efficiency factor.

    2. Conductive heat transfer (Qc) between the spacecraft and the atmosphere: :s
    $$Q_c = thermal\_conductivity \times \frac{T_s - atmo\_T}{capsule\_length}$$
    where $thermal\_conductivity$ is the thermal conductivity of the heat shield material, $T_s$ is the spacecraft temperature, $atmo\_T$ is the atmospheric temperature, and $capsule\_length$ is the length of the capsule.

    3. Radiative heat transfer (Qr) between the spacecraft and the atmosphere: :s
    $$Q_r = emissivity \times \sigma \times (T_s^4 - atmo\_T^4)$$
    where $emissivity$ is the emissivity of the heat shield material, $\sigma$ is the Stefan-Boltzmann constant, $T_s$ is the spacecraft temperature, and $atmo\_T$ is the atmospheric temperature.

    The net heat transfer (Q_net) is the sum of the heat generated minus the conductive and radiative heat transfers: :s
    $$Q_{net} = Q - Q_c - Q_r$$

    The temperature change (dT) is calculated as the net heat transfer divided by the product of the spacecraft mass and specific heat capacity: :s
    $$dT = \frac{Q_{net}}{spacecraft\_m \times specific\_heat\_capacity}$$

    The updated spacecraft temperature is obtained by adding the temperature change to the current temperature: :s
    $$T_s = T_s + dT$$

    The model iterates through these calculations for a specified number of iterations, taking into account the altitude, velocity, atmospheric temperature, and drag acceleration experienced by the spacecraft. The final output includes the conductive, radiative, and net heat transfers, as well as the updated spacecraft temperature and the temperature change.

    '''

fig9 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

fig9.add_trace(go.Scatter(x=t_sol, y=T_aw_data, name='Spacecraft Temperature at Stagnation Point (K)'), row=1, col=1, secondary_y=False)
fig9.add_trace(go.Scatter(x=t_sol, y=dT_data, name='Temperature rate of change (K)'), row=1, col=1, secondary_y=False)
fig9.add_trace(go.Scatter(x=t_sol, y=q_net_data, name='Total heat transfer (W)'), row=1, col=1, secondary_y=True)
fig9.add_trace(go.Scatter(x=t_sol, y=q_r_data, name='Radiation heat transfer (W)'), row=1, col=1, secondary_y=True)
fig9.add_trace(go.Scatter(x=t_sol, y=q_gen_data, name='Total heat generated (W)'), row=1, col=1, secondary_y=True)

fig9.update_layout(
    xaxis_title='Time (s)',
    yaxis=dict(
        title="Temperature (K)",
        side="left",
    ),
    autosize=True,
    margin=dict(l=0, r=0, t=60, b=0),
    legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"),
    hovermode="x unified",
    yaxis2=dict(
        title="Heat (W)",
        side="right",
    ),
)

st.plotly_chart(fig9, use_container_width=True)


# Explain atmospheric model used
#--------------------------------------------------------------------------------
st.subheader('Atmospheric Model')
st.write('This simulation uses a simplified atmospheric model based on the NRLMSISE-00 and works by dividing the atmosphere into layers with specific temperature gradients and base pressures. The temperature and pressure at a given altitude are calculated, followed by the atmospheric density. The model then incorporates the latitude and solar activity factors to provide more accurate results for density and temperature.')
with st.expander("Click here to learn more about this simulator's atmospheric model"):
    r'''
    This atmospheric model is a simplified version of the NRLMSISE-00 model. It estimates the density and temperature of Earth's atmosphere as a function of altitude, latitude, and solar activity.

    The model divides the atmosphere into different layers, with each layer having specific temperature gradients and base pressures:

    - Troposphere (0 to 11 km)
    - Stratosphere (11 to 47 km)
    - Mesosphere (47 to 84.8 km)
    - Thermosphere (84.8 km to 150 km)

    The temperature at a given altitude is calculated using the temperature gradient and the base temperature for the corresponding layer:

    $$T = T_{base} + \Delta h * \frac{dT}{dh}$$

    where:
    - $T_{base}$ is the base temperature for the layer
    - $\Delta h$ is the altitude difference from the base altitude of the layer
    - $\frac{dT}{dh}$ is the temperature gradient for the layer

    The pressure at the given altitude is computed using either the barometric formula or the hypsometric equation, depending on whether the temperature gradient is zero:

    If the temperature gradient is zero ($\frac{dT}{dh} = 0$), the pressure is calculated using the barometric formula:

    $$P = P_{base} * \exp \left( -\frac{g M \Delta h}{R T_{base}} \right)$$

    Otherwise, if the temperature gradient is not zero, the pressure is calculated using the hypsometric equation:

    $$P = P_{base} * \left( \frac{T}{T_{base}} \right) ^{-\frac{g M}{R \frac{dT}{dh}}}$$


    where:
    - $P_{base}$ is the base pressure for the layer
    - $g$ is Earth's gravity
    - $M$ is the molar mass of Earth's air
    - $R$ is the specific gas constant for Earth's air
    - $T_{base}$ is the base temperature for the layer
    - $\frac{dT}{dh}$ is the temperature gradient for the layer

    In the exosphere (above 150 km), the pressure is exponentially decreased with altitude based on a scale height parameter:

    $$P *= \exp \left( -\frac{h - h_{exo}}{H} \right)$$

    where:
    - $h_{exo}$ is the altitude where the exosphere begins (150 km)
    - $H$ is the scale height

    The latitude factor is applied to account for the variation in atmospheric pressure due to latitude:

    $$P *= 1 + \frac{0.01 * |latitude|}{90}$$

    The density is then computed from the pressure and temperature:

    $$\rho = \frac{P}{R_{gas} * T}$$

    where:
    - $R_{gas}$ is the specific gas constant for Earth's air
    '''

altitudes_graph = np.linspace(0, 1000000, num=1000)
temperatures = []
densities = []
temperatures = np.zeros(altitudes_graph.shape)
densities = np.zeros(altitudes_graph.shape)

for i, altitude in enumerate(altitudes_graph):
    rho, T = atmosphere_model(altitude, 0 ,epoch.jd)
    temperatures[i] = T
    densities[i] = rho

# Create a Plotly chart with two x-axes
fig_atmo = subplots.make_subplots(specs=[[{"secondary_y": True}]])
fig_atmo.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})

# Add temperature trace
fig_atmo.add_trace(go.Scatter(x=temperatures, y=altitudes_graph, name='Temperature (K)', mode='lines', line=dict(color='red')), secondary_y=False)

# Add density trace
fig_atmo.add_trace(go.Scatter(x=densities, y=altitudes_graph, name='Density (kg/m³)', mode='lines',line=dict(color='green')), secondary_y=False)
fig_atmo.data[1].update(xaxis='x2')
for layer in atmosphere_layers:
        fig_atmo.add_shape(type='rect', x0=0, x1=max(temperatures), y0=layer[0], y1=layer[1], yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor=layer[2], opacity=0.3, name=layer[3])
        fig_atmo.add_annotation(x=0, y=layer[1], text=layer[3], xanchor='left', yanchor='bottom', font=dict(size=10,), showarrow=False)
# Update layout
fig_atmo.update_layout(
    title='Atmospheric Temperature and Density vs. Altitude',
    xaxis_title='Temperature (K)',
    xaxis2_title='Density (kg/m³)',
    yaxis_title='Altitude (m)',
    legend_title='Parameters',
    height = 800,
    hovermode="y unified"
)

# Run Streamlit app
st.plotly_chart(fig_atmo, use_container_width=True)

st.write("To address a more realistic atmospheric model, this simulator also includes a simple version of solar activity and includes it in the calculation of atmospheric density at high altitudes.")

# last 10 years of solar cycle
with st.expander("Click here to see how the atmospheric model accounts for solar activity"):
    r'''
        To account for the effect of solar activity, a solar activity factor is calculated based on the F10.7 index. A simple sinusoidal model is used to estimate the F10.7 value from the Julian date:

    $$F10.7 = F10.7_{avg} + A * \sin \left( \frac{t_{cycle}}{T_{cycle}} * 2\pi \right)$$

    where:
    - $F10.7_{avg}$ is the average F10.7 value
    - $A$ is the amplitude of the solar cycle
    - $t_{cycle}$ is the time since the last solar minimum in months
    - $T_{cycle}$ is the solar cycle period in months

    The solar activity factor is then calculated:

    $$factor = 1 + \frac{F10.7 - F10.7_{avg}}{F10.7_{avg}}$$

    Finally, the solar activity factor is applied to the calculated density:

    $$\rho *= factor$$

    The `atmosphere_model` function returns the density and temperature at the given altitude, latitude, and solar activity.

    '''
jd_start_sim = epoch.jd
jd_end_sim = epoch.jd + tf / (24 * 3600)
solar_dates_past = np.linspace(jd_start_sim - 365 * 20, jd_start_sim, num=int(365.3 * 10))
solar_data_past = np.array([solar_activity_factor(date) for date in solar_dates_past])
solar_dates_sim = np.linspace(jd_start_sim, jd_end_sim, num=int(tf))
solar_data_sim = np.array([solar_activity_factor(date) for date in solar_dates_sim])

# Convert solar dates to datetime
solar_dates_past = [Time(date, format='jd').datetime for date in solar_dates_past]
solar_dates_sim = [Time(date, format='jd').datetime for date in solar_dates_sim]

fig_solar = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1)

# Add historical solar factor data trace
fig_solar.add_trace(go.Scatter(
    x=solar_dates_past, 
    y=solar_data_past, 
    name='Historical Solar Activity Factor (Past 10 years)',
    mode='lines',
    line=dict(color='yellow'),
    fill='tozeroy',
    fillcolor='rgba(255, 255, 0, 0.3)'
), row=1, col=1)

# Add mission time solar factor data trace
fig_solar.add_trace(go.Scatter(
    x=solar_dates_sim,
    y=solar_data_sim,
    name='Mission Time Solar Activity Factor (Simulation)',
    mode='lines',
    line=dict(color='red'),
    fill='tozeroy',
    fillcolor='rgba(255, 0, 0, 0.3)'
), row=2, col=1)

fig_solar.update_layout(
    title='Solar Activity Factor: Historical vs. Mission Time',
    height=600,
    legend=dict(
        title=dict(text='Legend Title'),
        x=0.01,
        y=0.99,
        font=dict(size=12, color='black'),
        orientation='h'  # 'v' for vertical, 'h' for horizontal
    ), showlegend=False,
    hovermode="x unified"
)

fig_solar.update_xaxes(title_text='Date', row=1, col=1)
fig_solar.update_yaxes(title_text='Solar Activity Factor', row=1, col=1)

fig_solar.update_xaxes(title_text='Date', row=2, col=1)
fig_solar.update_yaxes(title_text='Solar Activity Factor', row=2, col=1)

st.plotly_chart(fig_solar, use_container_width=True)


'''
That's it for this dashboard. Try more combinations to see the effects of different parameters on the trajectory. Also, try landing with low Gs (your spacecraft will thank you...)!

You can reach me on [Twitter](https://twitter.com/JohnMontenegro) or [Github](https://github.com/JMMonte) or [Linkedin](https://www.linkedin.com/in/jmontenegrodesign/).
You can also visit my [personal website](https://monte-negro.space/).
'''


# about this app
st.sidebar.markdown('''
**About this app**

This app means to show the power of mixing streamlit, plotly, poliastro with scipy as the simulation engine.:s
All the code is available on Github and is free to use.:s
Many of these equations are a best effort to implement and are likely not the most accurate in their current form.:s
Therefore, any feedback is greatly welcome.
You can reach me on [Twitter](https://twitter.com/JohnMontenegro) or [Github](https://github.com/JMMonte) or [Linkedin](https://www.linkedin.com/in/jmontenegrodesign/).
You can also visit my [personal website](https://monte-negro.space/).
''')