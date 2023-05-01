from poliastro.bodies import Earth
from astropy import units as u
from astropy.time import Time, TimeDelta
import base64
import cartopy.feature as cfeature
from copy import deepcopy
from coordinate_converter import (eci_to_ecef, ecef_to_geodetic, ecef_to_enu, ecef_distance, haversine_distance)
import datetime
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from poliastro.twobody import Orbit
from spacecraft_model import SpacecraftModel
from spacecraft_visualization import SpacecraftVisualization
import streamlit as st
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from numba import jit
from spacecraft_model import atmosphere_model
import plotly.subplots as subplots

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

# Begin the app
#--------------------------------------------
# main variables
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

st.set_page_config(layout="wide", page_title="Spacecraft Reentry Simulation", page_icon="☄️")
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
        mass = st.number_input("Spacecraft mass (kg)", value=500.0, step=10.0, min_value=0.1, key="mass", help="Spacecraft mass (kg) denotes the total weight of a spacecraft, including its structure, fuel, and payload. It plays a vital role in orbital and reentry trajectory planning, as it influences propulsion requirements, momentum, and heating rates. A lower mass can ease maneuvering and reduce fuel consumption (e.g., Sputnik 1), while a higher mass can pose challenges for propulsion and deceleration (e.g., International Space Station). Accurate knowledge of spacecraft mass is essential for efficient trajectory planning and mission success.")
        area = st.number_input("Cross section area (m^2)", value=20, help="The cross-sectional area (A) refers to a spacecraft's projected area perpendicular to the direction of motion during orbital and reentry trajectories. By adjusting A in the number input, you can evaluate its influence on drag forces, deceleration, heating, and trajectory accuracy.:s Smaller cross-sectional areas lead to reduced drag forces (e.g., Mercury capsule, A ~ 1.2 m²), promoting stability and requiring less deceleration. Larger cross-sectional areas increase drag (e.g., SpaceX's Starship, A ~ 354.3 m²), aiding in deceleration but potentially increasing heating rates.:sProperly managing cross-sectional area based on the spacecraft's design ensures optimized flight paths and successful mission outcomes.")
        codrag = st.number_input("Drag coefficient", value=1.3, min_value=0.0, help="The drag coefficient (Cd) quantifies a spacecraft's aerodynamic resistance during orbital and reentry trajectories. By adjusting Cd in the number input, you can explore its impact on the spacecraft's deceleration, heating, and trajectory accuracy.:s Lower Cd values indicate reduced aerodynamic drag (e.g., Mars Science Laboratory, Cd ~ 0.9), leading to smoother reentry and longer deceleration times. Higher Cd values result in increased drag (e.g., Apollo Command Module, Cd ~ 1.3), causing more rapid deceleration and potentially higher heating rates.:s Optimizing Cd based on the spacecraft's shape and design helps ensure efficient trajectory planning and mission success.")
        Cp = st.number_input("Heat transfer coefficient", value=500.0, min_value=0.0, help='''
        The heat transfer coefficient (Cp) represents a spacecraft's ability to absorb, conduct, and radiate heat during orbital and reentry trajectories. By adjusting Cp in the number input, you can simulate how the spacecraft's thermal performance affects its mission outcome.:s Lower Cp values indicate reduced heat transfer (e.g., Apollo Command Module, Cp ~ 300 W), leading to slower heating rates and prolonged reentry durations. Higher Cp values imply increased heat transfer (e.g., Space Shuttle, Cp ~ 800 W), resulting in faster heating rates and shorter reentry durations.:s Carefully selecting Cp based on spacecraft materials and design ensures safe and accurate trajectory planning for successful missions.''')
    with st.expander("Edit initial state", expanded=True):
        v = st.number_input("Orbital Velocity (m/s)", value=7.5e3, step=1e2, key="velocity", help="Orbital velocity (V) is the speed required for a spacecraft to maintain a stable orbit around a celestial body. By adjusting V in the number input, you can analyze its impact on orbit design, altitude, period, and mission objectives.:s For example, geostationary satellites orbit at a higher altitude than low Earth orbit spacecraft, but they have a lower orbital velocity (e.g., V ~ 3.07 km/s). The geostationary transfer orbit, on the other hand, is a high-velocity maneuver orbit used to transfer a spacecraft from low Earth orbit to geostationary orbit. This transfer orbit has a higher velocity than geostationary orbit (e.g., V ~ 10.3 km/s at perigee).:s Selecting an appropriate orbital velocity based on mission requirements and spacecraft capabilities ensures efficient orbit design and mission success.")
        azimuth = st.number_input("Azimuth (degrees)", value=90.0, min_value=0.0, max_value=360.0, step=1.0, key="azimuth", help="Azimuth represents the spacecraft's angle relative to a reference direction during orbital and reentry trajectories. By adjusting the azimuth in the number input, you can simulate how the spacecraft's orientation affects its mission outcome.:s Properly managing the spacecraft's azimuth is crucial for achieving optimal trajectory accuracy and minimizing aerodynamic drag. For example, during reentry, a steeper azimuth angle can result in higher heating rates due to increased deceleration, while a shallower angle can lead to a longer reentry duration.:s Historic missions such as Apollo 11 and the Space Shuttle program used specific azimuth angles to achieve their mission objectives. Apollo 11 had a roll angle of 69.5 degrees during reentry, while the Space Shuttle typically used an azimuth angle of around 40 degrees for its deorbit burn.:s Selecting the appropriate azimuth angle depends on the spacecraft's objectives and design. Properly managing the azimuth angle can help ensure safe and accurate trajectory planning for successful missions.")
        gamma = st.number_input("Flight path angle or gamma (degrees)", value=-5.0, min_value=-90.0, max_value=90.0, step=1.0, key="gamma", help="The flight path angle or gamma (degrees) represents the angle between the spacecraft's velocity vector and the local horizontal during orbital and reentry trajectories. By adjusting the flight path angle in the number input, you can simulate how the spacecraft's angle affects its mission outcome.:s Lower flight path angles (e.g., SpaceX's Dragon spacecraft, gamma ~ -12 degrees) result in steeper trajectories, leading to higher deceleration and increased heating rates during reentry. Higher flight path angles (e.g., Apollo Command Module, gamma ~ -6 degrees) result in shallower trajectories, leading to lower deceleration and reduced heating rates during reentry.:s Properly managing the flight path angle ensures optimized trajectory planning for successful missions, balancing the need for deceleration and minimizing heating effects.")
        lat = st.number_input("Latitude (deg)", value=45.0, min_value=-90.0, max_value=90.0, step=1.0, key="latitude")
        lon = st.number_input("Longitude (deg)", value=-75.0, min_value=-180.0, max_value=180.0, step=1.0, key="longitude")
        alt = st.number_input("Altitude (km)", value=500.0, step=100.0, key="altitude", help="Orbital altitude refers to the distance between the spacecraft and the Earth's surface during orbital trajectory planning. By changing the orbital altitude in the number input, you can simulate how it affects the spacecraft's orbital period, velocity, and energy requirements.:s Lower orbital altitudes (e.g., Low Earth Orbit, ~400 km) result in shorter orbital periods and higher spacecraft velocities. Higher orbital altitudes (e.g., Geostationary Orbit, ~36,000 km) lead to longer orbital periods and lower spacecraft velocities.:s The selected orbital altitude must consider the mission objectives, such as Earth observation, communication, or space exploration, and the spacecraft's capabilities, such as propulsion and power requirements. Careful planning of the orbital altitude ensures the successful accomplishment of the mission goals.")
        clock = st.time_input("Spacecraft Clock", value=datetime.time(20, 00), key="clock", help="The start time of the mission simulation.")
        calendar = st.date_input("Spacecraft Calendar", value=datetime.date.today(), key="calendar", help="The start date of the mission simulation.")

    # convert datetime to astropy time
    spacecraft_datetime_string = f"{calendar} {clock.hour}:{clock.minute}:{clock.second}"
    
    epoch = Time(spacecraft_datetime_string, format="iso", scale='tdb')
    
    gmst0 = epoch.sidereal_time('mean', 'greenwich').to_value(u.rad) # get the greenwich mean sidereal time
    
    y0 = spacecraft.get_initial_state(v=v, lat=lat, lon=lon, alt=alt * 1000, azimuth=azimuth,gamma=gamma, gmst=gmst0)
    spacecraft = SpacecraftModel(Cd=codrag, A=area,m=mass, epoch=epoch, Cp=Cp, gmst0=gmst0)
    # Define integration parameters
    orbit = Orbit.from_vectors(Earth, y0[0:3] * u.m, y0[3:6] * u.m / u.s, epoch)
    with sidebar.expander("Initial Orbit parameters"):
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
    st.subheader("Simulation Parameters")
    ts = 0 # initial time in seconds
    tf = st.number_input("Simulation duration (s)", min_value=0 , value=3700, step=1, key="tf", help="How long do you want to simulate the spacecraft?")  # final time in seconds
    dt = st.number_input("Time step (s)", min_value=0 , value=10, step=1, key="dt", help="The simulation will be broken down into a time step. Shorter timesteps give more precision but will increase the processing time.")  # time step in seconds
    sim_type = st.selectbox("Solver method", ["RK45", "RK23","DOP853","Radau","BDF","LSODA"], key="sim_type", help="The integration method to be used by the simulation physics solver. Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used for non-stiff problems and implicit methods ('Radau', 'BDF') for stiff problems. Among Runge-Kutta methods, 'DOP853' is recommended for solving with high precision (low values of `rtol` and `atol`).:s If not sure, first try to run 'RK45'. If it makes unusually many iterations, diverges, or fails, your problem is likely to be stiff and you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal choice, but it might be somewhat less convenient to work with as it wraps old Fortran code.:s You can also pass an arbitrary class derived from `OdeSolver` which implements the solver.")
    # Extract vectors for charts
    x_pos, y_pos, z_pos = y0[0:3] # Extract the position components
    x_vel, y_vel, z_vel = y0[3:6] # Extract the velocity components

    # Scale factor for the velocity vector
    scale_factor = 500  # Adjust this value to scale the velocity vector
    vel_arrow = visualization.create_3d_arrow(x_pos, y_pos, z_pos, x_pos + x_vel * scale_factor, y_pos + y_vel * scale_factor, z_pos + z_vel * scale_factor, 'green', 'Velocity vector') # Velocity vector scaled
    pos_arrow = visualization.create_3d_arrow(0, 0, 0, x_pos, y_pos, z_pos, 'red', 'Position vector') # Position vector

    # Update time span and t_eval based on ts
    t_span = (ts, tf)  # time span tuple
    t_eval = np.arange(ts, tf, dt)  # time array for output
    # get orbit from vectors

# Simulation
#--------------------------------------------
# Run the simulation

def update_progress(progress, elapsed_time):
    progress_bar.progress(progress,f"🔥 Cooking your TPS... {elapsed_time:.2f} seconds elapsed")
progress_bar = st.empty()

if run_simulation:
    progress_bar = st.progress(0)
    image = st.image("https://freight.cargo.site/t/original/i/9f858547e3d3296fcbdbb516f7788fa4f0f7413b754ffd259c5f954726f380ce/reentry.gif", use_column_width=True)
    sol = spacecraft.run_simulation(t_span, y0, t_eval, progress_callback=update_progress) # Run the simulation
    progress_bar.empty()
    image.empty()

if not run_simulation:
    # 3D Earth figure
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

additional_data = sol.additional_data # Compute additional parameters

# Unpack additional data
velocity = additional_data['velocity']
total_acceleration = additional_data['acceleration']
earth_grav_acceleration = additional_data['gravitational_acceleration']
j2_acceleration = additional_data['J2_acceleration']
moon_acceleration = additional_data['moon_acceleration']
drag_acceleration = additional_data['drag_acceleration']
altitude = additional_data['altitude']
sun_acceleration = additional_data['sun_acceleration']
heat_rate = additional_data['heat_rate']

# normalize each acceleration vector
velocity_norm = np.linalg.norm(velocity, axis=0)
total_acceleration_norm = np.linalg.norm(total_acceleration, axis=0)
earth_grav_acceleration_norm = np.linalg.norm(earth_grav_acceleration, axis=0)
j2_acceleration_norm = np.linalg.norm(j2_acceleration, axis=0)
moon_acceleration_norm = np.linalg.norm(moon_acceleration, axis=0)
drag_acceleration_norm = np.linalg.norm(drag_acceleration, axis=0)
sun_acceleration_norm = np.linalg.norm(sun_acceleration, axis=0)

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

# get last position
touchdown_r = r_eci[-1]

# Upload/download simulation data
#--------------------------------------------
data = {
    't': t_sol,
    'x': r_eci[0],
    'y': r_eci[1],
    'z': r_eci[2],
    'vx': sol.y[3],
    'vy': sol.y[4],
    'vz': sol.y[5],
    'altitude': altitude,
    'heat_rate': heat_rate,
    'velocity_x': [velocity[i][0] for i in range(len(velocity))],
    'velocity_y': [velocity[i][1] for i in range(len(velocity))],
    'velocity_z': [velocity[i][2] for i in range(len(velocity))],
    'total_acceleration_x': [total_acceleration[i][0] for i in range(len(total_acceleration))],
    'total_acceleration_y': [total_acceleration[i][1] for i in range(len(total_acceleration))],
    'total_acceleration_z': [total_acceleration[i][2] for i in range(len(total_acceleration))],
    'earth_grav_acceleration_x': [earth_grav_acceleration[i][0] for i in range(len(earth_grav_acceleration))],
    'earth_grav_acceleration_y': [earth_grav_acceleration[i][1] for i in range(len(earth_grav_acceleration))],
    'earth_grav_acceleration_z': [earth_grav_acceleration[i][2] for i in range(len(earth_grav_acceleration))],
    'j2_acceleration_x': [j2_acceleration[i][0] for i in range(len(j2_acceleration))],
    'j2_acceleration_y': [j2_acceleration[i][1] for i in range(len(j2_acceleration))],
    'j2_acceleration_z': [j2_acceleration[i][2] for i in range(len(j2_acceleration))],
    'moon_acceleration_x': [moon_acceleration[i][0] for i in range(len(moon_acceleration))],
    'moon_acceleration_y': [moon_acceleration[i][1] for i in range(len(moon_acceleration))],
    'moon_acceleration_z': [moon_acceleration[i][2] for i in range(len(moon_acceleration))],
    'drag_acceleration_x': [drag_acceleration[i][0] for i in range(len(drag_acceleration))],
    'drag_acceleration_y': [drag_acceleration[i][1] for i in range(len(drag_acceleration))],
    'drag_acceleration_z': [drag_acceleration[i][2] for i in range(len(drag_acceleration))],
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

    trajectory_trace = SpacecraftVisualization.create_3d_scatter(r_eci[0], r_eci[1], r_eci[2], heat_rate, name='Simulated trajectory', colorscale='Agsunset')
    fig3.add_trace(trajectory_trace)

    marker_color = 'purple' if altitude_event_times.size > 0 else 'red'
    marker_name = 'Touchdown' if altitude_event_times.size > 0 else 'Final position'
    fig3.add_trace(go.Scatter3d(x=[r_eci[0, -1]], y=[r_eci[1, -1]], z=[r_eci[2, -1]], mode='markers', marker=dict(size=6, color=marker_color), name=marker_name))

    if altitude_event_times.size > 0:
        fig3.update_layout(scene=dict(annotations=[dict(x=r_eci[0, -1], y=r_eci[1, -1], z=r_eci[2, -1], text="Touchdown", showarrow=True)]))

    impact_time = t_sol[-1]

    if crossing_points is not None:
        crossing_points_r_x = np.float64(crossing_points_r[0])
        crossing_points_r_y = np.float64(crossing_points_r[1])
        crossing_points_r_z = np.float64(crossing_points_r[2])
        fig3.add_trace(go.Scatter3d(x=[crossing_points_r_x], y=[crossing_points_r_y], z=[crossing_points_r_z], mode='markers', marker=dict(size=6, color='orange'), name='Karman line crossing'))

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
    # Normalize heat_rate data
    vmin, vmax = np.min(heat_rate), np.max(heat_rate)
    normalized_heat_rate = (heat_rate - vmin) / (vmax - vmin)

    # Calculate tick values and tick text for the subdivisions
    num_subdivisions = 10
    heat_rate_tickvals = np.linspace(0, 1, num_subdivisions)
    heat_rate_ticktext = [f"{vmin + tick * (vmax - vmin):.3E}" for tick in heat_rate_tickvals]

    colormap = cm.get_cmap('plasma')
    custom_colorscale = mpl_to_plotly_colormap(colormap)
    fig_colorscale = go.Figure()
    fig_colorscale.add_trace(go.Heatmap(
        x=sol.t,  # Add this line to set x values to sol.t
        z=[normalized_heat_rate],
        text=[[f"{value:.3E} W" for value in heat_rate]],  # Update text values to include units
        hoverinfo='x+y+text',
        colorscale=custom_colorscale,
        showscale=True,
        colorbar=dict(
            title="Heat rate [W]",
            titleside="bottom",
            x=0.5,
            lenmode="fraction",
            len=1,
            yanchor="top",
            y=-1.1,
            thicknessmode="pixels",
            thickness=20,
            orientation="h",
            tickvals=heat_rate_tickvals,
            ticktext=heat_rate_ticktext,
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
    col2.warning(f"⚠️ You're a fireball! Crossing the Karman line at {crossing_points} seconds after start intial time, experiencing a maximum deceleration of {max(gs_acceleration):.2f} G")

else:
    col2.success("Still flying high")
    # calculate final time of simulation using astropy


final_time = epoch + TimeDelta(impact_time, format='sec')
col2.warning(f"🌡️ The spacecraft reached a heat rate of {max(heat_rate):.3E} W during simulation. You can see what parts of the orbit were the hottest in the 3d plot above👆.")
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
    fig4.update_layout(xaxis_title='Time (s)', yaxis_title='Altitude (m)', legend=dict(y=1.3, yanchor="top", xanchor="left", x=0, orientation="h"), hovermode="x unified")
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

# Custom function to convert matplotlib colormap to Plotly colorscale
altitudes = altitudes / 1000 # convert to km
colormap = cm.get_cmap('plasma')
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
    vertical_velocity_ecef = v_ecef_vals[:, 2]  # Vertical velocity in ECEF frame

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

    ground_velocity_magnitudes = np.linalg.norm(sol.y[3:5, :], axis=0)
    
    orbital_velocity = np.linalg.norm(sol.y[3:6, :], axis=0)
    
    fig5.add_trace(go.Scatter(x=t_sol, y=orbital_velocity, mode='lines', line=dict(color='white', width=2), name='Orbital Velocity'))

    fig5.add_trace(go.Scatter(x=t_sol, y=ground_velocity_ecef, mode='lines', line=dict(color='orange', width=2), name='Ground Velocity (ECEF)'))
    fig5.add_trace(go.Scatter(x=t_sol, y=vertical_velocity_ecef, mode='lines', line=dict(color='yellow', width=2), name='Vertical Velocity (ECEF)'))

    fig5.add_trace(go.Scatter(x=t_sol, y=ground_velocity_geodetic, mode='lines', line=dict(color='purple', width=2), name='Ground Velocity (Geodetic)'))
    fig5.add_trace(go.Scatter(x=t_sol, y=vertical_velocity_geodetic, mode='lines', line=dict(color='brown', width=2), name='Vertical Velocity (Geodetic)'))

    fig5.add_trace(go.Scatter(x=t_sol, y=sol.y[3, :], mode='lines', line=dict(color='blue', width=2), name='X Velocity'))
    fig5.add_trace(go.Scatter(x=t_sol, y=sol.y[4, :], mode='lines', line=dict(color='red', width=2), name='Y Velocity'))
    fig5.add_trace(go.Scatter(x=t_sol, y=sol.y[5, :], mode='lines', line=dict(color='green', width=2), name='Z Velocity'))

    # calculate highest value and lowest in the chart given allthe data
    max_velocity = max(np.max(ground_velocity_ecef), np.max(vertical_velocity_ecef), np.max(ground_velocity_geodetic), np.max(vertical_velocity_geodetic), np.max(sol.y[3, :]), np.max(sol.y[4, :]), np.max(sol.y[5, :]))
    min_velocity = min(np.min(ground_velocity_ecef), np.min(vertical_velocity_ecef), np.min(ground_velocity_geodetic), np.min(vertical_velocity_geodetic), np.min(sol.y[3, :]), np.min(sol.y[4, :]), np.min(sol.y[5, :]))

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
    accelerations = [
        (np.linalg.norm(total_acceleration, axis=1), 'Total acceleration'),
        (np.linalg.norm(earth_grav_acceleration, axis=1), 'Earth grav. acceleration'),
        (np.linalg.norm(j2_acceleration, axis=1), 'J2 acceleration'),
        (np.linalg.norm(moon_acceleration, axis=1), 'Moon grav. acceleration'),
        (np.linalg.norm(drag_acceleration, axis=1), 'Drag acceleration'),
        (np.linalg.norm(sun_acceleration, axis=1), 'Sun grav. acceleration')
    ]

    fig8 = go.Figure([go.Scatter(x=sol.t, y=acc[0], name=acc[1]) for acc in accelerations])
    max_accel = max(np.max(acc[0]) for acc in accelerations)
    min_accel = min(np.min(acc[0]) for acc in accelerations)

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
st.subheader('Heat rate over time')
'''
This graph represents the simplified reentry heating rate ($q$) as a function of spacecraft velocity ($V$) in the Earth-Centered Inertial (ECI) reference frame. The equation used to estimate the heating rate is:

$$
q = 0.5 * rho * V^3 * C_p * A
$$

where:
- $q$: heat rate (W)
- $\rho$: atmospheric density (kg/m³)
- $V$: spacecraft velocity (m/s) in the ECI frame
- $C_p$: spacecraft heat transfer coefficient (W)
- $A$: spacecraft cross-sectional area (m²)

The model assumes constant atmospheric density, spacecraft parameters (e.g., cross-sectional area, heat transfer coefficient), and specific heat ratio. It is important to note that this is a simplified model and may not provide accurate heating predictions for real reentry scenarios. For more accurate estimations, consider using more complex models or simulation tools.

'''

fig9 = go.Figure()
fig9.add_trace(go.Scatter(x=sol.t, y=heat_rate, name='Heat rate'))
fig9.update_layout(
    xaxis_title='Time (s)',
    yaxis_title='Heat rate (W)',
    autosize=True,
    margin=dict(l=0, r=0, t=60, b=0),
    legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig9, use_container_width=True)


# Explain atmospheric model used
#--------------------------------------------------------------------------------
st.subheader('The Atmospheric model')
r'''
The atmospheric model used in this simulation folows the US Standard Atmosphere and is a combination of linear and exponential models to estimate the temperature, pressure, and density of Earth's atmosphere at a given altitude. The model divides the atmosphere into different layers based on altitude breakpoints and uses different temperature gradients for each layer.
'''
r'''**Linear Model:**:s
The linear model is used within the altitude intervals defined by `ALTITUDE_BREAKPOINTS`. For each interval, the temperature and pressure are calculated as follows:
'''
col5, col6 = st.columns(2)
with col5.expander("Temperature"):
    r'''
    The temperature at a given altitude is calculated using the base temperature of the interval and the temperature gradient.

     $$T = T_{base} + L \times (\text{altitude} - \text{altitude}_\text{base})$$

    where $T$ is the temperature, $T_{base}$ is the base temperature of the interval, $L$ is the temperature gradient, and $\text{altitude}_\text{base}$ is the base altitude of the interval.
    '''
with col6.expander("Pressure"):
    r'''
    The pressure at a given altitude is calculated using the base pressure of the interval and the temperature obtained in the previous step.

   - If the temperature gradient is zero:
   
     $$P = P_{base} \times \exp\left(-\frac{g \times M \times (\text{altitude} - \text{altitude}_\text{base})}{R \times T_{base}}\right)$$

   - If the temperature gradient is not zero:
   
     $$P = P_{base} \times \left(\frac{T}{T_{base}}\right)^{-\frac{g \times M}{R \times L}}$$

   where $P$ is the pressure, $P_{base}$ is the base pressure of the interval, $g$ is Earth's gravity, $M$ is Earth's air molar mass, $R$ is Earth's gas constant, and $T$ is the temperature obtained in the previous step.
    '''

r'''**Exponential Model:**

If the altitude is higher than the highest altitude breakpoint, the exponential model is used to approximate the temperature and pressure:'''
col7, col8 = st.columns(2)
with col7.expander("Temperature"):
    r'''
    The temperature at this altitude is assumed to be constant and equal to the base temperature of the highest altitude breakpoint.

     $$T = T_{base, last}$$
    '''
with col8.expander("Pressure"):
    r'''
    The pressure at this altitude is calculated using the base pressure of the highest altitude breakpoint and an exponential decay.

     $$P = P_{base, last} \times \exp\left(-\frac{\text{altitude} - \text{altitude}_\text{base, last}}{H}\right)$$

    where $H$ is the scale height.'''

r'''**Density Calculation:**

The density at any altitude is calculated using the ideal gas law:

 $$\rho = \frac{P}{R_\text{gas} \times T}$$

where $\rho$ is the density, $P$ is the pressure, $R_\text{gas}$ is the specific gas constant, and $T$ is the temperature.

This atmospheric model provides an approximation of the density and temperature at different altitudes in Earth's atmosphere, taking into account the changes in temperature gradients and pressure throughout the atmospheric layers.
'''
altitudes = np.linspace(0, 100000, num=1000)
temperatures = []
densities = []
temperatures = np.zeros(altitudes.shape)
densities = np.zeros(altitudes.shape)

for i, altitude in enumerate(altitudes):
    rho, T = atmosphere_model(altitude)
    temperatures[i] = T
    densities[i] = rho

# Create a Plotly chart with two x-axes
fig = subplots.make_subplots(specs=[[{"secondary_y": True}]])
fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})

# Add temperature trace
fig.add_trace(go.Scatter(x=temperatures, y=altitudes, name='Temperature (K)', mode='lines'), secondary_y=False)

# Add density trace
fig.add_trace(go.Scatter(x=densities, y=altitudes, name='Density (kg/m³)', mode='lines'), secondary_y=True)
fig.data[1].update(xaxis='x2')

# Update layout
fig.update_layout(
    title='Atmospheric Temperature and Density vs. Altitude',
    xaxis_title='Temperature (K)',
    xaxis2_title='Density (kg/m³)',
    yaxis_title='Altitude (m)',
    legend_title='Parameters',
    height = 800,
)

# Run Streamlit app
st.plotly_chart(fig, use_container_width=True)

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