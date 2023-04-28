import random
from astropy import units as u
from astropy.time import Time, TimeDelta
import base64
import cartopy.feature as cfeature
from copy import deepcopy
from coordinate_converter import CoordinateConverter
import datetime
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from spacecraft_model import SpacecraftModel
from spacecraft_visualization import SpacecraftVisualization
import streamlit as st


# Special functions
#--------------------------------------------
class LoadedSolution:
    def __init__(self, t, y):
        self.t = t
        self.y = y

def filter_results_by_altitude(sol, altitude):
    valid_indices = [i for i, alt in enumerate(altitude) if alt >= 0]

    sol = deepcopy(sol)
    sol.y = sol.y[:, valid_indices]
    sol.t = sol.t[valid_indices]
    sol.additional_data = [sol.additional_data[i] for i in valid_indices]

    return sol

# Create a download link for the simulated data
def make_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
#--------------------------------------------

# Begin the app
#--------------------------------------------
# main variables
spacecraft = SpacecraftModel()
visualization = SpacecraftVisualization()
convert = CoordinateConverter()
data_directory = "data"
coastline_feature = cfeature.COASTLINE
country_feature = cfeature.BORDERS
earth_rotation_deg_speed = convert.body_rotation_speed(Earth,'deg/s')
earth_rotational_speed = convert.body_rotation_speed(Earth, 'm/s')
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
        The heat transfer coefficient (Cp) represents a spacecraft's ability to absorb, conduct, and radiate heat during orbital and reentry trajectories. By adjusting Cp in the number input, you can simulate how the spacecraft's thermal performance affects its mission outcome.:s Lower Cp values indicate reduced heat transfer (e.g., Apollo Command Module, Cp ~ 300 W/m²·K), leading to slower heating rates and prolonged reentry durations. Higher Cp values imply increased heat transfer (e.g., Space Shuttle, Cp ~ 800 W/m²·K), resulting in faster heating rates and shorter reentry durations.:s Carefully selecting Cp based on spacecraft materials and design ensures safe and accurate trajectory planning for successful missions.''')
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
    gmst0 = epoch.sidereal_time('mean', 'greenwich').to_value(u.deg) # get the greenwich mean sidereal time

    y0 = spacecraft.get_initial_state(v=v, lat=lat, lon=lon, alt=alt * 1000, azimuth=azimuth,gamma=gamma, attractor=Earth, gmst=gmst0)
    spacecraft = SpacecraftModel(Cd=codrag, A=area,m=mass, epoch=epoch, Cp=Cp)
    # Define integration parameters
    st.subheader("Simulation Parameters")
    ts = 0 # initial time in seconds
    tf = st.number_input("Simulation duration (s)", min_value=0 , value=3700, step=1, key="tf", help="How long do you want to simulate the spacecraft?")  # final time in seconds
    dt = st.number_input("Time step (s)", min_value=0 , value=10, step=1, key="dt", help="The simulation will be broken down into a time step. Shorter timesteps give more precision but will increase the processing time.")  # time step in seconds

    # Extract vectors for charts
    x_pos, y_pos, z_pos = y0[0:3] # Extract the position components
    x_vel, y_vel, z_vel = y0[3:6] # Extract the velocity components

    # Scale factor for the velocity vector
    scale_factor = 500  # Adjust this value to scale the velocity vector
    vel_arrow = visualization.create_3d_arrow(x_pos, y_pos, z_pos, x_pos + x_vel * scale_factor, y_pos + y_vel * scale_factor, z_pos + z_vel * scale_factor, 'blue', 'Velocity vector') # Velocity vector scaled
    pos_arrow = visualization.create_3d_arrow(0, 0, 0, x_pos, y_pos, z_pos, 'red', 'Position vector') # Position vector

    # Update time span and t_eval based on ts
    t_span = (ts, tf)  # time span tuple
    t_eval = np.arange(ts, tf, dt)  # time array for output
    # get orbit from vectors
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
#--------------------------------------------

# Simulation
#--------------------------------------------
# Run the simulation

def update_progress(progress, elapsed_time):
    progress_bar.progress(progress,f"🔥 Burning some heatshields... {elapsed_time:.2f} seconds elapsed")
progress_bar = st.empty()

if run_simulation:
    progress_bar = st.progress(0)
    image = st.image("https://freight.cargo.site/t/original/i/9f858547e3d3296fcbdbb516f7788fa4f0f7413b754ffd259c5f954726f380ce/reentry.gif", use_column_width=True)
    sol = spacecraft.run_simulation(t_span, y0, t_eval, progress_callback=update_progress) # Run the simulation
    progress_bar.empty()
    image.empty()

if not run_simulation:
    # begin 3d earth figure
    fig2 = go.Figure()
    fig2 = go.Figure(layout=layout)
    orbit_trace = visualization.plot_orbit_3d(orbit,color='green',name='Classical orbit',dash= 'dot') # convert orbit into scatter3d plotly trace
    fig2.add_trace(orbit_trace)

    # Add position and velocity arrows to the plot
    for trace in pos_arrow:
        fig2.add_trace(trace)
    for trace in vel_arrow:
        fig2.add_trace(trace)

    # Recalculate the coastline traces based on the updated gmst0
    country_traces = visualization.get_geo_traces(country_feature, gmst0)
    coastline_traces = visualization.get_geo_traces(coastline_feature, gmst0)
    fig2.add_trace(spheroid_mesh)
    for trace in coastline_traces:
        trace.showlegend = False
        fig2.add_trace(trace)
        
    for trace in country_traces:
        trace.showlegend = False
        fig2.add_trace(trace)

    lat_lines = visualization.create_latitude_lines(gmst=gmst0)
    for lat_line in lat_lines:
        lat_line.showlegend = False
        fig2.add_trace(lat_line)

    lon_lines = visualization.create_longitude_lines(gmst=gmst0)
    for lon_line in lon_lines:
        lon_line.showlegend = False
        fig2.add_trace(lon_line)

    # set legend to bottom left
    fig2.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))
    #zooming in
    fig2.update_layout(scene_camera=dict(eye=dict(x=-0.5, y=0.6, z=1)))
    # keep axis aspect ratio always equal
    fig2.update_layout(scene_aspectmode='data')
    st.plotly_chart(fig2, use_container_width=True, equal_axes=True)
    st.stop()

#--------------------------------------------
# unpack the solution
t_sol = sol.t # Extract the time array
eci_coords = sol.y[0:3] # Extract the ECI coordinates
# coordinates as ECEF
ecef_coords = convert.eci_to_ecef(eci_coords[0], eci_coords[1], eci_coords[2], gmst0)
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
gs_acceleration = total_acceleration_norm / 9.81

# compute downrange distance
downrange = convert.ecef_distance(ecef_coords[0][0], ecef_coords[1][0], ecef_coords[2][0], ecef_coords[0], ecef_coords[1], ecef_coords[2])
crossing_points_downrange, crossing_points = visualization.find_crossing_points(t_sol, downrange, altitude, threshold=100000)

closest_indices = np.abs(np.subtract.outer(t_sol, crossing_points)).argmin(axis=0)
crossing_points_r_v = sol.y[:, closest_indices]
crossing_points_r = crossing_points_r_v[:3]

# Get touchdown array
altitude_event_times = sol.t_events[0]

# flatten the array
touchdown_time = np.int16(t_sol[-1])

# get last position
touchdown_r = eci_coords[-1]

# Upload/download simulation data
#--------------------------------------------
data = {
    't': t_sol,
    'x': eci_coords[0],
    'y': eci_coords[1],
    'z': eci_coords[2],
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
fig3 = go.Figure()
fig3 = go.Figure(layout=layout)
orbit_trace = visualization.plot_orbit_3d(orbit,color='green',name='Classical orbit',dash= 'dot') # convert orbit into scatter3d plotly trace
fig3.add_trace(orbit_trace)

for trace in pos_arrow:
    fig3.add_trace(trace)
for trace in vel_arrow:
    fig3.add_trace(trace)

# Add the new trajectory trace with altitude-based coloring
trajectory_trace = SpacecraftVisualization.create_3d_scatter(eci_coords[0], eci_coords[1], eci_coords[2], heat_rate,name='Simulated trajectory', colorscale='Agsunset')
fig3.add_trace(trajectory_trace)
#add colorbar for heat rate (heat_rate)
# Detect impact
#--------------------------------------------
if altitude_event_times.size > 0:
    fig3.add_trace(go.Scatter3d(x=[eci_coords[0, -1]], y=[eci_coords[1, -1]], z=[eci_coords[2, -1]], mode='markers', marker=dict(size=6, color='purple'), name='Touchdown'))
    fig3.update_layout(scene=dict(annotations=[dict(x=eci_coords[0, -1], y=eci_coords[1, -1], z=eci_coords[2, -1], text="Touchdown", showarrow=True)]))
    impact_time = t_sol[-1]
else:
    fig3.add_trace(go.Scatter3d(
        x=[eci_coords[0, -1]],
        y=[eci_coords[1, -1]],
        z=[eci_coords[2, -1]],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Final position'))
    impact_time = t_sol[-1]


# Detect karman line crossing and add a marker for each crossing in the plot
crossing_points_r_x = np.float64(crossing_points_r[0])
crossing_points_r_y = np.float64(crossing_points_r[1])
crossing_points_r_z = np.float64(crossing_points_r[2])

# get list of all points lower than 100km altitude
altitude_lower_than_100km = []
for i in range(len(altitude)):
    if altitude[i] < 100000:
        altitude_lower_than_100km.append(i)

#--------------------------------------------
if crossing_points is not None:
    for crossing_point in crossing_points:
        fig3.add_trace(go.Scatter3d(x=[crossing_points_r_x], y=[crossing_points_r_y], z=[crossing_points_r_z], mode='markers', marker=dict(size=6, color='orange'), name='Karman line crossing'))


# Update gmst0 based on the impact time or the final time of the simulation
gmst = gmst0 + earth_rotation_deg_speed * impact_time
with st.spinner("Making things pretty..."):
    spheroid_mesh = visualization.create_spheroid_mesh()
    # Recalculate the coastline traces based on the updated gmst0
    country_traces = visualization.get_geo_traces(country_feature, gmst)
    coastline_traces = visualization.get_geo_traces(coastline_feature, gmst)
    fig3.add_trace(spheroid_mesh)
    for trace in coastline_traces:
        trace.showlegend = False
        fig3.add_trace(trace)
        
    for trace in country_traces:
        trace.showlegend = False
        fig3.add_trace(trace)

    lat_lines = visualization.create_latitude_lines(gmst=gmst)
    for lat_line in lat_lines:
        lat_line.showlegend = False
        fig3.add_trace(lat_line)

    lon_lines = visualization.create_longitude_lines(gmst=gmst)
    for lon_line in lon_lines:
        lon_line.showlegend = False
        fig3.add_trace(lon_line)


    # set legend to bottom left
    fig3.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))
    #zooming in
    fig3.update_layout(scene_camera=dict(eye=dict(x=-0.5, y=0.6, z=1)))
    # keep axis aspect ratio always equal
    fig3.update_layout(scene_aspectmode='data')
    st.plotly_chart(fig3, use_container_width=True, equal_axes=True)

# Calculate impact time
#--------------------------------------------
st.subheader("Crash Detection")
col2, col3 = st.columns(2)
duration = datetime.timedelta(seconds=impact_time.astype(float))
if altitude_event_times.size > 0:
    # get location of impact in ECEF
    impact_point_ecef = np.array([eci_coords[0, -1], eci_coords[1, -1], eci_coords[2, -1]])
    # get location of impact in lat, lon, alt
    impact_point_lat_lon_alt = convert.ecef_to_geo(impact_point_ecef[0], impact_point_ecef[1], impact_point_ecef[2])
    # break down the lat, lon, alt
    impact_point_lat = impact_point_lat_lon_alt[0]
    impact_point_lon = impact_point_lat_lon_alt[1]
    impact_point_alt = impact_point_lat_lon_alt[2]
    col2.warning(f"⚠️ Touchdown detected {duration} (hh,mm,ss) after start intial time, experiencing a maximum deceleration of {max(gs_acceleration)} G")
    col2.info(f"📍 Touchdown detected at {impact_point_lat}ºN, {impact_point_lon}ºE")
else:
    col2.success("No reentry and landing detected")
    # calculate final time of simulation using astropy


final_time = epoch + TimeDelta(impact_time, format='sec')
col3.info(f"⏰ The simulation start time was {epoch} and ended on: {final_time}, with a total time simulated of: {duration} (hh,mm,ss)")
col2.info(f"🛰️ The spacecraft was at an altitude of {altitude[-1]}m at the end of the simulation")
#--------------------------------------------


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

fig4 = go.Figure()
# set y axis range to 0 to max altitude
# add trendline
z = np.polyfit(t_sol, altitude, 1)
p = np.poly1d(z)
# add karman line
fig4.add_trace(go.Scatter(x=t_sol, y=[100000]*len(t_sol), mode='lines', line=dict(color='rgba(255,255,255,0.5)', width=2, dash= 'dot'), name='Karman Line'))
# add the main layers of the atmosphere as rectangles
fig4.add_shape(type='rect', x0=0, x1=max(t_sol), y0=0, y1=11000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(196, 245, 255, 1)', opacity=0.3, name='Troposphere')
fig4.add_shape(type='rect', x0=0, x1=max(t_sol), y0=11000, y1=47000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(0, 212, 255, 1)', opacity=0.3, name='Stratosphere')
fig4.add_shape(type='rect', x0=0, x1=max(t_sol), y0=47000, y1=86000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(4, 132, 202, 1)', opacity=0.3, name='Mesosphere')
fig4.add_shape(type='rect', x0=0, x1=max(t_sol), y0=86000, y1=690000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(9, 9, 121, 1)', opacity=0.3, name='Thermosphere')
fig4.add_shape(type='rect', x0=0, x1=max(t_sol), y0=690000, y1=1000000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(2, 1, 42, 1)', opacity=0.3, name='Exosphere')
# Add layer names as annotations
fig4.add_annotation(x=0, y=11000, text='Troposphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig4.add_annotation(x=0, y=47000, text='Stratosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig4.add_annotation(x=0, y=86000, text='Mesosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig4.add_annotation(x=0, y=690000, text='Thermosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig4.add_annotation(x=0, y=1000000, text='Exosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)

fig4.add_trace(go.Scatter(x=t_sol, y=altitude, mode='lines', line=dict(color='#ff00f7', width=2), name='Altitude (m)'))
fig4.add_trace(go.Scatter(x=t_sol, y=p(t_sol), mode='lines', line=dict(color='cyan', width=2, dash= 'dot'), name=f'Trendline{p}'))

if altitude_event_times.size > 0:
    # Add touchdown line
    fig4.add_trace(go.Scatter(x=[touchdown_time], y=np.linspace(0, max(altitude), len(t_sol)), mode='lines', line=dict(color='rgba(0, 255, 0, 0.5)', width=2, dash='dot'), name='Touchdown'))
    # Add annotation for touchdown
    fig4.add_annotation(x=[touchdown_time], y=max(altitude), text='Touchdown', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

# show trendline equation in graph legend of trendline
if crossing_points is not None:
    for idx, crossing_point in enumerate(crossing_points):
        fig4.add_shape(type='line', x0=crossing_point, x1=crossing_point, y0=0, y1=max(altitude), yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot'))
        fig4.add_annotation(x=crossing_point, y=max(altitude), text=f'Crossing Karman line {idx+1}', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

fig4.update_yaxes(range=[0, max(altitude)])
fig4.update_layout(xaxis_title='Time (s)', yaxis_title='Altitude (m)', legend=dict(y=1.3, yanchor="top", xanchor="left", x=0, orientation="h"), hovermode="x unified")
st.plotly_chart(fig4, use_container_width=True)
#--------------------------------------------

# Plot the velocity vs time
#--------------------------------------------
st.subheader("Velocity vs Time")
''' 
Here you can see the velocity of the spacecraft both in absolute magnitude as well as in the x, y, and z directions.
One particularly useful measure is the ground velocity vs the orbital velocity.
'''
fig5 = go.Figure()
g_velx, g_vely, g_velz = sol.y[3:6]
# Calculate the GMST for each time step
gmst_t = gmst0 + sol.t * earth_rotation_deg_speed

# Calculate the orbital velocities (in ECI)
orbital_velocities = [np.linalg.norm(sol.y[3:6, i]) for i in range(sol.t.size)]

# make sure that in right unit
v_ecef = np.array([CoordinateConverter.eci_to_ecef(g_velx[i], g_vely[i], g_velz[i], gmst_t[i]) for i in range(sol.t.size)])
w_ECEF = np.array([0, 0, earth_rotation_deg_speed]) * np.pi / 180  # Convert to rad/s

# Calculate the Earth's rotational velocities in m/s
earth_rotational_velocities = [np.array([-earth_rotational_speed * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
                                         earth_rotational_speed * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
                                         0]) for lon in gmst_t]

# Calculate the ground velocities
ground_velocities = np.array([v_ecef[i] - earth_rotational_velocities[i] for i in range(sol.t.size)])

# Calculate the magnitude of the ground velocities
ground_velocity_magnitudes = [np.linalg.norm(ground_velocities[i]) for i in range(sol.t.size)]

max_velocity = max(max(ground_velocity_magnitudes), max(orbital_velocities),max(sol.y[3, :]), max(sol.y[4, :]), max(sol.y[5, :]))
min_velocity = min(min(ground_velocity_magnitudes), min(orbital_velocities),min(sol.y[3, :]), min(sol.y[4, :]), min(sol.y[5, :]))

# add it to the plot
fig5.add_trace(go.Scatter(x=t_sol, y=ground_velocity_magnitudes, mode='lines', line=dict(color='Purple', width=2), name='Ground Velocity'))
fig5.add_trace(go.Scatter(x=t_sol, y=orbital_velocities, mode='lines', line=dict(color='white', width=2), opacity=0.5, name='Orbital Velocity'))
fig5.add_trace(go.Scatter(x=t_sol, y=sol.y[3, :], mode='lines', line=dict(color='blue', width=2), name='X Velocity'))
fig5.add_trace(go.Scatter(x=t_sol, y=sol.y[4, :], mode='lines', line=dict(color='red', width=2), name='Y Velocity'))
fig5.add_trace(go.Scatter(x=t_sol, y=sol.y[5, :], mode='lines', line=dict(color='green', width=2), name='Z Velocity'))
if crossing_points is not None:
    for idx, crossing_point in enumerate(crossing_points):
        fig5.add_shape(type='line', x0=crossing_point, x1=crossing_point, y0=min_velocity, y1=max_velocity, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot'))
        fig5.add_annotation(x=crossing_point, y=max_velocity, text=f'Crossing Karman line {idx+1}', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

if altitude_event_times.size > 0:
    # Add touchdown line
    fig5.add_shape(type='line', x0=[touchdown_time], x1=[touchdown_time], y0=min_velocity, y1=max_velocity, yref='y', xref='x', line=dict(color='rgba(0, 255, 0, 0.5)', width=2, dash='dot'))
    # Add annotation for touchdown
    fig5.add_annotation(x=[touchdown_time], y=max_velocity, text='Touchdown', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

fig5.update_layout(xaxis_title='Time (s)', yaxis_title='Velocity (m/s)',legend=dict(y=1.2, yanchor="top", xanchor="left", x=0, orientation="h"),hovermode="x unified")
st.plotly_chart(fig5, use_container_width=True)
#--------------------------------------------

# Plot the downrange vs altitude
#--------------------------------------------
st.subheader("Downrange vs Altitude")
'''
Here you can see the downrange distance of the spacecraft from the launch site as a function of altitude.
Downrange is being measured in absolute distance from the starting point.
'''
# Calculate downrange using ecef_distance function
# Begin plotting
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=downrange, y=altitude, mode='lines', line=dict(color='purple', width=2), name='Altitude'))
# add karman line at 100km altitude
fig6.add_trace(go.Scatter(x=[0, max(downrange)], y=[100000]*2, mode='lines', line=dict(color='rgba(255,255,255,0.5)', width=2, dash= 'dot'), name='Karman Line'))
fig6.add_shape(type='rect', x0=0, x1=max(downrange), y0=0, y1=11000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(196, 245, 255, 1)', opacity=0.3, name='Troposphere')
fig6.add_shape(type='rect', x0=0, x1=max(downrange), y0=11000, y1=47000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(0, 212, 255, 1)', opacity=0.3, name='Stratosphere')
fig6.add_shape(type='rect', x0=0, x1=max(downrange), y0=47000, y1=86000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(4, 132, 202, 1)', opacity=0.3, name='Mesosphere')
fig6.add_shape(type='rect', x0=0, x1=max(downrange), y0=86000, y1=690000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(9, 9, 121, 1)', opacity=0.3, name='Thermosphere')
fig6.add_shape(type='rect', x0=0, x1=max(downrange), y0=690000, y1=1000000, yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor='rgba(2, 1, 42, 1)', opacity=0.3, name='Exosphere')
# Add layer names as annotations
fig6.add_annotation(x=0, y=11000, text='Troposphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig6.add_annotation(x=0, y=47000, text='Stratosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig6.add_annotation(x=0, y=86000, text='Mesosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig6.add_annotation(x=0, y=690000, text='Thermosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)
fig6.add_annotation(x=0, y=1000000, text='Exosphere', xanchor='left', yanchor='bottom', font=dict(size=10), showarrow=False)

if crossing_points_downrange is not None:
    for idx, crossing_point in enumerate(crossing_points_downrange):
        fig6.add_shape(type='line', x0=crossing_point, x1=crossing_point, y0=0, y1=max(altitude), yref='y', xref='x', line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot'))
        fig6.add_annotation(x=crossing_point, y=max(altitude), text=f'Crossing Karman line {idx+1}', showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

fig6.update_yaxes(range=[0, max(altitude)])
fig6.update_layout(legend=dict(y=1.2, yanchor="top", xanchor="left", x=0, orientation="h"))
fig6.update_layout(xaxis_title='Downrange (m)', yaxis_title='Altitude (m)', hovermode="x unified")
st.plotly_chart(fig6, use_container_width=True)
#--------------------------------------------


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
col1, col2 = st.columns(2)

fig7 = go.Figure()
latitudes = []
longitudes = []

for i in range(len(sol.t)):
    # Update gmst for each time step of the simulation
    gmst = (gmst0 + earth_rotation_deg_speed * sol.t[i]) * np.pi / 180  # Convert gmst to radians

    # Convert the position data into ECEF coordinates
    ecef_coords = convert.eci_to_ecef(sol.y[0, i], sol.y[1, i], sol.y[2, i], gmst)

    # Convert the ECEF coordinates into lat, lon, alt
    lat_lon_alt = convert.ecef_to_geo(ecef_coords[0], ecef_coords[1], ecef_coords[2])

    # Break down the lat, lon, alt
    lat_lon = lat_lon_alt[0:2]

    # Multiply the lat and lon by 180/pi to convert from radians to degrees
    lat_lon = np.multiply(lat_lon, 180/np.pi)

    latitudes.append(lat_lon[0])
    longitudes.append(lat_lon[1])

# Calculate the solar zenith angle at the final time considering the Earth's tilt
sza, lat_grid, lon_grid = convert.solar_zenith_angle(final_time)
# Calculate the night side overlay coordinates
night_side_lons, night_side_lats = convert.night_side_coordinates(sza, lat_grid, lon_grid)

# Add the night side overlay to the map
fig7.add_trace(go.Scattergeo(
    lon=night_side_lons,
    lat=night_side_lats,
    mode='markers',
    marker=dict(color='rgba(0, 0, 70, 1)', size=3),
    hoverinfo='none',  # Disable hover effect
    showlegend=False,
    name='Night side',
))

# Add a single trace for the ground track
fig7.add_trace(go.Scattergeo(
    lon=longitudes,
    lat=latitudes,
    mode='lines+markers',
    marker=dict(color='red', size=2),
    # map resolution
    showlegend=True,
    name='Groundtrack',
))

# add point for starting point and another for final position
final_lat = latitudes[-1]
final_lon = longitudes[-1]
fig7.add_trace(go.Scattergeo(
    lon=[final_lon],
    lat=[final_lat],
    mode='markers',
    marker=dict(color='orange', size=10),
    showlegend=True,
    name='Final position'
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
    )
)
fig7.update_geos(resolution=110)
fig7.update_layout(legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"))
st.plotly_chart(fig7, use_container_width=True)
#--------------------------------------------------------------------------------

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
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=sol.t, y=np.linalg.norm(total_acceleration, axis=1), name='Total acceleration'))
    fig8.add_trace(go.Scatter(x=sol.t, y=np.linalg.norm(earth_grav_acceleration, axis=1), name='Earth grav. acceleration'))
    fig8.add_trace(go.Scatter(x=sol.t, y=np.linalg.norm(j2_acceleration, axis=1), name='J2 acceleration'))
    fig8.add_trace(go.Scatter(x=sol.t, y=np.linalg.norm(moon_acceleration, axis=1), name='Moon grav. acceleration'))
    fig8.add_trace(go.Scatter(x=sol.t, y=np.linalg.norm(drag_acceleration, axis=1), name='Drag acceleration'))
    fig8.add_trace(go.Scatter(x=sol.t, y=np.linalg.norm(sun_acceleration, axis=1), name='Sun grav. acceleration'))
    max_accel = max(np.max(total_acceleration), np.max(earth_grav_acceleration), np.max(j2_acceleration), np.max(moon_acceleration), np.max(drag_acceleration))
    min_accel = min(np.min(total_acceleration), np.min(earth_grav_acceleration), np.min(j2_acceleration), np.min(moon_acceleration), np.min(drag_acceleration))
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
- $C_p$: spacecraft heat transfer coefficient (W/m²·K)
- $A$: spacecraft cross-sectional area (m²)

The model assumes constant atmospheric density, spacecraft parameters (e.g., cross-sectional area, heat transfer coefficient), and specific heat ratio. It is important to note that this is a simplified model and may not provide accurate heating predictions for real reentry scenarios. For more accurate estimations, consider using more complex models or simulation tools.

'''

fig9 = go.Figure()
fig9.add_trace(go.Scatter(x=sol.t, y=heat_rate, name='Heat rate'))
fig9.update_layout(
    xaxis_title='Time (s)',
    yaxis_title='Heat rate (W/m^2)',
    autosize=True,
    margin=dict(l=0, r=0, t=60, b=0),
    legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig9, use_container_width=True)

'''
That's it for this dashboard. Try more combinations to see the effects of different parameters on the trajectory. Also, try landing with low Gs (your spacecraft will thank you...)!

You can reach me on [Twitter](https://twitter.com/JohnMontenegro) or [Github](https://github.com/JMMonte) or [Linkedin](https://www.linkedin.com/in/jmontenegrodesign/).
You can also visit my [personal website](https://monte-negro.space/).
'''

#--------------------------------------------------------------------------------
# about this app
st.sidebar.markdown('''
**About this app**

This app means to show the power of mixing streamlit, plotly, poliastro with scipy as the simulation engine.
All the code is available on Github and is free to use.
Many of these equations are a best effort to implement and are likely not the most accurate in their current form.
Therefore, any feedback is greatly welcome.
You can reach me on [Twitter](https://twitter.com/JohnMontenegro) or [Github](https://github.com/JMMonte) or [Linkedin](https://www.linkedin.com/in/jmontenegrodesign/).
You can also visit my [personal website](https://monte-negro.space/).
''')