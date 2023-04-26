from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_sun
from astropy.coordinates import EarthLocation, AltAz
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
import time

# Special functions
#--------------------------------------------
class LoadedSolution:
    def __init__(self, t, y):
        self.t = t
        self.y = y

def filter_results_by_altitude(sol, R):
    altitudes = [data['altitude'] for data in sol.additional_data]
    valid_indices = [i for i, alt in enumerate(altitudes) if alt >= 0]

    filtered_sol = deepcopy(sol)
    filtered_sol.y = sol.y[:, valid_indices]
    filtered_sol.t = sol.t[valid_indices]
    filtered_sol.additional_data = [sol.additional_data[i] for i in valid_indices]

    return filtered_sol

# Create a download link for the simulated data
def make_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
#--------------------------------------------

# Begin the app
#--------------------------------------------
st.set_page_config(layout="wide", page_title="Spacecraft Reentry Simulation", page_icon="☄️")
st.title("Spacecraft Reentry Simulation")
if "impact" not in st.session_state:
    st.session_state.impact = False

spacecraft = SpacecraftModel()
visualization = SpacecraftVisualization()
convert = CoordinateConverter()
data_directory = "data"
coastline_feature = cfeature.COASTLINE
country_feature = cfeature.BORDERS
earth_rotation_deg_speed = (2 * np.pi * u.rad / (Earth.rotational_period.to(u.s)).to(u.s)).to_value(u.deg/u.s)

# Sidebar user inputs
#--------------------------------------------
with st.sidebar:
    st.title("Mission Parameters")
    # Define initial state (position and velocity vectors)
    with st.expander("🚀 Welcome! Read me first"):
        r'''
        This app aims to simulate the complex dynamics of a spacecraft orbits around the Earth.
        
        It takes into account the Earth's rotation, J2 perturbations, atmospheric drag and the Moon's gravity while predicting the spacecraft's trajectory.
        
        The simulation uses the amazing [poliastro](https://docs.poliastro.space/en/stable/) library, as well as [astropy](https://www.astropy.org/) and [streamlit](https://streamlit.io/).
        
        Made with ❤️ by [João Montenegro](https://monte-negro.space/).
        '''
    with st.expander("Edit Spacecraft initial state", expanded=True):
        v = st.number_input("Ground Velocity (m/s)", value=7.5e3, step=1e3, key="velocity")
        azimuth = st.number_input("Azimuth (degrees)", value=90.0, min_value=0.0, max_value=360.0, step=1.0, key="azimuth")
        lat = st.number_input("Latitude (deg)", value=45.0, min_value=-90.0, max_value=90.0, step=1.0, key="latitude")
        lon = st.number_input("Longitude (deg)", value=-75.0, min_value=-180.0, max_value=180.0, step=1.0, key="longitude")
        alt = st.number_input("Altitude (km)", value=500.0, step=100.0, key="altitude")
        alt = alt * 1000
        clock = st.time_input("Spacecraft Clock", value=datetime.time(8, 45), key="clock")
        spacecraft_clock = [clock.hour, clock.minute, clock.second]
        
        # f"GMST: {gmst}"
        calendar = st.date_input("Spacecraft Calendar", value=datetime.date.today(), key="calendar")

    st.subheader("Simulation Parameters")

    # Define integration parameters
    ts = st.number_input("Start time (s)", min_value=0 , value=0, step=1, key="ts")  # initial time in seconds
    tf = st.number_input("Simulation duration (s)", min_value=0 , value=1000, step=1, key="tf")  # final time in seconds
    dt = st.number_input("Time step (s)", min_value=0 , value=10, step=1, key="dt")  # time step in seconds

    # convert datetime to astropy time
    datetime_spacecraft = datetime.datetime.combine(calendar, clock)
    datetime_spacecraft = Time(datetime_spacecraft, scale='utc')
    gmst0 = datetime_spacecraft.sidereal_time('mean', 'greenwich').to_value(u.deg)

    # f"Date time: {datetime_spacecraft}"
    time = Time(datetime_spacecraft, format='ymdhms')
    datetime_str = time.to_value('iso', subfmt='date_hms')
    epoch = Time(datetime_str, format='iso', scale='utc')

    # Update time span and t_eval based on ts
    t_span = (ts, tf)  # time span tuple
    t_eval = np.arange(ts, tf, dt)  # time array for output
    y0 = spacecraft.get_initial_state(v=v, lat=lat, lon=lon, alt=alt, azimuth=azimuth, attractor=Earth, gmst=gmst0)
#--------------------------------------------

# Simulation
#--------------------------------------------
x_pos, y_pos, z_pos = y0[0:3] # Extract the position components
x_vel, y_vel, z_vel = y0[3:6] # Extract the velocity components

# Run the simulation
sol = spacecraft.run_simulation(t_span, y0, t_eval, previous_sol=None)
# clean the solution
R = spacecraft.R
filtered_sol = filter_results_by_altitude(sol, R)

# Convert the numpy arrays to a pandas DataFrame
data = {
    't': sol.t,
    'x': sol.y[0],
    'y': sol.y[1],
    'z': sol.y[2],
    'vx': sol.y[3],
    'vy': sol.y[4],
    'vz': sol.y[5]
}
df = pd.DataFrame(data)

# Display the download link in the Streamlit app
st.sidebar.markdown(make_download_link(df, 'simulated_data.csv', 'Download simulated data'), unsafe_allow_html=True)

# Add a file uploader for loading previously saved data
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df_loaded = pd.read_csv(uploaded_file)
    
    # Convert DataFrame columns to numpy arrays
    x = df_loaded["x"].to_numpy()
    y = df_loaded["y"].to_numpy()
    z = df_loaded["z"].to_numpy()
    vx = df_loaded["vx"].to_numpy()
    vy = df_loaded["vy"].to_numpy()
    vz = df_loaded["vz"].to_numpy()
    
    # Create a new LoadedSolution object with the loaded data
    filtered_sol = LoadedSolution(t=df_loaded["t"].to_numpy(), y=np.stack((x, y, z, vx, vy, vz)))


# unpack the solution
t_sol = sol.t
eci_coords = sol.y[0:3]
ecef_coords = convert.eci_to_ecef(eci_coords[0], eci_coords[1], eci_coords[2], gmst0)

# Scale factor for the velocity vector
scale_factor = 500  # Adjust this value to scale the velocity vector
earth_center = [0, 0, 0]  # Center of the Earth
vel_arrow = visualization.create_3d_arrow(x_pos, y_pos, z_pos, x_pos + x_vel * scale_factor, y_pos + y_vel * scale_factor, z_pos + z_vel * scale_factor, 'blue', 'Velocity vector') # Velocity vector scaled
pos_arrow = visualization.create_3d_arrow(0, 0, 0, x_pos, y_pos, z_pos, 'red', 'Position vector') # Position vector
# get orbit from vectors
orbit = Orbit.from_vectors(Earth, y0[0:3] * u.m, y0[3:6] * u.m / u.s, epoch)

# Plots
#--------------------------------------------
# Create the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    margin=dict(r=0, l=0, t=0, b=0),
    height=600,
)

fig3 = go.Figure()
fig3 = go.Figure(layout=layout)
orbit_trace = visualization.plot_orbit_3d(orbit,color='green',name='Projected orbit') # convert orbit into scatter3d plotly trace
fig3.add_trace(orbit_trace)
# Add position and velocity arrows to the plot
for trace in pos_arrow:
    fig3.add_trace(trace)
    
for trace in vel_arrow:
    fig3.add_trace(trace)
    
altitude = np.array([np.linalg.norm(pos) - Earth.R.to(u.m).value for pos in eci_coords.T])
impact_index = np.where(altitude <= 0)[0]
colorscale = "plasma"
fig3.add_trace(go.Scatter3d(x=filtered_sol.y[0], y=filtered_sol.y[1], z=filtered_sol.y[2], mode='lines', line=dict(colorscale=colorscale, width=2), name='Simualted trajectory'))

# Detect impact
#--------------------------------------------
if impact_index.size > 0:
    st.session_state.impact = True
    impact_index = impact_index[0]
    impact_point = go.Scatter3d(
        x=[sol.y[0, impact_index]],
        y=[sol.y[1, impact_index]],
        z=[sol.y[2, impact_index]],
        mode="markers",
        marker=dict(size=6, color="purple"),
        name="Impact",
    )
    fig3.add_trace(impact_point)
    impact_time = t_sol[impact_index]
else:
    st.session_state.impact = False
    fig3.add_trace(go.Scatter3d(x=[sol.y[0, -1]], y=[sol.y[1, -1]], z=[sol.y[2, -1]], mode='markers', marker=dict(size=6, color='red'), name='Final position'))
    impact_time = t_sol[-1]
#--------------------------------------------

# Update gmst0 based on the impact time or the final time of the simulation
gmst = gmst0 + earth_rotation_deg_speed * impact_time

country_traces = visualization.get_geo_traces(country_feature, gmst0)
spheroid_mesh = visualization.create_spheroid_mesh()
# Recalculate the coastline traces based on the updated gmst0
coastline_traces = visualization.get_geo_traces(coastline_feature, gmst0)
fig3.add_trace(spheroid_mesh)
for trace in coastline_traces:
    trace.showlegend = False
    fig3.add_trace(trace)
    
for trace in country_traces:
    trace.showlegend = False
    fig3.add_trace(trace)

lat_lines = SpacecraftVisualization.create_latitude_lines(gmst=gmst0)
for lat_line in lat_lines:
    lat_line.showlegend = False
    fig3.add_trace(lat_line)

lon_lines = SpacecraftVisualization.create_longitude_lines(gmst=gmst0)
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
if st.session_state.impact:
    # calculate impact time in format (days, hours, minutes, seconds)
    # get location of impact in ECEF
    impact_point_ecef = convert.eci_to_ecef(sol.y[0, impact_index], sol.y[1, impact_index], sol.y[2, impact_index], gmst0)
    # get location of impact in lat, lon, alt
    impact_point_lat_lon_alt = convert.ecef_to_geo(impact_point_ecef[0], impact_point_ecef[1], impact_point_ecef[2])
    # break down the lat, lon, alt
    impact_point_lat = impact_point_lat_lon_alt[0]
    impact_point_lon = impact_point_lat_lon_alt[1]
    impact_point_alt = impact_point_lat_lon_alt[2]
    col2.warning(f"⚠️ Reentry and landing detected in {duration} (hh,mm,ss)")
    col2.write(f"📍 Touchdown detected at {impact_point_lat}ºN, {impact_point_lon}ºE, we last heard from vehicle at: {impact_point_alt}m")
else:
    col2.success("No reentry and landing detected")
    # calculate final time of simulation using astropy


final_time = datetime_spacecraft + TimeDelta(impact_time, format='sec')
col3.info(f"The simulation start time was {datetime_spacecraft} and ended on: {final_time}, with a total time simulated of: {duration} (hh,mm,ss)")
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
fig4.add_trace(go.Scatter(x=t_sol, y=altitude, mode='lines', line=dict(color='blue', width=2), name='Altitude (m)'))
fig4.update_layout(xaxis_title='Time (s)', yaxis_title='Altitude (m)')
# set y axis range to 0 to max altitude
fig4.update_yaxes(range=[0, max(altitude)])
# add trendline
z = np.polyfit(t_sol, altitude, 1)
p = np.poly1d(z)
fig4.add_trace(go.Scatter(x=t_sol, y=p(t_sol), mode='lines', line=dict(color='red', width=2), name=f'Trendline{p}'))
# add karman line
fig4.add_trace(go.Scatter(x=t_sol, y=[100000]*len(t_sol), mode='lines', line=dict(color='rgba(255,255,255,0.5)', width=2, dash= 'dot'), name='Karman Line'))
# add the main layers of the atmosphere
fig4.add_trace(go.Scatter(x=t_sol, y=[0]*len(t_sol), mode='lines', line=dict(color='rgba(145, 187, 255,1)', width=2, dash= 'dot'), name='Troposphere'))
fig4.add_trace(go.Scatter(x=t_sol, y=[11000]*len(t_sol), mode='lines', line=dict(color='rgba(64, 127, 230,1)', width=2, dash= 'dot'), name='Stratosphere'))
fig4.add_trace(go.Scatter(x=t_sol, y=[47000]*len(t_sol), mode='lines', line=dict(color='rgba(26, 86, 184,1)', width=2, dash= 'dot'), name='Mesosphere'))
fig4.add_trace(go.Scatter(x=t_sol, y=[100000]*len(t_sol), mode='lines', line=dict(color='rgba(7, 40, 94,1)', width=2, dash= 'dot'), name='Thermosphere'))
fig4.add_trace(go.Scatter(x=t_sol, y=[1000000]*len(t_sol), mode='lines', line=dict(color='rgba(56, 28, 122,1)', width=2, dash= 'dot'), name='Exosphere'))
# show trendline equation in graph legend of trendline
fig4.update_layout(legend=dict(y=1.2, yanchor="top", xanchor="left", x=0, orientation="h"))
# add gradient fill to the chart's background from altitude 0m to 1000000m (1000km) indicating the atmosphere.

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
g_velx, g_vely, g_velz = filtered_sol.y[3:6]
# Calculate the GMST for each time step
gmst_t = gmst0 + filtered_sol.t * earth_rotation_deg_speed

# make sure that in right unit
v_ecef = np.array([CoordinateConverter.eci_to_ecef(g_velx[i], g_vely[i], g_velz[i], gmst_t[i]) for i in range(filtered_sol.t.size)])
w_ECEF = np.array([0, 0, earth_rotation_deg_speed]) * np.pi / 180  # Convert to rad/s


# Calculate the ground velocities
# Calculate the Earth's rotational velocities
earth_rotational_speed = 463.8  # m/s at the equator
earth_rotational_velocities = [np.array([-earth_rotational_speed * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
                                         earth_rotational_speed * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
                                         0]) for lon in gmst_t]

# Calculate the ground velocities
ground_velocities = [np.linalg.norm(v_ecef[i] - earth_rotational_velocities[i]) for i in range(filtered_sol.t.size)]

# Calculate the orbital velocities
orbital_velocities = (filtered_sol.y[3, :]**2 + filtered_sol.y[4, :]**2 + filtered_sol.y[5, :]**2)**0.5
# add it to the plot
fig5.add_trace(go.Scatter(x=t_sol, y=ground_velocities, mode='lines', line=dict(color='Purple', width=2), name='Ground Velocity'))
fig5.add_trace(go.Scatter(x=t_sol, y=orbital_velocities, mode='lines', line=dict(color='white', width=2), opacity=0.5, name='Orbital Velocity'))
fig5.update_layout(title='Norm Velocity vs Time', xaxis_title='Time (s)', yaxis_title='Velocity (m/s)')
fig5.add_trace(go.Scatter(x=t_sol, y=filtered_sol.y[3, :], mode='lines', line=dict(color='blue', width=2), name='X Velocity'))
fig5.add_trace(go.Scatter(x=t_sol, y=filtered_sol.y[4, :], mode='lines', line=dict(color='red', width=2), name='Y Velocity'))
fig5.add_trace(go.Scatter(x=t_sol, y=filtered_sol.y[5, :], mode='lines', line=dict(color='green', width=2), name='Z Velocity'))
fig5.update_layout(xaxis_title='Time (s)', yaxis_title='Velocity (m/s)')
fig5.update_layout(legend=dict(y=1.15, yanchor="top", xanchor="left", x=0, orientation="h"))
st.plotly_chart(fig5, use_container_width=True)
#--------------------------------------------

# Plot the downrange vs altitude
#--------------------------------------------
st.subheader("Downrange vs Altitude")
'''
Here you can see the downrange distance of the spacecraft from the launch site as a function of altitude.
Downrange is being measured in absolute distance from the starting point.
'''
# convert to ECEF (Earth Centered Earth Fixed) coordinates
ecef_coords = convert.eci_to_ecef(filtered_sol.y[0], filtered_sol.y[1], filtered_sol.y[2], gmst0)
# convert to lat, lon, alt
lat_lon_alt = convert.ecef_to_geo(ecef_coords[0], ecef_coords[1], ecef_coords[2])
# break down the lat, lon, alt
latitude = lat_lon_alt[0]
longitude = lat_lon_alt[1]
altitude = lat_lon_alt[2]
# convert to downrange
downrange = np.sqrt((latitude - latitude[0])**2 + (longitude - longitude[0])**2)
# convert to meters
downrange *= 111139 # 1 deg lat = 111139 m
# Plot the downrange vs altitude
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=downrange, y=altitude, mode='lines', line=dict(color='blue', width=2), name='Altitude'))
# add karman line at 100km altitude
fig6.add_trace(go.Scatter(x=[0, max(downrange)], y=[100000]*2, mode='lines', line=dict(color='rgba(255,255,255,0.5)', width=2, dash= 'dot'), name='Karman Line'))
fig6.update_layout(xaxis_title='Downrange (m)', yaxis_title='Altitude (m)')
fig6.update_yaxes(range=[0, max(altitude)])
fig6.update_layout(legend=dict(y=1.15, yanchor="top", xanchor="left", x=0, orientation="h"))
st.plotly_chart(fig6, use_container_width=True)
#--------------------------------------------


#Plot a groundtrack
#--------------------------------------------
# add a streamlit selectbox to select the map projection
# complete list of map projections: https://plotly.com/python/map-projections/
st.header('Groundtrack Projection')
'''
Here you can see the groundtrack of the spacecraft as a function of time.
Groundtrack's are a way to visualize the path of a spacecraft on a map in reference to the Earth's surface.
To do this we need to adjust our original frame of reference (Earth-Centered Inertial) to a new frame of reference (Earth-Centered Earth-Fixed).
'''
projection_list = ['equirectangular', 'mercator', 'orthographic', 'natural earth', 'kavrayskiy7', 'miller', 'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant', 'conic equal area', 'conic conformal', 'conic equidistant', 'gnomonic', 'stereographic', 'mollweide', 'hammer', 'transverse mercator', 'albers usa', 'winkel tripel', 'aitoff', 'sinusoidal']
col1, col2 = st.columns(2)
projection = col1.selectbox('Select a map projection', projection_list, key='projection')
# add a streamlit selectbox to select the map resolution
resolution = col2.selectbox('Select a map resolution (m)', [110, 50], key='resolution')

fig7 = go.Figure()
latitudes = []
longitudes = []

for i in range(len(filtered_sol.t)):
    # Update gmst for each time step of the simulation
    gmst = (gmst0 + earth_rotation_deg_speed * filtered_sol.t[i]) * np.pi / 180  # Convert gmst to radians

    # Convert the position data into ECEF coordinates
    ecef_coords = convert.eci_to_ecef(filtered_sol.y[0, i], filtered_sol.y[1, i], filtered_sol.y[2, i], gmst)

    # Convert the ECEF coordinates into lat, lon, alt
    lat_lon_alt = convert.ecef_to_geo(ecef_coords[0], ecef_coords[1], ecef_coords[2])

    # Break down the lat, lon, alt
    lat_lon = lat_lon_alt[0:2]

    # Multiply the lat and lon by 180/pi to convert from radians to degrees
    lat_lon = np.multiply(lat_lon, 180/np.pi)

    latitudes.append(lat_lon[0])
    longitudes.append(lat_lon[1])

# Calculate the solar zenith angle at the final datetime considering the Earth's tilt
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

# Calculate the night side overlay coordinates
def night_side_coordinates(sza, lat_grid, lon_grid):
    night_side_mask = sza > 90

    night_side_lats = lat_grid[night_side_mask].flatten().tolist()
    night_side_lons = lon_grid[night_side_mask].flatten().tolist()

    return night_side_lons, night_side_lats

# Get the final time of the simulation
final_time = datetime_spacecraft + TimeDelta(impact_time, format='sec')

# Calculate the solar zenith angle at the final time considering the Earth's tilt
sza, lat_grid, lon_grid = solar_zenith_angle(final_time)

# Calculate the night side overlay coordinates
night_side_lons, night_side_lats = night_side_coordinates(sza, lat_grid, lon_grid)

# add a selectbox for night side overlay
night_side = st.checkbox('Show night side shadow', key='night_side')
if night_side:
    # Add the night side overlay to the map
    fig7.add_trace(go.Scattergeo(
        lon=night_side_lons,
        lat=night_side_lats,
        mode='markers',
        marker=dict(color='rgba(0, 0, 50, 1)', size=3),
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
        projection=dict(type=projection),
        lonaxis=dict(range=[-180, 180], showgrid=True, gridwidth=0.5, gridcolor='rgba(0, 0, 255, 0.5)'),
        lataxis=dict(range=[-90, 90], showgrid=True, gridwidth=0.5, gridcolor='rgba(0, 0, 255, 0.5)'),
    )
)
fig7.update_geos(resolution=resolution)
fig7.update_layout(legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"))
st.plotly_chart(fig7, use_container_width=True)
#--------------------------------------------------------------------------------

# Plot an acceleration vs time graph
# include acceleration (final) gravitational_acceleration, drag_acceleration,moon_acceleration, and J2_acceleration.
st.subheader('Acceleration vs Time')
'''
Last but not least, we can plot the acceleration vs time graph. This will show us how the acceleration changes over time. We can see that the acceleration is initially very high, but then decreases as the spacecraft gets further away from the Earth.
In this simulation we are taking into account:

- Earth's gravitational acceleration
- Drag acceleration
- Moon's gravitational acceleration
- J2 acceleration

In our starting scenario (in Low Earth Orbit), you can see that the total acceleration is mainly affected by the Earth's gravitational acceleration. However, you can click on the legend to hide the total acceleration to adjust the graphs y axis so the other accelerations are visible.
'''
fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=filtered_sol.t, y=filtered_sol.y[1] * (-1), name='Total acceleration'))
fig8.add_trace(go.Scatter(x=filtered_sol.t, y=filtered_sol.y[2], name='Gravitational acceleration'))
fig8.add_trace(go.Scatter(x=filtered_sol.t, y=filtered_sol.y[3], name='J2 acceleration'))
fig8.add_trace(go.Scatter(x=filtered_sol.t, y=filtered_sol.y[4], name='Moon acceleration'))
fig8.add_trace(go.Scatter(x=filtered_sol.t, y=filtered_sol.y[5], name='Drag acceleration'))
fig8.update_layout(
    xaxis_title='Time (s)',
    yaxis_title='Acceleration (m/s^2)',
    autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
)
fig8.update_layout(legend=dict(y=1.1, yanchor="top", xanchor="left", x=0, orientation="h"))
st.plotly_chart(fig8, use_container_width=True)
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