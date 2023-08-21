from poliastro.bodies import Earth
from astropy import units as u
from astropy.time import Time, TimeDelta
import cartopy.feature as cfeature
from coordinate_converter import (eci_to_ecef, ecef_to_geodetic, haversine_distance)
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from poliastro.twobody import Orbit
from spacecraft_model import *
from spacecraft_visualization import *
import streamlit as st
import matplotlib as mpl
from constants import *
from copy_text import *

def update_progress(progress, elapsed_time):
    progress_bar.progress(progress,f"🔥 Cooking your TPS... {elapsed_time:.2f} seconds elapsed")

# Initialize the spacecraft model
spacecraft = SpacecraftModel()

# Setup the page
st.set_page_config(
    page_title="Spacecraft Reentry Simulator",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fix session state bug
st.session_state.update(st.session_state)

# Default values for session state
defaults = {
    'mass': 5000.0,
    'area': 14.0,
    'codrag': 1.3,
    'v': 7540.0,
    'azimuth': 90.0,
    'gamma': -2.0,
    'lat': 45.0,
    'lon': -75.0,
    'alt_init': 500.0,
    'clock': datetime.time(20, 00),
    'calendar': datetime.date.today(),
    'tf': 3700,
    'dt': 10,
    'sim_type': ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
    'iter_fact': 3.0,
    'max_points': 10000
}

# Set defaults in session state if not present
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Spacecraft Reentry Simulator")

# SIDEBAR
sidebar = st.sidebar
with st.sidebar:
    st.title("Mission Parameters")  # Define initial state (position and velocity vectors)
    '''To get started, edit the mission parameters and click the "Run Simulation" button.'''
    with st.expander("ℹ Help"):
        HELP_TEXT

    run_simulation = st.button("Run Simulation")

    # Style customization for the Run Simulation button
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(204, 49, 49);
    }
    </style>""", unsafe_allow_html=True)

    with st.expander("Edit spacecraft"):
        mass = st.number_input("Spacecraft mass (kg)", value=st.session_state.mass, min_value=0.0,help=INPUTS["mass"]["help_text"])
        area = st.number_input("Cross section area (m^2)", value=st.session_state.area, min_value=0.0, help=INPUTS["area"]["help_text"])
        codrag = st.number_input("Drag coefficient", value=st.session_state.codrag, min_value=0.0, help=INPUTS["codrag"]["help_text"])
        material_names = [*MATERIALS]
        selected_material = st.selectbox("Heat shield material", material_names, index=2, help=INPUTS["selected_material"]["help_text"])
        selected_material_obj = MATERIALS[selected_material]
        material_properties = list(selected_material_obj.values())
        st.write(f"Thermal conductivity (k):{MATERIALS[selected_material]['thermal_conductivity']} W/m*K")
        st.write(f"Specific heat capacity (c):{MATERIALS[selected_material]['specific_heat_capacity']} J/kg*K")
        st.write(f"Emissivity (e):{MATERIALS[selected_material]['emissivity']}")
        st.write(f"Ablation_efficiency:{MATERIALS[selected_material]['ablation_efficiency']}")

    with st.expander("Edit initial state", expanded=True):
        v = st.number_input("Orbital Velocity (m/s)", value=st.session_state.v, step=100.0, help=INPUTS["v"]["help_text"])
        azimuth = st.number_input("Azimuth (degrees)", value=st.session_state.azimuth, min_value=0.0, max_value=360.0, step=1.0, help=INPUTS["azimuth"]["help_text"])
        gamma = st.number_input("Flight path angle or gamma (degrees)", value=st.session_state.gamma, min_value=-90.0, max_value=90.0, step=1.0, help=INPUTS["gamma"]["help_text"])
        lat = st.number_input("Latitude (deg)", value=st.session_state.lat, min_value=-90.0, max_value=90.0, step=1.0)
        lon = st.number_input("Longitude (deg)", value=st.session_state.lon, min_value=-180.0, max_value=180.0, step=1.0)
        alt_init = st.number_input("Altitude (km)", value=st.session_state.alt_init, step=100.0, help=INPUTS["alt"]["help_text"])
        clock = st.time_input("Spacecraft Clock", value=st.session_state.clock, help=INPUTS["clock"]["help_text"])
        calendar = st.date_input("Spacecraft Calendar", value=st.session_state.calendar, help=INPUTS["calendar"]["help_text"])

    # convert datetime to astropy time
    spacecraft_datetime_string = f"{calendar} {clock.hour}:{clock.minute}:{clock.second}"
    
    epoch = Time(spacecraft_datetime_string, format="iso", scale='tdb')
    
    gmst0 = epoch.sidereal_time('mean', 'greenwich').to_value(u.rad) # get the greenwich mean sidereal time
    y0 = spacecraft.get_initial_state(v=v, lat=lat, lon=lon, alt=alt_init * 1000, azimuth=azimuth,gamma=gamma, gmst=gmst0)

    with st.expander("Simulation Parameters"):
        ts = 0 # initial time in seconds
        tf = st.number_input("Simulation duration (s)", min_value=0 , value=st.session_state.tf, step=1, help=INPUTS["tf"]["help_text"])
        dt = st.number_input("Time step (s)", min_value=0 , value=st.session_state.dt, step=1, help=INPUTS["dt"]["help_text"])
        sim_type = st.selectbox("Solver method", st.session_state.sim_type, help=INPUTS["sim_type"]["help_text"])
        iter_fact = st.number_input("Iteration slowdown", value=st.session_state.iter_fact, min_value=0.0, help=INPUTS["iter_fact"]["help_text"])
        max_points = st.number_input("Maximum number of points", value=st.session_state.max_points, min_value=0, help=INPUTS["max_points"]["help_text"])

    # Update session state values after collecting all the input values
    st.session_state.update({
        'mass': mass,
        'area': area,
        'codrag': codrag,
        'v': v,
        'azimuth': azimuth,
        'gamma': gamma,
        'lat': lat,
        'lon': lon,
        'alt_init': alt_init,
        'clock': clock,
        'calendar': calendar,
        'tf': tf,
        'dt': dt,
        'sim_type': sim_type,
        'iter_fact': iter_fact,
        'max_points': max_points
    })

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

    # Update time span and t_eval based on ts
    t_span = (ts, tf)  # time span tuple
    t_eval = np.arange(ts, tf, dt)  # time array for output

    ABOUT_APP

# -------------------------------------------
# SIMULATION FLOW
#--------------------------------------------

# Run the simulation
progress_bar = st.empty()

if not run_simulation:
    # 3D Earth figure
    with st.spinner("Loading 3D Earth figure..."):
        sim = None
        earth_viz = st.plotly_chart(visualize_orbit(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, alt_init, orbit, gmst0, epoch, sim), use_container_width=True, equal_axes=True)

elif run_simulation:
    progress_bar = st.progress(0)
    sim = spacecraft.run_simulation(t_span, y0, t_eval, progress_callback=update_progress) # Run the simulation

    #--------------------------------------------
    # Filter the data to a maximum of 150000 points
    progress_bar.empty()
    with st.spinner("Filtering data..."):
        if len(sim.t) > max_points:
            np.random.seed(0)
            indices = np.random.choice(len(sim.t), size=abs(len(sim.t) - max_points), replace=False)
            sim.t = np.delete(sim.t, indices)
            sim.y = np.delete(sim.y, indices, axis=1)
            if sim.additional_data:
                for key, value in sim.additional_data.items():
                    if value.ndim == 1:
                        if np.max(indices) < value.shape[0]:
                            sim.additional_data[key] = np.delete(value, indices)
                        else:
                            print(f"Index {np.max(indices)} is out of bounds for axis 0 with size {value.shape[0]} for key {key}")
                    elif value.ndim == 2:
                        if np.max(indices) < value.shape[1]:
                            sim.additional_data[key] = np.delete(value, indices, axis=1)
                        else:
                            print(f"Index {np.max(indices)} is out of bounds for axis 1 with size {value.shape[1]} for key {key}")
                    else:
                        print(f"Value array for key {key} has an unexpected dimension")

    with st.spinner("Loading simulation data..."):
        #--------------------------------------------
        # unpack the solution
        t_sol = sim.t # Extract the time array
        r_eci = sim.y[0:3] # Extract the ECI coordinates
        v_eci = sim.y[3:6] # Extract the ECI velocity components
        additional_data = sim.additional_data # Compute additional parameters

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
        velocity_norm = np.linalg.norm(v_eci, axis=0)

        total_acceleration_norm = np.linalg.norm(total_acceleration, axis=1)
        earth_grav_acceleration_norm = np.linalg.norm(earth_grav_acceleration, axis=1)
        j2_acceleration_norm = np.linalg.norm(j2_acceleration, axis=1)
        moon_acceleration_norm = np.linalg.norm(moon_acceleration, axis=1)
        drag_acceleration_norm = np.linalg.norm(drag_acceleration, axis=1)
        sun_acceleration_norm = np.linalg.norm(sun_acceleration, axis=1)

        accelerations = {
            'Total acceleration': total_acceleration_norm,
            'Earth grav. acceleration': earth_grav_acceleration_norm,
            'J2 acceleration': j2_acceleration_norm,
            'Moon grav. acceleration': moon_acceleration_norm,
            'Drag acceleration': drag_acceleration_norm,
            'Sun grav. acceleration': sun_acceleration_norm
        }
        
        # Convert ECEF to geodetic coordinates
        latitudes, longitudes = geodetic_coords[:, 0], geodetic_coords[:, 1]
        altitudes = geodetic_coords[:, 2]

        # Compute 
        velocities = compute_velocities(geodetic_coords, v_ecef_vals, t_sol, altitudes, sim, velocity_norm)
        
        max_velocity = max(np.max(vel) for vel in velocities.values())
        min_velocity = min(np.min(vel) for vel in velocities.values())

        altitudes = altitudes / 1000 # convert to km
        colormap = mpl.colormaps.get_cmap('viridis')
        custom_colorscale = mpl_to_plotly_colormap(colormap)
        vmin, vmax = np.min(altitudes), np.max(altitudes)
        normalized_altitude = (altitudes - vmin) / (vmax - vmin)

        # convert to g's
        gs_acceleration = total_acceleration_norm / 9.80665 # convert to g's

        # compute downrange distance
        downrange_distances = [0]

        for i in range(1, len(geodetic_coords)):
            lat1, lon1, _ = geodetic_coords[i - 1]
            lat2, lon2, _ = geodetic_coords[i]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            downrange_distances.append(downrange_distances[-1] + distance)
        crossing_points_downrange, crossing_points = SpacecraftVisualization.find_crossing_points(t_sol, downrange_distances, altitude, threshold=100000)

        closest_indices = np.abs(np.subtract.outer(t_sol, crossing_points)).argmin(axis=0)

        # Get touchdown array
        altitude_event_times = sim.t_events[0]

        # flatten the array
        touchdown_time = np.int16(t_sol[-1])
        impact_time = t_sol[-1]

        duration = datetime.timedelta(seconds=impact_time.astype(float))
        # get location of impact in ECEF
        last_r_ecef = np.array([r_eci[0, -1], r_eci[1, -1], r_eci[2, -1]])
        # get location of impact in lat, lon, alt
        last_r_geo = ecef_to_geodetic(last_r_ecef[0], last_r_ecef[1], last_r_ecef[2])
        # break down the lat, lon
        last_r_lat = last_r_geo[0]
        last_r_lon = last_r_geo[1]


        # Upload/download simulation data
        #--------------------------------------------
        arrays_to_match = [
            velocity_norm, altitude, drag_acceleration_norm, T_aw_data,
            r_eci[0], r_eci[1], r_eci[2], sim.y[3], sim.y[4], sim.y[5]
        ] + [
            [vec[i] for i in range(len(vec))]
            for vec in [v_eci, total_acceleration, earth_grav_acceleration,
                        j2_acceleration, moon_acceleration, drag_acceleration]
            for dim in range(3)
        ]

        matched_arrays = [match_array_length(arr, len(sim.t)) for arr in arrays_to_match]

        data = {
            't': t_sol,
            'altitude': matched_arrays[1],
            'drag_acceleration': matched_arrays[2],
            'spacecraft_temperature': matched_arrays[3],
            'x': matched_arrays[4],
            'y': matched_arrays[5],
            'z': matched_arrays[6],
            'vx': matched_arrays[7],
            'vy': matched_arrays[8],
            'vz': matched_arrays[9],
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

    # -------------------------------------------
    # PLOTS
    # -------------------------------------------

    with st.spinner("Generating trajectory 3d plot..."):

        st.plotly_chart(visualize_orbit(
                        x_pos, y_pos, z_pos,
                        x_vel, y_vel, z_vel,
                        alt_init,
                        orbit,
                        gmst0,
                        epoch,
                        sim,
                        altitude_event_times,
                        crossing_points,
                        impact_time,
                        closest_indices,
                        ), use_container_width=True, equal_axes=True)
        
        # --------------------------------------------

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

        # --------------------------------------------
        # GRAPH
        # --------------------------------------------

        plot_heatmap(t_sol, normalized_spacecraft_temperature, T_aw_data, custom_colorscale, spacecraft_temperature_tickvals, spacecraft_temperature_ticktext)

    #--------------------------------------------
    # FLIGHT SUMMARY
    #--------------------------------------------

    st.subheader("Crash Detection")
    col2, col3 = st.columns(2)

    if altitude_event_times > 0:

        col2.info(f"📍 Touchdown detected at {last_r_lat}ºN, {last_r_lon}ºE")
        col2.error(f"⚠️ Touchdown detected {duration} (hh,mm,ss) after start intial time.")
    if len(crossing_points) > 0:
        col2.warning(f"⚠️ You're a fireball! Crossing the Karman line at {', '.join([str(item) for item in crossing_points])} seconds after start intial time, experiencing a maximum deceleration of {max(gs_acceleration):.2f} G")

    else:
        col2.success("Still flying high")
        # calculate final time of simulation using astropy

    final_time = epoch + TimeDelta(impact_time, format='sec')
    col2.warning(f"🌡️ The spacecraft reached a temperature of {max(T_aw_data):.3E} K during simulation. You can see what parts of the orbit were the hottest in the 3d plot above👆.")
    col3.info(f"⏰ The simulation start time was {epoch} and ended on: {final_time}, with a total time simulated of: {duration} (hh,mm,ss)")
    col3.info(f"🛰️ The spacecraft was at a ground speed of {np.around(np.linalg.norm(v_ecef[-1]),2)}m/s and at an altitude of {altitude[-1]:.2f}m at the end of the simulation")

    #--------------------------------------------
    # CHARTS
    #--------------------------------------------
    # ALTITUDE VS TIME
    #--------------------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Altitude vs Time", "Downrange vs Altitude", "Groundtrack", "Velocity vs Time", "Perturbations vs Time", "Heat Flux vs Time", "Atmospheric model"])

    with tab1:
        st.subheader("Altitude vs Time")
        ALTITUDE_VS_TIME # label from copy_text.py

        with st.spinner("Generating altitude vs time graph..."):
            fig4 = go.Figure()

            z = np.polyfit(t_sol, altitude, 1) # fit a linear trendline
            p = np.poly1d(z) # create a polynomial function based on the linear trendline

            for layer_y0, layer_y1, layer_color, layer_name in ATMO_LAYERS:
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

    #--------------------------------------------
    # DOWNRANGE VS ALTITUDE
    #--------------------------------------------

    with tab2:
        st.subheader("Downrange vs Altitude")
        DOWNRAGE_VS_ALTITUDE # label from copy_text.py

        # Calculate downrange using ecef_distance function
        with st.spinner("Generating Downrange vs Altitude graph..."):

            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=downrange_distances, y=altitude, mode='lines', line=dict(color='purple', width=2), name='Altitude'))
            fig6.add_trace(go.Scatter(x=[0, max(downrange_distances)], y=[100000]*2, mode='lines', line=dict(color='rgba(255,255,255,0.5)', width=2, dash= 'dot'), name='Karman Line'))

            for layer in ATMO_LAYERS:
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

    #--------------------------------------------
    #GROUNDTRACK
    #--------------------------------------------
    # add a streamlit selectbox to select the map projection
    # complete list of map projections: https://plotly.com/python/map-projections/
    #'equirectangular', 'mercator', 'orthographic', 'natural earth', 'kavrayskiy7', 'miller', 'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant', 'conic equal area', 'conic conformal', 'conic equidistant', 'gnomonic', 'stereographic', 'mollweide', 'hammer', 'transverse mercator', 'albers usa', 'winkel tripel', 'aitoff', 'sinusoidal']

    with tab3:
        st.subheader('Groundtrack Projection')
        GROUNDTRACK # label from copy_text.py

        with st.spinner("Generating Groundtrack map..."):

            # Number of subdivisions in the color scale
            num_subdivisions = 10

            # Calculate tick values and tick text for the subdivisions
            tickvals = np.linspace(0, 1, num_subdivisions)
            ticktext = [f"{vmin + tick * (vmax - vmin):.2f}" for tick in tickvals]

            # Add the final position label
            final_lat_str = f"{latitudes[-1]:.5f}"
            final_lon_str = f"{longitudes[-1]:.5f}"
            final_position_label = f"Final position<br>Lat: {final_lat_str}º North,<br>Lon: {final_lon_str}º East,<br>{altitudes[-1]:.2f} km Altitude"

            # Add a single trace for the ground track
            plot_ground_track(longitudes, latitudes, normalized_altitude, custom_colorscale, tickvals, ticktext, colormap, final_position_label, st)


    #--------------------------------------------
    # VELOCITY VS TIME
    #--------------------------------------------

    with tab4:
        st.subheader("Velocity vs Time")
        VELOCITY_VS_TIME # label from copy_text.py

        with st.spinner("Generating Velocity vs time graph..."):

            fig5 = go.Figure([go.Scatter(x=t_sol, y=velocities[vel], mode='lines', name=vel) for vel in velocities.keys()])

            if crossing_points is not None:
                crossing_texts = [f'Crossing Karman line {idx+1}' for idx, _ in enumerate(crossing_points)]
                add_annotations(fig5, crossing_points, crossing_texts, min_velocity, max_velocity, 'rgba(255, 0, 0, 0.5)')

            if altitude_event_times.size > 0:
                add_annotations(fig5, [touchdown_time], ['Touchdown'], min_velocity, max_velocity, 'rgba(0, 255, 0, 0.5)')

            fig5.update_layout(xaxis_title='Time (s)', yaxis_title='Velocity (m/s)',legend=dict(y=1.2, yanchor="top", xanchor="left", x=0, orientation="h"),hovermode="x unified")
            st.plotly_chart(fig5, use_container_width=True)


    #--------------------------------------------
    # PERTURBATIONS OVER TIME
    #--------------------------------------------
    # include:
    # acceleration (final) gravitational_acceleration,
    # drag_acceleration,
    # moon_acceleration, 
    # and J2_acceleration.

    with tab5:
        st.subheader('Perturbations over time')
        PERTURBATIONS_TEXT # label from copy_text.py

        with st.spinner('Loading accelerations graph...'):

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


    #--------------------------------------------
    # TEMPERATURE MODEL
    #--------------------------------------------

    with tab6:
        st.subheader('Spacecraft Temperature over time')
        with st.expander("Click here to learn more about this simulator's temperature model"):
            TEMPERATURE_MODEL_TEXT # label from copy_text.py

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

    #--------------------------------------------
    # ATMOSPHERIC MODEL
    #--------------------------------------------

    with tab7:
        st.subheader('Atmospheric Model')
        st.write('This simulation uses a simplified atmospheric model based on the NRLMSISE-00 and works by dividing the atmosphere into layers with specific temperature gradients and base pressures. The temperature and pressure at a given altitude are calculated, followed by the atmospheric density. The model then incorporates the latitude and solar activity factors to provide more accurate results for density and temperature.')
        with st.expander("Click here to learn more about this simulator's atmospheric model"):
            ATMOSPHERIC_MODEL_TEXT

        altitudes_graph = np.linspace(0, 1000000, num=1000)
        temperatures = []
        densities = []
        solar_factors = []
        temperatures = np.zeros(altitudes_graph.shape)
        densities = np.zeros(altitudes_graph.shape)
        solar_factors = np.zeros(altitudes_graph.shape)

        for i, altitude in enumerate(altitudes_graph):
            rho, T = atmosphere_model(altitude, 0 ,epoch.jd)
            solar_factor = solar_activity_factor(epoch.jd, altitude)
            temperatures[i] = T
            densities[i] = rho
            solar_factors[i] = solar_factor

        # Create a Plotly chart with two x-axes
        fig_atmo = make_subplots(rows=1, cols=3, subplot_titles=("Temperature (K)", "Solar Factor", "Density (kg/m³)"))

        # Add temperature trace
        fig_atmo.add_trace(go.Scatter(x=temperatures, y=altitudes_graph, name='Temperature (K)', mode='lines', line=dict(color='red')), row=1, col=1)

        # Add solar factors trace
        fig_atmo.add_trace(go.Scatter(x=solar_factors, y=altitudes_graph, name='Solar Factor', mode='lines', line=dict(color='#fcba03')), row=1, col=2)

        # Add density trace
        fig_atmo.add_trace(go.Scatter(x=densities, y=altitudes_graph, name='Density (kg/m³)', mode='lines', line=dict(color='green')), row=1, col=3)

        x_ranges = [(0, max(temperatures)), (min(solar_factors), max(solar_factors)), (min(densities), max(densities))]

        for layer in ATMO_LAYERS:
            for col in range(1, 4):
                x_range = x_ranges[col-1]
                fig_atmo.add_shape(type='rect', x0=x_range[0], x1=x_range[1], y0=layer[0], y1=layer[1], yref='y', xref=f'x{col}',
                                line=dict(color='rgba(255, 0, 0, 0)', width=0), fillcolor=layer[2], opacity=0.3, name=layer[3], row=1, col=col)
                if col == 1:
                    fig_atmo.add_annotation(x=0, y=layer[1], text=layer[3], xanchor='left', yanchor='bottom', font=dict(size=10,), showarrow=False, xref=f'x{col}', yref=f'y{col}')

        # Update layout
        fig_atmo.update_layout(
            title='Atmospheric Temperature, Solar Factor, and Density vs. Altitude',
            yaxis_title='Altitude (m)',
            legend_title='Parameters',
            height = 800,
            hovermode="y unified"
        )

        # Update x axes titles
        fig_atmo.update_xaxes(title_text="Temperature (K)", row=1, col=1)
        fig_atmo.update_xaxes(title_text="Solar Factor", row=1, col=2)
        fig_atmo.update_xaxes(title_text="Density (kg/m³)", row=1, col=3)

        # Run Streamlit app
        st.plotly_chart(fig_atmo, use_container_width=True)

        st.write("To address a more realistic atmospheric model, this simulator also includes a simple version of solar activity and includes it in the calculation of atmospheric density at high altitudes.")

        # last 10 years of solar cycle
        with st.expander("Click here to see how the atmospheric model accounts for solar activity"):
            SOLAR_CYCLE_TEXT # label from copy_text.py

        jd_start_sim = epoch.jd
        jd_end_sim = epoch.jd + tf / (24 * 3600)
        solar_dates_past = np.linspace(jd_start_sim - 365 * 20, jd_start_sim, num=int(365.3 * 10))
        solar_data_past = np.array([solar_activity_factor(date, altitude) for date in solar_dates_past])
        solar_dates_sim = np.linspace(jd_start_sim, jd_end_sim, num=int(tf))
        solar_data_sim = np.array([solar_activity_factor(date, altitude) for date in solar_dates_sim])

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
