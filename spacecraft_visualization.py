import plotly.graph_objects as go
import numpy as np
from numba import jit, njit
from coordinate_converter import (ecef_to_eci, geodetic_to_spheroid)
from spacecraft_model import *
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
import shapely.geometry as sgeom
import plotly.express as px
import pvlib
import cartopy.feature as cfeature
from constants import *
import streamlit as st

base_radius = 1
height_ratio = 1.315
top_radius_ratio = 0.5
truncation_ratio = 0.8

metallic_colorscale = [
    [0, "gray"],
    [0.5, "lightgray"],
    [1, "lightblue"]
]

heatshield_colorscale = [
    [0, "orange"],
    [0.5, "red"],
    [1, "black"]
]

def visualize_orbit(
    x_pos, y_pos, z_pos,
    x_vel, y_vel, z_vel,
    alt,
    orbit, 
    gmst0,
    epoch,
    data=None,
    altitude_event_times=None,
    crossing_points=None,
    impact_time=None,
    closest_indices=None,
    country_feature=cfeature.BORDERS,
    coastline_feature=cfeature.COASTLINE
):
    scale_factor = 200  # Adjust this value to scale the velocity vector
    vel_arrow = SpacecraftVisualization.create_3d_arrow(x_pos, y_pos, z_pos, x_pos + x_vel * scale_factor, y_pos + y_vel * scale_factor, z_pos + z_vel * scale_factor, 'green', 'Velocity vector') # Velocity vector scaled
    pos_arrow = SpacecraftVisualization.create_3d_arrow(0, 0, 0, x_pos, y_pos, z_pos, 'red', 'Position vector') # Position vector

    fig = go.Figure()

    # Initial conditions
    orbit_trace = SpacecraftVisualization.plot_orbit_3d(orbit, color='#05FF7A', name='Classical orbit', dash='dot')
    fig.add_trace(orbit_trace)
    fig.add_traces(pos_arrow + vel_arrow)

    # Calculate GMST
    if impact_time is not None:
        gmst = gmst0 + EARTH_OMEGA * impact_time
        crossing_points_r_v = data.y[:, closest_indices]
        crossing_points_r = crossing_points_r_v[:3]
    else:
        gmst = gmst0

    # Add the Earth and other geographical features
    spheroid_mesh = SpacecraftVisualization.create_spheroid_mesh(epoch)
    fig.add_trace(spheroid_mesh)
    country_traces = SpacecraftVisualization.get_geo_traces(country_feature, gmst)
    coastline_traces = SpacecraftVisualization.get_geo_traces(coastline_feature, gmst)
    lat_lines = SpacecraftVisualization.create_latitude_lines(gmst=gmst)
    lon_lines = SpacecraftVisualization.create_longitude_lines(gmst=gmst)

    for trace in country_traces + coastline_traces + lat_lines + lon_lines:
        trace.showlegend = False
        fig.add_trace(trace)
    
    # Add periapsis and apoapsis points
    periapsis_ECI, apoapsis_ECI = periapsis_apoapsis_points(orbit)
    # Calculate the altitude of periapsis and apoapsis points
    periapsis_altitude = (orbit.r_p.to_value('m') - EARTH_R) / 1000
    apoapsis_altitude = (orbit.r_a.to_value('m') - EARTH_R) / 1000

    # Add the periapsis marker
    fig.add_trace(go.Scatter3d(x=[periapsis_ECI[0]],
                                y=[periapsis_ECI[1]],
                                z=[periapsis_ECI[2]],
                                mode='markers',
                                marker=dict(size=5, color='red', symbol='circle'),
                                name='Periapsis'))

    # Add the apoapsis marker
    fig.add_trace(go.Scatter3d(x=[apoapsis_ECI[0]],
                                y=[apoapsis_ECI[1]],
                                z=[apoapsis_ECI[2]],
                                mode='markers',
                                marker=dict(size=5, color='#05FF7A', symbol='circle'),
                                name='Apoapsis'))
    

    # Add the starting point marker
    fig.add_trace(go.Scatter3d(x=[x_pos],
                                y=[y_pos],
                                z=[z_pos],
                                mode='markers',
                                marker=dict(size=5, color='#fcba03', symbol='circle'),
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
                    textfont=dict(color=["red", "#05FF7A", "#fcba03"], size=12),
                    textposition="bottom center",
                    hoverinfo="none",
                    showlegend=False
    )
    # Add annotations to figure
    fig.add_trace(annotations_trace)

    # Simulation data
    if data is not None:
        r_eci = data.y[0:3]
        t_sol = data.t
        T_aw_data = data.additional_data['spacecraft_temperature']
        trajectory_trace = SpacecraftVisualization.create_3d_scatter(
            r_eci[0], r_eci[1], r_eci[2], T_aw_data, name='Simulated trajectory', colorscale='Agsunset'
        )
        fig.add_trace(trajectory_trace)

        # Final position or touchdown
        marker_color = 'purple' if altitude_event_times.size > 0 else 'red'
        marker_name = 'Touchdown' if altitude_event_times.size > 0 else 'Final position'
        fig.add_trace(go.Scatter3d(
            x=[r_eci[0, -1]], y=[r_eci[1, -1]], z=[r_eci[2, -1]], mode='markers',
            marker=dict(size=6, color=marker_color), name=marker_name
        ))

        # Karman line crossing
        if crossing_points is not None:
            crossing_points_r_x = np.float64(crossing_points_r[0])
            crossing_points_r_y = np.float64(crossing_points_r[1])
            crossing_points_r_z = np.float64(crossing_points_r[2])
            fig.add_trace(go.Scatter3d(
                x=[crossing_points_r_x], y=[crossing_points_r_y], z=[crossing_points_r_z],
                mode='markers+text', marker=dict(size=6, color='orange'),
                text=["Karman line crossing"], textposition="bottom center", showlegend=False
            ))

    fig.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    margin=dict(r=0, l=0, t=0, b=0),
    height=600,)
    fig.update_layout(scene_camera=dict(eye=dict(x=-0.5, y=0.6, z=1)))
    fig.update_layout(scene_aspectmode='data')
    
    return fig

def plot_heatmap(
    t_sol, normalized_spacecraft_temperature, T_aw_data, 
    custom_colorscale, spacecraft_temperature_tickvals, spacecraft_temperature_ticktext
):
    fig_colorscale = go.Figure()
    fig_colorscale.add_trace(go.Heatmap(
        x=t_sol,
        z=[normalized_spacecraft_temperature],
        text=[[f"{value:.3E} K" for value in T_aw_data]],
        hoverinfo='x+y+text',
        colorscale=custom_colorscale,
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

@njit
def compute_vertex_indices(num_lat, num_lon):
    vertex_indices = []
    for i in range(num_lat - 1):
        for j in range(num_lon - 1):
            vertex_indices.append([i * num_lon + j, i * num_lon + j + 1, (i + 1) * num_lon + j])
            vertex_indices.append([i * num_lon + j + 1, (i + 1) * num_lon + j + 1, (i + 1) * num_lon + j])
    return np.array(vertex_indices).T

def compute_velocities(geodetic_coords, v_ecef_vals, t_sol, altitudes, sim, velocity_norm):
    ground_velocity_ecef = np.linalg.norm(v_ecef_vals[:, :2], axis=1)
    vertical_velocity_ecef = np.gradient(altitudes, t_sol)
    ground_velocity_geodetic = np.zeros(len(t_sol))
    vertical_velocity_geodetic = np.zeros(len(t_sol))
    
    for i, (lat, lon, _) in enumerate(geodetic_coords):
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        rotation_matrix = np.array([
            [-np.sin(lon_rad), -np.cos(lon_rad) * np.sin(lat_rad), np.cos(lon_rad) * np.cos(lat_rad)],
            [np.cos(lon_rad), -np.sin(lon_rad) * np.sin(lat_rad), np.sin(lon_rad) * np.cos(lat_rad)],
            [0, np.cos(lat_rad), np.sin(lat_rad)]
        ])

        v_geodetic = np.dot(rotation_matrix, v_ecef_vals[i])
        ground_velocity_geodetic[i] = np.linalg.norm(v_geodetic[:2])
        vertical_velocity_geodetic[i] = v_geodetic[2]
    
    return {
        'Orbital Velocity': velocity_norm,
        'Ground Velocity (ECEF)': ground_velocity_ecef,
        'Vertical Velocity (ECEF)': vertical_velocity_ecef,
        'Ground Velocity (Geodetic)': ground_velocity_geodetic,
        'Vertical Velocity (Geodetic)': vertical_velocity_geodetic,
        'X Velocity': sim.y[3, :],
        'Y Velocity': sim.y[4, :],
        'Z Velocity': sim.y[5, :]
    }

def add_annotations(fig, x_positions, texts, min_velocity, max_velocity, color):
    for x_pos, text in zip(x_positions, texts):
        fig.add_shape(type='line', x0=x_pos, x1=x_pos, y0=min_velocity, y1=max_velocity, yref='y', xref='x', line=dict(color=color, width=2, dash='dot'))
        fig.add_annotation(x=x_pos, y=max_velocity, text=text, showarrow=True, font=dict(size=10), xanchor='center', yshift=10)

def plot_ground_track(longitudes, latitudes, normalized_altitude, custom_colorscale, tickvals, ticktext, colormap, final_position_label, st):
    # Add a single trace for the ground track
    fig7 = go.Figure()
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

    # Add point for starting point and another for final position
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

class SpacecraftVisualization:
    @staticmethod
    def create_geo_trace(geometry, gmst):
        '''
        Creates a plotly trace for a geometry object in geodetic coordinates
        :param geometry: geometry object
        :param gmst: Greenwich Mean Sidereal Time in degrees
        :return: plotly trace
        '''
        lons, lats = geometry.xy
        N = len(lats)
        x_eci, y_eci, z_eci = np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):
            x, y, z = geodetic_to_spheroid(lats[i], lons[i], alt=0)
            r_ecef = np.array([x, y, z])

            r_eci = ecef_to_eci(r_ecef , gmst)
            x_eci[i], y_eci[i], z_eci[i] = r_eci

        trace = go.Scatter3d(x=x_eci, y=y_eci, z=z_eci, mode='lines', line=dict(color='blue', width=2), hoverinfo='none')
        return trace

    @staticmethod
    def get_geo_traces(feature, gmst):
        '''
        Creates a list of plotly traces for a feature object
        :param feature: feature object
        :param gmst: Greenwich Mean Sidereal Time in degrees
        :return: list of plotly traces
        '''
        trace_list = []
        for geometry in feature.geometries():
            if geometry.geom_type == 'MultiLineString':
                for line_string in geometry.geoms:
                    trace_list.append(SpacecraftVisualization.create_geo_trace(line_string, gmst))
            else:
                trace_list.append(SpacecraftVisualization.create_geo_trace(geometry, gmst))

        return trace_list
    
    @staticmethod
    def create_latitude_lines(gmst):
        traces = []

        for lat in range(-90, 91, 30):
            geometry = sgeom.LineString([(lon, lat) for lon in range(-180, 181, 1)])
            lons, lats = geometry.xy
            N = len(lats)
            x_eci, y_eci, z_eci = np.zeros(N), np.zeros(N), np.zeros(N)

            for i in range(N):
                x, y, z = geodetic_to_spheroid(lats[i], lons[i], alt=0)
                r_ecef = np.array([x, y, z])

                r_eci = ecef_to_eci(r_ecef, gmst)
                x_eci[i], y_eci[i], z_eci[i] = r_eci

            trace = go.Scatter3d(x=x_eci, y=y_eci, z=z_eci, mode='lines', line=dict(color='blue', width=2), hoverinfo='none')
            traces.append(trace)
        return traces

    @staticmethod
    def create_longitude_lines(gmst):
        traces = []

        for lon in range(-180, 180, 30):
            geometry = sgeom.LineString([(lon, lat) for lat in range(-90, 91, 1)])
            lons, lats = geometry.xy
            N = len(lats)
            x_eci, y_eci, z_eci = np.zeros(N), np.zeros(N), np.zeros(N)

            for i in range(N):
                x, y, z = geodetic_to_spheroid(lats[i], lons[i], alt=0)
                r_ecef = np.array([x, y, z])

                r_eci = ecef_to_eci(r_ecef, gmst)
                x_eci[i], y_eci[i], z_eci[i] = r_eci

            trace = go.Scatter3d(x=x_eci, y=y_eci, z=z_eci, mode='lines', line=dict(color='blue', width=2), hoverinfo='none')
            traces.append(trace)
        return traces

    @staticmethod
    def create_spheroid_mesh(epoch, N=50):
        '''
        Creates a plotly mesh trace for a spheroid
        :param epoch: epoch object
        :param N: number of points to use for each latitude and longitude line
        :return: plotly mesh trace
        '''
        EARTH_COLOR_SCALE = [
            (0.0, '#00144F'), # Dark blue   
            (0.35, '#03172E'),   
            (0.48, '#001963'),  
            (0.52, '#0048A5'),  
            (0.75, '#2F78FF'),
            (1.0, '#659BFF'), # Light blue
        ]
        latitude, longitude = np.meshgrid(np.linspace(-90, 90, N), np.linspace(-180, 180, N))
        current_time = epoch.datetime

        # Create a numpy array with latitude and longitude values
        lat_long_arr = np.column_stack((latitude.flatten(), longitude.flatten()))

        # Calculate the solar position for each pair of latitude and longitude
        solar_position = []
        for lat, lon in lat_long_arr:
            solar_position.append(pvlib.solarposition.get_solarposition(current_time, lat, lon))
        zenith_values = [pos.zenith for pos in solar_position]
        zenith = np.array(zenith_values).reshape(latitude.shape)

        # Normalize solar zenith angles
        normalized_zenith = (zenith - zenith.min()) / (zenith.max() - zenith.min())

        lat = np.linspace(-90, 90, N)
        lon = np.linspace(-180, 180, N)
        lat_grid, lon_grid = np.meshgrid(lat, lon)

        lat_rad_grid = np.radians(lat_grid)
        alt_grid = np.zeros_like(lat_rad_grid)

        x, y, z = geodetic_to_spheroid(lat_grid, lon_grid, alt_grid)
        norm_zenith_flat = normalized_zenith.flatten()

        # Calculate the vertex indices for the mesh triangles
        num_lon = len(lon)
        num_lat = len(lat)
        
        vertex_indices = compute_vertex_indices(num_lat, num_lon)

        return go.Mesh3d(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            i=vertex_indices[0], j=vertex_indices[1], k=vertex_indices[2],
            intensity=norm_zenith_flat,
            colorscale=EARTH_COLOR_SCALE,
            colorbar=None,
            showscale=False,
            alphahull=0, opacity=1.0, hoverinfo='none')

    @staticmethod
    def create_3d_arrow(x_start, y_start, z_start, x_end, y_end, z_end, color, name):
        # Arrow line trace
        line_trace = go.Scatter3d(
            x=[x_start, x_end],
            y=[y_start, y_end],
            z=[z_start, z_end],
            mode='lines',
            line=dict(color=color),
            hoverinfo="none",
            name=name,
        )

        # Arrowhead trace
        arrowhead_trace = go.Cone(
            x=[x_end],
            y=[y_end],
            z=[z_end],
            u=[x_end - x_start],
            v=[y_end - y_start],
            w=[z_end - z_start],
            sizemode='absolute',
            sizeref=200000,
            anchor='tip',
            colorscale=[[0, color], [1, color]],
            showscale=False,
            hoverinfo="none",
            name=name
        )

        return line_trace, arrowhead_trace
    
    @staticmethod
    def plot_orbit_3d(orbit, num_points=1000, color='blue', name=None, dash='solid'):
        '''
        Creates a plotly trace for a 3D orbit
        :param orbit: Orbit object
        :param num_points: number of points to use for the trace
        :param color: color of the trace
        :param name: name of the trace
        :return: plotly trace
        '''
        # Get position data
        time_values = np.linspace(0, orbit.period.to(u.s).value, num_points) * u.s
        positions = np.array([orbit.propagate(t).represent_as(CartesianRepresentation).xyz.to(u.m).value for t in time_values])

        # Create a 3D scatter plot
        scatter = go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                            mode='lines', line=dict(width=3, color=color,dash=dash), name=name)

        return scatter

    @staticmethod
    def create_3d_scatter(x, y, z, colors, name, colorscale='Viridis'):
        '''
        Creates a plotly trace for a 3D scatter plot
        :param x: x data
        :param y: y data
        :param z: z data
        :param colors: color data
        :param name: name of the trace
        :return: plotly trace
        '''
        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=colors, width=4,colorscale=colorscale),
            name=name,
        )
        return scatter
    
    @staticmethod
    def find_crossing_points(t_sol, downrange, altitude, threshold=100000):
        '''
        Finds the crossing points of a given threshold altitude
        :param t_sol: time array
        :param downrange: downrange array
        :param altitude: altitude array
        :param threshold: threshold altitude
        :return: crossing points downrange and time
        '''
        crossing_points_downrange = []
        crossing_points_time = []
        for i in range(1, len(altitude)):
            if (altitude[i] >= threshold and altitude[i-1] < threshold) or (altitude[i] <= threshold and altitude[i-1] > threshold):
                crossing_points_downrange.append(downrange[i])
                crossing_points_time.append(t_sol[i])
        return crossing_points_downrange, crossing_points_time