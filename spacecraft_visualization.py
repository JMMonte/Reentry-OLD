import plotly.graph_objects as go
import numpy as np
from poliastro.bodies import Earth
from coordinate_converter import (ecef_to_eci, geodetic_to_spheroid)
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from poliastro.twobody import Orbit
import shapely.geometry as sgeom
import plotly.express as px


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

def create_cone_mesh(height_ratio, top_radius_ratio, truncation_ratio, num_points=100):
    u = np.linspace(1 - truncation_ratio, 1, num_points)
    v = np.linspace(0, 2 * np.pi, num_points)
    
    u_grid, v_grid = np.meshgrid(u, v)
    
    x = u_grid * np.cos(v_grid)
    y = u_grid * np.sin(v_grid)
    
    z = -height_ratio * u_grid
    x_top = top_radius_ratio * x + (1 - top_radius_ratio) * x[::-1]
    
    return np.concatenate([x, x_top]), np.concatenate([y, y[::-1]]), np.concatenate([z, z[::-1]])


def create_half_sphere_mesh(radius, height_offset, num_points=100):
    u = np.linspace(0, 0.5 * np.pi, num_points)
    v = np.linspace(0, 2 * np.pi, num_points)
    
    u_grid, v_grid = np.meshgrid(u, v)
    
    x = radius * np.sin(u_grid) * np.cos(v_grid)
    y = radius * np.sin(u_grid) * np.sin(v_grid)
    z = -0.25 * radius * np.cos(u_grid) - height_offset
    
    return x, y, z

def create_circular_cap(radius, height, num_points=100):
    u = np.linspace(0, radius, num_points)
    v = np.linspace(0, 2 * np.pi, num_points)

    u_grid, v_grid = np.meshgrid(u, v)

    x = u_grid * np.cos(v_grid)
    y = u_grid * np.sin(v_grid)
    z = -height * np.ones_like(u_grid)

    return x, y, z

def rotate_around_y(x, y, z, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    x_rot = x * cos_angle - z * sin_angle
    z_rot = x * sin_angle + z * cos_angle
    
    return x_rot, y, z_rot

def generate_turbulent_lines(base_radius, num_lines=40, num_points=150, randomness=0.2, max_length=2):
    lines = []
    
    for i in range(num_lines):
        angle = 2 * np.pi * i / num_lines
        x_start = -base_radius*0.8 * np.cos(angle)
        y_start = -base_radius*0.8 * np.sin(angle)
        z_start = -height_ratio*1.2
        
        x = [x_start]
        y = [y_start]
        z = [z_start]
        
        for j in range(1, num_points):
            step = max_length * j / num_points
            x.append(x_start - step * np.cos(angle) + randomness * (np.random.rand() - 0.5))
            y.append(y_start - step * np.sin(angle) + randomness * (np.random.rand() - 0.5))
            z.append(z_start + step + randomness * (np.random.rand() - 0.5))
            
        lines.append((x, y, z))
        
    return lines


def create_3d_line(x, y, z, colorscale='Plasma', showlegend=False):
    colors = np.linspace(0, 1, len(x))
    line_color = [colorscale[int(c * (len(colorscale) - 1))] for c in colors]

    line = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
        line=dict(width=2, color=line_color),
        showlegend=showlegend,
        marker=dict(color=colors, colorscale=colorscale, showscale=False)
    )
    return line

def create_capsule ():
    cap_radius = base_radius * (1 - truncation_ratio) * top_radius_ratio * 2
    cap_height = height_ratio * (1 - truncation_ratio)

    turbulent_lines = generate_turbulent_lines(base_radius)
    x_cap, y_cap, z_cap = create_circular_cap(cap_radius, cap_height)
    x_cone, y_cone, z_cone = create_cone_mesh(height_ratio, top_radius_ratio, truncation_ratio)
    x_shield, y_shield, z_shield = create_half_sphere_mesh(base_radius, height_ratio)


    angle_degrees = 45
    x_cone_rot, y_cone_rot, z_cone_rot = rotate_around_y(x_cone, y_cone, z_cone, angle_degrees)
    x_cap_rot, y_cap_rot, z_cap_rot = rotate_around_y(x_cap, y_cap, z_cap, angle_degrees)
    x_shield_rot, y_shield_rot, z_shield_rot = rotate_around_y(x_shield, y_shield, z_shield, angle_degrees)
    plasma_colorscale = px.colors.sequential.Plasma[::-1]

    fig = go.Figure()
    for line in turbulent_lines:
        x_line, y_line, z_line = rotate_around_y(np.array(line[0]), np.array(line[1]), np.array(line[2]), angle_degrees)
        fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=z_line, mode='lines', line=dict(width=2, color="blue"), showlegend=False))

    for line in turbulent_lines:
        x_line, y_line, z_line = rotate_around_y(np.array(line[0]), np.array(line[1]), np.array(line[2]), angle_degrees)
        fig.add_trace(create_3d_line(x_line, y_line, z_line, colorscale=plasma_colorscale))
        

    fig.add_trace(go.Surface(x=x_cone_rot, y=y_cone_rot, z=z_cone_rot, colorscale=metallic_colorscale, showscale=False))
    fig.add_trace(go.Surface(x=x_shield_rot, y=y_shield_rot, z=z_shield_rot, colorscale=heatshield_colorscale, showscale=False))
    fig.add_trace(go.Surface(x=x_cap_rot, y=y_cap_rot, z=z_cap_rot, colorscale=metallic_colorscale, showscale=False))

    x_range = [-10, 20]
    y_range = [-15, 15]
    z_range = [-20, 10]

    fig.update_layout(
        scene=dict(
            aspectratio=dict(x=1, y=1, z=1),  # Manually set the aspect ratio
            xaxis=dict(range=x_range, visible=False),
            yaxis=dict(range=y_range, visible=False),
            zaxis=dict(range=z_range, visible=False),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=-0.15, y=0, z=0.15),
                eye=dict(x=-0.15, y=-0.2, z=0.2)
            )
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig



class SpacecraftVisualization:
    @staticmethod
    def create_geo_trace(geometry, gmst):
        '''
        Creates a plotly trace for a geometry object
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
    def create_spheroid_mesh(N=50):
        '''
        Creates a plotly mesh trace for a spheroid
        :param N: number of points to use for each latitude and longitude line
        :param attractor: attractor object
        :return: plotly mesh trace
        '''
        lat = np.linspace(-90, 90, N)
        lon = np.linspace(-180, 180, N)
        lat_grid, lon_grid = np.meshgrid(lat, lon)

        lat_rad_grid = np.radians(lat_grid)
        alt_grid = np.zeros_like(lat_rad_grid)

        x, y, z = geodetic_to_spheroid(lat_grid, lon_grid, alt_grid)

        return go.Mesh3d(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            alphahull=0, color='rgb(0,0,100)', opacity=0.9, hoverinfo='none')

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
            sizeref=500000,
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