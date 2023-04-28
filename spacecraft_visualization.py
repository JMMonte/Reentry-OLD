import plotly.graph_objects as go
import numpy as np
from poliastro.bodies import Earth
from coordinate_converter import CoordinateConverter
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
import colorsys

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
        x, y, z = CoordinateConverter.geo_to_ecef(np.deg2rad(lons), np.deg2rad(lats))
        x, y, z = CoordinateConverter.ecef_to_eci(x, y, z, gmst)
        trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=2),hoverinfo='none')
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
    def create_latitude_lines(N=50, gmst=0):
        '''
        Creates a list of plotly traces for latitude lines
        :param N: number of points to use for each latitude line
        :param gmst: Greenwich Mean Sidereal Time in degrees
        :return: list of plotly traces
        '''
        lat_lines = []
        lon = np.linspace(-180, 180, N)
        lat_space = np.linspace(-90, 90, N // 2)
        for lat in lat_space:
            lons = np.full_like(lon, lat)
            x, y, z = CoordinateConverter.geo_to_ecef(np.deg2rad(lon), np.deg2rad(lons))
            x, y, z = CoordinateConverter.ecef_to_eci(x, y, z, gmst)
            lat_lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', hoverinfo='none', line=dict(color='blue', width=1)))
        return lat_lines

    @staticmethod
    def create_longitude_lines(N=50, gmst=0):
        '''
        Creates a list of plotly traces for longitude lines
        :param N: number of points to use for each longitude line
        :param gmst: Greenwich Mean Sidereal Time in degrees
        :return: list of plotly traces
        '''
        lon_lines = []
        lat = np.linspace(-90, 90, N)
        lon_space = np.linspace(-180, 180, N)
        for lon in lon_space:
            lons = np.full_like(lat, lon)
            x, y, z = CoordinateConverter.geo_to_ecef(np.deg2rad(lons), np.deg2rad(lat))
            x, y, z = CoordinateConverter.ecef_to_eci(x, y, z, gmst)
            lon_lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', hoverinfo='none', line=dict(color='blue', width=1)))
        return lon_lines

    @staticmethod
    def create_spheroid_mesh(N=50, attractor=Earth):
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
        lon_rad_grid = np.radians(lon_grid)
        alt_grid = np.zeros_like(lat_rad_grid)

        x, y, z = CoordinateConverter.geo_to_ecef(lon_rad_grid, lat_rad_grid, alt_grid)

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
    
