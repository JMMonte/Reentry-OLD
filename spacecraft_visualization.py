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
        trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=2))
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
            lat_lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=1)))
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
            lon_lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=1)))
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
            alphahull=0, color='rgb(0,0,100)', opacity=0.9)

    @staticmethod
    def create_3d_arrow(x_start, y_start, z_start, x_end, y_end, z_end, color, name):
        '''
        Creates a plotly trace for a 3D arrow
        :param x_start: x coordinate of the arrow start
        :param y_start: y coordinate of the arrow start
        :param z_start: z coordinate of the arrow start
        :param x_end: x coordinate of the arrow end
        :param y_end: y coordinate of the arrow end
        :param z_end: z coordinate of the arrow end
        :param color: color of the arrow
        :param name: name of the arrow
        :return: plotly trace
        '''
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
        arrowhead_length_ratio = 0.1  # Adjust this value to change the arrowhead length
        arrowhead_width_ratio = 0.05  # Adjust this value to change the arrowhead width

        arrow_vector = np.array([x_end - x_start, y_end - y_start, z_end - z_start])
        arrow_length = np.linalg.norm(arrow_vector)
        arrow_unit_vector = arrow_vector / arrow_length

        arrowhead_length = arrow_length * arrowhead_length_ratio
        arrowhead_base = np.array([x_end, y_end, z_end]) - arrowhead_length * arrow_unit_vector

        cross_product1 = np.cross(arrow_unit_vector, np.array([1, 0, 0]))
        if np.linalg.norm(cross_product1) == 0:
            cross_product1 = np.cross(arrow_unit_vector, np.array([0, 1, 0]))

        cross_product2 = np.cross(arrow_unit_vector, cross_product1)

        arrowhead_width = arrow_length * arrowhead_width_ratio
        corner1 = arrowhead_base + arrowhead_width * (cross_product1 / np.linalg.norm(cross_product1))
        corner2 = arrowhead_base + arrowhead_width * (cross_product2 / np.linalg.norm(cross_product2))
        corner3 = arrowhead_base - arrowhead_width * (cross_product1 / np.linalg.norm(cross_product1))
        corner4 = arrowhead_base - arrowhead_width * (cross_product2 / np.linalg.norm(cross_product2))

        arrowhead_trace = go.Mesh3d(
            x=[x_end, corner1[0], corner2[0], corner3[0], corner4[0]],
            y=[y_end, corner1[1], corner2[1], corner3[1], corner4[1]],
            z=[z_end, corner1[2], corner2[2], corner3[2], corner4[2]],
            i=[0, 0, 0, 0],
            j=[1, 2, 3, 4],
            k=[2, 3, 4, 1],
            color=color,
            name=name,
            hoverinfo="none"
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
    def hsl_to_rgb(h, s, l):
        return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h / 360, l, s))
    
    @staticmethod
    def create_3d_scatter(x, y, z, altitudes, karman_line=100000,name=None):

        colors = [SpacecraftVisualization.altitude_to_color(alt) for alt in altitudes]
        rgb_colors = [SpacecraftVisualization.hsl_to_rgb(hue, 1, 0.5) for hue in colors]

        line_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color=rgb_colors, width=2),
            showlegend=True,
            name=name,
            hoverinfo="none"
        )
        return line_trace
    
    @staticmethod
    def altitude_to_color(altitude, karman_line=100000):
        # Color scale from red (0) to yellow (0.5) to green (1)
        normalized_altitude = altitude / karman_line
        hue = 2 * normalized_altitude  # 0 for red, 60 for yellow, 120 for green
        return hue
    
    @staticmethod
    def find_crossing_points(t_sol, downrange, altitude, threshold=100000):
        crossing_points_downrange = []
        crossing_points_time = []
        for i in range(1, len(altitude)):
            if (altitude[i] >= threshold and altitude[i-1] < threshold) or (altitude[i] <= threshold and altitude[i-1] > threshold):
                crossing_points_downrange.append(downrange[i])
                crossing_points_time.append(t_sol[i])
        return crossing_points_downrange, crossing_points_time
