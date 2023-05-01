import plotly.graph_objects as go
import numpy as np
from poliastro.bodies import Earth
from coordinate_converter import (ecef_to_eci, geodetic_to_spheroid)
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from poliastro.twobody import Orbit
import shapely.geometry as sgeom


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
    
# test this class
if __name__ == '__main__':
    # Create a figure
    fig = go.Figure()

    # Add the spheroid mesh trace
    fig.add_trace(SpacecraftVisualization.create_spheroid_mesh())

    # Add the latitude and longitude lines
    fig.add_traces(SpacecraftVisualization.create_latitude_lines())
    fig.add_traces(SpacecraftVisualization.create_longitude_lines())

    # Add the Earth texture
    fig.add_trace(SpacecraftVisualization.create_earth_texture())

    # Add the orbit trace
    fig.add_trace(SpacecraftVisualization.plot_orbit_3d(Orbit.circular(Earth, 400 * u.km), num_points=1000, color='red'))

    # Add the satellite trace
    fig.add_trace(SpacecraftVisualization.create_3d_scatter([0], [0], [0], ['red'], 'Satellite'))

    # Set the scene
    fig.update_layout(scene=dict(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=dict(text='')),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=dict(text='')),
        zaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=dict(text='')),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    ))

    # Show the figure
    fig.show()