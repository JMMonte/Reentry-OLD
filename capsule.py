import numpy as np
import plotly.graph_objects as go
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
        title="Space Capsule 3D Mesh",
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
