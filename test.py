# Required imports
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.earth.atmosphere import COESA76

# Initial conditions
mass = 3500 * u.kg
reentry_angle = -1 * u.deg
initial_altitude = 120 * u.km
initial_velocity = 7.8 * u.km / u.s
drag_coefficient = 1.5
reference_area = 10 * u.m**2

# Function to compute drag force
def drag_force(altitude, velocity):
    _, _, density, _ = COESA76.properties(altitude)
    dynamic_pressure = 0.5 * density * velocity**2
    drag_force = drag_coefficient * reference_area * dynamic_pressure
    return drag_force.to(u.N)

# Function to compute gravitational force
def gravitational_force(altitude, mass):
    gravitational_constant = Earth.k.to(u.m**3 / u.kg / u.s**2)
    earth_radius = Earth.R.to(u.m)
    gravitational_force = gravitational_constant * mass / (earth_radius + altitude)**2
    return gravitational_force.to(u.N)

# Function to propagate the orbit
def propagate_orbit(initial_conditions, time_step, max_simulation_time):
    # Initialize orbit
    r0 = Earth.R + initial_altitude
    v0 = initial_velocity
    orbit = Orbit.from_classical(Earth, r0, 0 * u.one, reentry_angle, 0 * u.deg, 0 * u.deg, 0 * u.deg)
    
    # Initialize simulation variables
    time = [0]
    altitude = [initial_altitude]
    velocity = [initial_velocity]
    deceleration = [0]

    while (altitude[-1] > 0 * u.km) and (time[-1] < max_simulation_time):
        # Compute drag and gravitational forces
        drag = drag_force(altitude[-1], velocity[-1])
        gravity = gravitational_force(altitude[-1], mass)

        # Update deceleration, altitude, and velocity
        new_deceleration = (drag - gravity) / mass
        new_altitude = orbit.r + orbit.r.dot(time_step)
        new_velocity = orbit.v + orbit.v.dot(time_step)
        
        # Append values to their respective lists
        time.append(time[-1] + time_step)
        altitude.append(new_altitude)
        velocity.append(new_velocity)
        deceleration.append(new_deceleration)

    return time, altitude, velocity, deceleration

# Simulation parameters
time_step = 1 * u.s
max_simulation_time = 1800 * u.s

# Run the simulation
time, altitude, velocity, deceleration = propagate_orbit(
    (mass, reentry_angle, initial_altitude, initial_velocity, drag_coefficient, reference_area),
    time_step,
    max_simulation_time
)

# Plot results
plt.figure()
plt.plot(time, altitude)
plt.xlabel('Time [s]')
plt.ylabel('Altitude [km]')
plt.title('Time vs. Altitude')
plt.show()

plt.figure()
plt.plot(time, velocity)
plt.xlabel('Time [s]')
plt.ylabel('Velocity [km/s]')
plt.title('Time vs. Velocity')
plt.show()