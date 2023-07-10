import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import interp1d

# Constants
GRAIN_TO_KG = 0.00006479891
FPS_TO_MS = 0.3048
GRAVITY = 9.81
TIME_STEP = 0.01
FEET_TO_METERS = 0.3048
BULLET_DIAMETER = 0.009  # .224 caliber in meters
RHO_SEA_LEVEL = 1.225  # kg/m^3

# Projectile properties
mass_grains = 55
velocity_fps = 3200

# Convert to metric units
mass_kg = mass_grains * GRAIN_TO_KG
velocity_ms = velocity_fps * FPS_TO_MS
height_feet = 6
height_meters = height_feet * FEET_TO_METERS

# Ballistic coefficient dependent on velocity
velocities_fps = [3100, 2550, 2050, 1800]
coefficients = [0.271, 0.264, 0.25, 0.236, 0.224]
bc_func = interp1d(velocities_fps, coefficients, kind='cubic', fill_value='extrapolate')

# Initialize lists to store time, energy, distance, and height
times = [0]
energies = [0.5 * mass_kg * velocity_ms**2]
distances = [0]
heights = [height_meters]

# Initialize velocity and position
velocity_x = velocity_ms
velocity_y = 0
position_x = 0
position_y = height_meters

# Run simulation until the projectile hits the ground
while position_y > 0:
    # Calculate forces
    velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    drag_coefficient = bc_func(velocity / FPS_TO_MS) * math.pi * (BULLET_DIAMETER/2)**2
    drag_force = 0.5 * RHO_SEA_LEVEL * velocity**2 * drag_coefficient
    gravity_force = mass_kg * GRAVITY

    # Update velocities
    velocity_x -= (drag_force * velocity_x / velocity) / mass_kg * TIME_STEP
    velocity_y -= ((drag_force * velocity_y / velocity) / mass_kg + GRAVITY) * TIME_STEP

    # Update positions
    position_x += velocity_x * TIME_STEP
    position_y += velocity_y * TIME_STEP

    # Calculate energy
    energy = 0.5 * mass_kg * velocity**2

    # Store time, energy, distance, and height
    times.append(times[-1] + TIME_STEP)
    energies.append(energy)
    distances.append(position_x)
    heights.append(position_y)

# Create a figure and axes for energy plot
fig1, ax1 = plt.subplots()

# Plot energy over distance
ax1.set_xlabel('Distance (meters)')
ax1.set_ylabel('Energy (Joules)')
ax1.plot(distances, energies)

# Create a figure and axes for trajectory plot
fig2, ax2 = plt.subplots()

# Plot trajectory
ax2.set_xlabel('Distance (meters)')
ax2.set_ylabel('Height (meters)')
ax2.plot(distances, heights)

# Show the plots
plt.show()
