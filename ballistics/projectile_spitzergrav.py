import matplotlib.pyplot as plt
import numpy as np

# Constants
GRAIN_TO_KG = 0.00006479891
FPS_TO_MS = 0.3048
AIR_DENSITY_SEA_LEVEL = 1.225
DRAG_COEFFICIENT = 0.295  # For a spitzer-shaped projectile
GRAVITY = 9.81
TIME_STEP = 0.01

# Projectile properties
mass_grains = 55
velocity_fps = 3200

# Convert to metric units
mass_kg = mass_grains * GRAIN_TO_KG
velocity_ms = velocity_fps * FPS_TO_MS

# Initialize lists to store time, energy, distance, and height
times = [0]
energies = [0.5 * mass_kg * velocity_ms**2]
distances = [0]
heights = [0]

# Initialize velocity and position
velocity_x = velocity_ms  # Assuming the projectile is fired horizontally
velocity_y = 0
position_x = 0
position_y = 0

# Run simulation for 10 seconds or until the projectile hits the ground
while times[-1] < 10 and position_y >= 0:
    # Calculate forces
    velocity = np.sqrt(velocity_x**2 + velocity_y**2)  # Total velocity
    drag_force = 0.5 * AIR_DENSITY_SEA_LEVEL * DRAG_COEFFICIENT * velocity**2
    gravity_force = mass_kg * GRAVITY

    # Forces in x and y directions
    drag_force_x = drag_force * velocity_x / velocity
    drag_force_y = drag_force * velocity_y / velocity

    # Update velocities
    velocity_x -= drag_force_x/mass_kg * TIME_STEP
    velocity_y -= (drag_force_y/mass_kg + GRAVITY) * TIME_STEP

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

# Convert lists to NumPy arrays
times = np.array(times)
energies = np.array(energies)
distances = np.array(distances)
heights = np.array(heights)

# Create a figure and axes for energy plot
fig1, ax1 = plt.subplots()

# Plot energy over time
color = 'tab:red'
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Energy (Joules)', color=color)
ax1.plot(times, energies, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a figure and axes for trajectory plot
fig2, ax2 = plt.subplots()

# Plot trajectory
ax2.set_xlabel('Distance (meters)')
ax2.set_ylabel('Height (meters)')
ax2.plot(distances, heights)

# Show the plots
plt.show()
