import matplotlib.pyplot as plt
import numpy as np

# Constants
GRAIN_TO_KG = 0.00006479891
FPS_TO_MS = 0.3048
AIR_DENSITY_SEA_LEVEL = 1.225
DRAG_COEFFICIENT = 0.47  # For a sphere, will be less for a bullet-like shape
GRAVITY = 9.81
TIME_STEP = 0.01

# Projectile properties
mass_grains = 55
velocity_fps = 3200

# Convert to metric units
mass_kg = mass_grains * GRAIN_TO_KG
velocity_ms = velocity_fps * FPS_TO_MS

# Initialize lists to store time, energy, and distance
times = [0]
energies = [0.5 * mass_kg * velocity_ms**2]
distances = [0]

# Initialize velocity and position
velocity = velocity_ms
position = 0

# Run simulation for 10 seconds
while times[-1] < 10:
    # Calculate forces
    drag_force = 0.5 * AIR_DENSITY_SEA_LEVEL * DRAG_COEFFICIENT * velocity**2
    gravity_force = mass_kg * GRAVITY

    # Calculate total force
    total_force = drag_force + gravity_force

    # Update velocity, ensuring it doesn't go negative
    if total_force * TIME_STEP > mass_kg * velocity:
        velocity = 0
    else:
        velocity -= (total_force/mass_kg) * TIME_STEP

    # Update position
    position += velocity * TIME_STEP

    # Calculate energy
    energy = 0.5 * mass_kg * velocity**2

    # Store time, energy, and distance
    times.append(times[-1] + TIME_STEP)
    energies.append(energy)
    distances.append(position)

# Convert lists to NumPy arrays
times = np.array(times)
energies = np.array(energies)
distances = np.array(distances)

# Create a figure and axes
fig, ax1 = plt.subplots()

# Plot energy over time
color = 'tab:red'
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Energy (Joules)', color=color)
ax1.plot(times, energies, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second axes for the distance plot
ax2 = ax1.twinx()

# Plot distance over time
color = 'tab:blue'
ax2.set_ylabel('Distance (meters)', color=color)
ax2.plot(times, distances, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Show the plot
plt.show()
