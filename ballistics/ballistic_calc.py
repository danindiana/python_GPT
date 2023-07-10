import matplotlib.pyplot as plt
import numpy as np

# Constants
GRAIN_TO_KG = 0.00006479891
FPS_TO_MS = 0.3048

# Projectile properties
mass_grains = 55
velocity_fps = 3200

# Convert to metric units
mass_kg = mass_grains * GRAIN_TO_KG
velocity_ms = velocity_fps * FPS_TO_MS

# Calculate kinetic energy (Joules) and distance (meters) for each second up to 10 seconds
times = np.arange(0, 11, 1)
energies = 0.5 * mass_kg * velocity_ms**2 * np.ones_like(times)
distances = velocity_ms * times

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
