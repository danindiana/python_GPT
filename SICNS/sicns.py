import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def generate_heartbeat(duration_seconds=10, sampling_rate=1000):
    """Generates a simulated heartbeat waveform for the ICNS."""

    time = np.linspace(0, duration_seconds, duration_seconds * sampling_rate)

    # Baseline heart rate (adjust as needed)
    heart_rate = 70 
    frequency = heart_rate / 60  

    # Approximate wave components
    sinus_node_wave = 0.2 * np.sin(2 * np.pi * frequency * time)
    av_node_wave = 0.15 * np.sin(2 * np.pi * (frequency - 0.05) * time + 0.2)
    bundle_branches_wave = 0.1 * np.sin(2 * np.pi * (frequency - 0.1) * time + 0.4)
    purkinje_fibers_wave = 0.05 * np.sin(2 * np.pi * (frequency - 0.15) * time + 0.6)

    return time, sinus_node_wave, av_node_wave, bundle_branches_wave, purkinje_fibers_wave

# Generate the heartbeat data
time, sinus_node_wave, av_node_wave, bundle_branches_wave, purkinje_fibers_wave = generate_heartbeat()

# Combine wave components
heartbeat = sinus_node_wave + av_node_wave + bundle_branches_wave + purkinje_fibers_wave

# Animation Setup
fig, ax = plt.subplots(figsize=(12, 6))
sinus_line, = ax.plot([], [], 'r-', linewidth=2, label='Sinus Node Wave') 
av_line, = ax.plot([], [], 'g-', linewidth=2, label='AV Node Wave')
bundle_line, = ax.plot([], [], 'b-', linewidth=2, label='Bundle Branches Wave')
purkinje_line, = ax.plot([], [], 'm-', linewidth=2, label='Purkinje Fibers Wave')
heartbeat_line, = ax.plot([], [], 'k-', linewidth=2, label='Combined Heartbeat')

ax.set_xlim(0, 10)  
ax.set_ylim(-1.5, 1.5)  
ax.set_title("Simulated Intrinsic Cardiac Nervous System (ICNS) Waveform")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude (arbitrary units)")  
ax.grid(True)
ax.legend()

# Animation Function
def animate(i):
    sinus_line.set_data(time[:i], sinus_node_wave[:i])
    av_line.set_data(time[:i], av_node_wave[:i])
    bundle_line.set_data(time[:i], bundle_branches_wave[:i])
    purkinje_line.set_data(time[:i], purkinje_fibers_wave[:i])
    heartbeat_line.set_data(time[:i], heartbeat[:i])
    return sinus_line, av_line, bundle_line, purkinje_line, heartbeat_line

# Create Animation
ani = animation.FuncAnimation(fig, animate, frames=len(time), interval=10, blit=True, repeat=False) 

plt.show()
