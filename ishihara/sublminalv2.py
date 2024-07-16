import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
duration = 1 * b2.second
presentation_rate = 60 * b2.Hz
stimulus_frequency = 15 * b2.Hz
subliminal_duration = 10 * b2.ms
tau = 10 * b2.ms
threshold = 0.6 
v_reset = 0

# Neuron model
eqs = '''
dv/dt = (I-v)/tau : 1
I : 1 (shared)
'''
G = b2.NeuronGroup(1, eqs, threshold='v>threshold', reset='v=v_reset', method='exact')

# Input (Poisson spikes)
P = b2.PoissonGroup(1, rates=stimulus_frequency)
S = b2.Synapses(P, G, on_pre='v += 1.0')
S.connect()

# Stimulus current
stimulus_current = 1.2

# Function to turn stimulus on and off
@b2.network_operation(dt=1/presentation_rate)
def update_stimulus():
    if (b2.defaultclock.t < subliminal_duration) or (duration/2 < b2.defaultclock.t < duration/2 + subliminal_duration):
        G.I = stimulus_current
    else:
        G.I = 0

# Monitor membrane potential and spikes
state_monitor = b2.StateMonitor(G, 'v', record=0)
spike_monitor = b2.SpikeMonitor(G)

# Run simulation
net = b2.Network(G, P, S, update_stimulus, state_monitor, spike_monitor)
net.run(duration)

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'k-', linewidth=2)
spike_dots, = ax.plot([], [], 'r.', markersize=10)  # Initialize spike dots
ax.set_xlim(0, duration/b2.ms)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane Potential (V)')
ax.set_title('Subliminal Perception Test')

# Animation function
def animate(i):
    if i < len(state_monitor.t):
        # Update the line data with the current membrane potential
        line.set_data(state_monitor.t/b2.ms, state_monitor.v[0])
        
        # Update the spike dots
        current_spikes = spike_monitor.t[spike_monitor.t < state_monitor.t[i]]
        spike_dots.set_data(current_spikes/b2.ms, np.ones(len(current_spikes)))
        
    return line, spike_dots

# Create the animation
ani = FuncAnimation(fig, animate, frames=int(duration/b2.defaultclock.dt), blit=True, interval=1000/presentation_rate)
plt.show()
