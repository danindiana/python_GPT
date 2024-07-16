import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
duration = 10 * b2.second
presentation_rate = 60 * b2.Hz
stimulus_frequency = 15 * b2.Hz  # Frequency of the subliminal stimulus
subliminal_duration = 10 * b2.ms
tau = 10 * b2.ms        # Neuron time constant
v_reset = 0             # Reset voltage after spike
threshold = 0.8         # Firing threshold

# Neuron model
eqs = '''
dv/dt = (I-v)/tau : 1
I : 1 (shared)           # Input current
'''
G = b2.NeuronGroup(1, eqs, threshold='v>threshold', reset='v=v_reset')

# Create a Poisson input group
P = b2.PoissonGroup(1, rates=stimulus_frequency)

# Connect input to the neuron with a stronger synapse
S = b2.Synapses(P, G, on_pre='v += 1.0')  # Stronger input
S.connect()

# Set the stimulus current (stronger)
stimulus_current = 1.2

# Function to turn stimulus on and off
@b2.network_operation(dt=1/presentation_rate)
def update_stimulus():
    if (b2.defaultclock.t < subliminal_duration) or (duration/2 < b2.defaultclock.t < duration/2 + subliminal_duration):
        G.I = stimulus_current  # Stimulus on
    else:
        G.I = 0                # Stimulus off

# Spike monitor
M = b2.SpikeMonitor(G)

# Run simulation
net = b2.Network(G, P, S, update_stimulus)
net.run(duration)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(M.t/b2.ms, M.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Subliminal Perception Test')
plt.axvline(x=subliminal_duration/b2.ms, color='r', linestyle='--', label='Stimulus Onset')
plt.axvline(x=duration/2/b2.ms, color='r', linestyle='--')
plt.axvline(x=(duration/2 + subliminal_duration)/b2.ms, color='r', linestyle='--')
plt.legend()
plt.show()
