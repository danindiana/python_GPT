import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Seed for reproducibility (remove for true randomness)
# np.random.seed(42)  

# Generate a random array of numbers
data = np.random.rand(100, 100)

# Create a mask for the hidden pattern
pattern_mask = np.zeros((100, 100))

# Apply a hidden pattern to the data
for i in range(10):
    for j in range(10):
        if (i + j) % 3 == 0:
            data[i, j] += 0.2  # Subtle increase in value
            pattern_mask[i, j] = 1  # Mark pattern cells

# Visualize the data as a heatmap
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()

# Highlight the hidden pattern with a different color (e.g., white edges)
for i in range(10):
    for j in range(10):
        if pattern_mask[i, j] == 1:
            plt.gca().add_patch(Rectangle((j, i), 1, 1, linewidth=2, edgecolor='white', facecolor='none'))

# Add random rectangles to further obfuscate the pattern
for _ in range(15):
    x = np.random.randint(0, 10)
    y = np.random.randint(0, 10)
    width = np.random.randint(1, 4)
    height = np.random.randint(1, 4)
    plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='red', facecolor='none'))

# Remove axis labels and ticks
plt.axis('off')

# Show the plot
plt.show()
