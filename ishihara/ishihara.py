import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Seed for reproducibility (remove for true randomness)
np.random.seed(420)  

# Generate a random array of numbers
data = np.random.rand(100, 100)

# Apply a hidden pattern to the data
for i in range(10):
    for j in range(10):
        if (i + j) % 3 == 0:
            data[i, j] += 0.2  # Subtle increase in value

# Visualize the data as a heatmap
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()

# Add random rectangles to further obfuscate the pattern
for _ in range(15):
    x = np.random.randint(0, 10)
    y = np.random.randint(0, 10)
    width = np.random.randint(1, 4)
    height = np.random.randint(1, 4)
    plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none'))

# Remove axis labels and ticks
plt.axis('off')

# Show the plot
plt.show()
