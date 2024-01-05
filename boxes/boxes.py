import matplotlib.pyplot as plt
import numpy as np

# Define the number of random rectangles
num_rectangles = 5

# Create a figure and axis
fig, ax = plt.subplots()

# Generate and draw random rectangles
for _ in range(num_rectangles):
    x = np.random.rand()  # Random x-coordinate
    y = np.random.rand()  # Random y-coordinate
    width = np.random.rand() * 0.5  # Random width
    height = np.random.rand() * 0.5  # Random height

    # Create a Rectangle patch and add it to the axis
    rect = plt.Rectangle((x, y), width, height, color='blue', alpha=0.5)
    ax.add_patch(rect)

# Set axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Display the plot
plt.show()
