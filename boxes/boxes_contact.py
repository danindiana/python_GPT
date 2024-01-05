import matplotlib.pyplot as plt
import numpy as np

# Define the number of random rectangles
num_rectangles = 5

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize variables for the position of the first rectangle
x = np.random.rand()
y = np.random.rand()

# Generate and draw random rectangles that touch each other
for _ in range(num_rectangles):
    width = np.random.rand() * 0.2  # Random width
    height = np.random.rand() * 0.2  # Random height

    # Create a Rectangle patch and add it to the axis
    rect = plt.Rectangle((x, y), width, height, color='blue', alpha=0.5)
    ax.add_patch(rect)

    # Update the position of the next rectangle to touch the current one
    x += width + 0.02  # Adjust the spacing between rectangles (0.02 is an example value)

# Set axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Display the plot
plt.show()
