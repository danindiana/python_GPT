import matplotlib.pyplot as plt
import numpy as np

def pack_squares(container_size, square_sizes):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set axis limits
    ax.set_xlim(0, container_size)
    ax.set_ylim(0, container_size)

    # Initialize variables for position
    x, y = 0, 0

    # Iterate through square sizes and pack them
    for size in square_sizes:
        if x + size > container_size:  # Start a new row if the square doesn't fit horizontally
            x = 0
            y += size + 1  # Add 1 unit of spacing between rows

        # Create a Rectangle patch and add it to the axis
        rect = plt.Rectangle((x, y), size, size, color='blue', alpha=0.5, linewidth=0)
        ax.add_patch(rect)

        x += size + 1  # Add 1 unit of spacing between squares

    # Display the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()

# Define the size of the container square and the sizes of the squares to be packed
container_size = 10
square_sizes = [3, 2, 1, 2, 2, 1, 1, 1]

# Call the pack_squares function to visualize the square packing
pack_squares(container_size, square_sizes)
