import plotly.graph_objs as go
import random
from collections import defaultdict
import os
import inquirer
import ast

def list_directories_above_and_below(start_directory, max_depth=2):
    """List directories above and below the current directory up to a certain depth."""
    directories = set()  # Use a set to avoid duplicates
    
    print(f"Searching for directories below: {start_directory}")
    
    # Add current directory and subdirectories, up to a reasonable depth
    for root, dirs, _ in os.walk(start_directory):
        if root.count(os.sep) - start_directory.count(os.sep) > max_depth:
            continue
        for dir_name in dirs:
            directories.add(os.path.join(root, dir_name))
        print(f"Found subdirectory: {root}")

    # Add parent directories up to `max_depth`
    print(f"Searching for parent directories above: {start_directory}")
    parent_directory = start_directory
    for i in range(max_depth):
        parent_directory = os.path.dirname(parent_directory)
        directories.add(parent_directory)
        print(f"Found parent directory: {parent_directory}")
    
    return list(directories)  # Return as a list for menu selection

def get_python_files(directory):
    """Recursively collect all Python files starting from the chosen directory."""
    print(f"Scanning for Python files in: {directory}")
    python_files = []
    for root, _, files in os.walk(directory):
        if root.count(os.sep) - directory.count(os.sep) > 1:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
                print(f"Found Python file: {file}")
    if not python_files:
        print(f"No Python files found in: {directory}")
    return python_files

def get_function_and_class_names(file_path):
    """Extract function and class names from a Python file."""
    print(f"Analyzing file: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                tree = ast.parse(file.read(), filename=file_path)
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"Error in file {file_path}: {e}")
                return [], []  # Return empty if there's a syntax or decoding error
    except UnicodeDecodeError:
        print(f"UTF-8 decoding failed, skipping file: {file_path}")
        return [], []  # Skip files that can't be decoded

    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    print(f"Found {len(functions)} functions and {len(classes)} classes in file: {file_path}")
    return functions, classes

def visualize_structure_map_point_cloud_plotly(file_structure_map, show_detail=True):
    """Visualize the file structure map as a 3D point cloud using Plotly with clustering by directory and manual detail control."""
    print("\nStarting visualization using Plotly...")

    xs, ys, zs = [], [], []
    labels = []
    colors = []  # To store the color of each point

    # Separate different directories in 3D space
    directory_offset = 10  # Distance between different directories
    file_offset = 2        # Distance between files within the same directory

    # Iterate over directories and group points by directory
    print("Clustering points based on directories and files...")
    for dir_idx, (directory, items) in enumerate(file_structure_map.items()):
        dir_x = dir_idx * directory_offset
        dir_y = random.uniform(0, 10)  # Randomize y-coordinate to spread directories
        dir_z = random.uniform(0, 10)  # Randomize z-coordinate to spread directories

        for file_idx, (func_class_name, f_type) in enumerate(items):
            file_x = dir_x + file_idx * file_offset
            file_y = dir_y + random.uniform(-1, 1)
            file_z = dir_z + random.uniform(-1, 1)

            # File point
            xs.append(file_x)
            ys.append(file_y)
            zs.append(file_z)
            labels.append(f"File: {func_class_name}")
            colors.append('blue')  # Color for files

            # Functions/Classes points relative to the file (if detail is enabled)
            if show_detail:
                fx = file_x + random.uniform(-0.5, 0.5)
                fy = file_y + random.uniform(-0.5, 0.5)
                fz = file_z + random.uniform(-0.5, 0.5)

                xs.append(fx)
                ys.append(fy)
                zs.append(fz)
                labels.append(f"{f_type.capitalize()}: {func_class_name}")

                # Functions are green, Classes are red
                if f_type == 'function':
                    colors.append('green')  # Color for functions
                else:
                    colors.append('red')  # Color for classes

    print(f"Prepared {len(xs)} data points for visualization.")
    
    # Create 3D scatter plot
    scatter = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=labels, hoverinfo='text'
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        updatemenus=[{
            'buttons': [
                {
                    'args': [True],
                    'label': 'Show Detail',
                    'method': 'relayout'
                },
                {
                    'args': [False],
                    'label': 'Hide Detail',
                    'method': 'relayout'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }]
    )

    print("Rendering the Plotly figure...")
    fig = go.Figure(data=[scatter], layout=layout)
    
    try:
        fig.show()
        print("Plotly figure rendered successfully.")
    except Exception as e:
        print(f"Error while rendering Plotly figure: {e}")

def find_similar_python_files(directory):
    """Scan the chosen directory for Python files and compare their structures."""
    print(f"\nStarting similarity scan in directory: {directory}")
    python_files = get_python_files(directory)
    file_structure_map = defaultdict(list)
    
    # Build a structure map by storing functions and classes per file
    for file in python_files:
        functions, classes = get_function_and_class_names(file)
        for func in functions:
            file_structure_map[file].append((func, 'function'))
        for cls in classes:
            file_structure_map[file].append((cls, 'class'))
    
    print(f"Structure map built for {len(file_structure_map)} files.")
    
    # Visualize the structure map as a 3D point cloud with manual control for detail level
    visualize_structure_map_point_cloud_plotly(file_structure_map, show_detail=True)

def choose_directory(directories):
    """Display a menu for the user to choose a directory."""
    print(f"\nDisplaying directory choices to the user...")
    questions = [
        inquirer.List(
            'directory',
            message="Select the directory to scan",
            choices=directories
        )
    ]
    answers = inquirer.prompt(questions)
    print(f"\nUser selected directory: {answers['directory']}")
    return answers['directory']

if __name__ == "__main__":
    # Get the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    # List directories both above and below the current directory
    directories = list_directories_above_and_below(current_directory, max_depth=2)

    # Allow user to select a directory from the list
    chosen_directory = choose_directory(directories)

    # Find and display similar Python files, visualizing the structure as a 3D point cloud
    find_similar_python_files(chosen_directory)
