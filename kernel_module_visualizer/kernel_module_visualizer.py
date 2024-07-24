import subprocess
import pygraphviz as pgv

def get_kernel_modules():
    """Run lsmod and parse its output to get module names and dependencies."""
    lsmod_output = subprocess.check_output(['lsmod']).decode('utf-8')
    modules = {}
    for line in lsmod_output.split('\n')[1:]:  # Skip the first line which is the header
        parts = line.split()
        if not parts:
            continue
        module_name = parts[0]
        dependencies = parts[3].split(',') if len(parts) > 3 else []
        modules[module_name] = dependencies
    return modules

def build_graph(modules):
    """Build a graph from the modules and their dependencies."""
    G = pgv.AGraph(strict=False, directed=True)
    for module, dependencies in modules.items():
        G.add_node(module)
        for dep in dependencies:
            G.add_node(dep)
            G.add_edge(module, dep)
    return G

def visualize_kernel_modules():
    """Main function to visualize kernel modules."""
    modules = get_kernel_modules()
    G = build_graph(modules)
    G.layout(prog='dot')  # Use dot layout engine for hierarchical structures
    G.draw('kernel_modules.png')  # Save the visualization to a file

if __name__ == "__main__":
    visualize_kernel_modules()
