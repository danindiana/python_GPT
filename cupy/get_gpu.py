import cupy as cp

# Get the number of GPUs available
num_gpus = cp.cuda.runtime.getDeviceCount()

print(f"Number of GPUs detected: {num_gpus}")

# Print information about each GPU
for i in range(num_gpus):
    props = cp.cuda.runtime.getDeviceProperties(i)
    print(f"GPU {i}:")
    print(f"  Name: {props['name'].decode('utf-8')}")
    print(f"  Compute Capability: {props['major']}.{props['minor']}")
