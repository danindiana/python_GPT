import numpy as np

# Define the CRAM-like memory array (simulated as a NumPy array)
memory_array = np.array([0.5, 0.2, 0.8, 0.4, 0.6])

# Define input data (simulated as a NumPy array)
input_data = np.array([0.3, 0.7, 0.1, 0.9, 0.2])

# Perform CRAM-like computation (element-wise multiplication)
output_data = memory_array * input_data

# Sum the results to produce the final output
final_output = np.sum(output_data)

# Display the results
print("Memory Array:")
print(memory_array)
print("\nInput Data:")
print(input_data)
print("\nCRAM-like Computation Result:")
print(output_data)
print("\nFinal Output after Summation:")
print(final_output)
