def calculate_effective_parallelism_and_bandwidth():
    """
    Calculates effective parallelism, bandwidth/IO movement, and overhead for parallel matrix multiplication using RAFT.
    Includes PCIe Gen 4/5 calculations and refined overhead considerations.
    """

    try:
        # Inputs for matrix dimensions
        m = int(input("Enter the number of rows in matrix A (m): "))
        n = int(input("Enter the common dimension (columns in A and rows in B) (n): "))
        p = int(input("Enter the number of columns in matrix B (p): "))

        # Inputs for parallelism and overhead
        total_parallel_units = int(input("Enter the total number of parallel units (threads or GPUs): "))
        load_balancing_efficiency = float(input("Enter the load balancing efficiency (0.0 to 1.0): "))
        communication_overhead_factor = float(input("Enter the communication overhead factor (0.0 to 1.0, higher means more overhead): "))
        hardware_limitation_factor = float(input("Enter the hardware limitation factor (0.0 to 1.0, representing percentage of limitation): "))

        # Inputs for data transfer and PCIe
        data_size_gb = float(input("Enter the total data size to be moved (GB): "))
        io_operations = float(input("Enter the total number of IO operations: "))
        pcie_gen = int(input("Enter PCIe generation (4 or 5): "))
        pcie_lanes = int(input("Enter number of PCIe lanes: "))

        # Calculate effective parallelism
        effective_parallelism_load_balance = total_parallel_units * load_balancing_efficiency
        effective_parallelism_comm_overhead = effective_parallelism_load_balance * (1 - communication_overhead_factor)
        effective_parallelism_hardware_limit = effective_parallelism_comm_overhead * (1 - hardware_limitation_factor)

        # Calculate PCIe bandwidth
        if pcie_gen == 4:
            pcie_bandwidth_gbps = pcie_lanes * 2  # 2 GB/s per lane for Gen 4
        elif pcie_gen == 5:
            pcie_bandwidth_gbps = pcie_lanes * 4  # 4 GB/s per lane for Gen 5
        else:
            raise ValueError("Invalid PCIe generation. Must be 4 or 5.")

        # Calculate data transfer time
        data_transfer_time_seconds = data_size_gb / pcie_bandwidth_gbps

        # Calculate IOPS (IO Operations Per Second)
        iops = io_operations / data_transfer_time_seconds if data_transfer_time_seconds > 0 else 0

        # Calculate computational work for matrix multiplication
        total_operations = m * n * p  # Total multiplications and additions

        # Calculate parallel computational time
        parallel_computational_time = total_operations / effective_parallelism_hardware_limit

        # Overhead function f(P)
        # For simplicity, assume overhead is proportional to the number of parallel units
        overhead_factor = 0.1  # Example: 10% overhead
        overhead_time = overhead_factor * total_parallel_units

        # Total parallel time
        total_parallel_time = parallel_computational_time + overhead_time

        # Print results
        print("\nResults:")
        print(f"Effective Parallelism (Load Balancing): {effective_parallelism_load_balance:.2f}")
        print(f"Effective Parallelism (Communication Overhead): {effective_parallelism_comm_overhead:.2f}")
        print(f"Effective Parallelism (Hardware Limitations): {effective_parallelism_hardware_limit:.2f}")
        print(f"\nPCIe Bandwidth (GB/s): {pcie_bandwidth_gbps:.2f}")
        print(f"Data Transfer Time (seconds): {data_transfer_time_seconds:.2f}")
        print(f"IOPS: {iops:.2f}")
        print(f"\nMatrix Multiplication Computational Work: {total_operations} operations")
        print(f"Parallel Computational Time (seconds): {parallel_computational_time:.2f}")
        print(f"Overhead Time (seconds): {overhead_time:.2f}")
        print(f"Total Parallel Time (seconds): {total_parallel_time:.2f}")

        # Additional impact analysis
        print("\nImpact of factors:")
        print(f"Load balancing efficiency reduction: {total_parallel_units - effective_parallelism_load_balance:.2f}")
        print(f"Communication Overhead reduction: {effective_parallelism_load_balance - effective_parallelism_comm_overhead:.2f}")
        print(f"Hardware limitation reduction: {effective_parallelism_comm_overhead - effective_parallelism_hardware_limit:.2f}")

    except ValueError as e:
        print(f"Invalid input: {e}. Please enter numeric values and valid PCIe generation.")
    except ZeroDivisionError:
        print("Error: Data transfer time cannot be zero.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    calculate_effective_parallelism_and_bandwidth()
