import os

def estimate_chunk_size(target_dir):
  """Estimates the average chunk size in a directory."""

  # Ask the user for the total word count
  while True:
    try:
      total_word_count = int(input("Enter the total word count: "))
      break
    except ValueError:
      print("Invalid input. Please enter an integer.")

  # Count the number of files in the target directory
  num_files = len([f for f in os.listdir(target_dir) if f.endswith('.txt')])

  if num_files == 0:
    print("No .txt files found in the target directory.")
    return None

  # Estimate the average chunk size
  average_chunk_size = total_word_count / num_files
  return average_chunk_size

if __name__ == "__main__":
  # Ask the user for the target directory
  target_dir = input("Enter the target directory: ")

  average_chunk_size = estimate_chunk_size(target_dir)

  if average_chunk_size:
    print(f"Estimated average chunk size: {average_chunk_size:.2f} words")
