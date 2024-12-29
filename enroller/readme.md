# Text to Vector DB Project

This project is designed to efficiently manage and store text files in a SQLite database. It sets up a structured Python environment for handling large text corpora, enabling integration with a retrieval-augmented generator (RAG) system.

## Features

- **User-Friendly Initialization**:
  - Prompts the user for project directory and name.
  - Creates the project folder with correct permissions if it doesn't exist.
  - Sets up a Python 3.12 virtual environment in the project directory.

- **Database Management**:
  - Initializes a SQLite database (`text_files.db`) if it doesn't already exist.
  - Avoids duplicate entries using content hashing.

- **File Enrollment**:
  - Processes `.txt` files in a target directory.
  - Offers recursive or non-recursive directory processing.

- **Informative Console Output**:
  - Provides step-by-step feedback during setup and file enrollment.

## Requirements

- Python 3.12
- SQLite
- A Unix-based environment (tested on Ubuntu 22.04)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Run the script:
   ```bash
   python3 script.py
   ```

3. Follow the on-screen prompts to configure the project directory and enroll text files.

## Usage

1. **Set up the project:**
   - The script will prompt for a project root directory and name.
   - If the project directory does not exist, the script will offer to create it.

2. **Set up the virtual environment:**
   - If this is the first run, the script will create a Python virtual environment (`venv`) inside the project directory.

3. **Initialize the database:**
   - If the SQLite database is not already present, it will be initialized automatically.

4. **Enroll text files:**
   - Specify the target directory containing `.txt` files.
   - Choose whether to process the directory recursively.
   - The script will store the file path, content, and a hash of each file in the database, skipping duplicates.

## Example Output

- Directory setup:
  ```
  Enter the project root directory [/home/jeb/programs]:
  Enter the project name [text_to_vector_db]:
  Directory /home/jeb/programs/text_to_vector_db does not exist. Create it? (y/n): y
  [INFO] Project directory created at: /home/jeb/programs/text_to_vector_db
  ```

- Virtual environment setup:
  ```
  [INFO] Setting up Python virtual environment...
  [INFO] Virtual environment created at: /home/jeb/programs/text_to_vector_db/venv
  ```

- Database initialization:
  ```
  [INFO] Database initialized at: /home/jeb/programs/text_to_vector_db/text_files.db
  ```

- File enrollment:
  ```
  Enter the path to the directory containing text files: /path/to/files
  Do you want to process files recursively (y/n)? y
  [INFO] Enrolled file: /path/to/files/example1.txt
  [INFO] Enrolled file: /path/to/files/example2.txt
  [INFO] File enrollment complete.
  ```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

This project was developed to streamline the storage and management of text corpora for RAG systems and similar applications. Special thanks to users who contributed feedback and suggestions for improvement.
