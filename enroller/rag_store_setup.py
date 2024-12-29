import os
import sqlite3
import hashlib
from pathlib import Path
import json
from datetime import datetime
import subprocess

# Constants
DEFAULT_PROJECT_ROOT = "/home/jeb/programs"
DEFAULT_PROJECT_NAME = "text_to_vector_db"
DB_NAME = "text_files.db"

# Ensure the project directory exists and is correctly permissioned
def setup_project_directory():
    project_root = input(f"Enter the project root directory [{DEFAULT_PROJECT_ROOT}]: ").strip() or DEFAULT_PROJECT_ROOT
    project_name = input(f"Enter the project name [{DEFAULT_PROJECT_NAME}]: ").strip() or DEFAULT_PROJECT_NAME
    project_path = Path(project_root) / project_name
    
    if not project_path.exists():
        create = input(f"Directory {project_path} does not exist. Create it? (y/n): ").strip().lower()
        if create != 'y':
            print("[INFO] Project directory creation aborted.")
            exit()
        project_path.mkdir(parents=True, exist_ok=True)
        os.chmod(project_path, 0o755)
        print(f"[INFO] Project directory created at: {project_path}")
    else:
        print(f"[INFO] Using existing project directory at: {project_path}")

    # Set up Python virtual environment
    venv_path = project_path / "venv"
    if not venv_path.exists():
        print("[INFO] Setting up Python virtual environment...")
        subprocess.run(["python3.12", "-m", "venv", str(venv_path)])
        print(f"[INFO] Virtual environment created at: {venv_path}")
    else:
        print(f"[INFO] Virtual environment already exists at: {venv_path}")

    return project_path

# Initialize SQLite database
def setup_database(project_path):
    db_path = project_path / DB_NAME
    db_exists = db_path.exists()
    conn = sqlite3.connect(db_path)
    if not db_exists:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                content TEXT,
                hash TEXT UNIQUE,
                created_at TEXT
            )
        """)
        conn.commit()
        print(f"[INFO] Database initialized at: {db_path}")
    else:
        print(f"[INFO] Database already exists at: {db_path}")
    return conn

# Generate a hash for file content
def hash_content(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# Enroll text files into the database
def enroll_files(target_dir, conn, recursive):
    cursor = conn.cursor()
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_hash = hash_content(content)
                    cursor.execute("""
                        INSERT INTO text_files (file_path, content, hash, created_at) 
                        VALUES (?, ?, ?, ?)
                    """, (str(file_path), content, file_hash, datetime.now().isoformat()))
                    print(f"[INFO] Enrolled file: {file_path}")
                except sqlite3.IntegrityError:
                    print(f"[WARN] File {file_path} is already enrolled.")
                except Exception as e:
                    print(f"[ERROR] Error processing file {file_path}: {e}")
        if not recursive:
            break
    conn.commit()
    print("[INFO] File enrollment complete.")

# Main script execution
def main():
    project_path = setup_project_directory()
    conn = setup_database(project_path)

    while True:
        target_dir = input("Enter the path to the directory containing text files: ").strip()
        if not Path(target_dir).exists():
            print(f"[ERROR] Directory {target_dir} does not exist. Please try again.")
            continue

        recursive = input("Do you want to process files recursively (y/n)? ").strip().lower()
        if recursive not in ['y', 'n']:
            print("[ERROR] Invalid input. Please enter 'y' or 'n'.")
            continue

        enroll_files(target_dir, conn, recursive == 'y')
        break

    conn.close()
    print("[INFO] All operations completed successfully.")

if __name__ == "__main__":
    main()
