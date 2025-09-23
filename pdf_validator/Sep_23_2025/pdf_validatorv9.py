import os
import sys
import logging
import time
import re
import shutil
import uuid
import signal
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
import hashlib
import multiprocessing
import queue
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# Create a global multiprocessing context
# Use 'spawn' for better cross-platform compatibility and stability
MP_CONTEXT = multiprocessing.get_context('spawn')

# Global process tracking registry
_active_processes: Set[multiprocessing.Process] = set()
_processes_lock = threading.Lock()

def register_process(process: multiprocessing.Process) -> None:
    """Register a process for global tracking.
    
    Args:
        process: The process to register
    """
    with _processes_lock:
        _active_processes.add(process)

def unregister_process(process: multiprocessing.Process) -> None:
    """Unregister a process from global tracking.
    
    Args:
        process: The process to unregister
    """
    with _processes_lock:
        if process in _active_processes:
            _active_processes.remove(process)

def cleanup_processes() -> None:
    """Clean up all registered processes that are still running.
    
    This ensures we don't leave orphaned processes when we exit.
    """
    logging.info(f"Cleaning up processes: {len(_active_processes)} active processes found")
    with _processes_lock:
        processes_to_clean = list(_active_processes)
    
    for process in processes_to_clean:
        try:
            if process.is_alive():
                logging.warning(f"Terminating lingering process {process.pid}")
                process.terminate()
                process.join(timeout=0.5)
                
                # Force kill if needed
                if process.is_alive():
                    logging.warning(f"Force killing process {process.pid}")
                    process.kill()
                    process.join(timeout=0.5)
            
            # Always unregister regardless of whether it was alive
            unregister_process(process)
        except Exception as e:
            logging.error(f"Error cleaning up process: {e}")
            # Still try to unregister
            unregister_process(process)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logging.warning(f"Signal {signum} received. Cleaning up processes and exiting...")
    
    # Perform cleanup
    try:
        # First clean up any active processes
        cleanup_processes()
        
        # Also clean up the resource tracker to prevent semaphore leaks
        # This is a bit of a hack but helps prevent resource warnings
        import gc
        gc.collect()
        
        # Shutdown the resource tracker cleanly if possible
        try:
            import multiprocessing.resource_tracker
            multiprocessing.resource_tracker._resource_tracker._stop = True  
            multiprocessing.resource_tracker._resource_tracker.join(1)
        except Exception:
            pass
            
        logging.info("Cleanup complete, exiting now")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
    
    # Use os._exit instead of sys.exit to avoid threading exceptions during shutdown
    os._exit(1)

# Use a more robust PDF validation library
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("PyMuPDF (fitz) is not installed. PDF validation will be basic.")

# Optional: Try to import Pillow for additional image-based validation
try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Configure the number of worker processes/threads
# Default to number of CPU cores minus 1 (leave one core for the main process)
CPU_COUNT = max(1, multiprocessing.cpu_count() - 1)

# Constants for filename handling
MAX_FILENAME_LENGTH = 110  # Allow some buffer below filesystem limits
MAX_PATH_LENGTH = 4000     # Allow buffer below PATH_MAX limit (typically 4096)
FILENAME_SAFE_CHARS = re.compile(r'[^a-zA-Z0-9_.-]')  # Characters to replace in filenames

# Constants for resource management
MAX_PDF_SIZE_MB = 100      # Skip files larger than this many MB to prevent memory issues
MAX_MEMORY_PERCENT = 80    # Threshold to pause processing if memory usage exceeds this percentage
MAX_BATCH_PROCESSING_TIME = 10 * 60  # Maximum time (seconds) for a batch before watchdog forces continuation

# Use the already defined _active_processes and _processes_lock from above
# No need to redefine them

# Register a cleanup function to run on normal program exit
atexit_registered = False
def ensure_cleanup():
    """Ensure cleanup runs on exit."""
    global atexit_registered
    if not atexit_registered:
        import atexit
        atexit.register(cleanup_processes)
        atexit_registered = True
ensure_cleanup()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pdf_validation_worker(file_path, check_rendering, quick_mode, fix_long_names, result_queue):
    """Worker function to validate a PDF file and put the result in a queue.
    
    Args:
        file_path: Path to the PDF file to validate
        check_rendering: Whether to perform rendering checks
        quick_mode: Whether to use quick validation mode
        fix_long_names: Whether to fix long filenames
        result_queue: Queue to store the result
    """
    try:
        # Validate the file and put result in queue
        result = validate_file_wrapper(file_path, check_rendering, quick_mode, fix_long_names)
        result_queue.put(result)
    except Exception as e:
        # Catch exceptions to ensure we always put something in the queue
        logging.error(f"Worker error processing {file_path}: {e}")
        result_queue.put((file_path, False, None))

def process_file_with_timeout(file_path: str, check_rendering: bool, quick_mode: bool, fix_long_names: bool, timeout: int = 60) -> Tuple[str, bool, Optional[str]]:
    """Process a single file with a timeout to prevent hanging.
    
    Args:
        file_path: Path to the PDF file to validate
        check_rendering: Whether to perform rendering checks
        quick_mode: Whether to use quick validation mode
        fix_long_names: Whether to fix long filenames
        timeout: Maximum seconds to wait for processing
        
    Returns:
        Tuple of (file_path, is_valid, original_path)
    """
    # Use a queue to get the result from the worker process
    # Use our global MP_CONTEXT to ensure consistent context
    result_queue = MP_CONTEXT.Queue()
    
    # Use our global MP_CONTEXT for process creation
    process = MP_CONTEXT.Process(
        target=pdf_validation_worker,
        args=(file_path, check_rendering, quick_mode, fix_long_names, result_queue)
    )
    process.daemon = True  # Set as daemon so it gets killed if parent exits
    
    # Register the process for global tracking
    register_process(process)
    
    # Start the process
    try:
        process.start()
    except Exception as e:
        logging.error(f"Failed to start process for {file_path}: {e}")
        unregister_process(process)
        return (file_path, False, None)
    
    try:
        # Wait for result with timeout
        result = result_queue.get(block=True, timeout=timeout)
        return result
    except (queue.Empty, Exception) as e:
        logging.error(f"Timeout or error processing {file_path}: {e}")
        return (file_path, False, None)
    finally:
        # Clean up
        try:
            # First attempt a clean shutdown
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
                
            # Force kill if needed
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
                
            # Unregister from global tracking
            unregister_process(process)
        except Exception as e:
            logging.error(f"Error cleaning up process for {file_path}: {e}")
            # Still try to unregister
            unregister_process(process)

def get_memory_usage() -> float:
    """Get the current memory usage of the process as a percentage.
    
    Returns:
        Memory usage as a percentage (0-100)
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_percent()
    except (ImportError, Exception):
        # If psutil is not available, assume memory is OK
        return 0.0

def is_file_too_large(file_path: str, max_size_mb: int = MAX_PDF_SIZE_MB) -> bool:
    """Check if a file is larger than the maximum size.
    
    Args:
        file_path: Path to the file to check
        max_size_mb: Maximum size in megabytes
        
    Returns:
        True if the file is too large, False otherwise
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb > max_size_mb
    except Exception:
        return False

def safe_print(text: str):
    """Safely print text, handling Unicode encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Print with error handling for problematic characters
        print(text.encode('utf-8', errors='replace').decode('utf-8'))

def is_filename_too_long(file_path: str) -> bool:
    """Check if a filename is too long for the filesystem.
    
    Args:
        file_path: Full path to the file
        
    Returns:
        True if the filename is too long, False otherwise
    """
    path_obj = Path(file_path)
    filename = path_obj.name
    
    # Check if the filename itself is too long
    if len(filename) > MAX_FILENAME_LENGTH:
        return True
    
    # Check if the full path is too long
    if len(file_path) > MAX_PATH_LENGTH:
        return True
        
    return False

def sanitize_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """Create a safe, shortened version of a filename.
    
    Args:
        filename: Original filename to sanitize
        max_length: Maximum length for the sanitized filename
        
    Returns:
        A sanitized version of the filename
    """
    # Get base name and extension
    base, ext = os.path.splitext(filename)
    
    # Replace unsafe characters with underscores
    base = FILENAME_SAFE_CHARS.sub('_', base)
    
    # Shorten the base if needed (preserve extension)
    available_length = max_length - len(ext)
    if len(base) > available_length:
        # Keep the start and end of the filename, replace middle with hash
        # This preserves some meaningful parts of the filename
        if available_length < 20:
            # For very short available lengths, just use a hash
            hash_part = hashlib.md5(base.encode()).hexdigest()[:8]
            base = hash_part
        else:
            # Keep some of the start and end
            start_len = (available_length - 10) // 2
            end_len = available_length - 10 - start_len
            hash_part = hashlib.md5(base.encode()).hexdigest()[:8]
            base = base[:start_len] + '_' + hash_part + '_' + base[-end_len:]
    
    return base + ext

def fix_long_filename(file_path: str) -> Tuple[str, bool]:
    """Fix a filename that's too long by creating a new file with a shorter name.
    
    Args:
        file_path: Original file path
        
    Returns:
        Tuple of (new_file_path, was_renamed)
    """
    if not is_filename_too_long(file_path):
        return file_path, False
    
    try:
        original_path = Path(file_path)
        dir_path = original_path.parent
        filename = original_path.name
        ext = original_path.suffix
        
        # Create a shorter name
        new_filename = sanitize_filename(filename)
        
        # If still too long or conflicts exist, use a UUID-based name
        if len(new_filename) > MAX_FILENAME_LENGTH or os.path.exists(dir_path / new_filename):
            short_uuid = str(uuid.uuid4())[:8]
            new_filename = f"pdf_{short_uuid}{ext}"
        
        new_file_path = str(dir_path / new_filename)
        
        # Create a copy of the file with the new name
        shutil.copy2(file_path, new_file_path)
        
        # Log the rename operation
        logging.info(f"Renamed long filename:\n  From: {file_path}\n  To: {new_file_path}")
        
        # Return the new path
        return new_file_path, True
        
    except Exception as e:
        logging.error(f"Error fixing long filename {file_path}: {e}")
        return file_path, False

def validate_pdf_robust(file_path: str, check_rendering: bool = True, quick_mode: bool = True) -> bool:
    """Validates a PDF file using PyMuPDF, falling back to basic validation if unavailable.
    Checks if the PDF can be opened and if the pages can be rendered.
    
    Args:
        file_path: Path to the PDF file to validate
        check_rendering: Whether to perform additional rendering quality checks
        quick_mode: If True, uses faster but less thorough validation (default: True)
        
    Returns:
        Boolean indicating if the PDF is valid
    """
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(file_path)
            
            # Check if document is encrypted and needs a password
            if doc.is_encrypted:
                logging.error(f"PDF is encrypted and requires a password: {file_path}")
                doc.close()
                return False
                
            # Check if document has valid pages
            if doc.page_count == 0:
                logging.error(f"PDF has no pages: {file_path}")
                doc.close()
                return False
            
            # Try to access and render each page (or at least sample some pages)
            max_pages_to_check = min(5, doc.page_count)  # Limit check to first 5 pages for performance
            for page_num in range(max_pages_to_check):
                try:
                    page = doc[page_num]
                    # Try to get page text - this will fail if page can't be rendered
                    _ = page.get_text()
                except Exception as e:
                    logging.error(f"Failed to access page {page_num} in {file_path}: {e}")
                    doc.close()
                    return False
            
            # Basic validation passed, now do the additional rendering check if requested
            doc.close()
            
            if check_rendering:
                # Run the more comprehensive rendering quality check
                quality_results = check_pdf_render_quality(file_path, quick_mode=quick_mode)
                if quality_results["render_issues"]:
                    issues = "; ".join(quality_results["issues"])
                    logging.error(f"PDF rendering issues in {file_path}: {issues}")
                    return False
                    
            return True
        except Exception as e:
            logging.error(f"PyMuPDF validation failed for {file_path}: {e}")
            return False
    else:
        return validate_pdf_basic(file_path, quick_mode)

def validate_pdf_basic(file_path: str, quick_mode: bool = True) -> bool:
    """Validates a PDF file by checking its header, trailer, and basic structure.
    This is a more thorough basic validation when PyMuPDF isn't available.
    
    Args:
        file_path: Path to the PDF file to validate
        quick_mode: If True, performs only essential checks for speed
    """
    try:
        # Get file size first - very fast check
        file_size = os.path.getsize(file_path)
        if file_size < 1000:  # Most valid PDFs are at least 1KB
            logging.error(f"PDF file too small, likely truncated: {file_path}")
            return False
            
        with open(file_path, 'rb') as f:
            # Check PDF header - always do this minimal check
            header = f.read(4)
            if header != b'%PDF':
                logging.error(f"Invalid PDF header in {file_path}")
                return False
                
            if quick_mode:
                # In quick mode, just do a few essential checks
                # Jump to the end to check for EOF marker
                f.seek(max(0, file_size - 1024))
                trailer = f.read(1024)
                return b'%%EOF' in trailer
            else:
                # In thorough mode, read the entire file and do more checks
                # Continue reading the rest of the file
                content = header + f.read()
                
                # Check for PDF trailer
                if not (b'%%EOF' in content[-1024:]):
                    logging.error(f"Missing PDF trailer in {file_path}")
                    return False
                
                # Check for corrupted xref table (common issue)
                if b'xref' not in content:
                    logging.error(f"Missing xref table in {file_path}")
                    return False
                    
                return True
    except Exception as e:
        logging.error(f"Error opening or reading {file_path}: {e}")
        return False

def delete_files(file_paths: List[str]) -> int:
    """Deletes a list of files and handles potential errors."""
    deleted_count = 0
    for file_path in file_paths:
        try:
            os.remove(file_path)
            logging.info(f"Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")
    return deleted_count

def check_pdf_render_quality(file_path: str, quick_mode: bool = False) -> Dict[str, Any]:
    """Performs advanced checks on PDF rendering quality.
    
    This function attempts to detect common rendering issues like:
    - Missing fonts
    - Corrupt images
    - Invalid page dimensions
    - Malformed content streams
    
    Args:
        file_path: Path to the PDF file to check
        quick_mode: If True, uses faster but less thorough checks
        
    Returns:
        A dictionary containing quality metrics and issue flags
    """
    results = {
        "render_issues": False,
        "issues": []
    }
    
    if not HAS_PYMUPDF:
        results["issues"].append("PyMuPDF not available for render quality check")
        return results
        
    try:
        doc = fitz.open(file_path)
        
        # Check document-level problems
        if doc.is_repaired:
            results["issues"].append("Document needed repair during loading")
            results["render_issues"] = True
            if quick_mode:  # Early return if in quick mode and we already found issues
                doc.close()
                return results
        
        # Analyze a much smaller sample of pages for speed
        pages_to_check = []
        if doc.page_count <= 3 or not quick_mode:
            # For very small documents, still check all pages
            pages_to_check = list(range(min(3, doc.page_count)))
        else:
            # For quick mode on larger documents, just check first and last page
            pages_to_check = [0]
            if doc.page_count > 1:
                pages_to_check.append(doc.page_count - 1)
        
        # For very large documents in quick mode, limit even more
        if quick_mode and doc.page_count > 100:
            pages_to_check = [0]  # Just check the first page for huge documents
        
        for page_idx in pages_to_check:
            try:
                page = doc[page_idx]
                
                # Check for abnormal page dimensions
                if page.rect.width == 0 or page.rect.height == 0:
                    results["issues"].append(f"Page {page_idx+1} has invalid dimensions")
                    results["render_issues"] = True
                    if quick_mode:  # Early return in quick mode
                        break
                
                # In quick mode, skip the more intensive checks if we've already found issues
                if quick_mode and results["render_issues"]:
                    break
                
                # Check for missing fonts - simpler check in quick mode
                if not quick_mode:
                    text_dict = page.get_text("dict")
                    if "error" in text_dict:
                        results["issues"].append(f"Text extraction error on page {page_idx+1}")
                        results["render_issues"] = True
                        if quick_mode:
                            break
                else:
                    # Faster check in quick mode
                    try:
                        _ = page.get_text("text")
                    except Exception:
                        results["issues"].append(f"Text extraction error on page {page_idx+1}")
                        results["render_issues"] = True
                        break
                
                # Check for content rendering - more efficient in quick mode
                try:
                    # Use a much smaller rendering size for quicker processing
                    scale = 0.2 if quick_mode else 1.0
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                    
                    # Only do the more intensive PIL analysis if not in quick mode
                    if HAS_PIL and not quick_mode:
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Check if image is mostly empty or single color
                        colors = img.getcolors(maxcolors=256)
                        if colors is not None and len(colors) < 5:
                            # Very few colors might indicate rendering problems for some documents
                            dominant_color = max(colors, key=lambda x: x[0])[1]
                            if dominant_color[0] == dominant_color[1] == dominant_color[2]:  # grayscale
                                # If over 98% of image is one gray color, it might be a failed render
                                most_common_count = max(colors, key=lambda x: x[0])[0]
                                if most_common_count / (pix.width * pix.height) > 0.98:
                                    results["issues"].append(f"Page {page_idx+1} may have rendered as blank")
                                    results["render_issues"] = True
                except Exception as e:
                    results["issues"].append(f"Failed to render page {page_idx+1}: {str(e)}")
                    results["render_issues"] = True
                    if quick_mode:
                        break
                    
            except Exception as e:
                results["issues"].append(f"Error analyzing page {page_idx+1}: {str(e)}")
                results["render_issues"] = True
                if quick_mode:
                    break
        
        doc.close()
    except Exception as e:
        results["issues"].append(f"Error during rendering quality check: {str(e)}")
        results["render_issues"] = True
        
    return results

def find_duplicate_files(file_paths: List[str]) -> Tuple[List[str], dict]:
    """Find duplicate PDF files based on file size and first 1024 bytes of content.
    
    Args:
        file_paths: List of paths to PDF files to check for duplicates.
        
    Returns:
        A tuple containing a list of duplicate files and a dictionary mapping
        each file to its duplicate group identifier.
    """
    # Dictionary to store file signature (size + content hash) -> paths
    size_content_map = {}
    duplicate_files = []
    duplicate_groups = {}
    
    for file_path in file_paths:
        try:
            # Get file size
            size = os.path.getsize(file_path)
            
            # Read first 1024 bytes for content comparison
            with open(file_path, 'rb') as f:
                content_start = f.read(1024)
                
            # Create a signature combining size and content start
            signature = (size, hash(content_start))
            
            # Check if we've seen this signature before
            if signature in size_content_map:
                duplicate_files.append(file_path)
                group_id = size_content_map[signature][0]  # Use first file as group identifier
                duplicate_groups[file_path] = group_id
            else:
                size_content_map[signature] = [file_path]
                
        except Exception as e:
            logging.error(f"Error checking for duplicates in {file_path}: {e}")
    
    return duplicate_files, duplicate_groups

def validate_file_wrapper(file_path: str, check_rendering: bool = True, quick_mode: bool = True,
                        fix_long_names: bool = True) -> Tuple[str, bool, Optional[str]]:
    """Wrapper function for parallel processing that validates a single file.
    
    Args:
        file_path: Path to the PDF file to validate
        check_rendering: Whether to perform additional rendering quality checks
        quick_mode: If True, uses faster but less thorough validation
        fix_long_names: Whether to fix filenames that are too long
        
    Returns:
        Tuple of (file_path, is_valid, original_path)
    """
    try:
        logging.info(f"Validating: {file_path}")
        
        # Check if file is too large to process safely
        if is_file_too_large(file_path):
            logging.warning(f"Skipping oversized file ({file_path}): exceeds {MAX_PDF_SIZE_MB}MB limit")
            return (file_path, False, None)
            
        # Check if filename is too long and fix if needed
        original_path = file_path
        renamed = False
        
        if fix_long_names and is_filename_too_long(file_path):
            file_path, was_renamed = fix_long_filename(file_path)
            renamed = was_renamed
        
        # Validate the file (original or renamed)
        is_valid = validate_pdf_robust(file_path, check_rendering, quick_mode)
        result_str = "Valid" if is_valid else "Invalid"
        logging.info(f"  {result_str}")
        
        if renamed:
            return (file_path, is_valid, original_path)
        else:
            return (file_path, is_valid, None)
            
    except Exception as e:
        # Catch all exceptions so they don't crash the worker process
        logging.error(f"Error processing file {file_path}: {str(e)}")
        # Return the file as invalid since we couldn't process it
        return (file_path, False, None)

def scan_and_validate_pdf_files(directory_path: str, delete_invalid: bool = False, recursive: bool = False, 
                              detect_duplicates: bool = False, check_rendering: bool = True,
                              num_workers: int = None, batch_size: int = 100, 
                              quick_mode: bool = True, fix_long_names: bool = True) -> Tuple[List[str], List[str], int, List[str], int, List[str]]:
    """Scans, validates, and optionally deletes PDF files.
    
    Args:
        directory_path: The directory to scan.
        delete_invalid: Whether to delete invalid files.
        recursive: Whether to scan subdirectories.
        detect_duplicates: Whether to detect duplicate files.
        check_rendering: Whether to perform additional rendering quality checks.
        num_workers: Number of parallel workers (default: CPU_COUNT)
        batch_size: Number of files to process in each batch
        quick_mode: If True, uses faster but less thorough validation
        fix_long_names: If True, renames files with excessively long names
        
    Returns:
        A tuple containing lists of valid files, invalid files, number of deleted invalid files,
        duplicate files, number of deleted duplicate files, and renamed files.
    """
    start_time = time.time()
    valid_files = []
    invalid_files = []
    renamed_files = []
    
    if num_workers is None:
        num_workers = CPU_COUNT
    
    # Find all PDF files
    logging.info("Finding PDF files...")
    if recursive:
        file_list = [os.path.join(root, file)
                    for root, _, files in os.walk(directory_path)
                    for file in files if file.lower().endswith('.pdf')]
    else:
        file_list = [os.path.join(directory_path, file)
                    for file in os.listdir(directory_path)
                    if file.lower().endswith('.pdf') and os.path.isfile(os.path.join(directory_path, file))]
    
    total_files = len(file_list)
    logging.info(f"Found {total_files} PDF files to process")
    
    if not file_list:
        return valid_files, invalid_files, 0, [], 0
    
    # Process files in parallel using a process pool
    # We use process pool instead of thread pool since PyMuPDF operations are CPU-intensive
    logging.info(f"Processing with {num_workers} workers...")
    
    # Create a partial function with fixed parameters
    validation_fn = partial(validate_file_wrapper, check_rendering=check_rendering, quick_mode=quick_mode, fix_long_names=fix_long_names)
    
    # Function to process a single batch with error handling
    def process_batch(batch_files):
        batch_results = []
        batch_start_time = time.time()
        batch_timeout = 600  # 10 minutes total batch timeout 
        
        # Check memory usage before starting the batch
        memory_usage = get_memory_usage()
        if memory_usage > MAX_MEMORY_PERCENT:
            logging.warning(f"High memory usage detected ({memory_usage:.1f}%). Pausing briefly...")
            # Force garbage collection
            import gc
            gc.collect()
            # Sleep to allow system to recover
            time.sleep(5)
        
        # Adjust batch size based on memory usage
        if memory_usage > 70:
            # Reduce effective batch size for this iteration
            process_files = batch_files[:len(batch_files)//2]
            logging.info(f"Memory pressure: Processing reduced batch of {len(process_files)} files")
        else:
            process_files = batch_files
        
        # Use ThreadPool to manage worker processes - this gives us more control
        # than ProcessPoolExecutor and prevents deadlocks
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all files to the executor
            futures = []
            for file_path in process_files:
                # Each thread will spawn a single process with timeout management
                future = executor.submit(
                    process_file_with_timeout, 
                    file_path, 
                    check_rendering, 
                    quick_mode, 
                    fix_long_names,
                    90  # Generous timeout in seconds
                )
                futures.append((future, file_path))
            
            # Set up a watchdog timer for the entire batch
            def watchdog_timer():
                nonlocal futures
                time.sleep(batch_timeout)
                # If this function completes, the batch has taken too long
                logging.warning(f"Batch watchdog timer expired after {batch_timeout} seconds")
                # Trigger cleanup of all processes
                cleanup_processes()
            
            # Start watchdog in a separate thread
            watchdog = threading.Thread(target=watchdog_timer, daemon=True)
            watchdog.start()
            
            # Collect results as they complete
            for future, file_path in futures:
                # Check if we've exceeded the batch timeout
                if time.time() - batch_start_time > batch_timeout:
                    logging.warning(f"Batch timeout exceeded after {batch_timeout} seconds")
                    break
                    
                try:
                    # Wait for the result with timeout
                    result = future.result(timeout=120)  # Slightly longer than the process timeout
                    batch_results.append(result)
                except concurrent.futures.TimeoutError:
                    logging.error(f"Processing timed out for {file_path}")
                    batch_results.append((file_path, False, None))
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    batch_results.append((file_path, False, None))
        
        # Clean up any orphaned processes after batch completion
        cleanup_processes()
        
        # If we only processed a partial batch due to memory constraints,
        # process the rest sequentially with careful timeouts
        if len(process_files) < len(batch_files):
            remaining = batch_files[len(process_files):]
            logging.info(f"Processing remaining {len(remaining)} files sequentially")
            
            # Calculate how much time we have left for sequential processing
            elapsed_time = time.time() - batch_start_time
            remaining_time = max(60, batch_timeout - elapsed_time)  # At least 60 seconds
            per_file_timeout = min(60, remaining_time / len(remaining))  # Cap at 60s per file
            
            for file_path in remaining:
                try:
                    # Direct sequential processing with timeout
                    result = process_file_with_timeout(
                        file_path, 
                        check_rendering, 
                        quick_mode, 
                        fix_long_names,
                        int(per_file_timeout)  # Timeout in seconds
                    )
                    batch_results.append(result)
                except Exception as exc:
                    logging.error(f"Sequential processing: File {file_path} generated an exception: {exc}")
                    batch_results.append((file_path, False, None))
                
                # Check if we've exceeded the batch timeout
                if time.time() - batch_start_time > batch_timeout:
                    logging.warning("Sequential processing timeout exceeded")
                    break
        
        # Final cleanup before returning
        cleanup_processes()
        
        return batch_results
    
    # Process files in smaller batches to provide progress updates
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(file_list) + batch_size - 1)//batch_size}")
        
        try:
            # Process the batch with error handling
            results = process_batch(batch)
            # Reset failure counter on success
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logging.error(f"Batch processing failed: {e}")
            
            # If too many consecutive failures, reduce batch size and workers
            if consecutive_failures >= max_consecutive_failures:
                logging.warning("Multiple batch failures detected. Reducing batch size and worker count.")
                batch_size = max(5, batch_size // 2)
                num_workers = max(1, num_workers // 2)
                logging.info(f"Adjusted settings: batch_size={batch_size}, workers={num_workers}")
                consecutive_failures = 0  # Reset counter after adjustment
                
            # Process this batch sequentially as a fallback
            results = []
            for file_path in batch:
                try:
                    result = validate_file_wrapper(file_path, check_rendering, quick_mode, fix_long_names)
                    results.append(result)
                except Exception as exc:
                    logging.error(f"Failed to process {file_path}: {exc}")
                    results.append((file_path, False, None))
        
        # Process results from this batch
        for result in results:
            file_path, is_valid, original_path = result
            
            if original_path:  # This file was renamed
                renamed_files.append((original_path, file_path))
                # The original file should be considered invalid (it will be replaced by the renamed one)
                invalid_files.append(original_path)
                
            # Process the validation result of the current file
            if is_valid:
                valid_files.append(file_path)
            else:
                invalid_files.append(file_path)
        
        # Print progress
        processed = min(i + batch_size, total_files)
        elapsed = time.time() - start_time
        files_per_second = processed / elapsed if elapsed > 0 else 0
        logging.info(f"Progress: {processed}/{total_files} files ({files_per_second:.1f} files/sec)")
        
        # Give the system a brief moment to recover
        time.sleep(0.1)
    
    deleted_invalid_count = delete_files(invalid_files) if delete_invalid else 0
    
    # Handle duplicate detection - use thread pool since this is more I/O bound
    duplicate_files = []
    deleted_duplicate_count = 0
    
    if detect_duplicates and valid_files:
        logging.info("Checking for duplicate files...")
        
        # Process duplicate detection using ThreadPoolExecutor for I/O bound operations
        if len(valid_files) > 1000:
            # For large numbers of files, use threads to speed up duplicate detection
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Split files into chunks for parallel processing
                chunk_size = max(100, len(valid_files) // num_workers)
                chunks = [valid_files[i:i + chunk_size] for i in range(0, len(valid_files), chunk_size)]
                
                # Process each chunk in parallel
                chunk_results = list(executor.map(find_duplicate_files, chunks))
                
                # Merge results
                all_duplicates = []
                all_groups = {}
                for dups, groups in chunk_results:
                    all_duplicates.extend(dups)
                    all_groups.update(groups)
                    
                duplicate_files, duplicate_groups = all_duplicates, all_groups
        else:
            # For smaller sets, just process directly
            duplicate_files, duplicate_groups = find_duplicate_files(valid_files)
            
        # Handle deletion
        if duplicate_files and delete_invalid:  # Reuse delete_invalid flag for duplicates
            deleted_duplicate_count = delete_files(duplicate_files)
            # Remove deleted duplicates from valid files
            valid_files = [f for f in valid_files if f not in duplicate_files]
    
    # Print performance summary
    elapsed = time.time() - start_time
    total_processed = len(valid_files) + len(invalid_files)
    files_per_sec = total_processed / elapsed if elapsed > 0 else 0
    logging.info(f"Performance: Processed {total_processed} files in {elapsed:.1f} seconds ({files_per_sec:.1f} files/sec)")
    
    if renamed_files:
        logging.info(f"Renamed {len(renamed_files)} files with excessively long names")
    
    return valid_files, invalid_files, deleted_invalid_count, duplicate_files, deleted_duplicate_count, renamed_files

def get_user_confirmation(prompt: str, default_value: str = "y") -> bool:
    """Gets user confirmation with a default value, handling input more robustly."""
    default_value = default_value.lower()
    if default_value not in ("y", "n"):
        raise ValueError("default_value must be 'y' or 'n'")
    
    while True:
        alt = "n" if default_value == "y" else "y"
        prompt_str = f"{prompt} ({default_value.upper()}/{alt}): "
        response = input(prompt_str).strip().lower()
        if not response:
            return default_value == "y"
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Invalid input. Please enter 'y', 'n', 'yes', or 'no'.")

def write_file_list(filename: str, valid_files: List[str], invalid_files: List[str], 
                duplicate_files: List[str] = None, renamed_files: List[Tuple[str, str]] = None):
    try:
        with open(filename, "w", encoding="utf-8", errors="replace") as file:  # Added error handling
            file.write("Valid PDF files:\n")
            file.writelines(f"{path}\n" for path in valid_files)
            file.write("\nInvalid PDF files:\n")
            file.writelines(f"{path}\n" for path in invalid_files)
            
            if duplicate_files:
                file.write("\nDuplicate PDF files:\n")
                file.writelines(f"{path}\n" for path in duplicate_files)
                
            if renamed_files:
                file.write("\nRenamed PDF files (original → new):\n")
                file.writelines(f"{orig} → {new}\n" for orig, new in renamed_files)
                
        logging.info(f"File list written to {filename}")
    except Exception as e:
        logging.error(f"Error writing to {filename}: {e}")

if __name__ == "__main__":
    # Set up signal handlers for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    target_directory = input("Enter the target directory path: ")
    if not os.path.isdir(target_directory):
        print("Invalid directory path.")
        sys.exit(1)
    
    recursive_scan = get_user_confirmation("Do you want to scan recursively?")
    delete_invalid = get_user_confirmation("Do you want to delete invalid/corrupted files?")
    detect_duplicates = get_user_confirmation("Do you want to detect/delete duplicate files?", "y")  # Default is Y
    check_rendering = get_user_confirmation("Do you want to perform advanced rendering checks?", "y")  # Default is Y
    quick_mode = get_user_confirmation("Do you want to use quick mode for faster processing?", "y")  # Default is Y
    fix_long_names = get_user_confirmation("Do you want to fix/rename files with excessively long names?", "y")  # Default is Y
    
    # Inform user of size limits
    print(f"\nNote: Files larger than {MAX_PDF_SIZE_MB}MB will be skipped to prevent memory issues.")
    
    # Check if psutil is available for memory monitoring
    try:
        import psutil
        print("Memory monitoring is enabled.")
    except ImportError:
        print("Memory monitoring is disabled (psutil not installed).")
        print("For better stability with large PDF collections, install psutil:")
        print("pip install psutil")
    
    # Determine optimal number of workers based on system resources
    num_workers = CPU_COUNT
    # Adjust batch size based on available memory and number of workers
    batch_size = max(20, 200 // num_workers)  # Smaller batch size for more frequent updates
    
    print(f"\nScanning and validating PDF files using {num_workers} workers...")
    start_time = time.time()
    
    valid_files, invalid_files, deleted_invalid_count, duplicate_files, deleted_duplicate_count, renamed_files = scan_and_validate_pdf_files(
        target_directory, delete_invalid, recursive_scan, detect_duplicates, check_rendering,
        num_workers=num_workers, batch_size=batch_size, quick_mode=quick_mode, fix_long_names=fix_long_names
    )
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.1f} seconds")
    
    print("\nValid PDF files:")
    for file_path in valid_files:
        safe_print(file_path)  # Use safe_print function
    
    print("\nInvalid PDF files:")
    for file_path in invalid_files:
        safe_print(file_path)  # Use safe_print function
    
    if detect_duplicates:
        print("\nDuplicate PDF files:")
        for file_path in duplicate_files:
            safe_print(file_path)  # Use safe_print function
    
    write_file_list("validated_files.txt", valid_files, invalid_files, 
                 duplicate_files if detect_duplicates else None,
                 renamed_files if renamed_files else None)
    
    if delete_invalid:
        print(f"\nNumber of invalid files deleted: {deleted_invalid_count}")
        
    if detect_duplicates and delete_invalid:
        print(f"Number of duplicate files deleted: {deleted_duplicate_count}")
        
    if renamed_files:
        print(f"\nNumber of files renamed due to excessive name length: {len(renamed_files)}")
        print("\nRenamed files (original → new):")
        for orig, new in renamed_files[:10]:  # Show first 10 renamed files
            safe_print(f"  {orig} → {new}")
        if len(renamed_files) > 10:
            print(f"  ... and {len(renamed_files) - 10} more files")
            
        # Write renamed files to a separate log file
        with open("renamed_files.txt", "w", encoding="utf-8", errors="replace") as f:
            f.write("Original path → New path\n")
            f.write("-------------------------------------------\n")
            for orig, new in renamed_files:
                f.write(f"{orig} → {new}\n")
        print("\nFull list of renamed files written to renamed_files.txt")
    
    # Final cleanup of any remaining processes
    cleanup_processes()
    logging.info("All processes cleaned up successfully")