# PDF Validation and Cleanup Script

A comprehensive PDF validation and cleanup script with robust parallel processing capabilities.

## Features

### 🔄 **Robust Parallel Processing**
- Uses both process and thread pools with timeout management
- Consistent multiprocessing context using spawn method
- Proper worker function isolation for serialization

### 💾 **Memory Management**
- Implements memory monitoring and adaptive batch processing
- Efficient resource handling for large-scale operations

### 📁 **File Safety**
- Handles long filenames and large files
- Manages filesystem constraints effectively
- Safe file operations with proper error handling

### ✅ **PDF Validation**
- Multiple validation levels:
  - Basic validation
  - Rendering checks
  - Quality assessments

### 🔍 **Duplicate Detection**
- Finds duplicate files based on file size
- Content-based duplicate detection

### 🧹 **Graceful Cleanup**
- Proper process termination and resource cleanup
- Global process tracking with registration/unregistration
- Robust cleanup function for lingering processes

### 🎯 **User-Friendly Interface**
- Interactive prompts and progress reporting
- Clear logging and error messages

## Technical Implementation

### Multiprocessing Context
```python
MP_CONTEXT = multiprocessing.get_context('spawn')
```
- Ensures all multiprocessing objects use the same context
- Prevents "SemLock created in a fork context" errors

### Worker Function Isolation
- `pdf_validation_worker()` function properly isolated for serialization
- Clear separation of concerns between main process and workers

### Process Management
- Global tracking of processes with registration/unregistration
- Robust cleanup processes with proper termination
- Comprehensive error handling around process creation

### Signal Handling
- Enhanced signal handler for thorough resource cleanup
- Multiprocessing resource tracker cleanup to prevent semaphore leaks
- Uses `os._exit()` for cleaner termination instead of `sys.exit()`

### Error Handling
- Comprehensive error handling throughout the codebase
- Proper logging of worker process errors
- Prevents main process hanging due to worker failures

## Usage

The script provides an interactive command-line interface with progress reporting and comprehensive validation capabilities for PDF files.

# - Python 3.12.9

# - Package Version
------- -------
- pip     25.2
- PyMuPDF 1.25.3
