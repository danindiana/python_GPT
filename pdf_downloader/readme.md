# PDF Downloader

A Python script that scans websites for PDF files and downloads them, with support for JavaScript-rendered pages and recursive scanning.

## Features

- 🚀 **Selenium-powered** - Handles JavaScript-rendered content
- 📂 **Automatic directory creation** - Saves PDFs to `downloaded_pdfs/`
- 🔄 **Recursive scanning** - Option to follow links within the same domain
- ⏱ **Delay handling** - Waits for JavaScript execution
- 📝 **Comprehensive logging** - Detailed activity log in `pdf_downloader.log`
- 🛡 **Same-domain protection** - Only follows links from the original domain by default

## Prerequisites

- Python 3.6+
- Chrome browser installed
- ChromeDriver (see installation below)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-downloader.git
   cd pdf-downloader
Install required packages:

bash
pip install -r requirements.txt
Download ChromeDriver:

Download from ChromeDriver website (match your Chrome version)

Place chromedriver executable in the project directory or in your PATH

Usage
Run the script:

bash
python pdf_downloader.py
You'll be prompted for:

Target URL (e.g., "https://example.com")

Whether to scan recursively (y/n)

Maximum recursion depth (optional)

Configuration
You can modify these defaults in the script:

DEFAULT_DOWNLOAD_DIR: Change output directory

time.sleep(5): Adjust JavaScript wait time

Chrome options in Options()

Example Output
text
Enter the target URL: https://example.edu/documents
Do you want to recursively search within the same domain? (y/n): y
Enter the maximum recursion depth (e.g., 2, press Enter for no limit): 2

Scanning: https://example.edu/documents
Downloaded: document1.pdf
Downloaded: research-paper.pdf
Scanning: https://example.edu/documents/archive
Downloaded: historical-record.pdf

Scanning and downloading complete.
Troubleshooting
Common Issues:

WebDriverException: Ensure ChromeDriver version matches your Chrome browser

Timeout errors: Increase wait time in the script

Missing PDFs: Site may require authentication or have anti-bot measures

Check pdf_downloader.log for detailed error information.

License
MIT License - Free for personal and commercial use
