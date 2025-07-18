import os
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from urllib.parse import urljoin, urlparse
from collections import deque
import warnings
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# --- Configure Selenium WebDriver ---
# This looks for 'chromedriver' in the same directory as the script.
# Make sure you have downloaded it and placed it here.
service = Service() 
options = Options()
# Run in "headless" mode (no browser window will pop up)
options.add_argument("--headless") 
options.add_argument("--log-level=3") # Suppress console messages from Chrome

# Configure logging
logging.basicConfig(filename='pdf_downloader.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Default directory to save PDF files
DEFAULT_DOWNLOAD_DIR = "downloaded_pdfs"

def create_download_directory():
    """Create the default download directory if it doesn't exist."""
    if not os.path.exists(DEFAULT_DOWNLOAD_DIR):
        os.makedirs(DEFAULT_DOWNLOAD_DIR)
        logging.info(f"Created directory: {DEFAULT_DOWNLOAD_DIR}")
    else:
        logging.info(f"Directory already exists: {DEFAULT_DOWNLOAD_DIR}")

def get_page_source_with_selenium(url):
    """Use Selenium to get the fully rendered page source after JavaScript execution."""
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get(url)
        # Wait a few seconds for JavaScript to load the content
        time.sleep(5) 
        return driver.page_source
    except Exception as e:
        log_error(f"Selenium failed to get page source for {url}: {e}")
        return None
    finally:
        driver.quit()

def log_error(message):
    """Log error messages."""
    logging.error(message)
    print(f"[ERROR] {message}")


def download_pdf(url, filename):
    """Download a PDF file."""
    filepath = os.path.join(DEFAULT_DOWNLOAD_DIR, filename)
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logging.info(f"Downloaded: {filename}")
        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        log_error(f"Error downloading {filename}: {e}")


def is_same_domain(base_url, target_url):
    """Check if the target URL is within the same domain."""
    return urlparse(base_url).netloc == urlparse(target_url).netloc


def scan_and_download_pdfs(target_url, recursive=False, max_depth=None):
    """Scan for PDFs and download."""
    url_queue = deque([(target_url, 0)])
    visited_urls = set()

    while url_queue:
        current_url, depth = url_queue.popleft()

        if current_url in visited_urls or (max_depth is not None and depth > max_depth):
            continue
        visited_urls.add(current_url)

        logging.info(f"Scanning: {current_url}")
        print(f"Scanning: {current_url}")

        # --- USE SELENIUM TO GET THE HTML ---
        page_source = get_page_source_with_selenium(current_url)
        if not page_source:
            continue # Skip if Selenium failed

        soup = BeautifulSoup(page_source, 'lxml')

        for link in soup.find_all('a', href=True):
            href = link['href']
            # On this site, the base tag makes relative links tricky.
            # We will always join from the main domain root.
            full_url = urljoin(f"https://{urlparse(target_url).netloc}", href)

            if href.lower().endswith('.pdf'):
                pdf_filename = os.path.basename(full_url)
                download_pdf(full_url, pdf_filename)
            elif recursive and is_same_domain(target_url, full_url) and full_url not in visited_urls:
                # Add a check to avoid re-queueing already visited URLs
                url_queue.append((full_url, depth + 1))
        
        # Be respectful, add a small delay (already done by Selenium's load time)

def main():
    """Main function."""
    target_url = input("Enter the target URL: ").strip()
    recursive = input("Do you want to recursively search within the same domain? (y/n): ").strip().lower() == 'y'
    max_depth = None
    if recursive:
        try:
            depth_input = input("Enter the maximum recursion depth (e.g., 2, press Enter for no limit): ").strip()
            if depth_input:
                max_depth = int(depth_input)
        except ValueError:
            log_error("Invalid maximum depth. Continuing without depth limit.")

    create_download_directory()
    scan_and_download_pdfs(target_url, recursive, max_depth)
    print("\nScanning and downloading complete.")

if __name__ == "__main__":
    main()
