import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import json
from tqdm import tqdm

# --- Configuration ---
DEFAULT_DOWNLOAD_DIR = "downloaded_pdfs"
SESSION_FILE = "session.json"
# List of file extensions/protocols to ignore when crawling
IGNORED_EXTENSIONS = ('.zip', '.rar', '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.mp4', 'mailto:', 'javascript:')

# --- Selenium WebDriver Setup ---
def setup_driver():
    """Initializes the Selenium WebDriver."""
    service = Service()
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--log-level=3")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    return webdriver.Chrome(service=service, options=options)

# --- Logging ---
logging.basicConfig(filename='pdf_downloader.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_error(message):
    """Logs an error message to the console and file."""
    logging.error(message)
    print(f"\n[ERROR] {message}")

# --- Core Functions ---
def create_download_directory():
    """Creates the download directory if it doesn't exist."""
    if not os.path.exists(DEFAULT_DOWNLOAD_DIR):
        os.makedirs(DEFAULT_DOWNLOAD_DIR)

def get_page_source_with_selenium(url, wait_time):
    """Uses Selenium to get the fully rendered page source."""
    driver = setup_driver()
    try:
        driver.get(url)
        time.sleep(wait_time)
        return driver.page_source
    except Exception as e:
        log_error(f"Selenium failed for {url}: {e}")
        return None
    finally:
        driver.quit()

def download_pdf(url, filename):
    """Downloads a PDF with a progress bar and handles filename conflicts."""
    # Resolve filename conflicts
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(DEFAULT_DOWNLOAD_DIR, new_filename)):
        new_filename = f"{base} ({counter}){extension}"
        counter += 1
    
    filepath = os.path.join(DEFAULT_DOWNLOAD_DIR, new_filename)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=f"Downloading {new_filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False # Remove bar after completion
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                bar.update(size)
                
        logging.info(f"Downloaded: {new_filename}")
        # A small confirmation can be nice
        print(f"Downloaded: {new_filename}")
    except requests.exceptions.RequestException as e:
        log_error(f"Failed to download {new_filename}: {e}")

def is_valid_url(url, base_domain):
    """Checks if a URL is valid for crawling."""
    if url.lower().endswith(IGNORED_EXTENSIONS) or url.startswith(('mailto:', 'javascript:')):
        return False
    
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ('http', 'https'):
        return False
        
    return urlparse(base_domain).netloc == parsed_url.netloc

def scan_and_download_pdfs(target_url, recursive, max_depth, wait_time, visited_urls):
    """Scans a website for PDFs and downloads them, with progress tracking."""
    url_queue = deque([(target_url, 0)])
    pbar = tqdm(total=len(url_queue), desc="Scanning Pages", unit="page")

    base_domain = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"

    try:
        while url_queue:
            current_url, depth = url_queue.popleft()

            if current_url in visited_urls or (max_depth is not None and depth > max_depth):
                pbar.update(1)
                continue

            logging.info(f"Scanning: {current_url}")
            visited_urls.add(current_url)

            page_source = get_page_source_with_selenium(current_url, wait_time)
            if not page_source:
                pbar.update(1)
                continue

            soup = BeautifulSoup(page_source, 'lxml')
            new_links = 0
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href).split('#')[0] # Join and remove fragments

                if full_url not in visited_urls and full_url not in [item[0] for item in url_queue]:
                    if href.lower().endswith('.pdf'):
                        pdf_filename = os.path.basename(urlparse(full_url).path)
                        if pdf_filename:
                            download_pdf(full_url, pdf_filename)
                    elif recursive and is_valid_url(full_url, base_domain):
                        url_queue.append((full_url, depth + 1))
                        new_links += 1
            
            pbar.total = len(visited_urls) + len(url_queue)
            pbar.update(1)
            pbar.set_postfix(found=new_links)

    except KeyboardInterrupt:
        print("\nScan interrupted by user. Saving progress...")
    finally:
        pbar.close()
        with open(SESSION_FILE, 'w') as f:
            json.dump(list(visited_urls), f)
        print(f"Scan progress saved to {SESSION_FILE}")

def main():
    """Main function to drive the PDF downloader."""
    create_download_directory()
    visited_urls = set()

    if os.path.exists(SESSION_FILE):
        if input(f"Found {SESSION_FILE}. Resume previous scan? (y/n): ").strip().lower() == 'y':
            with open(SESSION_FILE, 'r') as f:
                visited_urls = set(json.load(f))
            print(f"Resuming with {len(visited_urls)} previously visited URLs.")

    target_url = input("Enter the target URL: ").strip()
    recursive = input("Recursively search domain? (y/n): ").strip().lower() == 'y'
    
    max_depth = None
    if recursive:
        try:
            depth_input = input("Enter max recursion depth (e.g., 2, press Enter for no limit): ").strip()
            if depth_input: max_depth = int(depth_input)
        except ValueError:
            log_error("Invalid depth. Continuing without limit.")
            
    try:
        wait_input = input("Enter wait time for JS loading in seconds (default: 5): ").strip()
        wait_time = int(wait_input) if wait_input else 5
    except ValueError:
        log_error("Invalid wait time. Using default of 5 seconds.")
        wait_time = 5
        
    scan_and_download_pdfs(target_url, recursive, max_depth, wait_time, visited_urls)
    print("\nScanning and downloading complete.")

if __name__ == "__main__":
    main()
