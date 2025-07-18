import os
import re
import time
import random
import logging
import requests
from urllib.parse import urljoin, urlparse, unquote
from collections import deque
from typing import Optional, Tuple
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DEFAULT_DOWNLOAD_DIR = "downloaded_pdfs"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
REQUEST_DELAY = 1.0  # seconds
REQUEST_JITTER = 0.5  # seconds
RENDER_TIMEOUT = 10  # seconds

# Configure logging
logging.basicConfig(
    filename='pdf_downloader.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SecurePDFDownloader:
    def __init__(self):
        self.download_dir = DEFAULT_DOWNLOAD_DIR
        self.visited_urls = set()
        self.session = requests.Session()
        self.create_download_directory()
        self.driver = self._init_selenium()

    def _init_selenium(self) -> webdriver.Chrome:
        """Initialize Selenium WebDriver with configured options"""
        service = Service()
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-blink-features=AutomationControlled")
        return webdriver.Chrome(service=service, options=options)

    def create_download_directory(self) -> None:
        """Create download directory with proper permissions"""
        os.makedirs(self.download_dir, exist_ok=True)
        logging.info(f"Download directory: {os.path.abspath(self.download_dir)}")

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove dangerous characters"""
        filename = unquote(filename)
        filename = re.sub(r'[\\/*?:"<>|\x00-\x1f]', '_', filename)
        filename = filename.strip('. ')
        return filename[:255]  # Limit filename length

    def get_safe_filename(self, url: str, headers: Optional[dict] = None) -> str:
        """Generate a safe filename from URL or Content-Disposition header"""
        if headers and 'content-disposition' in headers:
            match = re.search(
                r'filename\*?=["\']?(?:UTF-\d[\'"]*)?([^"\'\s;]+)',
                headers['content-disposition'],
                re.IGNORECASE
            )
            if match:
                return self.sanitize_filename(match.group(1))

        path = urlparse(url).path
        filename = os.path.basename(path) or 'document.pdf'
        return self.sanitize_filename(filename)

    def is_pdf_link(self, url: str) -> bool:
        """Check if URL points to a PDF using multiple methods"""
        try:
            # Fast check for common PDF extensions
            if url.lower().endswith(('.pdf', '.pdf?')):
                return True

            # Verify with HEAD request
            response = self.session.head(
                url,
                allow_redirects=True,
                timeout=10,
                headers={'Accept': 'application/pdf'}
            )
            
            content_type = response.headers.get('content-type', '').lower()
            return (
                response.status_code == 200 and
                'application/pdf' in content_type
            )
        except (requests.RequestException, ValueError):
            return False

    def get_rendered_page(self, url: str) -> Optional[str]:
        """Get fully rendered page source with smart waiting"""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, RENDER_TIMEOUT).until(
                lambda d: d.find_elements(By.XPATH, '//a[contains(@href, ".pdf")]') or
                         d.execute_script('return document.readyState') == 'complete'
            )
            return self.driver.page_source
        except TimeoutException:
            logging.warning(f"Timeout rendering page: {url}")
            return None
        except Exception as e:
            logging.error(f"Error rendering page {url}: {str(e)}")
            return None

    def download_pdf(self, url: str) -> Tuple[bool, str]:
        """Safely download PDF with size limits and proper filename handling"""
        try:
            # Polite delay between requests
            time.sleep(REQUEST_DELAY + random.uniform(0, REQUEST_JITTER))

            with self.session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()

                # Verify content type
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' not in content_type:
                    return False, "Not a PDF file"

                # Check file size
                size = int(response.headers.get('content-length', 0))
                if size > MAX_FILE_SIZE:
                    return False, f"File too large ({size} bytes)"

                # Generate safe filename
                filename = self.get_safe_filename(url, response.headers)
                filepath = os.path.join(self.download_dir, filename)

                # Ensure we don't escape download directory
                if not os.path.abspath(filepath).startswith(os.path.abspath(self.download_dir)):
                    return False, "Path traversal attempt detected"

                # Save file with progress
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"Downloaded: {filename}")
                return True, filename

        except requests.RequestException as e:
            return False, f"Download failed: {str(e)}"

    def is_same_domain(self, base_url: str, target_url: str) -> bool:
        """Check if URLs belong to the same domain"""
        base_domain = urlparse(base_url).netloc.lower()
        target_domain = urlparse(target_url).netloc.lower()
        return base_domain == target_domain

    def scan_and_download(self, start_url: str, recursive: bool = False, max_depth: Optional[int] = None) -> None:
        """Main scanning and downloading logic"""
        queue = deque([(start_url, 0)])
        download_count = 0

        while queue:
            current_url, depth = queue.popleft()

            if current_url in self.visited_urls:
                continue
            self.visited_urls.add(current_url)

            if max_depth is not None and depth > max_depth:
                continue

            logging.info(f"Scanning: {current_url} (depth {depth})")
            print(f"\nScanning: {current_url}")

            # Get rendered page content
            page_source = self.get_rendered_page(current_url)
            if not page_source:
                continue

            # Parse HTML and find links
            soup = BeautifulSoup(page_source, 'lxml')
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href or href.startswith(('mailto:', 'tel:', 'javascript:')):
                    continue

                full_url = urljoin(current_url, href)

                # Handle PDF links
                if self.is_pdf_link(full_url):
                    success, message = self.download_pdf(full_url)
                    if success:
                        download_count += 1
                        print(f"✓ Downloaded: {message}")
                    else:
                        print(f"✗ Failed: {message}")
                # Queue internal links for recursive scanning
                elif recursive and self.is_same_domain(start_url, full_url):
                    queue.append((full_url, depth + 1))

        print(f"\nDownload complete. Total PDFs downloaded: {download_count}")

    def cleanup(self) -> None:
        """Clean up resources"""
        self.driver.quit()
        self.session.close()

def main():
    """Main entry point"""
    print("=== PDF Downloader ===")
    target_url = input("Enter target URL: ").strip()
    recursive = input("Recursive search? (y/n): ").lower() == 'y'
    max_depth = None

    if recursive:
        depth_input = input("Max depth (Enter for unlimited): ").strip()
        if depth_input.isdigit():
            max_depth = int(depth_input)

    downloader = SecurePDFDownloader()
    try:
        downloader.scan_and_download(target_url, recursive, max_depth)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        print(f"\nError: {str(e)}")
    finally:
        downloader.cleanup()

if __name__ == "__main__":
    main()
