#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import os
import sys
import random
import logging
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
from typing import Tuple, Optional, Set
from dataclasses import dataclass

# --- Configuration ---

# List of User-Agent strings to rotate to mimic different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Maximum content size to download (10MB)
MAX_CONTENT_SIZE = 10 * 1024 * 1024

# Status codes that should be retried
RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530}

# Status codes that indicate permanent failures (don't retry)
PERMANENT_FAILURE_CODES = {400, 404, 405, 406, 409, 410, 411, 412, 413, 414, 415, 416, 417, 422, 451}

# Common PDF MIME types
PDF_MIME_TYPES = {
    'application/pdf',
    'application/x-pdf',
    'application/acrobat',
    'applications/vnd.pdf',
    'text/pdf',
    'text/x-pdf'
}

# File extensions that might contain PDFs or links to PDFs
CRAWLABLE_EXTENSIONS = {'.html', '.htm', '.php', '.asp', '.aspx', '.jsp', '.cfm', ''}

# Common words that might indicate PDF content
PDF_INDICATORS = {
    'pdf', 'document', 'paper', 'report', 'manual', 'guide', 'whitepaper', 
    'brochure', 'catalog', 'datasheet', 'specification', 'download'
}

@dataclass
class CrawlStats:
    """Statistics for the crawling process"""
    urls_processed: int = 0
    pdfs_downloaded: int = 0
    pdfs_found: int = 0
    pages_crawled: int = 0
    errors: int = 0

@dataclass
class FetchResult:
    """Result of fetching a URL"""
    success: bool
    content: Optional[bytes] = None
    content_type: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None

# --- Setup Logging ---

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('crawler.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- Helper Functions ---

def find_default_json_file():
    """Tries to find a .json file in the current directory to suggest as a default."""
    for file in os.listdir("."):
        if file.endswith(".json"):
            return file
    return None

def is_valid_url(url: str) -> bool:
    """Basic check to see if a URL is valid and not a local file path or javascript link."""
    try:
        parsed = urlparse(url)
        # Check for a valid scheme (http/https) and a network location (domain).
        return all([parsed.scheme, parsed.netloc]) and parsed.scheme in ["http", "https"]
    except (ValueError, AttributeError):
        return False

def is_pdf_url(url: str) -> bool:
    """Checks if a URL likely points to a PDF file based on URL path."""
    try:
        path = urlparse(url).path.lower()
        return path.endswith('.pdf') or '/pdf/' in path
    except (ValueError, AttributeError):
        return False

def should_crawl_url(url: str, base_domains: Set[str], max_depth: int, current_depth: int) -> bool:
    """Determines if a URL should be crawled based on domain and depth limits."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check if it's within allowed domains (if domain filtering is enabled)
        if base_domains and not any(domain.endswith(base_domain) for base_domain in base_domains):
            return False
        
        # Check depth limit
        if current_depth >= max_depth:
            return False
        
        # Check if extension is crawlable
        path = parsed.path.lower()
        extension = os.path.splitext(path)[1]
        
        return extension in CRAWLABLE_EXTENSIONS
        
    except (ValueError, AttributeError):
        return False

def extract_pdf_links(soup: BeautifulSoup, base_url: str) -> Set[str]:
    """Extract all potential PDF links from HTML content."""
    pdf_links = set()
    
    # Look for direct PDF links in various attributes
    selectors_and_attrs = [
        ('a[href]', 'href'),
        ('link[href]', 'href'),
        ('iframe[src]', 'src'),
        ('embed[src]', 'src'),
        ('object[data]', 'data'),
        ('form[action]', 'action'),
    ]
    
    for selector, attr in selectors_and_attrs:
        for element in soup.select(selector):
            url = element.get(attr)
            if url:
                abs_url = urljoin(base_url, url)
                if is_valid_url(abs_url) and (is_pdf_url(abs_url) or might_be_pdf_link(element, url)):
                    pdf_links.add(abs_url)
    
    # Look for PDF links in JavaScript (basic pattern matching)
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string:
            # Look for URLs ending in .pdf in JavaScript
            pdf_matches = re.findall(r'["\']([^"\']*\.pdf[^"\']*)["\']', script.string, re.IGNORECASE)
            for match in pdf_matches:
                abs_url = urljoin(base_url, match)
                if is_valid_url(abs_url):
                    pdf_links.add(abs_url)
    
    return pdf_links

def extract_crawlable_links(soup: BeautifulSoup, base_url: str) -> Set[str]:
    """Extract links that should be crawled for more content."""
    crawlable_links = set()
    
    # Look for regular navigation links
    for link in soup.find_all('a', href=True):
        url = urljoin(base_url, link['href'])
        if is_valid_url(url) and not is_pdf_url(url):
            # Filter out obvious non-content links
            href = link['href'].lower()
            if not any(skip in href for skip in ['mailto:', 'tel:', 'javascript:', '#', 'logout', 'login']):
                crawlable_links.add(url)
    
    return crawlable_links

def might_be_pdf_link(element, url: str) -> bool:
    """Check if a link element might lead to a PDF based on context clues."""
    url_lower = url.lower()
    
    # Check URL for PDF indicators
    if any(indicator in url_lower for indicator in PDF_INDICATORS):
        return True
    
    # Check link text and surrounding context
    text_content = element.get_text().lower() if hasattr(element, 'get_text') else ''
    
    # Check various attributes for PDF indicators
    attrs_to_check = ['title', 'alt', 'class', 'id']
    for attr in attrs_to_check:
        attr_value = element.get(attr, '').lower() if hasattr(element, 'get') else ''
        if any(indicator in attr_value for indicator in PDF_INDICATORS):
            return True
    
    # Check if link text suggests PDF content
    if any(indicator in text_content for indicator in PDF_INDICATORS):
        return True
    
    # Check for download-related attributes
    if hasattr(element, 'get') and element.get('download'):
        return True
    
    return False

def is_pdf_content(content_type: str) -> bool:
    """Checks if content type indicates PDF content."""
    if not content_type:
        return False
    return any(pdf_type in content_type.lower() for pdf_type in PDF_MIME_TYPES)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe for filesystem."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length and ensure it's not empty
    filename = filename[:255]
    if not filename or filename.startswith('.'):
        filename = f"document_{int(datetime.now().timestamp())}.pdf"
    
    return filename

def get_retry_delay(retry_after_header: Optional[str], attempt: int) -> float:
    """Calculate retry delay, respecting Retry-After header if present."""
    if retry_after_header:
        try:
            return float(retry_after_header)
        except ValueError:
            pass
    
    # Exponential backoff with jitter
    base_delay = min(2 ** attempt, 60)  # Cap at 60 seconds
    jitter = random.uniform(0.1, 0.5)
    return base_delay + jitter

# --- Core Asynchronous Functions ---

async def fetch_url(session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> FetchResult:
    """
    Fetches the content of a single URL with intelligent retry logic.
    Returns a FetchResult object with detailed information about the request.
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    for attempt in range(max_retries):
        try:
            async with session.get(
                url, 
                headers=headers, 
                timeout=aiohttp.ClientTimeout(total=30, connect=10),
                allow_redirects=True
            ) as response:
                
                status = response.status
                content_type = response.headers.get("Content-Type", "")
                
                logger.debug(f"[{status}] Fetched: {url}")
                
                if status == 200:
                    # Check content size before reading
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > MAX_CONTENT_SIZE:
                        logger.warning(f"Content too large ({content_length} bytes) for {url}")
                        return FetchResult(
                            success=False, 
                            status_code=status, 
                            error_message=f"Content too large: {content_length} bytes"
                        )
                    
                    content = await response.read()
                    
                    # Double-check actual content size
                    if len(content) > MAX_CONTENT_SIZE:
                        logger.warning(f"Downloaded content too large ({len(content)} bytes) for {url}")
                        return FetchResult(
                            success=False, 
                            status_code=status, 
                            error_message=f"Downloaded content too large: {len(content)} bytes"
                        )
                    
                    return FetchResult(
                        success=True,
                        content=content,
                        content_type=content_type,
                        status_code=status
                    )
                
                elif status in PERMANENT_FAILURE_CODES:
                    logger.debug(f"Permanent failure ({status}) for {url}")
                    return FetchResult(
                        success=False,
                        status_code=status,
                        error_message=f"Permanent failure: HTTP {status}"
                    )
                
                elif status == 403:
                    if attempt == 0:
                        logger.debug(f"Forbidden ({status}) for {url}. Trying different User-Agent...")
                        headers["User-Agent"] = random.choice(USER_AGENTS)
                        await asyncio.sleep(get_retry_delay(None, attempt))
                        continue
                    else:
                        logger.debug(f"Persistent forbidden ({status}) for {url}")
                        return FetchResult(
                            success=False,
                            status_code=status,
                            error_message=f"Forbidden: HTTP {status}"
                        )
                
                elif status == 401:
                    if attempt == 0:
                        logger.debug(f"Unauthorized ({status}) for {url}. Retrying...")
                        await asyncio.sleep(get_retry_delay(None, attempt))
                        continue
                    else:
                        logger.debug(f"Persistent unauthorized ({status}) for {url}")
                        return FetchResult(
                            success=False,
                            status_code=status,
                            error_message=f"Unauthorized: HTTP {status}"
                        )
                
                elif status == 429:
                    retry_after = response.headers.get('Retry-After')
                    delay = get_retry_delay(retry_after, attempt)
                    logger.warning(f"Rate limited ({status}) for {url}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                
                elif status in RETRYABLE_STATUS_CODES:
                    delay = get_retry_delay(None, attempt)
                    logger.debug(f"Server error ({status}) for {url}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                
                else:
                    logger.debug(f"Unexpected status ({status}) for {url}")
                    return FetchResult(
                        success=False,
                        status_code=status,
                        error_message=f"Unexpected status: HTTP {status}"
                    )
                    
        except asyncio.TimeoutError:
            delay = get_retry_delay(None, attempt)
            logger.debug(f"Timeout for {url}. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            continue
            
        except aiohttp.ClientError as e:
            delay = get_retry_delay(None, attempt)
            logger.debug(f"Client error for {url}: {e}. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return FetchResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    # All retries exhausted
    logger.debug(f"All retries exhausted for {url}")
    return FetchResult(
        success=False,
        error_message="All retries exhausted"
    )

async def download_pdf(session: aiohttp.ClientSession, url: str, save_dir: str, stats: CrawlStats) -> bool:
    """Downloads a PDF from a URL and saves it to the specified directory."""
    logger.info(f"Downloading PDF: {url}")
    
    result = await fetch_url(session, url)
    
    if not result.success:
        logger.error(f"Failed to download PDF from {url}: {result.error_message}")
        stats.errors += 1
        return False
    
    # Verify it's actually a PDF
    if not (is_pdf_url(url) or is_pdf_content(result.content_type)):
        # Check if content starts with PDF signature
        if not (result.content and result.content.startswith(b'%PDF')):
            logger.warning(f"URL {url} doesn't appear to be a PDF (Content-Type: {result.content_type})")
            return False
    
    try:
        # Generate filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename or not filename.lower().endswith('.pdf'):
            filename = f"document_{int(datetime.now().timestamp())}.pdf"
        
        filename = sanitize_filename(filename)
        save_path = os.path.join(save_dir, filename)
        
        # Check if file already exists
        counter = 1
        original_save_path = save_path
        while os.path.exists(save_path):
            name, ext = os.path.splitext(original_save_path)
            save_path = f"{name}_{counter}{ext}"
            counter += 1
        
        with open(save_path, "wb") as f:
            f.write(result.content)
        
        logger.info(f"Successfully downloaded '{os.path.basename(save_path)}' ({len(result.content)} bytes)")
        stats.pdfs_downloaded += 1
        return True
        
    except OSError as e:
        logger.error(f"Could not save PDF file for {url}: {e}")
        stats.errors += 1
        return False

async def crawl_worker(
    url_queue: asyncio.Queue,
    pdf_queue: asyncio.Queue, 
    session: aiohttp.ClientSession, 
    visited_urls: Set[str], 
    save_dir: str,
    stats: CrawlStats,
    base_domains: Set[str],
    max_depth: int
):
    """A worker task that crawls URLs to find PDFs and more crawlable content."""
    while True:
        try:
            url, depth = await url_queue.get()

            if url in visited_urls:
                url_queue.task_done()
                continue
            
            visited_urls.add(url)
            stats.urls_processed += 1

            # Print progress periodically
            if stats.urls_processed % 50 == 0:
                logger.info(f"Progress: {stats.urls_processed} URLs processed, {stats.pdfs_found} PDFs found, {stats.pdfs_downloaded} downloaded")

            result = await fetch_url(session, url)
            
            if not result.success:
                stats.errors += 1
                url_queue.task_done()
                continue
            
            # Check if this is a PDF
            if is_pdf_content(result.content_type) or (result.content and result.content.startswith(b'%PDF')):
                await pdf_queue.put(url)
                stats.pdfs_found += 1
                url_queue.task_done()
                continue
            
            # Parse HTML content to find PDFs and more pages to crawl
            if result.content_type and "text/html" in result.content_type:
                try:
                    soup = BeautifulSoup(result.content, "html.parser")
                    stats.pages_crawled += 1
                    
                    # Extract PDF links
                    pdf_links = extract_pdf_links(soup, url)
                    for pdf_link in pdf_links:
                        if pdf_link not in visited_urls:
                            await pdf_queue.put(pdf_link)
                            stats.pdfs_found += 1
                    
                    # Extract more pages to crawl (if not at max depth)
                    if depth < max_depth:
                        crawlable_links = extract_crawlable_links(soup, url)
                        for link in crawlable_links:
                            if (link not in visited_urls and 
                                should_crawl_url(link, base_domains, max_depth, depth + 1)):
                                await url_queue.put((link, depth + 1))
                    
                    if pdf_links:
                        logger.info(f"Found {len(pdf_links)} PDF links on {url}")
                        
                except Exception as e:
                    logger.error(f"Error parsing HTML from {url}: {e}")
                    stats.errors += 1
            
            url_queue.task_done()
            
            # Be polite to servers
            await asyncio.sleep(random.uniform(0.5, 1.5))

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Unhandled exception in crawl worker for {url}: {e}")
            stats.errors += 1
            url_queue.task_done()

async def pdf_download_worker(
    pdf_queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    save_dir: str,
    stats: CrawlStats,
    downloaded_pdfs: Set[str]
):
    """Worker dedicated to downloading PDFs."""
    while True:
        try:
            url = await pdf_queue.get()
            
            if url not in downloaded_pdfs:
                downloaded_pdfs.add(url)
                await download_pdf(session, url, save_dir, stats)
            
            pdf_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Unhandled exception in PDF download worker: {e}")
            stats.errors += 1
            pdf_queue.task_done()

# --- Main Function ---

async def main():
    """Main function to set up and run the crawler."""
    print("Advanced PDF Crawler")
    print("=" * 40)
    
    # --- User Prompts ---
    default_file = find_default_json_file()
    prompt_msg = f"Enter the JSON file path (default: {default_file}): " if default_file else "Enter the JSON file path: "
    json_file = input(prompt_msg).strip() or default_file
    
    if not json_file or not os.path.exists(json_file):
        print(f"ERROR: File not found at '{json_file}'")
        sys.exit(1)

    num_workers_str = input("Enter number of crawling workers (default: 5): ").strip()
    num_workers = int(num_workers_str) if num_workers_str.isdigit() else 5
    
    pdf_workers_str = input("Enter number of PDF download workers (default: 3): ").strip()
    pdf_workers = int(pdf_workers_str) if pdf_workers_str.isdigit() else 3

    max_depth_str = input("Enter maximum crawl depth (default: 3): ").strip()
    max_depth = int(max_depth_str) if max_depth_str.isdigit() else 3

    stay_on_domain = input("Stay on same domains? (y/n, default: y): ").strip().lower()
    stay_on_domain = stay_on_domain != 'n'

    save_dir = input("Enter save directory (default: pdf_downloads): ").strip() or "pdf_downloads"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"INFO: PDFs will be saved in the '{save_dir}/' directory.")
    print(f"INFO: Maximum crawl depth: {max_depth}")
    print(f"INFO: Domain restriction: {'Yes' if stay_on_domain else 'No'}")
    print(f"INFO: Log file will be created as 'crawler.log'")

    # --- Load URLs from JSON with Error Recovery ---
    data = []
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)
            logger.info(f"Successfully loaded {len(data)} entries from {json_file}")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON file '{json_file}' is malformed. Attempting recovery...")
        try:
            content_up_to_error = content[:e.pos]
            last_separator_pos = content_up_to_error.rfind('},\n')
            
            if last_separator_pos != -1:
                end_of_valid_json = last_separator_pos + 1
                salvaged_content = content_up_to_error[:end_of_valid_json] + '\n]'
                data = json.loads(salvaged_content)
                logger.info(f"Recovered {len(data)} entries from corrupted file.")
            else:
                logger.error("Could not find valid object separator to recover from.")
                sys.exit(1)
                
        except Exception as recovery_e:
            logger.error(f"Recovery attempt failed: {recovery_e}")
            sys.exit(1)
    except IOError as e:
        logger.error(f"Could not read file '{json_file}': {e}")
        sys.exit(1)

    # Extract URLs and determine base domains
    initial_urls = set()
    base_domains = set()
    
    for item in data:
        if isinstance(item.get("urls"), list):
            for url in item["urls"]:
                if is_valid_url(url):
                    initial_urls.add(url)
                    if stay_on_domain:
                        domain = urlparse(url).netloc.lower()
                        base_domains.add(domain)

    if not initial_urls:
        logger.warning("No valid URLs found in the JSON file.")
        return

    logger.info(f"Found {len(initial_urls)} valid URLs to start crawling")
    if stay_on_domain:
        logger.info(f"Will stay on domains: {', '.join(sorted(base_domains))}")

    # --- Setup and Run Crawler ---
    url_queue = asyncio.Queue()
    pdf_queue = asyncio.Queue()
    
    # Add initial URLs with depth 0
    for url in initial_urls:
        await url_queue.put((url, 0))
    
    visited_urls = set()
    downloaded_pdfs = set()
    stats = CrawlStats()
    
    # Configure connection limits
    connector = aiohttp.TCPConnector(
        limit_per_host=2,  # Very conservative to be polite
        limit=50,
        ssl=False,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    ) as session:
        
        logger.info(f"Starting crawler with {num_workers} crawl workers and {pdf_workers} download workers...")
        
        # Create crawling workers
        crawl_workers = [
            asyncio.create_task(crawl_worker(
                url_queue, pdf_queue, session, visited_urls, save_dir, 
                stats, base_domains, max_depth
            ))
            for _ in range(num_workers)
        ]
        
        # Create PDF download workers
        download_workers = [
            asyncio.create_task(pdf_download_worker(
                pdf_queue, session, save_dir, stats, downloaded_pdfs
            ))
            for _ in range(pdf_workers)
        ]

        try:
            await url_queue.join()
            logger.info("All URLs have been crawled. Finishing PDF downloads...")
            await pdf_queue.join()
            logger.info("All PDFs have been processed.")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down gracefully...")
        finally:
            # Cancel all workers
            for worker in crawl_workers + download_workers:
                worker.cancel()
            
            # Wait for workers to finish
            await asyncio.gather(*crawl_workers, *download_workers, return_exceptions=True)
            
            logger.info(f"Crawler finished. Final stats:")
            logger.info(f"- URLs processed: {stats.urls_processed}")
            logger.info(f"- Pages crawled: {stats.pages_crawled}")
            logger.info(f"- PDFs found: {stats.pdfs_found}")
            logger.info(f"- PDFs downloaded: {stats.pdfs_downloaded}")
            logger.info(f"- Errors: {stats.errors}")
            
            print(f"\n🏁 Crawler finished!")
            print(f"📊 Final Statistics:")
            print(f"   • URLs processed: {stats.urls_processed}")
            print(f"   • Pages crawled: {stats.pages_crawled}")
            print(f"   • PDFs found: {stats.pdfs_found}")
            print(f"   • PDFs downloaded: {stats.pdfs_downloaded}")
            print(f"   • Errors encountered: {stats.errors}")
            print(f"📁 Check '{save_dir}/' for downloaded PDFs")
            print(f"📋 Check 'crawler.log' for detailed logs")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
