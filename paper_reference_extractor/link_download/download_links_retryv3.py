import json
import requests
import os
import urllib.parse
import signal
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup  # For parsing HTML and extracting links

# Global flag for graceful shutdown
shutdown_flag = False

# List of User-Agent strings to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
]

# List of proxies (if available)
PROXIES = [
    # Add your proxy URLs here, e.g., "http://user:pass@proxy1:port", "http://user:pass@proxy2:port"
]

def signal_handler(sig, frame):
    """Handles interrupt signals for graceful shutdown."""
    global shutdown_flag
    print("\nGracefully shutting down...")
    shutdown_flag = True
    sys.exit(0)

def find_default_json_file():
    """Searches the current directory for JSON files and suggests the first one found."""
    default_directory = os.getcwd()
    for file in os.listdir(default_directory):
        if file.endswith(".json"):
            return os.path.join(default_directory, file)
    return None

def prompt_for_json_file():
    """Prompts the user for the JSON file path, suggesting a default if found."""
    default_json_file = find_default_json_file()
    if default_json_file:
        user_input = input(f"Enter the path to the JSON file (default: {default_json_file}): ").strip()
        if not user_input:
            return default_json_file
        return user_input
    else:
        return input("Enter the path to the JSON file: ").strip()

def prompt_for_file_type():
    """Prompts the user for the file type to download, defaulting to PDF."""
    file_type = input("Enter the file type to download (e.g., pdf, html, etc.) [default: pdf]: ").strip().lower()
    return file_type if file_type else "pdf"

def get_random_user_agent():
    """Returns a random User-Agent string."""
    return random.choice(USER_AGENTS)

def get_random_proxy():
    """Returns a random proxy from the list (if available)."""
    return random.choice(PROXIES) if PROXIES else None

def extract_links_from_html(html_content, base_url):
    """Extracts all links from an HTML page."""
    soup = BeautifulSoup(html_content, "html.parser")
    links = set()
    for tag in soup.find_all("a", href=True):
        link = urllib.parse.urljoin(base_url, tag["href"])
        links.add(link)
    return links

def download_file(session, url, download_directory, max_attempts=3):
    """Downloads a single file with retries."""
    parsed_url = urllib.parse.urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        print(f"Skipping invalid URL: {url}")
        return None

    filename = os.path.join(download_directory, os.path.basename(parsed_url.path) or "downloaded_file")
    if not os.path.splitext(filename)[1]:
        filename = filename + ".html"

    for attempt in range(max_attempts):
        if shutdown_flag:
            return None
        try:
            # Add a random delay between requests
            time.sleep(random.uniform(1, 3))  # Random delay between 1 and 3 seconds

            # Rotate User-Agent and proxy
            headers = {"User-Agent": get_random_user_agent()}
            proxies = {"http": get_random_proxy(), "https": get_random_proxy()} if get_random_proxy() else None

            response = session.get(url, headers=headers, proxies=proxies, stream=True, timeout=10)
            response.raise_for_status()
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded: {url} to {filename}")
            return filename
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    print(f"Failed to download {url} after {max_attempts} attempts.")
    return None

def crawl_and_download(session, start_url, download_directory, file_type, max_depth=3):
    """Crawls from the start URL to a specified depth and downloads files of the specified type."""
    visited_urls = set()
    urls_to_crawl = [(start_url, 0)]  # (url, current_depth)

    while urls_to_crawl and not shutdown_flag:
        current_url, current_depth = urls_to_crawl.pop(0)
        if current_url in visited_urls or current_depth > max_depth:
            continue

        visited_urls.add(current_url)
        print(f"Crawling: {current_url} (Depth: {current_depth})")

        try:
            # Add a random delay between requests
            time.sleep(random.uniform(1, 3))  # Random delay between 1 and 3 seconds

            # Rotate User-Agent and proxy
            headers = {"User-Agent": get_random_user_agent()}
            proxies = {"http": get_random_proxy(), "https": get_random_proxy()} if get_random_proxy() else None

            response = session.get(current_url, headers=headers, proxies=proxies, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error crawling {current_url}: {e}")
            continue

        # If the response is HTML, extract links and add them to the queue
        if "text/html" in response.headers.get("Content-Type", ""):
            links = extract_links_from_html(response.content, current_url)
            for link in links:
                if link not in visited_urls:
                    urls_to_crawl.append((link, current_depth + 1))

        # If the response is a file of the specified type, download it
        elif file_type in response.headers.get("Content-Type", "") or current_url.endswith(f".{file_type}"):
            download_file(session, current_url, download_directory)

def download_links_from_json(json_file, file_type, max_workers=5, crawl_depth=3):
    """Downloads links from a JSON file with optional crawling."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Error: JSON file does not contain a list of dictionaries.")
            return

        download_directory = "downloaded_files"
        os.makedirs(download_directory, exist_ok=True)

        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in data:
                if 'urls' in item and isinstance(item['urls'], list):
                    for url in item['urls']:
                        if shutdown_flag:
                            break
                        # Start crawling from each URL
                        futures.append(executor.submit(crawl_and_download, session, url, download_directory, file_type, crawl_depth))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

        print("Download process completed.")

    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Prompt for JSON file with default suggestion
    json_file = prompt_for_json_file()

    # Prompt for file type to download, defaulting to PDF
    file_type = prompt_for_file_type()

    # Start the download process
    download_links_from_json(json_file, file_type, crawl_depth=4)  # Set crawl depth here