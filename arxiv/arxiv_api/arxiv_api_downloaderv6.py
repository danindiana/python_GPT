import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
import re
import json
import signal
import sys
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# --- Configuration ---
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Python-arXiv-Downloader/2.2"
BASE_DIR = Path(os.getcwd()).absolute()
DOWNLOAD_DIR = BASE_DIR / "arxiv_downloads"
STATE_FILE = DOWNLOAD_DIR / ".arxiv_downloader_state.json"
LOG_FILE = DOWNLOAD_DIR / "arxiv_downloader.log"

RATE_LIMIT_SECONDS = 3       # Delay between every PDF attempt (success or failure)
API_RATE_SECONDS = 1         # Minimum delay between API calls
API_CHUNK_SIZE = 100
NETWORK_TIMEOUT = 30
SIGNAL_CHECK_INTERVAL = 0.2
MIN_PDF_BYTES = 1_000        # Files smaller than this are considered corrupt
MAX_RETRIES = 3              # Per-file download retries

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")

class Logger:
    """Writes to stdout and to a persistent log file simultaneously."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "a", buffering=1)

    def _write(self, line: str):
        print(line)
        self._fh.write(line + "\n")

    def info(self, msg: str):  self._write(f"[{_ts()}] [INFO ] {msg}")
    def ok(self, msg: str):    self._write(f"[{_ts()}] [OK   ] {msg}")
    def warn(self, msg: str):  self._write(f"[{_ts()}] [WARN ] {msg}")
    def error(self, msg: str): self._write(f"[{_ts()}] [ERROR] {msg}")
    def debug(self, msg: str): self._write(f"[{_ts()}] [DEBUG] {msg}")

    def close(self):
        self._fh.close()

log: Logger  # initialised in ArxivDownloader.__init__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_socket_timeout(timeout: int = NETWORK_TIMEOUT):
    socket.setdefaulttimeout(timeout)

def clean_filename(name: str) -> str:
    name = re.sub(r'\s+', ' ', name)
    name = name.replace('$', '').replace('{', '').replace('}', '').replace('\\', '')
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.strip()[:150]

def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def extract_query(user_input: str) -> str:
    user_input = user_input.strip()
    if user_input.startswith("http"):
        try:
            parsed = urllib.parse.urlparse(user_input)
            qs = urllib.parse.parse_qs(parsed.query)
            if "query" in qs:        return qs["query"][0]
            elif "search_query" in qs: return qs["search_query"][0]
        except Exception:
            pass
    return user_input


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

class ControlFlags:
    def __init__(self):
        self.shutdown_requested = False
        self.suspend_requested = False

    def _handle_shutdown(self, signum, frame):
        if self.shutdown_requested:
            log.warn("Forced exit.")
            sys.exit(1)
        log.warn(f"Signal {signum} received — finishing current file then saving state. Ctrl-C again to force-quit.")
        self.shutdown_requested = True

    def _handle_suspend(self, signum, frame):
        log.warn("SIGTSTP received — checkpointing before suspend.")
        self.suspend_requested = True

    def install(self):
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        if hasattr(signal, "SIGTSTP"):
            signal.signal(signal.SIGTSTP, self._handle_suspend)


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class ArxivDownloader:
    def __init__(self):
        global log
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        log = Logger(LOG_FILE)
        self.flags = ControlFlags()
        self.ns = {
            "atom":       "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        }
        set_socket_timeout(NETWORK_TIMEOUT)

    # ------------------------------------------------------------------
    # Rate-limited sleep (interruptible)
    # ------------------------------------------------------------------

    def _rate_sleep(self, seconds: float, label: str = ""):
        msg = f"Waiting {seconds}s" + (f" ({label})" if label else "")
        log.debug(msg)
        end = time.time() + seconds
        while time.time() < end:
            if self.flags.shutdown_requested or self.flags.suspend_requested:
                return
            time.sleep(SIGNAL_CHECK_INTERVAL)

    # ------------------------------------------------------------------
    # PDF download with retries and full telemetry
    # ------------------------------------------------------------------

    def _download_pdf(self, url: str, dest_path: str) -> bool:
        """
        Download a PDF to dest_path.  Uses a .tmp sidecar to ensure the
        destination is only written if the download is fully verified.
        Returns True on success, False on any failure.
        """
        temp_path = dest_path + ".tmp"
        dest = Path(dest_path)

        # Clean up any leftover temp file from a previous crash
        if os.path.exists(temp_path):
            log.debug(f"Removing stale temp file: {temp_path}")
            try:
                os.remove(temp_path)
            except OSError as e:
                log.error(f"Cannot remove stale temp file {temp_path}: {e}")
                return False

        for attempt in range(1, MAX_RETRIES + 1):
            if self.flags.shutdown_requested or self.flags.suspend_requested:
                return False

            log.debug(f"  Download attempt {attempt}/{MAX_RETRIES}: {url}")
            t0 = time.time()

            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as res:
                    http_status = res.status
                    ctype       = res.headers.get("Content-Type", "").lower()
                    clength     = res.headers.get("Content-Length", "unknown")
                    log.debug(f"  HTTP {http_status} | Content-Type: {ctype} | Content-Length: {clength}")

                    # ArXiv should return application/pdf; reject HTML error pages etc.
                    if "pdf" not in ctype and "octet-stream" not in ctype:
                        log.warn(f"  Unexpected Content-Type '{ctype}' — skipping (not a PDF).")
                        return False

                    bytes_written = 0
                    with open(temp_path, "wb") as f:
                        while True:
                            if self.flags.shutdown_requested or self.flags.suspend_requested:
                                log.warn("  Interrupted mid-download — discarding partial file.")
                                return False
                            chunk = res.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            bytes_written += len(chunk)

            except urllib.error.HTTPError as e:
                log.error(f"  HTTP error {e.code} {e.reason} on attempt {attempt}")
            except urllib.error.URLError as e:
                log.error(f"  URL error on attempt {attempt}: {e.reason}")
            except OSError as e:
                log.error(f"  OS/network error on attempt {attempt}: {e}")
            except Exception as e:
                log.error(f"  Unexpected error on attempt {attempt}: {type(e).__name__}: {e}")
            else:
                # ---- Verify the downloaded file ----
                elapsed = time.time() - t0
                actual_size = os.path.getsize(temp_path)

                if actual_size < MIN_PDF_BYTES:
                    log.warn(
                        f"  File too small ({human_bytes(actual_size)}) — likely an error page. Discarding."
                    )
                    os.remove(temp_path)
                    # Don't retry; ArXiv served something intentionally small
                    return False

                # Atomically move temp → final destination
                try:
                    os.replace(temp_path, dest_path)
                except OSError as e:
                    log.error(f"  Could not move {temp_path} → {dest_path}: {e}")
                    return False

                log.ok(
                    f"  Saved {human_bytes(actual_size)} in {elapsed:.1f}s "
                    f"→ {dest.name}"
                )
                return True

            # Back-off before retry
            if attempt < MAX_RETRIES:
                self._rate_sleep(RATE_LIMIT_SECONDS * attempt, label=f"retry back-off")

        log.error(f"  All {MAX_RETRIES} attempts failed for {url}")
        return False

    # ------------------------------------------------------------------
    # ArXiv API query
    # ------------------------------------------------------------------

    def _query_api(self, safe_query: str, start: int, size: int) -> Optional[bytes]:
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query={safe_query}&start={start}&max_results={size}"
        )
        log.debug(f"API request: {url}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as res:
                    data = res.read()
                    log.debug(f"API response: {len(data)} bytes")
                    return data
            except urllib.error.HTTPError as e:
                log.error(f"API HTTP {e.code} on attempt {attempt}: {e.reason}")
            except urllib.error.URLError as e:
                log.error(f"API URL error on attempt {attempt}: {e.reason}")
            except Exception as e:
                log.error(f"API error on attempt {attempt}: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                self._rate_sleep(API_RATE_SECONDS * attempt, label="api retry")
        return None

    # ------------------------------------------------------------------
    # PDF URL extraction
    # ------------------------------------------------------------------

    def _extract_pdf_url(self, entry) -> Optional[str]:
        for link in entry.findall("atom:link", self.ns):
            t = link.attrib.get("title", "")
            mt = link.attrib.get("type", "")
            if t == "pdf" or mt == "application/pdf":
                url = link.attrib.get("href", "").replace("arxiv.org", "export.arxiv.org")
                if not url.endswith(".pdf"):
                    url += ".pdf"
                return url
        return None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self, state: Dict):
        tmp = str(STATE_FILE) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, STATE_FILE)
            log.debug(f"State saved → {STATE_FILE}")
        except OSError as e:
            log.error(f"Could not save state: {e}")

    # ------------------------------------------------------------------
    # Suspend (SIGTSTP / SIGSTOP)
    # ------------------------------------------------------------------

    def _handle_suspend_loop(self, state: Dict):
        self._save_state(state)
        log.info("State saved. Suspending process (fg to resume).")
        if hasattr(signal, "SIGSTOP"):
            self.flags.suspend_requested = False
            os.kill(os.getpid(), signal.SIGSTOP)
            # Execution resumes here after `fg`
            log.info("Resumed from suspend.")

    # ------------------------------------------------------------------
    # Session state builder
    # ------------------------------------------------------------------

    def _build_new_state(self) -> Optional[Dict]:
        raw_q = input("Queries (comma-separated): ").strip()
        if not raw_q:
            log.warn("No query entered — exiting.")
            return None
        limit_s = input(f"Max results per query [50]: ").strip()
        limit = int(limit_s) if limit_s.isdigit() else 50

        topics = []
        for raw in raw_q.split(","):
            cq = extract_query(raw)
            safe = urllib.parse.quote(f"all:{cq}" if ":" not in cq else cq)
            topics.append({
                "query":          cq,
                "safe_query":     safe,
                "start_index":    0,
                "total_processed": 0,
                "downloaded":     0,
                "skipped_exist":  0,
                "failed":         0,
                "done":           False,
                "total_avail":    None,
            })
        return {
            "max_results":        limit,
            "topics":             topics,
            "active_topic_index": 0,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self.flags.install()

        log.info(f"arXiv Downloader starting — saving to {DOWNLOAD_DIR}")
        log.info(f"Full log: {LOG_FILE}")

        # ---- Load or build state ----
        state = None
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                log.info(f"Resuming from saved state: {STATE_FILE}")
                # Migrate state written by older versions of this script —
                # fill in any keys that weren't tracked back then.
                TOPIC_DEFAULTS = {
                    "downloaded":    0,
                    "skipped_exist": 0,
                    "failed":        0,
                    "total_avail":   None,
                }
                migrated = False
                for t in state.get("topics", []):
                    for key, default in TOPIC_DEFAULTS.items():
                        if key not in t:
                            t[key] = default
                            migrated = True
                if migrated:
                    log.info("State migrated: added missing counter keys from old format.")
            except Exception as e:
                log.error(f"Could not load state file ({e}) — starting fresh.")
                state = None

        if state is None:
            state = self._build_new_state()
            if not state:
                return

        total_topics = len(state["topics"])

        try:
            for idx in range(state.get("active_topic_index", 0), total_topics):
                state["active_topic_index"] = idx
                topic = state["topics"][idx]

                if topic.get("done"):
                    log.info(f"Topic {idx+1}/{total_topics} already complete: '{topic['query']}'")
                    continue

                log.info(
                    f"── Topic {idx+1}/{total_topics}: '{topic['query']}' "
                    f"(processed so far: {topic['total_processed']}, "
                    f"target: {state['max_results']}) ──"
                )

                while (
                    not topic.get("done")
                    and topic["total_processed"] < state["max_results"]
                    and not self.flags.shutdown_requested
                ):
                    remaining = state["max_results"] - topic["total_processed"]
                    batch_size = min(API_CHUNK_SIZE, remaining)

                    log.info(
                        f"Fetching API batch: start={topic['start_index']} "
                        f"size={batch_size} remaining={remaining}"
                    )

                    xml_data = self._query_api(
                        topic["safe_query"], topic["start_index"], batch_size
                    )
                    if xml_data is None:
                        log.error("API query failed after retries — stopping this topic.")
                        break

                    try:
                        root = ET.fromstring(xml_data)
                    except ET.ParseError as e:
                        log.error(f"XML parse error: {e}")
                        break

                    entries = root.findall("atom:entry", self.ns)

                    if topic.get("total_avail") is None:
                        avail_tag = root.find("opensearch:totalResults", self.ns)
                        topic["total_avail"] = int(avail_tag.text) if avail_tag is not None else 0
                        log.info(f"arXiv reports {topic['total_avail']} total results for this query.")

                    cap = min(state["max_results"], topic["total_avail"])
                    log.info(f"Batch returned {len(entries)} entries (cap: {cap}).")

                    if not entries:
                        log.info("No more entries — topic complete.")
                        topic["done"] = True
                        break

                    for entry in entries:
                        # Check signals at the top of every entry
                        if self.flags.suspend_requested:
                            self._handle_suspend_loop(state)
                        if self.flags.shutdown_requested:
                            log.warn("Shutdown requested — saving state.")
                            self._save_state(state)
                            return

                        # ---- Extract metadata ----
                        title_el = entry.find("atom:title", self.ns)
                        title    = (title_el.text or "untitled").strip()
                        safe_title = clean_filename(title)
                        pdf_url    = self._extract_pdf_url(entry)
                        arxiv_id_el = entry.find("atom:id", self.ns)
                        arxiv_id    = (arxiv_id_el.text or "").strip()

                        topic["total_processed"] += 1
                        topic["start_index"]     += 1

                        counter = (
                            f"[{topic['total_processed']}/{cap}]"
                        )

                        if not pdf_url:
                            log.warn(f"{counter} No PDF link found for: {safe_title!r} ({arxiv_id})")
                            topic["failed"] += 1
                            continue

                        dest_path = str(DOWNLOAD_DIR / f"{safe_title}.pdf")

                        log.debug(f"{counter} arXiv ID : {arxiv_id}")
                        log.debug(f"{counter} Title    : {safe_title!r}")
                        log.debug(f"{counter} PDF URL  : {pdf_url}")
                        log.debug(f"{counter} Dest     : {dest_path}")

                        if os.path.exists(dest_path):
                            size = human_bytes(os.path.getsize(dest_path))
                            log.info(f"{counter} EXISTS ({size}): {safe_title}")
                            topic["skipped_exist"] += 1
                            # Still save state periodically even when skipping
                            if topic["skipped_exist"] % 10 == 0:
                                self._save_state(state)
                            continue

                        log.info(f"{counter} Downloading: {safe_title}")
                        success = self._download_pdf(pdf_url, dest_path)

                        if success:
                            topic["downloaded"] += 1
                        else:
                            topic["failed"] += 1
                            log.warn(f"{counter} Download failed — continuing to next paper.")

                        # Always rate-limit after each attempt, success or not,
                        # to avoid hammering arXiv and triggering bans.
                        self._rate_sleep(RATE_LIMIT_SECONDS, label="inter-download")
                        self._save_state(state)

                    if topic["total_processed"] >= state["max_results"]:
                        topic["done"] = True

                    # Rate-limit between API batch calls too
                    self._rate_sleep(API_RATE_SECONDS, label="inter-batch")

                # ---- Per-topic summary ----
                log.info(
                    f"Topic '{topic['query']}' summary — "
                    f"downloaded: {topic.get('downloaded', 0)}, "
                    f"already existed: {topic.get('skipped_exist', 0)}, "
                    f"failed: {topic.get('failed', 0)}, "
                    f"total processed: {topic['total_processed']}"
                )
                self._save_state(state)

                if self.flags.shutdown_requested:
                    break

            if not self.flags.shutdown_requested:
                log.info("All topics finished.")
                # Summarise across all topics
                total_dl  = sum(t.get("downloaded", 0)    for t in state["topics"])
                total_ex  = sum(t.get("skipped_exist", 0) for t in state["topics"])
                total_err = sum(t.get("failed", 0)        for t in state["topics"])
                log.info(
                    f"Session total — downloaded: {total_dl}, "
                    f"already existed: {total_ex}, failed: {total_err}"
                )
                if STATE_FILE.exists():
                    STATE_FILE.unlink()
                    log.info("State file removed (clean finish).")
            else:
                log.info("Exited early — state saved for resumption.")

        except Exception as e:
            log.error(f"Fatal exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self._save_state(state)

        finally:
            log.close()


if __name__ == "__main__":
    ArxivDownloader().run()
