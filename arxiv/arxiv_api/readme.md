# arXiv Paper Downloader

A robust, production-grade Python script for bulk downloading academic papers from arXiv.org based on search queries. Features intelligent resume capability, comprehensive error handling, and respectful rate limiting.

## Features

- 📚 **Bulk Download**: Download multiple papers across different search queries
- 🔄 **Resumable**: Automatically saves progress and can resume interrupted sessions
- 🛡️ **Error Resilient**: Retries failed downloads, validates PDFs, handles network timeouts
- ⏱️ **Rate Limiting**: Respects arXiv's servers with configurable delays
- 📊 **Progress Tracking**: Real-time logging and download statistics
- 🎯 **Signal Handling**: Gracefully handles Ctrl+C, SIGTERM, and SIGTSTP (suspend)
- 📝 **Comprehensive Logging**: Writes to both console and persistent log file
- 🔍 **Smart Query Parsing**: Accepts both raw queries and arXiv URL formats

## Installation

### Prerequisites
- Python 3.6 or higher
- Internet connection

### Setup
1. Download the script:
```bash
wget https://raw.githubusercontent.com/yourusername/arxiv-downloader/main/arxiv_downloader.py
# or clone the repository
git clone https://github.com/yourusername/arxiv-downloader.git
cd arxiv-downloader
```

2. Make it executable (optional):
```bash
chmod +x arxiv_downloader.py
```

## Usage

### Basic Usage

Run the script and follow the prompts:
```bash
python arxiv_downloader.py
```

You'll be asked for:
- **Queries**: Comma-separated search terms (e.g., `quantum computing, machine learning, astrophysics`)
- **Max results**: Maximum number of papers to download per query (default: 50)

### Example Session

```
$ python arxiv_downloader.py
Queries (comma-separated): transformer neural networks, reinforcement learning
Max results per query [50]: 25

[10:30:15] [INFO ] arXiv Downloader starting — saving to /home/user/arxiv_downloads
[10:30:15] [INFO ] Full log: /home/user/arxiv_downloads/arxiv_downloader.log
[10:30:15] [INFO ] ── Topic 1/2: 'transformer neural networks' (processed so far: 0, target: 25) ──
[10:30:16] [INFO ] Fetching API batch: start=0 size=25 remaining=25
[10:30:18] [INFO ] arXiv reports 342 total results for this query.
[10:30:18] [INFO ] Batch returned 25 entries (cap: 25).
[10:30:19] [INFO ] [1/25] Downloading: Attention Is All You Need
[10:30:22] [OK   ]   Saved 2.3 MB in 3.1s → Attention_Is_All_You_Need.pdf
...
```

### Resume After Interruption

If the script is interrupted (Ctrl+C, system crash, etc.), simply run it again in the same directory:
```bash
python arxiv_downloader.py
```

The script will automatically detect the saved state and resume from where it left off.

## Configuration

You can modify these constants at the top of the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `USER_AGENT` | "Mozilla/5.0 ..." | User agent for HTTP requests |
| `DOWNLOAD_DIR` | "./arxiv_downloads" | Directory for downloaded PDFs |
| `RATE_LIMIT_SECONDS` | 3 | Delay between PDF downloads |
| `API_RATE_SECONDS` | 1 | Delay between API calls |
| `API_CHUNK_SIZE` | 100 | Papers per API request |
| `NETWORK_TIMEOUT` | 30 | HTTP timeout in seconds |
| `MIN_PDF_BYTES` | 1000 | Minimum valid PDF size |
| `MAX_RETRIES` | 3 | Download retry attempts |

## Output Structure

```
arxiv_downloads/
├── Attention_Is_All_You_Need.pdf
├── Deep_Reinforcement_Learning.pdf
├── BERT_Pre-training.pdf
├── .arxiv_downloader_state.json      # Resume state (auto-generated)
└── arxiv_downloader.log               # Persistent log file
```

## Advanced Features

### Signal Handling

- **Ctrl+C / SIGTERM**: Gracefully finishes current download and saves state
- **Ctrl+Z (SIGTSTP)**: Saves state and suspends process (resume with `fg`)
- **Double Ctrl+C**: Force exit immediately

### Query Formats

The script accepts multiple query formats:
```python
# Raw search
"quantum computing"

# arXiv search URL
"https://arxiv.org/search/?query=quantum+computing&searchtype=all"

# Advanced arXiv query
"https://export.arxiv.org/api/query?search_query=all:quantum+AND+cat:quant-ph"
```

### Log Levels

The console output shows:
- `[INFO]` - Normal operation messages
- `[OK]` - Successful downloads
- `[WARN]` - Non-critical issues
- `[ERROR]` - Failures requiring attention
- `[DEBUG]` - Detailed debug information

## Error Handling

The script handles various error conditions gracefully:
- **Network timeouts**: Automatic retries with backoff
- **Invalid PDFs**: Detects and discards HTML error pages
- **API failures**: Retries with exponential backoff
- **Disk full**: Stops with appropriate error message
- **Interrupted downloads**: Discards partial files

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- [ ] Add concurrent download option
- [ ] Implement metadata export (JSON, BibTeX)
- [ ] Add command-line argument parsing
- [ ] Create GUI interface
- [ ] Add email notification on completion
- [ ] Support for other preprint servers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for personal use only. Please respect arXiv's [terms of service](https://arxiv.org/help/api/tou) and:
- Implement appropriate delays between requests
- Don't use for systematic bulk downloading
- Consider arXiv's server load
- Cite papers appropriately in your work

## Acknowledgments

- arXiv.org for providing the API
- The Python community for excellent libraries
- All researchers sharing their work openly

## Support

If you encounter any issues:
1. Check the log file: `arxiv_downloads/arxiv_downloader.log`
2. Ensure you have a stable internet connection
3. Try reducing `MAX_RESULTS` if experiencing timeouts
4. Open an issue on GitHub with the error log

---

**Note**: Always verify you have the right to download and use papers from arXiv. Respect copyright and licensing terms of individual papers.
