#!/usr/bin/env python3
"""
media_subsystem_forensics.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Forensic detection script for threat model:
  - PipeWire / WirePlumber LOTL & C2 abuse
  - GStreamer plugin loader abuse, LOTL pipelines, parser exploitation
  - Combined exfiltration/persistence chains

Targets: Ubuntu 22.04 (works on any systemd-based Linux with PipeWire/GStreamer)
Run as:  python3 media_subsystem_forensics.py [--all-users] [--json] [--quiet]

Author:  Generated for red-team / blue-team forensic audit
License: MIT
"""

import os
import sys
import re
import json
import glob
import stat
import time
import hashlib
import shutil
import argparse
import subprocess
import pwd
import grp
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ── Terminal colours ──────────────────────────────────────────────────────────
class C:
    RED    = "\033[91m"
    ORANGE = "\033[38;5;208m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    @staticmethod
    def disable():
        for attr in ('RED','ORANGE','YELLOW','GREEN','CYAN','BLUE','BOLD','DIM','RESET'):
            setattr(C, attr, '')

# ── Severity constants ────────────────────────────────────────────────────────
CRIT = "CRITICAL"
HIGH = "HIGH"
MED  = "MEDIUM"
LOW  = "LOW"
INFO = "INFO"

SEV_COLOUR = {
    CRIT: C.RED,
    HIGH: C.ORANGE,
    MED:  C.YELLOW,
    LOW:  C.CYAN,
    INFO: C.DIM,
}

SEV_ORDER = {CRIT: 4, HIGH: 3, MED: 2, LOW: 1, INFO: 0}

findings = []  # global accumulator

# ── Utility helpers ───────────────────────────────────────────────────────────

def banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║    Media Subsystem Forensics  ·  PipeWire / WirePlumber / GStreamer  ║
║    Ubuntu 22.04 Compromise Detection Script                          ║
╚══════════════════════════════════════════════════════════════════════╝{C.RESET}
  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}
  EUID    : {os.geteuid()} ({'root' if os.geteuid() == 0 else 'non-root — some checks limited'})
""")


def section(title: str):
    print(f"\n{C.BOLD}{C.BLUE}{'─'*70}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'─'*70}{C.RESET}")


def finding(severity: str, category: str, title: str, detail: str,
            path: str = None, evidence: str = None):
    """Record and print a finding."""
    rec = {
        "severity": severity,
        "category": category,
        "title": title,
        "detail": detail,
        "path": str(path) if path else None,
        "evidence": evidence,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    findings.append(rec)
    col = SEV_COLOUR.get(severity, '')
    sev_tag = f"{col}[{severity:8s}]{C.RESET}"
    print(f"  {sev_tag} {C.BOLD}{title}{C.RESET}")
    print(f"            {C.DIM}{detail}{C.RESET}")
    if path:
        print(f"            {C.YELLOW}Path    :{C.RESET} {path}")
    if evidence:
        for line in evidence.strip().splitlines()[:6]:
            print(f"            {C.DIM}> {line[:110]}{C.RESET}")


def ok(msg: str):
    print(f"  {C.GREEN}[OK      ]{C.RESET} {msg}")


def run(cmd: str, timeout: int = 10) -> str:
    """Run shell command, return stdout (never raises)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def file_age_days(path) -> float:
    """Return age of file in days (mtime)."""
    try:
        return (time.time() - os.path.getmtime(path)) / 86400
    except Exception:
        return 9999.0


def sha256(path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def get_home_dirs(all_users: bool):
    """Return list of (username, home_path) tuples to examine."""
    homes = []
    if all_users and os.geteuid() == 0:
        for pw in pwd.getpwall():
            h = pw.pw_dir
            if h.startswith('/home') or h == '/root':
                if os.path.isdir(h):
                    homes.append((pw.pw_name, Path(h)))
    else:
        homes.append((os.environ.get('USER', 'unknown'), Path.home()))
    return homes


def grep_file(path, patterns: list) -> list:
    """Return list of (lineno, line) matching any pattern in file."""
    matches = []
    try:
        with open(path, 'r', errors='replace') as f:
            for i, line in enumerate(f, 1):
                for pat in patterns:
                    if re.search(pat, line, re.IGNORECASE):
                        matches.append((i, line.rstrip()))
                        break
    except Exception:
        pass
    return matches


def is_suspicious_name(filename: str) -> bool:
    """Heuristic: does filename look like it's hiding?"""
    fn = os.path.basename(filename)
    suspicious = [
        r'^\.',                         # hidden file
        r'\.(sh|py|pl|rb)$',            # scripts masquerading as data
        r'^lib[a-z]{2,6}\.so$',         # too-short generic .so name
        r'(tmp|temp|cache|update|sync|helper|worker|agent|service)',
    ]
    return any(re.search(p, fn, re.I) for p in suspicious)


# ── Check modules ─────────────────────────────────────────────────────────────

def check_wireplumber_lua_scripts(home: Path, user: str):
    """Scan WirePlumber user config for injected Lua scripts."""
    wp_dirs = [
        home / '.config' / 'wireplumber',
        home / '.config' / 'wireplumber' / 'scripts',
        home / '.config' / 'wireplumber' / 'wireplumber.conf.d',
    ]
    # Patterns that suggest malicious Lua content
    danger_patterns = [
        r'os\.execute\s*\(',
        r'io\.popen\s*\(',
        r'io\.open\s*\(',
        r'require\s*[\(\'"](socket|http|https|curl|posix)',
        r'GLib\.spawn',
        r'Gio\.Subprocess',
        r'base64',
        r'curl\s',
        r'wget\s',
        r'bash\s',
        r'/tmp/',
        r'crontab',
        r'\.hidden',
        r'LD_PRELOAD',
        r'tcpclient',
        r'udp[sf]',
    ]

    found_scripts = []
    for d in wp_dirs:
        if d.exists():
            for p in d.rglob('*.lua'):
                found_scripts.append(p)
            for p in d.rglob('*.conf'):
                found_scripts.append(p)

    # Also check the main conf
    main_conf = home / '.config' / 'wireplumber' / 'wireplumber.conf'
    if main_conf.exists():
        found_scripts.append(main_conf)

    if not found_scripts:
        ok(f"[{user}] No WirePlumber user scripts found")
        return

    for script in found_scripts:
        matches = grep_file(script, danger_patterns)
        age = file_age_days(script)
        age_warn = age < 30  # modified in last 30 days

        if matches:
            evidence = '\n'.join(f"L{ln}: {l}" for ln, l in matches[:8])
            finding(CRIT, "WirePlumber/Lua",
                    f"Suspicious Lua script: {script.name}",
                    f"User {user}: WirePlumber Lua file contains dangerous function calls. "
                    f"File age: {age:.1f} days.",
                    path=script, evidence=evidence)
        elif age_warn:
            finding(MED, "WirePlumber/Lua",
                    f"Recently modified WirePlumber config: {script.name}",
                    f"User {user}: WirePlumber config modified {age:.1f} days ago — review content.",
                    path=script,
                    evidence=open(script, errors='replace').read()[:400] if script.stat().st_size < 4096 else None)
        else:
            finding(INFO, "WirePlumber/Lua",
                    f"Non-default WirePlumber script present: {script.name}",
                    f"User {user}: Exists but no obvious IOCs. Verify it is intentional.",
                    path=script)


def check_wireplumber_systemd_overrides(home: Path, user: str):
    """Check for systemd user unit overrides that modify WirePlumber startup."""
    override_dir = home / '.config' / 'systemd' / 'user' / 'wireplumber.service.d'
    if not override_dir.exists():
        ok(f"[{user}] No WirePlumber systemd overrides")
        return

    danger = [r'LD_PRELOAD', r'ExecStartPre', r'ExecStartPost',
               r'Environment.*=.*/', r'curl', r'wget', r'bash', r'/tmp']

    for conf in override_dir.glob('*.conf'):
        matches = grep_file(conf, danger)
        content = open(conf, errors='replace').read()
        if matches:
            evidence = '\n'.join(f"L{ln}: {l}" for ln, l in matches)
            finding(CRIT, "WirePlumber/Systemd",
                    "Malicious WirePlumber systemd override",
                    f"User {user}: Override injects code into WirePlumber service startup.",
                    path=conf, evidence=evidence)
        else:
            finding(MED, "WirePlumber/Systemd",
                    "Non-default WirePlumber systemd override found",
                    f"User {user}: Override drops down persistence surface — review carefully.",
                    path=conf, evidence=content[:300])


def check_pipewire_socket_connections():
    """Enumerate processes connected to the PipeWire socket."""
    section("PipeWire — Socket Connection Audit")

    # Find all PipeWire sockets
    sockets = glob.glob('/run/user/*/pipewire-0') + \
               glob.glob('/run/user/*/pipewire-0-manager')

    if not sockets:
        ok("No PipeWire sockets found (PipeWire may not be running)")
        return

    # Known-good process names that legitimately connect
    known_good = {
        'wireplumber', 'pipewire', 'pipewire-pulse', 'pipewire-media-session',
        'pulseaudio', 'gnome-shell', 'xdg-desktop-portal', 'gnome-settings-daemon',
        'firefox', 'chromium', 'chrome', 'obs', 'vlc', 'totem', 'rhythmbox',
        'pavucontrol', 'pw-cli', 'pw-play', 'pw-record', 'pactl', 'pacmd',
        'alsa-card-profiles', 'gsettings', 'gst-launch-1.0',
    }

    for sock in sockets:
        uid_match = re.search(r'/run/user/(\d+)/', sock)
        uid = uid_match.group(1) if uid_match else '?'
        try:
            username = pwd.getpwuid(int(uid)).pw_name
        except Exception:
            username = uid

        # Use ss to enumerate unix socket connections
        out = run(f"ss -xp 2>/dev/null | grep '{os.path.basename(sock)}'")
        if not out:
            ok(f"No external connections to {sock}")
            continue

        for line in out.splitlines():
            # Extract process name from ss output
            proc_match = re.search(r'users:\(\("([^"]+)"', line)
            pid_match  = re.search(r',pid=(\d+),', line)
            proc_name  = proc_match.group(1) if proc_match else 'unknown'
            pid        = pid_match.group(1) if pid_match else '?'

            if proc_name not in known_good:
                # Pull cmdline
                cmdline = run(f"cat /proc/{pid}/cmdline 2>/dev/null | tr '\\0' ' '")
                finding(HIGH, "PipeWire/Socket",
                        f"Unknown process connected to PipeWire socket",
                        f"Process '{proc_name}' (PID {pid}) has an open connection to "
                        f"{sock} (user: {username}). Not in known-good list.",
                        evidence=f"cmdline: {cmdline}\nss: {line[:120]}")


def check_gstreamer_plugin_dirs(home: Path, user: str):
    """Scan user GStreamer plugin directory for unexpected/malicious .so files."""
    plugin_dirs = [
        home / '.local' / 'share' / 'gstreamer-1.0' / 'plugins',
        home / '.local' / 'lib' / 'gstreamer-1.0',
        Path('/tmp'),
        Path('/var/tmp'),
    ]
    # Also check GST_PLUGIN_PATH from user environment
    for env_file in [home/'.bashrc', home/'.profile', home/'.bash_profile',
                     home/'.zshrc', home/'.config'/'environment.d'/'gst.conf']:
        if env_file.exists():
            for _, line in grep_file(env_file, [r'GST_PLUGIN_PATH']):
                # Extract directory paths
                for d in re.findall(r'[:/]([^\s:$"\']+)', line):
                    p = Path(d)
                    if p.exists() and p not in plugin_dirs:
                        plugin_dirs.append(p)

    for d in plugin_dirs:
        if not d.exists():
            continue
        so_files = list(d.glob('*.so')) + list(d.glob('libgst*.so*'))
        if not so_files:
            continue

        for so in so_files:
            age  = file_age_days(so)
            h    = sha256(so)
            name = so.name

            # Red flags
            red_flags = []
            if is_suspicious_name(str(so)):
                red_flags.append("suspicious filename pattern")
            if age < 7:
                red_flags.append(f"very recently modified ({age:.1f}d ago)")
            if str(d).startswith('/tmp') or str(d).startswith('/var/tmp'):
                red_flags.append("located in /tmp or /var/tmp")
            if not name.startswith('libgst') and d != Path('/tmp'):
                red_flags.append("non-standard libgst prefix")

            # Check if this .so is known to dpkg
            dpkg_out = run(f"dpkg -S {so} 2>/dev/null")
            in_dpkg = bool(dpkg_out)

            if not in_dpkg and red_flags:
                # Inspect ELF symbols for extra evidence
                symbols = run(f"nm -D --defined-only {so} 2>/dev/null | head -30")
                strings_out = run(f"strings {so} 2>/dev/null | grep -Ei "
                                  f"'(curl|wget|http|tcp|udp|bash|exec|system|popen|/tmp|base64|crontab|LD_PRELOAD|socket|connect|send|recv)' "
                                  f"| head -20")
                sev = CRIT if (str(d).startswith('/tmp') or 'recently' in ' '.join(red_flags)) else HIGH
                finding(sev, "GStreamer/Plugin",
                        f"Unrecognised GStreamer plugin: {name}",
                        f"User {user}: .so not in dpkg DB. Flags: {', '.join(red_flags)}. SHA256: {h[:16]}…",
                        path=so,
                        evidence=f"suspicious strings:\n{strings_out}" if strings_out else f"flags: {red_flags}")
            elif not in_dpkg:
                finding(MED, "GStreamer/Plugin",
                        f"Non-package GStreamer plugin: {name}",
                        f"User {user}: .so not tracked by dpkg but no immediate red flags. "
                        f"Age: {age:.1f}d. SHA256: {h[:16]}…",
                        path=so)
            else:
                ok(f"[{user}] Plugin OK (dpkg-tracked): {name}")


def check_gstreamer_env_injection(home: Path, user: str):
    """Detect GST_PLUGIN_PATH / GST_PLUGIN_SCANNER injection in shell configs."""
    config_files = [
        home / '.bashrc',
        home / '.bash_profile',
        home / '.profile',
        home / '.zshrc',
        home / '.zprofile',
        home / '.config' / 'environment.d' / 'gst.conf',
        home / '.pam_environment',
    ]
    # systemd user environment.d
    env_d = home / '.config' / 'environment.d'
    if env_d.exists():
        config_files += list(env_d.glob('*.conf'))

    danger = [
        r'GST_PLUGIN_PATH',
        r'GST_PLUGIN_SCANNER',
        r'GST_REGISTRY',
        r'GST_DEBUG_DUMP_DOT_DIR',
        r'LD_PRELOAD.*gst',
        r'LD_LIBRARY_PATH.*gst',
    ]

    found_any = False
    for cf in config_files:
        if not cf.exists():
            continue
        matches = grep_file(cf, danger)
        if matches:
            found_any = True
            for ln, line in matches:
                # Is the path pointed to somewhere suspicious?
                path_matches = re.findall(r'=([^\s"\']+)', line)
                suspicious_paths = [p for p in path_matches
                                    if any(x in p for x in ['/tmp', '/var/tmp', '.hidden',
                                                              'cache', '.local/lib'])]
                sev = CRIT if suspicious_paths else HIGH
                finding(sev, "GStreamer/EnvInjection",
                        f"GST environment variable set in shell config",
                        f"User {user}: '{line.strip()}' in {cf.name}. "
                        f"This persists on every shell start and affects all child processes.",
                        path=cf,
                        evidence=f"L{ln}: {line}")
    if not found_any:
        ok(f"[{user}] No GST_PLUGIN_PATH/SCANNER in shell configs")


def check_gst_launch_processes():
    """Detect running gst-launch-1.0 with network sinks or capture sources."""
    section("GStreamer — Active Pipeline Processes")

    # pgrep -a avoids the classic grep-catches-itself false positive
    procs = run("pgrep -a 'gst-launch' 2>/dev/null")
    if not procs:
        ok("No gst-launch-1.0 processes currently running")
        return
    # Reformat pgrep -a output (pid<TAB>cmdline) to match ps aux style for downstream parsing
    reformatted = []
    for line in procs.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            pid, cmdline = parts
            reformatted.append(f"unknown {pid} 0.0 0.0 0 0 ? S 00:00 0:00 {cmdline}")
    procs = '\n'.join(reformatted)

    # Dangerous elements in a pipeline
    net_sinks   = r'(souphttpclientsink|tcpclientsink|udpsink|rtmpsink|' \
                  r'rtspclientsink|multiudpsink|tcpserversink)'
    net_sources = r'(souphttpsrc|tcpclientsrc|udpsrc|tcpserversrc)'
    cap_sources = r'(pulsesrc|v4l2src|ximagesrc|pipewiresrc|alsasrc)'
    exfil_combo = re.compile(f'({cap_sources}).*({net_sinks})', re.I)
    c2_combo    = re.compile(f'({net_sources}).*filesink|fakesink', re.I)

    for line in procs.splitlines():
        pid_m = re.match(r'\S+\s+(\d+)', line)
        pid   = pid_m.group(1) if pid_m else '?'

        if exfil_combo.search(line):
            finding(CRIT, "GStreamer/ActiveExfil",
                    "ACTIVE audio/video exfiltration pipeline detected",
                    f"PID {pid}: gst-launch-1.0 pipeline captures A/V AND streams to network.",
                    evidence=line[:200])
        elif re.search(net_sinks, line, re.I) and re.search(cap_sources, line, re.I):
            finding(CRIT, "GStreamer/ActiveExfil",
                    "gst-launch-1.0 with capture + network sink",
                    f"PID {pid}: Possible live A/V exfiltration.",
                    evidence=line[:200])
        elif re.search(net_sinks, line, re.I) or re.search(net_sources, line, re.I):
            finding(HIGH, "GStreamer/NetworkPipeline",
                    "gst-launch-1.0 with unexpected network element",
                    f"PID {pid}: Pipeline uses network I/O — verify legitimacy.",
                    evidence=line[:200])
        elif re.search(cap_sources, line, re.I):
            finding(MED, "GStreamer/CapturePipeline",
                    "gst-launch-1.0 with A/V capture source",
                    f"PID {pid}: Pipeline capturing audio/video.",
                    evidence=line[:200])
        else:
            finding(INFO, "GStreamer/Pipeline",
                    "gst-launch-1.0 running (no obvious IOC)",
                    f"PID {pid}: Review pipeline manually.",
                    evidence=line[:160])

    # Check network connections for gst-launch PIDs
    for line in procs.splitlines():
        pid_m = re.match(r'\S+\s+(\d+)', line)
        if not pid_m:
            continue
        pid = pid_m.group(1)
        netstat = run(f"ss -tnp 2>/dev/null | grep 'pid={pid}'")
        if netstat:
            finding(CRIT, "GStreamer/C2Connection",
                    f"gst-launch-1.0 (PID {pid}) has active TCP connections",
                    "Live C2 or exfiltration channel suspected.",
                    evidence=netstat[:400])


def check_pw_record_processes():
    """Detect silent pw-record (PipeWire native capture) processes."""
    procs = run("pgrep -a 'pw-record' 2>/dev/null")
    if not procs:
        ok("No pw-record processes running")
        return
    for line in procs.splitlines():
        pid_m = re.match(r'\S+\s+(\d+)', line)
        pid   = pid_m.group(1) if pid_m else '?'
        finding(HIGH, "PipeWire/CaptureProcess",
                f"pw-record running (PID {pid})",
                "Silent audio capture via native PipeWire API — verify legitimacy.",
                evidence=line[:200])


def check_pipewire_metadata():
    """Check PipeWire metadata store for suspicious keys (C2 channel indicator)."""
    # pw-metadata is only accessible if PipeWire is running as current user
    meta_out = run("pw-metadata 2>/dev/null", timeout=5)
    if not meta_out:
        ok("pw-metadata: not accessible or PipeWire not running")
        return

    suspicious_keys = [r'cmd', r'exec', r'update', r'payload', r'beacon',
                       r'run', r'stage', r'base64', r'token', r'key']
    matches = []
    for line in meta_out.splitlines():
        if any(re.search(p, line, re.I) for p in suspicious_keys):
            matches.append(line)

    if matches:
        finding(CRIT, "PipeWire/Metadata",
                "Suspicious keys in PipeWire metadata store",
                "PipeWire metadata contains keys consistent with in-session C2 messaging.",
                evidence='\n'.join(matches[:10]))
    else:
        ok("PipeWire metadata: no suspicious keys found")


def check_gstreamer_registry_integrity():
    """
    Validate the GStreamer plugin registry cache against dpkg-known plugins.
    Any plugin in the registry that isn't in a package is suspicious.
    """
    section("GStreamer — Plugin Registry Integrity")

    registry_files = glob.glob(os.path.expanduser('~/.cache/gstreamer-1.0/registry.*.bin'))
    if not registry_files:
        ok("No GStreamer registry cache found (or not readable)")
        return

    for reg in registry_files:
        age = file_age_days(reg)
        if age < 1:
            finding(MED, "GStreamer/Registry",
                    f"GStreamer registry modified very recently",
                    f"Registry {os.path.basename(reg)} was modified {age*24:.1f} hours ago — "
                    f"may have been rebuilt after a plugin was planted.",
                    path=reg)
        else:
            ok(f"Registry age {age:.1f}d: {os.path.basename(reg)}")

    # gst-inspect-1.0 may not be in root's stripped PATH under sudo
    gst_inspect = (shutil.which('gst-inspect-1.0') or
                   '/usr/bin/gst-inspect-1.0' if os.path.exists('/usr/bin/gst-inspect-1.0') else None)
    if not gst_inspect:
        finding(LOW, "GStreamer/Registry",
                "gst-inspect-1.0 not found — plugin registry audit skipped",
                "Install gstreamer1.0-tools to enable full registry enumeration.")
        return

    out = run(f"{gst_inspect} --print-all 2>/dev/null | grep 'Filename:' | awk '{{print $2}}'",
              timeout=30)
    if not out:
        ok("gst-inspect-1.0 not available or returned nothing")
        return

    unknown_plugins = []
    for so_path in out.splitlines():
        if not so_path or not os.path.exists(so_path):
            continue
        dpkg_check = run(f"dpkg -S {so_path} 2>/dev/null")
        if not dpkg_check:
            strings_hit = run(
                f"strings {so_path} 2>/dev/null | "
                f"grep -Ei '(curl|wget|http|tcp|bash|exec|system|popen|/tmp|base64|socket)' "
                f"| head -10")
            unknown_plugins.append((so_path, strings_hit))

    if unknown_plugins:
        for path, strings_hit in unknown_plugins:
            sev = CRIT if strings_hit else HIGH
            finding(sev, "GStreamer/Registry",
                    f"Registered plugin not in dpkg: {os.path.basename(path)}",
                    f"This plugin is active in the registry but not tracked by any package.",
                    path=path,
                    evidence=f"suspicious strings:\n{strings_hit}" if strings_hit else None)
    else:
        ok("All registered GStreamer plugins tracked by dpkg")


def check_dpkg_integrity():
    """Run dpkg -V on GStreamer and PipeWire packages to detect tampered files."""
    section("Package Integrity — dpkg -V")
    packages = [
        'libgstreamer1.0-0',
        'gstreamer1.0-plugins-base',
        'gstreamer1.0-plugins-good',
        'gstreamer1.0-plugins-bad',
        'gstreamer1.0-plugins-ugly',
        'libpipewire-0.3-0',
        'pipewire',
        'pipewire-bin',
        'wireplumber',
        'libwireplumber-0.4-0',
        'gstreamer1.0-pipewire',
    ]
    for pkg in packages:
        out = run(f"dpkg -V {pkg} 2>/dev/null")
        if out:
            finding(CRIT, "PackageIntegrity",
                    f"dpkg integrity failure: {pkg}",
                    "One or more files in this package have been modified after installation.",
                    evidence=out[:400])
        else:
            # Check package is installed first
            installed = run(f"dpkg -l {pkg} 2>/dev/null | grep '^ii'")
            if installed:
                ok(f"Integrity OK: {pkg}")


def check_hidden_capture_files(home: Path, user: str):
    """Look for hidden audio/video capture files that may indicate silent recording."""
    search_dirs = [
        home,
        Path('/tmp'),
        Path('/var/tmp'),
        home / '.cache',
        home / '.local' / 'share',
    ]
    # Patterns: hidden files with media extensions
    media_exts = {'.mp3', '.wav', '.ogg', '.flac', '.mp4', '.mkv', '.avi',
                  '.webm', '.pcm', '.raw', '.au'}

    for d in search_dirs:
        if not d.exists():
            continue
        try:
            for entry in d.iterdir():
                name = entry.name
                ext  = Path(name).suffix.lower()
                if ext not in media_exts:
                    continue
                is_hidden = name.startswith('.')
                in_tmp    = str(d).startswith('/tmp') or str(d).startswith('/var/tmp')
                age       = file_age_days(entry)

                if (is_hidden or in_tmp) and age < 7:
                    sz = entry.stat().st_size if entry.exists() else 0
                    finding(HIGH, "CaptureArtifact",
                            f"Suspicious media file: {name}",
                            f"User {user}: {'Hidden' if is_hidden else 'In /tmp'} "
                            f"media file ({sz} bytes, {age:.1f}d old). "
                            f"May be silent recording output.",
                            path=entry)
        except PermissionError:
            pass


def check_persistence_vectors(home: Path, user: str):
    """Check crontab, autostart, and other persistence mechanisms for LOTL payloads."""
    # ── crontab ──
    crontab = run(f"crontab -l -u {user} 2>/dev/null")
    if crontab:
        gst_cron = re.search(r'(gst-launch|pw-record|pw-play|pipewire|wireplumber)', crontab)
        if gst_cron:
            finding(HIGH, "Persistence/Crontab",
                    f"Media tool in crontab for user {user}",
                    "Scheduled execution of GStreamer/PipeWire binary — likely persistence.",
                    evidence=crontab[:400])
        else:
            # General suspicious crontab check
            for line in crontab.splitlines():
                if re.search(r'(/tmp/|\.hidden|base64|curl|wget)', line, re.I):
                    finding(HIGH, "Persistence/Crontab",
                            f"Suspicious crontab entry for {user}",
                            "Crontab contains IOC-matching entry.",
                            evidence=line)

    # ── XDG autostart ──
    autostart_dirs = [
        home / '.config' / 'autostart',
        Path('/etc/xdg/autostart'),
    ]
    for d in autostart_dirs:
        if not d.exists():
            continue
        for desktop in d.glob('*.desktop'):
            content = open(desktop, errors='replace').read()
            matches = []
            for pat in [r'gst-launch', r'pw-record', r'pulsesrc', r'tcpclient',
                        r'souphttpclient', r'/tmp/', r'base64', r'curl', r'wget']:
                if re.search(pat, content, re.I):
                    matches.append(pat)
            if matches:
                finding(HIGH, "Persistence/Autostart",
                        f"Suspicious .desktop autostart: {desktop.name}",
                        f"Autostart entry matches IOCs: {matches}",
                        path=desktop, evidence=content[:300])

    # ── systemd user units ──
    unit_dirs = [
        home / '.config' / 'systemd' / 'user',
        Path(f'/run/user/{os.getuid()}/systemd/user'),
    ]
    for d in unit_dirs:
        if not d.exists():
            continue
        for unit in list(d.glob('*.service')) + list(d.glob('*.timer')):
            content = open(unit, errors='replace').read()
            iocs = re.findall(r'(gst-launch|pw-record|pulsesrc|tcpclient|souphttpclient'
                              r'|ximagesrc|v4l2src|/tmp/\S+|LD_PRELOAD|curl\s|wget\s)', content, re.I)
            if iocs:
                finding(CRIT if any(x in str(iocs) for x in ['tcpclient','souphttpclient']) else HIGH,
                        "Persistence/SystemdUser",
                        f"Suspicious systemd user unit: {unit.name}",
                        f"Unit file matches IOCs: {list(set(iocs))}",
                        path=unit, evidence=content[:400])

    # ── LD_PRELOAD checks ──
    ld_files = [home/'.bashrc', home/'.profile', Path('/etc/environment'),
                Path('/etc/ld.so.preload')]
    for f in ld_files:
        if not f.exists():
            continue
        matches = grep_file(f, [r'LD_PRELOAD'])
        for ln, line in matches:
            # LD_PRELOAD to a GStreamer or PipeWire .so is especially suspicious
            if re.search(r'(gst|pipewire|wireplumber|pulse)', line, re.I):
                finding(CRIT, "Persistence/LDPreload",
                        f"LD_PRELOAD pointing to media library in {f.name}",
                        "Injecting .so into every process via LD_PRELOAD.",
                        path=f, evidence=f"L{ln}: {line}")
            else:
                finding(HIGH, "Persistence/LDPreload",
                        f"LD_PRELOAD set in {f.name}",
                        "All child processes will load this library.",
                        path=f, evidence=f"L{ln}: {line}")


def check_network_beaconing():
    """Check for periodic or unusual outbound TCP connections from media processes."""
    section("Network — C2 Beacon Detection")

    media_procs = ['gst-launch-1.0', 'pw-record', 'pw-play', 'wireplumber',
                   'pipewire', 'gst-plugin-scanner', 'python3', 'bash']

    tcp_conns = run("ss -tnp 2>/dev/null | grep -E 'ESTAB|SYN_SENT'")
    if not tcp_conns:
        ok("No established/connecting TCP connections found")
        return

    for line in tcp_conns.splitlines():
        for proc in media_procs:
            if proc in line:
                # Extract destination
                dst = re.search(r'(\d+\.\d+\.\d+\.\d+:\d+)\s+users:', line)
                dst_str = dst.group(1) if dst else '?'
                # Flag non-LAN destinations from media tools
                if not re.search(r'^(127\.|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)', dst_str):
                    finding(HIGH, "Network/Beacon",
                            f"Media process with external TCP connection",
                            f"'{proc}' has established connection to {dst_str}",
                            evidence=line[:200])
                else:
                    finding(LOW, "Network/Beacon",
                            f"Media process with LAN TCP connection",
                            f"'{proc}' connected to {dst_str}",
                            evidence=line[:200])


def check_proc_maps_for_injected_gst():
    """Scan /proc/*/maps for any GStreamer plugin .so loaded from non-standard paths."""
    section("Process Memory Maps — Injected GStreamer Libraries")

    suspicious_count = 0
    for maps_path in glob.glob('/proc/*/maps'):
        try:
            pid = maps_path.split('/')[2]
            if not pid.isdigit():
                continue
            with open(maps_path, 'r') as f:
                content = f.read()
            # Look for libgst*.so loaded from outside standard lib dirs
            for match in re.finditer(r'(/[^\s]+libgst[^\s]+\.so[^\s]*)', content):
                so_path = match.group(1)
                if not re.match(r'^/usr/lib|^/usr/local/lib', so_path):
                    cmdline = run(f"cat /proc/{pid}/cmdline 2>/dev/null | tr '\\0' ' '")
                    suspicious_count += 1
                    finding(CRIT, "ProcMaps/InjectedLib",
                            f"GStreamer .so loaded from non-standard path",
                            f"PID {pid} has {so_path} mapped into memory.",
                            evidence=f"PID {pid} cmdline: {cmdline}\npath: {so_path}")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass

    if suspicious_count == 0:
        ok("No GStreamer .so loaded from non-standard paths")


def check_apparmor_status():
    """Report AppArmor status for PipeWire/GStreamer binaries."""
    section("AppArmor — Media Binary Confinement")

    binaries = [
        '/usr/bin/gst-launch-1.0',
        '/usr/libexec/gstreamer-1.0/gst-plugin-scanner',
        '/usr/bin/pw-record',
        '/usr/bin/wireplumber',
        '/usr/bin/pipewire',
    ]
    aa_enabled = run("aa-status --enabled 2>/dev/null ; echo $?")
    if aa_enabled.strip() != '0':
        finding(MED, "AppArmor",
                "AppArmor not enabled or not enforcing",
                "Without AppArmor, there are no MAC restrictions on media binaries.")
        return

    for binary in binaries:
        aa_out = run(f"aa-status 2>/dev/null | grep '{binary}'")
        if aa_out:
            ok(f"AppArmor profile active: {binary}")
        else:
            finding(LOW, "AppArmor",
                    f"No AppArmor profile: {binary}",
                    f"Binary has no MAC confinement — LOTL use is unrestricted.")


def check_auditd_coverage():
    """Verify auditd rules cover the key threat paths."""
    section("Auditd — Threat Coverage")

    if not shutil.which('auditctl'):
        finding(LOW, "Auditd",
                "auditctl not found — auditd may not be installed",
                "Without auditd, there is no syscall-level audit trail for media abuse.")
        return

    rules = run("auditctl -l 2>/dev/null")
    checks = {
        'gst-launch-1.0 exec watch': r'gst-launch',
        'pw-record exec watch':      r'pw-record',
        'wireplumber config watch':  r'wireplumber',
        'gstreamer plugin dir watch':r'gstreamer',
        'LD_PRELOAD env watch':      r'LD_PRELOAD',
    }
    for desc, pattern in checks.items():
        if re.search(pattern, rules, re.I):
            ok(f"Audit rule covers: {desc}")
        else:
            finding(LOW, "Auditd",
                    f"No auditd rule for: {desc}",
                    "Gap in audit coverage for this threat path.")


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(output_json: bool, quiet: bool):
    print(f"\n{C.BOLD}{C.CYAN}{'═'*70}")
    print("  FINDINGS SUMMARY")
    print(f"{'═'*70}{C.RESET}\n")

    by_sev = defaultdict(list)
    for f in findings:
        by_sev[f['severity']].append(f)

    for sev in [CRIT, HIGH, MED, LOW, INFO]:
        items = by_sev[sev]
        if not items:
            continue
        col = SEV_COLOUR[sev]
        print(f"  {col}{C.BOLD}{sev:8s}  ({len(items)}){C.RESET}")
        if not quiet or sev in (CRIT, HIGH):
            for f in items:
                print(f"    • {f['title']}")
                if f['path']:
                    print(f"      {C.DIM}{f['path']}{C.RESET}")
        print()

    total = len(findings)
    critical_count = len(by_sev[CRIT])
    high_count     = len(by_sev[HIGH])

    if critical_count > 0:
        verdict = f"{C.RED}{C.BOLD}SYSTEM LIKELY COMPROMISED — {critical_count} CRITICAL findings{C.RESET}"
    elif high_count > 0:
        verdict = f"{C.ORANGE}{C.BOLD}SUSPICIOUS — {high_count} HIGH severity findings require investigation{C.RESET}"
    else:
        verdict = f"{C.GREEN}{C.BOLD}NO CRITICAL/HIGH INDICATORS FOUND{C.RESET}"

    print(f"  Overall verdict : {verdict}")
    print(f"  Total findings  : {total}")
    print(f"  Scan completed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if output_json:
        out_path = f"/tmp/media_forensics_{int(time.time())}.json"
        with open(out_path, 'w') as jf:
            json.dump({
                "scan_time": datetime.now(timezone.utc).isoformat(),
                "host": run("hostname"),
                "euid": os.geteuid(),
                "findings": findings,
            }, jf, indent=2)
        print(f"  {C.CYAN}JSON report written: {out_path}{C.RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Forensic detection for PipeWire/WirePlumber/GStreamer LOTL/C2 compromise")
    parser.add_argument('--all-users', action='store_true',
                        help='Scan all home directories (requires root)')
    parser.add_argument('--json',   action='store_true', help='Write JSON report to /tmp/')
    parser.add_argument('--quiet',  action='store_true', help='Suppress INFO findings in summary')
    parser.add_argument('--no-color', action='store_true', help='Disable coloured output')
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        C.disable()

    banner()
    homes = get_home_dirs(args.all_users)

    # ── Per-user checks ────────────────────────────────────────────────────────
    for user, home in homes:
        section(f"User: {user}  ({home})")

        check_wireplumber_lua_scripts(home, user)
        check_wireplumber_systemd_overrides(home, user)
        check_gstreamer_plugin_dirs(home, user)
        check_gstreamer_env_injection(home, user)
        check_hidden_capture_files(home, user)
        check_persistence_vectors(home, user)

    # ── System-wide / process-level checks ────────────────────────────────────
    check_pipewire_socket_connections()

    section("PipeWire — Capture Process Audit")
    check_pw_record_processes()

    section("PipeWire — Metadata C2 Channel")
    check_pipewire_metadata()

    check_gst_launch_processes()
    check_gstreamer_registry_integrity()
    check_dpkg_integrity()
    check_network_beaconing()
    check_proc_maps_for_injected_gst()
    check_apparmor_status()
    check_auditd_coverage()

    print_summary(args.json, args.quiet)


if __name__ == '__main__':
    main()
