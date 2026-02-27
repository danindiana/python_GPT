 sudo python3 media_subsystem_forensics.py --all-users --json

[sudo] password for jeb: 

╔══════════════════════════════════════════════════════════════════════╗
║    Media Subsystem Forensics  ·  PipeWire / WirePlumber / GStreamer  ║
║    Ubuntu 22.04 Compromise Detection Script                          ║
╚══════════════════════════════════════════════════════════════════════╝
  Started : 2026-02-26 23:46:17 
  EUID    : 0 (root)


──────────────────────────────────────────────────────────────────────
  User: root  (/root)
──────────────────────────────────────────────────────────────────────
  [OK      ] [root] No WirePlumber user scripts found
  [OK      ] [root] No WirePlumber systemd overrides
  [OK      ] [root] No GST_PLUGIN_PATH/SCANNER in shell configs

──────────────────────────────────────────────────────────────────────
  User: jeb  (/home/jeb)
──────────────────────────────────────────────────────────────────────
  [OK      ] [jeb] No WirePlumber user scripts found
  [OK      ] [jeb] No WirePlumber systemd overrides
  [OK      ] [jeb] No GST_PLUGIN_PATH/SCANNER in shell configs

──────────────────────────────────────────────────────────────────────
  User: netdisco  (/home/netdisco)
──────────────────────────────────────────────────────────────────────
  [OK      ] [netdisco] No WirePlumber user scripts found
  [OK      ] [netdisco] No WirePlumber systemd overrides
  [OK      ] [netdisco] No GST_PLUGIN_PATH/SCANNER in shell configs

──────────────────────────────────────────────────────────────────────
  User: coder  (/home/coder)
──────────────────────────────────────────────────────────────────────
  [OK      ] [coder] No WirePlumber user scripts found
  [OK      ] [coder] No WirePlumber systemd overrides
  [OK      ] [coder] No GST_PLUGIN_PATH/SCANNER in shell configs

──────────────────────────────────────────────────────────────────────
  PipeWire — Socket Connection Audit
──────────────────────────────────────────────────────────────────────
  [OK      ] No PipeWire sockets found (PipeWire may not be running)

──────────────────────────────────────────────────────────────────────
  PipeWire — Capture Process Audit
──────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────
  PipeWire — Metadata C2 Channel
──────────────────────────────────────────────────────────────────────
  [OK      ] pw-metadata: not accessible or PipeWire not running

──────────────────────────────────────────────────────────────────────
  GStreamer — Active Pipeline Processes
──────────────────────────────────────────────────────────────────────
  [INFO    ] gst-launch-1.0 running (no obvious IOC)
            PID 123154: Review pipeline manually.
            > root      123154  0.0  0.0   2892  1536 pts/7    S+   23:46   0:00 /bin/sh -c ps aux --no-headers 2>/dev/null 
  [INFO    ] gst-launch-1.0 running (no obvious IOC)
            PID 123156: Review pipeline manually.
            > root      123156  0.0  0.0  10260  2304 pts/7    S+   23:46   0:00 grep gst-launch

──────────────────────────────────────────────────────────────────────
  GStreamer — Plugin Registry Integrity
──────────────────────────────────────────────────────────────────────
  [OK      ] Registry age 735.5d: registry.x86_64.bin
  [OK      ] gst-inspect-1.0 not available or returned nothing

──────────────────────────────────────────────────────────────────────
  Package Integrity — dpkg -V
──────────────────────────────────────────────────────────────────────
  [OK      ] Integrity OK: libgstreamer1.0-0
  [OK      ] Integrity OK: gstreamer1.0-plugins-base
  [OK      ] Integrity OK: gstreamer1.0-plugins-good
  [OK      ] Integrity OK: gstreamer1.0-plugins-bad
  [OK      ] Integrity OK: gstreamer1.0-plugins-ugly

──────────────────────────────────────────────────────────────────────
  Network — C2 Beacon Detection
──────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────
  Process Memory Maps — Injected GStreamer Libraries
──────────────────────────────────────────────────────────────────────
  [OK      ] No GStreamer .so loaded from non-standard paths

──────────────────────────────────────────────────────────────────────
  AppArmor — Media Binary Confinement
──────────────────────────────────────────────────────────────────────
  [MEDIUM  ] AppArmor not enabled or not enforcing
            Without AppArmor, there are no MAC restrictions on media binaries.

──────────────────────────────────────────────────────────────────────
  Auditd — Threat Coverage
──────────────────────────────────────────────────────────────────────
  [LOW     ] No auditd rule for: gst-launch-1.0 exec watch
            Gap in audit coverage for this threat path.
  [LOW     ] No auditd rule for: pw-record exec watch
            Gap in audit coverage for this threat path.
  [LOW     ] No auditd rule for: wireplumber config watch
            Gap in audit coverage for this threat path.
  [LOW     ] No auditd rule for: gstreamer plugin dir watch
            Gap in audit coverage for this threat path.
  [LOW     ] No auditd rule for: LD_PRELOAD env watch
            Gap in audit coverage for this threat path.

══════════════════════════════════════════════════════════════════════
  FINDINGS SUMMARY
══════════════════════════════════════════════════════════════════════

  MEDIUM    (1)
    • AppArmor not enabled or not enforcing

  LOW       (5)
    • No auditd rule for: gst-launch-1.0 exec watch
    • No auditd rule for: pw-record exec watch
    • No auditd rule for: wireplumber config watch
    • No auditd rule for: gstreamer plugin dir watch
    • No auditd rule for: LD_PRELOAD env watch

  INFO      (2)
    • gst-launch-1.0 running (no obvious IOC)
    • gst-launch-1.0 running (no obvious IOC)

  Overall verdict : NO CRITICAL/HIGH INDICATORS FOUND
  Total findings  : 8
  Scan completed  : 2026-02-26 23:46:23

  JSON report written: /tmp/media_forensics_1772171183.json

(venv) jeb@worlock:~/programs/python_programs/secops$ 

