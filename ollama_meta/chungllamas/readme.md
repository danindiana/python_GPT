Here’s a `README.md` file that describes the Ollama multi-GPU setup process as you outlined. I’ve generalized the instructions to make them reusable and anonymized specific details.

---

# Ollama Multi-GPU Setup

This repository contains instructions for configuring and running Ollama with multi-GPU support on a Linux system. By following these steps, you can enable Ollama to utilize multiple GPUs, enhancing the performance of models served on your machine.

## Prerequisites

1. **Linux System with Multiple GPUs**: Ensure that your system has compatible NVIDIA GPUs with CUDA support.
2. **Ollama Installed**: Ollama must be installed and accessible from `/usr/local/bin/ollama`.
3. **Systemd Service Setup**: Ollama should be configured to run as a systemd service for easier management.

## Instructions

### Step 1: Edit the `ollama.service` File

To configure Ollama to use multiple GPUs, edit the `ollama.service` file located at `/etc/systemd/system/ollama.service`.

```bash
sudo nano /etc/systemd/system/ollama.service
```

Update the service file as shown below:

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/path/to/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
Environment="OLLAMA_SCHED_SPREAD=1"
Environment="CUDA_VISIBLE_DEVICES=0,1"  # Adjust GPU IDs as necessary

[Install]
WantedBy=default.target
```

- **Environment Variables**:
  - `OLLAMA_SCHED_SPREAD=1`: Enables workload distribution across GPUs.
  - `CUDA_VISIBLE_DEVICES=0,1`: Specifies the GPU IDs Ollama should use. Adjust this to match the IDs of the GPUs you want to utilize.

### Step 2: Reload and Restart the Systemd Service

After updating the service file, reload the systemd daemon and restart the Ollama service to apply the changes.

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Step 3: Verify Service Status

Check the status of the Ollama service to ensure it is running correctly:

```bash
sudo systemctl status ollama
```

You should see output indicating that Ollama is running and has loaded the necessary GPU libraries.

### Step 4: Monitor GPU Usage

To confirm that both GPUs are being utilized by Ollama, start an inference task and monitor GPU usage with either `nvtop` or `nvidia-smi`.

```bash
nvtop
```

or

```bash
nvidia-smi
```

You should see activity on both GPUs, indicating that they are contributing to the model processing workload.

---

## Troubleshooting

- If only one GPU is in use, double-check that `OLLAMA_SCHED_SPREAD=1` and `CUDA_VISIBLE_DEVICES=0,1` are correctly set in the `ollama.service` file.
- Confirm that your GPUs are compatible with CUDA and that you have the latest NVIDIA drivers installed.
- Check the Ollama logs for additional details by running:

  ```bash
  journalctl -u ollama.service
  ```

---

## Additional Resources

For more information, refer to [Ollama’s documentation](https://github.com/ollama/ollama) on multi-GPU support and configuration.

--- 

