To create a flexible, user-friendly setup to control the `ollama` service, you can add a few shortcuts and scripts for convenience. Here’s a setup that will let you start, stop, restart, or check the status of the service easily.

### 1. **Alias Commands for Quick Service Control**

You can create aliases in your shell configuration file (e.g., `.bashrc` or `.zshrc`), so you don’t have to type `sudo systemctl` each time. Open your `.bashrc` or `.zshrc` file:

```bash
nano ~/.bashrc
```

Add the following aliases:

```bash
alias ollama-start="sudo systemctl start ollama.service"
alias ollama-stop="sudo systemctl stop ollama.service"
alias ollama-restart="sudo systemctl restart ollama.service"
alias ollama-status="sudo systemctl status ollama.service"
```

After adding these lines, source the file to activate the aliases:

```bash
source ~/.bashrc
```

Now, you can start, stop, restart, or check the status of the `ollama` service by running `ollama-start`, `ollama-stop`, `ollama-restart`, or `ollama-status`.

### 2. **Create a Script for Interactive Control**

If you prefer an interactive script, you can create one that provides an on/off toggle, restart, and status view in one go.

#### Step 1: Create the Script

Create a file named `ollama-control.sh` in your home directory:

```bash
nano ~/ollama-control.sh
```

Add the following script:

```bash
#!/bin/bash

echo "Choose an option for the ollama service:"
echo "1. Start"
echo "2. Stop"
echo "3. Restart"
echo "4. Status"
echo "5. Exit"

read -p "Enter your choice [1-5]: " choice

case $choice in
    1)
        sudo systemctl start ollama.service
        ;;
    2)
        sudo systemctl stop ollama.service
        ;;
    3)
        sudo systemctl restart ollama.service
        ;;
    4)
        sudo systemctl status ollama.service
        ;;
    5)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        ;;
esac
```

#### Step 2: Make the Script Executable

Make the script executable by running:

```bash
chmod +x ~/ollama-control.sh
```

#### Step 3: Run the Script

You can now run this script any time to interactively manage the service:

```bash
~/ollama-control.sh
```

### 3. **Optional: Add a Desktop Shortcut (for GUI)**

If you’re using a desktop environment (e.g., GNOME, KDE), you can create a desktop shortcut for even quicker access.

1. Create a `.desktop` file:
   ```bash
   nano ~/Desktop/OllamaControl.desktop
   ```

2. Add the following to the file:

   ```ini
   [Desktop Entry]
   Version=1.0
   Name=Ollama Control
   Comment=Start, stop, or restart the ollama service
   Exec=gnome-terminal -- ~/ollama-control.sh
   Icon=utilities-terminal
   Terminal=true
   Type=Application
   ```

3. Make the shortcut executable:

   ```bash
   chmod +x ~/Desktop/OllamaControl.desktop
   ```

Now, you have flexible and easy ways to control your `ollama` service!
