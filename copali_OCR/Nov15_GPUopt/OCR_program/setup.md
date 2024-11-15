Here’s a step-by-step guide to set up and run your program on a fresh Ubuntu 22.04 system.

---

### **Step 1: Update the System**
Ensure the system is updated to the latest packages.
```bash
sudo apt update && sudo apt upgrade -y
```

---

### **Step 2: Install Required Dependencies**
Install essential tools and libraries:
```bash
sudo apt install -y python3 python3-venv python3-pip git curl build-essential \
                    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
                    nvidia-driver-535 nvidia-cuda-toolkit
```

- **Key Packages:**
  - `python3`, `python3-venv`, `python3-pip`: For Python and virtual environment management.
  - `git`: To clone repositories.
  - `tesseract-ocr`, `tesseract-ocr-eng`, `libtesseract-dev`: For OCR functionality.
  - `nvidia-driver-535`, `nvidia-cuda-toolkit`: NVIDIA driver and CUDA for GPU acceleration.

---

### **Step 3: Verify Installed Software**

#### Check Python Version
```bash
python3 --version
```
Ensure Python 3.10 or higher is installed.

#### Check NVIDIA Driver and CUDA
```bash
nvidia-smi
nvcc --version
```

#### Check Tesseract Installation
```bash
tesseract --version
```
Ensure it outputs the version and confirms that `eng.traineddata` is available.

---

### **Step 4: Create and Activate a Virtual Environment**
Navigate to the project directory or create a new one:
```bash
mkdir ~/programs/copali
cd ~/programs/copali
```

Set up and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### **Step 5: Clone or Copy the Project**
If hosted on GitHub:
```bash
git clone https://github.com/your-repo/copali.git .
```

If provided as a ZIP file, unzip the files:
```bash
unzip /path/to/copali.zip -d .
```

---

### **Step 6: Install Python Dependencies**
Install project dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### **Step 7: Configure Environment Variables**

#### Set `TESSDATA_PREFIX`
Update `TESSDATA_PREFIX` to point to the Tesseract data directory:
```bash
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"
```

For a permanent change, add this to `~/.bashrc`:
```bash
echo 'export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"' >> ~/.bashrc
source ~/.bashrc
```

#### (Optional) CUDA GPU Configuration
If the program requires specific CUDA GPUs, ensure the correct GPU is selected in the script:
```python
import torch
torch.cuda.set_device(0)  # Replace '0' with the target GPU ID
```

---

### **Step 8: Run the System Information Script**
Run the system information script to validate the setup:
```bash
python system_check.py
```
Verify the output for any warnings or errors.

---

### **Step 9: Run the Main Program**
Execute the program:
```bash
python copali_main.py
```

---

### **Step 10: Debugging Common Issues**

#### Virtual Environment Not Activated
If you encounter `ModuleNotFoundError` or missing dependencies, ensure the virtual environment is active:
```bash
source venv/bin/activate
```

#### CUDA Errors
If CUDA memory errors occur:
- Reduce batch size in the script:
  ```python
  batch_size = 1
  ```

#### Missing Dependencies
Reinstall the dependencies:
```bash
pip install --force-reinstall -r requirements.txt
```

---

### **Summary of Commands**
Here’s the consolidated list of commands:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git curl build-essential \
                    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
                    nvidia-driver-535 nvidia-cuda-toolkit
python3 --version
nvidia-smi
tesseract --version

mkdir ~/programs/copali
cd ~/programs/copali
python3 -m venv venv
source venv/bin/activate

git clone https://github.com/your-repo/copali.git .
pip install --upgrade pip
pip install -r requirements.txt

export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"
echo 'export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"' >> ~/.bashrc
source ~/.bashrc

python system_check.py
python copali_main.py
```

This setup ensures your program is ready to run on Ubuntu 22.04. Let me know if you need further assistance!
