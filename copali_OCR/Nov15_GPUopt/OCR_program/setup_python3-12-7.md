Here's the updated guide for **Python 3.12.7** on **Ubuntu 22.04**.

---

### **Step 1: Update the System**
Ensure the system is updated to the latest packages:
```bash
sudo apt update && sudo apt upgrade -y
```

---

### **Step 2: Install Required Dependencies**
Install system libraries and tools:
```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip git curl build-essential \
                    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
                    nvidia-driver-535 nvidia-cuda-toolkit
```

- **Key Packages:**
  - `python3.12`, `python3.12-venv`, `python3.12-dev`: For Python 3.12.7 support.
  - `python3-pip`: Installs Python packages.
  - `git`, `curl`, `build-essential`: For cloning repositories and compiling extensions.
  - `tesseract-ocr`, `tesseract-ocr-eng`, `libtesseract-dev`: OCR functionality.
  - `nvidia-driver-535`, `nvidia-cuda-toolkit`: NVIDIA drivers and CUDA support.

---

### **Step 3: Verify Installed Software**

#### Check Python Version
Ensure Python 3.12.7 is installed:
```bash
python3.12 --version
```

#### Check NVIDIA Driver and CUDA
Verify the NVIDIA drivers and CUDA:
```bash
nvidia-smi
nvcc --version
```

#### Check Tesseract Installation
Ensure Tesseract OCR is installed:
```bash
tesseract --version
```

---

### **Step 4: Create and Activate a Virtual Environment**
Navigate to your project directory or create a new one:
```bash
mkdir ~/programs/copali
cd ~/programs/copali
```

Set up and activate a virtual environment using Python 3.12:
```bash
python3.12 -m venv venv
source venv/bin/activate
```

---

### **Step 5: Clone or Copy the Project**
If the project is hosted on GitHub, clone the repository:
```bash
git clone https://github.com/your-repo/copali.git .
```

If provided as a ZIP file, extract the files:
```bash
unzip /path/to/copali.zip -d .
```

---

### **Step 6: Install Python Dependencies**
Upgrade `pip` and install required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### **Step 7: Configure Environment Variables**

#### Set `TESSDATA_PREFIX`
Configure the `TESSDATA_PREFIX` environment variable:
```bash
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"
```

Make it permanent by adding it to your shell configuration file:
```bash
echo 'export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"' >> ~/.bashrc
source ~/.bashrc
```

#### (Optional) CUDA GPU Configuration
If the program uses CUDA, ensure the correct GPU is selected in the script:
```python
import torch
torch.cuda.set_device(0)  # Replace '0' with the target GPU ID
```

---

### **Step 8: Run the System Information Script**
Run the `system_check.py` script to validate your system configuration:
```bash
python system_check.py
```

Review the output for any errors or warnings.

---

### **Step 9: Run the Main Program**
Execute the main program:
```bash
python copali_main.py
```

---

### **Step 10: Debugging Common Issues**

#### Virtual Environment Not Activated
If you encounter `ModuleNotFoundError`, ensure the virtual environment is active:
```bash
source venv/bin/activate
```

#### Missing Dependencies
If packages are missing or out of date:
```bash
pip install --force-reinstall -r requirements.txt
```

#### CUDA Memory Errors
To address memory issues:
- Reduce batch size in the script:
  ```python
  batch_size = 1
  ```

---

### **Consolidated Commands**
Here’s the complete list of commands for Python 3.12.7:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip git curl build-essential \
                    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
                    nvidia-driver-535 nvidia-cuda-toolkit

python3.12 --version
nvidia-smi
nvcc --version
tesseract --version

mkdir ~/programs/copali
cd ~/programs/copali
python3.12 -m venv venv
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

This setup ensures that your program is ready to run on a host with Python 3.12.7 and Ubuntu 22.04. Let me know if further clarification is needed! 🚀
