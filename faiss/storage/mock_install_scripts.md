### **Software Components Documentation**

The RAG system involves multiple components and dependencies. Here’s the breakdown:

---

#### **Core Software Dependencies**
1. **Operating System:**
   - Ubuntu 22.04 (assumed installed)

2. **Core System Tools:**
   - `curl`: For downloading packages.
   - `git`: For version control and repository cloning.

3. **NVIDIA Drivers & CUDA Toolkit:**
   - NVIDIA GPU drivers for hardware support.
   - CUDA Toolkit for GPU-accelerated computation.

4. **PDF Tools:**
   - `poppler-utils` (`pdftotext`) for PDF to text conversion.

5. **Python Environment:**
   - Python 3.10+ for all scripting needs.
   - Virtual environment (`venv`) for dependency isolation.

6. **Python Libraries:**
   - `faiss-gpu`: GPU-accelerated similarity search.
   - `langchain`: Framework for managing documents and queries.
   - `sentence-transformers`: For embeddings generation.
   - `ollama`: Integration with Ollama LLM runtime.
   - `torch`: Deep learning library for model inference.

7. **Optional Tools:**
   - `nvidia-smi`: For monitoring GPU usage.
   - `htop`: For monitoring system performance.

---

### **Bash Checklist**

The following is a **checklist** for verifying the environment setup:

```bash
#!/bin/bash

echo "=== RAG System Setup Checklist ==="

# Check OS
echo -n "Checking Ubuntu version... "
if [[ "$(lsb_release -rs)" == "22.04" ]]; then
  echo "OK"
else
  echo "FAIL (Requires Ubuntu 22.04)"
  exit 1
fi

# Check for GPU
echo -n "Checking NVIDIA GPU... "
if command -v nvidia-smi &> /dev/null; then
  echo "OK"
else
  echo "FAIL (NVIDIA drivers not installed)"
  exit 1
fi

# Check for CUDA Toolkit
echo -n "Checking CUDA Toolkit... "
if nvcc --version &> /dev/null; then
  echo "OK"
else
  echo "FAIL (CUDA Toolkit not installed)"
  exit 1
fi

# Check core tools
echo -n "Checking curl... "
command -v curl &> /dev/null && echo "OK" || echo "FAIL"

echo -n "Checking git... "
command -v git &> /dev/null && echo "OK" || echo "FAIL"

# Check Python
echo -n "Checking Python 3.10+... "
if python3 --version | grep -q "3.10"; then
  echo "OK"
else
  echo "FAIL (Requires Python 3.10+)"
  exit 1
fi

# Check poppler-utils
echo -n "Checking poppler-utils... "
if command -v pdftotext &> /dev/null; then
  echo "OK"
else
  echo "FAIL (poppler-utils not installed)"
  exit 1
fi

# Verify Python Libraries
echo "Checking Python libraries in venv..."
if [[ -d "rag-env" ]]; then
  source rag-env/bin/activate
  for lib in faiss-gpu langchain sentence-transformers ollama torch; do
    echo -n "Checking $lib... "
    pip show $lib &> /dev/null && echo "OK" || echo "FAIL"
  done
  deactivate
else
  echo "FAIL (venv 'rag-env' not set up)"
  exit 1
fi

echo "=== Checklist Complete ==="
```

---

### **Mock Install Script**

Here’s a **mock script** to install and configure all dependencies:

```bash
#!/bin/bash

echo "=== RAG System Install Script ==="

# Step 1: Update and Install Core System Tools
echo "Updating system and installing core tools..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git poppler-utils python3 python3-pip python3-venv

# Step 2: Install NVIDIA Drivers and CUDA Toolkit
echo "Installing NVIDIA drivers and CUDA Toolkit..."
sudo apt install -y nvidia-driver-525
sudo apt install -y nvidia-cuda-toolkit
echo "Verifying NVIDIA setup..."
nvidia-smi
nvcc --version

# Step 3: Set Up Python Virtual Environment
echo "Setting up Python virtual environment..."
python3 -m venv rag-env
source rag-env/bin/activate
pip install --upgrade pip

# Step 4: Install Python Libraries
echo "Installing Python libraries..."
pip install faiss-gpu langchain sentence-transformers ollama torch

# Step 5: Verify Installations
echo "Verifying installations..."
for lib in faiss-gpu langchain sentence-transformers ollama torch; do
  pip show $lib &> /dev/null && echo "$lib installed successfully" || echo "Error installing $lib"
done
deactivate

# Step 6: Finalize
echo "RAG System installation complete!"
echo "To start using the system:"
echo "1. Activate the virtual environment: source rag-env/bin/activate"
echo "2. Run your RAG scripts."
```

---

### **Next Steps**
1. Save the **checklist script** as `rag-checklist.sh` and the **install script** as `rag-install.sh`.
2. Make both scripts executable:
   ```bash
   chmod +x rag-checklist.sh rag-install.sh
   ```
3. Run the scripts sequentially to validate and install the environment:
   ```bash
   ./rag-checklist.sh
   ./rag-install.sh
   ```
4. Start developing and deploying your RAG system with confidence!
