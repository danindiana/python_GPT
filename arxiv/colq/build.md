To build a multimodal document retrieval system using ColQwen and Vespa on your Ubuntu 22.04 server with NVIDIA GPUs, follow these steps:

**1. System Preparation**

- **Update Packages**: Ensure your system is up-to-date.

  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

- **Install Essential Tools**:

  ```bash
  sudo apt install -y build-essential curl git python3-pip
  ```

**2. NVIDIA Driver and CUDA Toolkit**

- **Verify NVIDIA Driver**: Your `nvidia-smi` output indicates that the NVIDIA driver version 565.57.01 and CUDA version 12.7 are installed. Ensure these are compatible with your GPU models (RTX 3080 and RTX 3060).

- **Install CUDA Toolkit**: If not already installed, download and install the CUDA Toolkit 12.7 from NVIDIA's official website.

**3. Python Environment**

- **Install Python 3.8+**: Ubuntu 22.04 comes with Python 3.10 by default. Verify the version:

  ```bash
  python3 --version
  ```

- **Set Up Virtual Environment**:

  ```bash
  python3 -m venv colqwen-vespa-env
  source colqwen-vespa-env/bin/activate
  ```

- **Upgrade `pip`**:

  ```bash
  pip install --upgrade pip
  ```

**4. Install PyTorch with CUDA Support**

- **Install PyTorch**: Use the following command to install PyTorch with CUDA support:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu127
  ```

**5. Install ColQwen and Dependencies**

- **Clone ColQwen Repository**:

  ```bash
  git clone https://github.com/ColPali/ColQwen.git
  cd ColQwen
  ```

- **Install Dependencies**:

  ```bash
  pip install -r requirements.txt
  ```

**6. Prepare PDF Documents**

- **Convert PDFs to Images**: Use `pdf2image` to convert PDF pages into images.

  ```bash
  pip install pdf2image
  ```

  ```python
  from pdf2image import convert_from_path

  images = convert_from_path('document.pdf')
  for i, image in enumerate(images):
      image.save(f'page_{i + 1}.png')
  ```

**7. Generate Embeddings with ColQwen**

- **Load ColQwen Model**: Refer to the ColQwen documentation for model loading instructions.

- **Generate Embeddings**: Process the images to generate embeddings.

**8. Set Up Vespa**

- **Install Docker**: Vespa requires Docker.

  ```bash
  sudo apt install -y docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  ```

- **Run Vespa Container**:

  ```bash
  docker run -d --name vespa -p 8080:8080 vespaengine/vespa
  ```

- **Configure Vespa Schema**: Define your document schema and ranking profiles as per your application's requirements.

**9. Index Data into Vespa**

- **Prepare Data**: Structure your data in JSON format, including embeddings and metadata.

- **Feed Data**: Use Vespa's HTTP API to feed data.

  ```bash
  curl -X POST http://localhost:8080/document/v1/namespace/documenttype/docid -H "Content-Type: application/json" --data-binary @data.json
  ```

**10. Query Vespa**

- **Formulate Queries**: Use Vespa's query language to search your indexed documents.

- **Retrieve and Display Results**: Process and display the search results as needed.

**11. Interpretability with ColQwen**

- **Visualize Relevant Patches**: Utilize ColQwen's interpretability features to highlight relevant document sections.

For detailed instructions and code examples, refer to the original article: 

Ensure all software versions are compatible and test each component individually before integrating them into the full system. 
