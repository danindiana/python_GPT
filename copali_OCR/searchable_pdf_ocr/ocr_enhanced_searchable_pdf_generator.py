import os
import pytesseract
from pytesseract import Output
import fitz  # PyMuPDF for direct text extraction
import torch
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
import subprocess
import time
import signal

# Set TESSDATA_PREFIX to the correct directory for Tesseract 5.4.1
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'

# Verify TESSDATA_PREFIX and eng.traineddata file
tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
if not os.path.exists(tessdata_path):
    raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
    raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")

def preprocess_image_for_ocr(image):
    """Preprocess the image for better OCR accuracy."""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.autocontrast(image)  # Increase contrast
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary threshold
    return image

def extract_text_with_boxes(image):
    """Extract text with bounding boxes."""
    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        return data  # Dictionary containing text and bounding box information
    except pytesseract.pytesseract.TesseractError as e:
        print(f"Tesseract OCR error: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}

def create_searchable_pdf(original_pdf, ocr_data, output_pdf):
    """Embed OCR text into the original PDF to make it searchable."""
    doc = fitz.open(original_pdf)
    
    for page_num, page in enumerate(doc, start=1):
        page_text = []
        for i, word in enumerate(ocr_data['text']):
            if word.strip():  # Ignore empty text
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                rect = fitz.Rect(x, y, x + w, y + h)
                page.insert_textbox(rect, word, fontsize=10, overlay=False)
                page_text.append(word)
        print(f"Page {page_num} processed with {len(page_text)} OCR words.")

    doc.save(output_pdf)
    print(f"Searchable PDF saved to {output_pdf}")

def extract_images_and_create_searchable_pdf(pdf_path, output_pdf, resize_factor=2):
    """Extract OCR data and create a searchable PDF."""
    pdf = PdfDocument(pdf_path)
    ocr_data = {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}
    
    for page_number, page in enumerate(pdf):
        width, height = page.get_size()
        bitmap = page.render()
        pil_image = bitmap.to_pil()

        # Resize for OCR processing
        new_width = int(width // resize_factor)
        new_height = int(height // resize_factor)
        pil_image = pil_image.resize((new_width, new_height))

        processed_image = preprocess_image_for_ocr(pil_image)
        page_ocr_data = extract_text_with_boxes(processed_image)

        # Aggregate OCR data
        for key in ocr_data.keys():
            ocr_data[key].extend(page_ocr_data[key])

    # Create a searchable PDF
    create_searchable_pdf(pdf_path, ocr_data, output_pdf)

def get_gpu_info():
    """Fetch GPU information using nvidia-smi."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpus = []
    for line in output.strip().split('\n'):
        index, name, utilization = line.split(', ')
        gpus.append((int(index), name, int(utilization)))
    return gpus

def select_gpu(gpus):
    """Prompt the user to select a GPU."""
    print("Available GPUs:")
    for i, (index, name, utilization) in enumerate(gpus):
        print(f"{i + 1}. GPU {index}: {name} (Utilization: {utilization}%)")

    while True:
        try:
            selection = int(input("Select the GPU you wish to use (enter the corresponding number): "))
            if 1 <= selection <= len(gpus):
                return gpus[selection - 1][0]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Ask the user for input and output directories
input_dir = input("Enter the path of the target directory containing PDF files: ")
output_dir = input("Enter the path of the output directory for processed text files: ")

# Verify the directories exist
if not os.path.isdir(input_dir):
    print("The target directory does not exist.")
    exit()
if not os.path.isdir(output_dir):
    print("The output directory does not exist.")
    exit()

# Fetch GPU information and prompt user to select a GPU
gpus = get_gpu_info()
selected_gpu = select_gpu(gpus)
torch.cuda.set_device(selected_gpu)

# Process all PDF files in the target directory
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

if not pdf_files:
    print("No PDF files found in the specified directory.")
    exit()

# Initialize a list to store skipped files
skipped_files = []

# Process each PDF file in the input directory
for pdf_file in pdf_files:
    if len(pdf_file) > 200:
        print(f"Skipping file {pdf_file} due to file name length exceeding 200 characters.")
        skipped_files.append(pdf_file)
        continue

    pdf_path = os.path.join(input_dir, pdf_file)
    output_pdf = os.path.join(output_dir, f"{pdf_file}_searchable.pdf")

    print(f"Processing images for {pdf_file}...")
    try:
        # Set a timeout for the processing of each PDF file
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # Set timeout to 5 minutes (300 seconds)
        start_time = time.time()
        extract_images_and_create_searchable_pdf(pdf_path, output_pdf, resize_factor=2)
        end_time = time.time()
        print(f"Processing time for {pdf_file}: {end_time - start_time:.2f} seconds")
        signal.alarm(0)  # Cancel the alarm
    except TimeoutError:
        print(f"Timeout: Skipping file {pdf_file} due to long processing time.")
        skipped_files.append(pdf_file)
        continue
    except Exception as e:
        print(f"Error processing PDF {pdf_file}: {e}")
        skipped_files.append(pdf_file)
        continue

    print(f"\nSearchable PDF saved to {output_pdf}")

# Final memory cleanup
torch.cuda.empty_cache()

# Display the list of skipped files
if skipped_files:
    print("\nThe following files were skipped:")
    for skipped_file in skipped_files:
        print(skipped_file)
else:
    print("\nNo files were skipped.")
