import os
import sys
import io
import fitz  # PyMuPDF
from PIL import Image
import torch
import pytesseract
import re
import warnings
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np

# --- Configuration ---
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# OCR Model Configuration
NOUGAT_MODEL_ID = "facebook/nougat-base"  # Optimized for academic papers
TROCR_MODEL_ID = "microsoft/trocr-large-printed"
GOT_MODEL_ID = "stepfun-ai/GOT-OCR2_0"

MIN_TEXT_LENGTH = 50
TESSERACT_CONFIG = r'--psm 6 -l eng+equ'

# Symbol correction mapping
SYMBOL_MAP = {
    r'D([^θ]|$)': r'Dθ\1',
    r'dn\(': r'dη(',
    r'C\s': '⊆ ',
    r'€': '∈',
    r'Pmono': 'Pₘₒₙₒ',
    r'X([0-9]+)': r'X_{\1}',
    r'>>': '≫',
    r'\.\.\.': '···',
    r'\\theta': 'θ',
    r'\\eta': 'η'
}

class NougatOCR:
    def __init__(self, device: str = "auto"):
        """Initialize Nougat model for academic paper OCR"""
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            print(f"Loading Nougat model on {self.device}...")
            
            self.processor = NougatProcessor.from_pretrained(NOUGAT_MODEL_ID)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                NOUGAT_MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            print(f"✓ Nougat model loaded (~{self._estimate_vram():.1f}GB VRAM)")
            
        except ImportError as e:
            print("!! Nougat dependencies missing. Install with:")
            print("pip install transformers torch torchvision")
            raise
        except Exception as e:
            print(f"!! Nougat model loading failed: {e}")
            raise

    def _estimate_vram(self) -> float:
        """Estimate VRAM usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**3
        return 0.0

    def process_image(self, image: Image.Image) -> str:
        """Process image with Nougat OCR"""
        try:
            # Preprocess image for Nougat
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.model.decoder.config.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                )
            
            # Decode output
            sequence = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            sequence = self.processor.post_process_generation(sequence, fix_markdown=False)
            
            return sequence.strip()
            
        except Exception as e:
            return f"[Nougat Error: {str(e)}]"

class TrOCR:
    def __init__(self, device: str = "auto"):
        """Initialize TrOCR model for general text OCR"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            print(f"Loading TrOCR model on {self.device}...")
            
            self.processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                TROCR_MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            print(f"✓ TrOCR model loaded (~{self._estimate_vram():.1f}GB VRAM)")
            
        except Exception as e:
            print(f"!! TrOCR loading failed: {e}")
            raise

    def _estimate_vram(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**3
        return 0.0

    def process_image(self, image: Image.Image) -> str:
        """Process image with TrOCR"""
        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
            
        except Exception as e:
            return f"[TrOCR Error: {str(e)}]"

class MathOCR:
    def __init__(self):
        """Initialize math OCR (fallback for equations)"""
        try:
            from pix2tex.cli import LatexOCR
            self.latex_model = LatexOCR()
            print("✓ LaTeX OCR initialized")
        except ImportError:
            print("!! pix2tex not available - math OCR disabled")
            self.latex_model = None

    def is_math_region(self, image: Image.Image) -> bool:
        """Detect if image contains math equations"""
        grayscale = image.convert('L')
        hist = grayscale.histogram()
        return sum(hist[-50:])/sum(hist) > 0.7

    def ocr_math(self, image_bytes: bytes) -> str:
        """Process math equations"""
        if not self.latex_model:
            return "[Math OCR unavailable]"
            
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if self.is_math_region(img):
                result = self.latex_model(img)
                return f"\\[{result}\\]"
            return pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        except Exception as e:
            return f"[Math OCR Error: {str(e)}]"

def clean_latex_text(text: str) -> str:
    """Convert LaTeX markup to readable plain text"""
    # Common LaTeX commands to plain text
    latex_replacements = {
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',  # fractions
        r'\\sqrt\{([^}]+)\}': r'√(\1)',  # square root
        r'\\overline\{([^}]+)\}': r'\1̄',  # overline
        r'\\underline\{([^}]+)\}': r'\1_',  # underline
        r'\\mathbf\{([^}]+)\}': r'\1',  # bold (remove formatting)
        r'\\mathrm\{([^}]+)\}': r'\1',  # roman (remove formatting)
        r'\\mathit\{([^}]+)\}': r'\1',  # italic (remove formatting)
        r'\\text\{([^}]+)\}': r'\1',  # text mode
        r'\\left\(': '(',  # left parenthesis
        r'\\right\)': ')',  # right parenthesis
        r'\\left\[': '[',  # left bracket
        r'\\right\]': ']',  # right bracket
        r'\\left\{': '{',  # left brace
        r'\\right\}': '}',  # right brace
        r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
        r'\\epsilon': 'ε', r'\\theta': 'θ', r'\\lambda': 'λ', r'\\mu': 'μ',
        r'\\pi': 'π', r'\\sigma': 'σ', r'\\tau': 'τ', r'\\phi': 'φ',
        r'\\omega': 'ω', r'\\Omega': 'Ω', r'\\infty': '∞',
        r'\\sum': '∑', r'\\prod': '∏', r'\\int': '∫',
        r'\\leq': '≤', r'\\geq': '≥', r'\\neq': '≠',
        r'\\approx': '≈', r'\\sim': '~', r'\\equiv': '≡',
        r'\\in': '∈', r'\\subset': '⊂', r'\\subseteq': '⊆',
        r'\\cup': '∪', r'\\cap': '∩', r'\\emptyset': '∅',
        r'\\cdot': '·', r'\\times': '×', r'\\div': '÷',
        r'\\pm': '±', r'\\mp': '∓',
        r'\

def extract_page_content(page: fitz.Page, ocr_model=None, math_ocr: Optional[MathOCR] = None, 
                        use_ocr: bool = True, ocr_type: str = "nougat", clean_latex: bool = False) -> str:
    """Extract content from PDF page using multiple methods"""
    
    # Method 1: Native PDF text extraction
    text_blocks = page.get_text("blocks", sort=True)
    native_text = ""
    for block in text_blocks:
        if len(block) >= 5:
            native_text += block[4] + "\n"
    
    # Method 2: OCR processing (if native text is insufficient)
    ocr_text = ""
    if use_ocr and ocr_model and len(native_text.strip()) < MIN_TEXT_LENGTH:
        try:
            # Convert page to image
            dpi = 300 if ocr_type == "nougat" else 200  # Nougat works better with higher DPI
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # Process with selected OCR model
            ocr_text = ocr_model.process_image(img)
            
        except Exception as e:
            ocr_text = f"[OCR Error: {str(e)}]"
    
    # Method 3: Math OCR fallback for equations
    math_text = ""
    if math_ocr and any(w in native_text.lower() for w in ['theorem', 'proof', 'equation', 'formula']):
        try:
            pix = page.get_pixmap(dpi=300, alpha=False)
            img_bytes = pix.tobytes("png")
            math_text = math_ocr.ocr_math(img_bytes)
        except Exception as e:
            math_text = f"[Math Error: {str(e)}]"
    
    # Combine results intelligently
    result = ""
    
    if native_text.strip():
        result += f"=== Native PDF Text ===\n{clean_ocr_text(native_text, remove_latex=clean_latex)}\n\n"
    
    if ocr_text.strip() and ocr_text != native_text.strip():
        model_name = ocr_type.title()
        cleaned_ocr = clean_ocr_text(ocr_text, remove_latex=clean_latex) if clean_latex else ocr_text
        result += f"=== {model_name} OCR ===\n{cleaned_ocr}\n\n"
    
    if math_text.strip() and "[Math OCR" not in math_text:
        cleaned_math = clean_ocr_text(math_text, remove_latex=clean_latex) if clean_latex else math_text
        result += f"=== Math LaTeX ===\n{cleaned_math}\n\n"
    
    return result if result.strip() else "[No content extracted]"

def process_pdf(pdf_path: Path, output_dir: Path, ocr_model=None, math_ocr: Optional[MathOCR] = None, 
               use_ocr: bool = True, ocr_type: str = "nougat", clean_latex: bool = False) -> bool:
    """Process a single PDF file"""
    try:
        print(f"Processing: {pdf_path.name}")
        
        doc = fitz.open(pdf_path)
        output_file = output_dir / f"{pdf_path.stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {pdf_path.name} ===\n")
            f.write(f"Pages: {len(doc)}\n")
            f.write(f"OCR Model: {ocr_type.title() if use_ocr and ocr_model else 'Disabled'}\n")
            f.write(f"LaTeX Cleaning: {'Enabled' if clean_latex else 'Disabled'}\n\n")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"  Page {page_num + 1}/{len(doc)}", end='\r')
                
                content = extract_page_content(page, ocr_model, math_ocr, use_ocr, ocr_type, clean_latex)
                
                if content.strip():
                    f.write(f"{'='*50}\n")
                    f.write(f"PAGE {page_num + 1}\n")
                    f.write(f"{'='*50}\n")
                    f.write(content)
                    f.write("\n\n")
        
        doc.close()
        print(f"  ✓ Completed: {output_file}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdf_path}: {e}")
        return False

def main():
    print("=== Academic PDF OCR Processor ===")
    
    parser = argparse.ArgumentParser(description='Process PDFs with specialized OCR models')
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('output_dir', help='Directory to save extracted text files')
    parser.add_argument('--ocr-model', choices=['nougat', 'trocr', 'none'], default='nougat',
                       help='OCR model to use (nougat=academic papers, trocr=general text)')
    parser.add_argument('--no-math', action='store_true', help='Disable math LaTeX OCR')
    parser.add_argument('--clean-latex', action='store_true', help='Convert LaTeX markup to plain text')
    parser.add_argument('--device', default='auto', help='Device for OCR model (cuda/cpu/auto)')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (0 or 1)')
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"✗ Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Set device with GPU selection
    if args.device == 'auto' and torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        print(f"GPU Available: True (using GPU {args.gpu_id})")
        print(f"VRAM Available: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.1f}GB")
    else:
        device = args.device
        print(f"GPU Available: {torch.cuda.is_available()}")
    
    # Initialize OCR model
    ocr_model = None
    if args.ocr_model != 'none':
        try:
            if args.ocr_model == 'nougat':
                ocr_model = NougatOCR(device=device)
            elif args.ocr_model == 'trocr':
                ocr_model = TrOCR(device=device)
        except Exception as e:
            print(f"Warning: OCR model disabled due to error: {e}")
    
    # Initialize math OCR
    math_ocr = None
    if not args.no_math:
        try:
            math_ocr = MathOCR()
        except Exception as e:
            print(f"Warning: Math OCR disabled due to error: {e}")
    
    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"✗ No PDF files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        if process_pdf(pdf_file, output_dir, ocr_model, math_ocr, 
                      args.ocr_model != 'none', args.ocr_model, args.clean_latex):
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main(): '',  # remove dollar signs
        r'\\': '',  # remove remaining backslashes
        r'\{': '', r'\}': '',  # remove remaining braces
    }
    
    # Apply replacements
    for pattern, replacement in latex_replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Clean up extra whitespace and formatting
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = re.sub(r'\(\s*\)', '', text)  # remove empty parentheses
    
    return text.strip()

def clean_ocr_text(text: str, remove_latex: bool = False) -> str:
    """Apply symbol corrections and formatting fixes"""
    if remove_latex:
        text = clean_latex_text(text)
    
    for pattern, replacement in SYMBOL_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'(\d)\s*([+\-*/])\s*(\d)', r'\1\2\3', text)
    return text.strip()

def extract_page_content(page: fitz.Page, ocr_model=None, math_ocr: Optional[MathOCR] = None, 
                        use_ocr: bool = True, ocr_type: str = "nougat") -> str:
    """Extract content from PDF page using multiple methods"""
    
    # Method 1: Native PDF text extraction
    text_blocks = page.get_text("blocks", sort=True)
    native_text = ""
    for block in text_blocks:
        if len(block) >= 5:
            native_text += block[4] + "\n"
    
    # Method 2: OCR processing (if native text is insufficient)
    ocr_text = ""
    if use_ocr and ocr_model and len(native_text.strip()) < MIN_TEXT_LENGTH:
        try:
            # Convert page to image
            dpi = 300 if ocr_type == "nougat" else 200  # Nougat works better with higher DPI
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # Process with selected OCR model
            ocr_text = ocr_model.process_image(img)
            
        except Exception as e:
            ocr_text = f"[OCR Error: {str(e)}]"
    
    # Method 3: Math OCR fallback for equations
    math_text = ""
    if math_ocr and any(w in native_text.lower() for w in ['theorem', 'proof', 'equation', 'formula']):
        try:
            pix = page.get_pixmap(dpi=300, alpha=False)
            img_bytes = pix.tobytes("png")
            math_text = math_ocr.ocr_math(img_bytes)
        except Exception as e:
            math_text = f"[Math Error: {str(e)}]"
    
    # Combine results intelligently
    result = ""
    
    if native_text.strip():
        result += f"=== Native PDF Text ===\n{clean_ocr_text(native_text)}\n\n"
    
    if ocr_text.strip() and ocr_text != native_text.strip():
        model_name = ocr_type.title()
        result += f"=== {model_name} OCR ===\n{ocr_text}\n\n"
    
    if math_text.strip() and "[Math OCR" not in math_text:
        result += f"=== Math LaTeX ===\n{math_text}\n\n"
    
    return result if result.strip() else "[No content extracted]"

def process_pdf(pdf_path: Path, output_dir: Path, ocr_model=None, math_ocr: Optional[MathOCR] = None, 
               use_ocr: bool = True, ocr_type: str = "nougat") -> bool:
    """Process a single PDF file"""
    try:
        print(f"Processing: {pdf_path.name}")
        
        doc = fitz.open(pdf_path)
        output_file = output_dir / f"{pdf_path.stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {pdf_path.name} ===\n")
            f.write(f"Pages: {len(doc)}\n")
            f.write(f"OCR Model: {ocr_type.title() if use_ocr and ocr_model else 'Disabled'}\n\n")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"  Page {page_num + 1}/{len(doc)}", end='\r')
                
                content = extract_page_content(page, ocr_model, math_ocr, use_ocr, ocr_type)
                
                if content.strip():
                    f.write(f"{'='*50}\n")
                    f.write(f"PAGE {page_num + 1}\n")
                    f.write(f"{'='*50}\n")
                    f.write(content)
                    f.write("\n\n")
        
        doc.close()
        print(f"  ✓ Completed: {output_file}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdf_path}: {e}")
        return False

def main():
    print("=== Academic PDF OCR Processor ===")
    
    parser = argparse.ArgumentParser(description='Process PDFs with specialized OCR models')
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('output_dir', help='Directory to save extracted text files')
    parser.add_argument('--ocr-model', choices=['nougat', 'trocr', 'none'], default='nougat',
                       help='OCR model to use (nougat=academic papers, trocr=general text)')
    parser.add_argument('--no-math', action='store_true', help='Disable math LaTeX OCR')
    parser.add_argument('--device', default='auto', help='Device for OCR model (cuda/cpu/auto)')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (0 or 1)')
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"✗ Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Set device with GPU selection
    if args.device == 'auto' and torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        print(f"GPU Available: True (using GPU {args.gpu_id})")
        print(f"VRAM Available: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.1f}GB")
    else:
        device = args.device
        print(f"GPU Available: {torch.cuda.is_available()}")
    
    # Initialize OCR model
    ocr_model = None
    if args.ocr_model != 'none':
        try:
            if args.ocr_model == 'nougat':
                ocr_model = NougatOCR(device=device)
            elif args.ocr_model == 'trocr':
                ocr_model = TrOCR(device=device)
        except Exception as e:
            print(f"Warning: OCR model disabled due to error: {e}")
    
    # Initialize math OCR
    math_ocr = None
    if not args.no_math:
        try:
            math_ocr = MathOCR()
        except Exception as e:
            print(f"Warning: Math OCR disabled due to error: {e}")
    
    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"✗ No PDF files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        if process_pdf(pdf_file, output_dir, ocr_model, math_ocr, 
                      args.ocr_model != 'none', args.ocr_model):
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
