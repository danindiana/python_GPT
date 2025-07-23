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
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np

# --- Configuration ---
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress MuPDF color space warnings
logging.getLogger("fitz").setLevel(logging.ERROR)

# --- Function Definitions ---

def check_pymupdf_version():
    """Check PyMuPDF version and capabilities"""
    try:
        version = fitz.version[0] if hasattr(fitz, 'version') else "Unknown"
        print(f"PyMuPDF version: {version}")

        # Test repair capability
        repair_support = False
        try:
            # Try to create a dummy document with repair parameter
            test_doc = fitz.open()
            test_doc.close()
            repair_support = True
        except:
            pass

        return {
            'version': version,
            'repair_support': repair_support
        }
    except Exception as e:
        print(f"Warning: Could not determine PyMuPDF capabilities: {e}")
        return {'version': 'Unknown', 'repair_support': False}

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

# LaTeX replacements dictionary
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
}

def clean_latex_text(text: str) -> str:
    """Convert LaTeX markup to readable plain text"""
    # Remove remaining LaTeX markup
    replacements = latex_replacements.copy()
    replacements.update({
        r'\$': '',  # remove dollar signs
        r'\\': '',  # remove remaining backslashes
        r'\{': '',  # remove remaining braces
        r'\}': '',  # remove remaining braces
    })

    # Apply replacements
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Clean up extra whitespace and formatting
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = re.sub(r'\(\s*\)', '', text)  # remove empty parentheses

    return text.strip()

def remove_ocr_contamination(text: str) -> str:
    """Remove OCR hallucinations like pinyin, CJK characters, and other script contamination"""
    
    # Define allowed character ranges for academic/technical documents
    allowed_patterns = [
        r'[\u0020-\u007F]',      # Basic ASCII (space to DEL)
        r'[\u00A0-\u00FF]',      # Latin-1 Supplement (accented chars)
        r'[\u0100-\u017F]',      # Latin Extended-A
        r'[\u0180-\u024F]',      # Latin Extended-B
        r'[\u1E00-\u1EFF]',      # Latin Extended Additional
        r'[\u2000-\u206F]',      # General Punctuation
        r'[\u2070-\u209F]',      # Superscripts and Subscripts
        r'[\u20A0-\u20CF]',      # Currency Symbols
        r'[\u2100-\u214F]',      # Letterlike Symbols
        r'[\u2150-\u218F]',      # Number Forms
        r'[\u2190-\u21FF]',      # Arrows
        r'[\u2200-\u22FF]',      # Mathematical Operators
        r'[\u2300-\u23FF]',      # Miscellaneous Technical
        r'[\u25A0-\u25FF]',      # Geometric Shapes
        r'[\u2600-\u26FF]',      # Miscellaneous Symbols
        r'[\u27C0-\u27EF]',      # Miscellaneous Mathematical Symbols-A
        r'[\u2980-\u29FF]',      # Miscellaneous Mathematical Symbols-B
        r'[\u2A00-\u2AFF]',      # Supplemental Mathematical Operators
        r'[\uFB00-\uFB4F]',      # Alphabetic Presentation Forms (ligatures)
    ]
    
    # Create pattern for allowed characters
    allowed_pattern = '|'.join(allowed_patterns)
    
    # Keep only allowed characters
    cleaned_chars = []
    for char in text:
        if re.match(f'^({allowed_pattern})$', char):
            cleaned_chars.append(char)
        else:
            # Replace problematic chars with space to avoid word concatenation
            cleaned_chars.append(' ')
    
    cleaned_text = ''.join(cleaned_chars)
    
    # Clean up excessive whitespace created by character removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # Max 2 consecutive newlines
    
    return cleaned_text.strip()

def clean_ocr_text(text: str, remove_latex: bool = False, filter_contamination: bool = True) -> str:
    """Apply symbol corrections and formatting fixes"""
    if remove_latex:
        text = clean_latex_text(text)
    
    for pattern, replacement in SYMBOL_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'(\d)\s*([+\-*/])\s*(\d)', r'\1\2\3', text)
    
    # Remove OCR contamination (pinyin, CJK characters, etc.)
    if filter_contamination:
        text = remove_ocr_contamination(text)
    
    return text.strip()

def find_pdf_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    """Find PDF files in directory with optional recursive search"""
    if recursive:
        pdf_files = list(input_dir.rglob("*.pdf"))
    else:
        pdf_files = list(input_dir.glob("*.pdf"))
    
    return sorted(pdf_files)

def prompt_user_confirmation(pdf_files: List[Path], recursive: bool) -> bool:
    """Prompt user for confirmation when processing files"""
    search_type = "recursively" if recursive else "in root directory only"
    
    print(f"\n=== PDF Processing Confirmation ===")
    print(f"Search mode: {search_type}")
    print(f"Files found: {len(pdf_files)}")
    
    if len(pdf_files) > 10:
        print("\nFirst 10 files:")
        for i, pdf_file in enumerate(pdf_files[:10]):
            print(f"  {i+1:2d}. {pdf_file.relative_to(pdf_files[0].parent.parent) if recursive else pdf_file.name}")
        print(f"  ... and {len(pdf_files) - 10} more files")
    else:
        print("\nFiles to process:")
        for i, pdf_file in enumerate(pdf_files):
            print(f"  {i+1:2d}. {pdf_file.relative_to(pdf_files[0].parent.parent) if recursive else pdf_file.name}")
    
    print(f"\nContinue processing? (y/N): ", end="")
    try:
        response = input().strip().lower()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return False

# --- OCR Model Classes ---

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

class Phi3VisionOCR:
    def __init__(self, device: str = "auto"):
        """Initialize Phi-3 Vision model for multimodal OCR"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            print(f"Loading Phi-3 Vision model on {self.device}...")
            
            model_id = "microsoft/Phi-3-vision-128k-instruct"
            
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                _attn_implementation="eager"  # More stable than flash attention
            )
            
            print(f"✓ Phi-3 Vision model loaded (~{self._estimate_vram():.1f}GB VRAM)")
            
        except ImportError as e:
            print("!! Phi-3 Vision dependencies missing. Install with:")
            print("pip install transformers torch torchvision")
            raise
        except Exception as e:
            print(f"!! Phi-3 Vision model loading failed: {e}")
            raise

    def _estimate_vram(self) -> float:
        """Estimate VRAM usage"""
        if torch.cuda.is_available() and "cuda" in self.device:
            device_id = int(self.device.split(':')[-1]) if ':' in self.device else 0
            return torch.cuda.memory_allocated(device_id) / 1024**3
        return 0.0

    def process_image(self, image: Image.Image) -> str:
        """Process image with Phi-3 Vision OCR"""
        try:
            # Prepare the prompt for OCR task
            messages = [
                {
                    "role": "user", 
                    "content": "<|image_1|>\nExtract all text from this image. Include equations, formulas, tables, and any other text content. Provide the text in a clear, readable format."
                }
            ]
            
            # Apply chat template
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                generation_args = {
                    "max_new_tokens": 1000,
                    "temperature": 0.1,
                    "do_sample": False,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                }
                
                generated_ids = self.model.generate(**inputs, **generation_args)
                
                # Decode response (remove input tokens)
                generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
                response = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            return response.strip()
            
        except Exception as e:
            return f"[Phi-3 Vision Error: {str(e)}]"

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

# --- Core Processing Functions ---

def get_page_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    """Get page image with fallback rendering for problematic PDFs"""
    try:
        # Primary rendering method
        pix = page.get_pixmap(dpi=dpi, alpha=False)
    except Exception as e:
        # Fallback methods for corrupted or problematic PDFs
        try:
            # Try with basic parameters only
            pix = page.get_pixmap(alpha=False)
        except Exception as e2:
            try:
                # Last resort - minimal parameters
                pix = page.get_pixmap()
            except Exception as e3:
                raise Exception(f"Failed to render page: Primary={e}, Fallback1={e2}, Fallback2={e3}")
        
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

def extract_page_content(page: fitz.Page, ocr_model=None, math_ocr: Optional[MathOCR] = None, 
                         use_ocr: bool = True, ocr_type: str = "nougat", clean_latex: bool = False,
                         filter_contamination: bool = True) -> str:
    """Extract content from PDF page using multiple methods"""
    
    # Method 1: Native PDF text extraction with fallback
    native_text = ""
    try:
        text_blocks = page.get_text("blocks", sort=True)
        for block in text_blocks:
            if len(block) >= 5:
                native_text += block[4] + "\n"
    except Exception:
        # Fallback to raw text extraction if blocks fail
        try:
            native_text = page.get_text("text")
        except Exception:
            native_text = ""
    
    # Method 2: OCR processing (if native text is insufficient)
    ocr_text = ""
    if use_ocr and ocr_model and len(native_text.strip()) < MIN_TEXT_LENGTH:
        try:
            # Convert page to image with appropriate DPI
            if ocr_type == "nougat":
                dpi = 300  # Nougat works better with higher DPI
            elif ocr_type == "phi3-vision":
                dpi = 200  # Phi-3 Vision is more flexible with resolution
            else:
                dpi = 200  # Default for TrOCR and others
                
            img = get_page_image(page, dpi)
            
            # Process with selected OCR model
            ocr_text = ocr_model.process_image(img)
            
        except Exception:
            ocr_text = ""
    
    # Method 3: Math OCR fallback for equations
    math_text = ""
    if math_ocr and any(w in native_text.lower() for w in ['theorem', 'proof', 'equation', 'formula']):
        try:
            # Use the same image generation function for consistency
            img = get_page_image(page, dpi=300)
            # Convert PIL image to bytes for math OCR
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            math_text = math_ocr.ocr_math(img_bytes)
        except Exception:
            math_text = ""
    
    # Combine results intelligently - clean text only
    result_parts = []
    
    # Use the best available text source
    if native_text.strip():
        cleaned_native = clean_ocr_text(native_text, remove_latex=clean_latex, filter_contamination=filter_contamination)
        if cleaned_native.strip():
            result_parts.append(cleaned_native)
    
    # Add OCR text if different and substantial
    if ocr_text.strip() and ocr_text != native_text.strip():
        cleaned_ocr = clean_ocr_text(ocr_text, remove_latex=clean_latex, filter_contamination=filter_contamination)
        if cleaned_ocr.strip() and cleaned_ocr not in str(result_parts):
            result_parts.append(cleaned_ocr)
    
    # Add math content if available
    if math_text.strip() and not any(error in math_text for error in ["[Math", "Error"]):
        cleaned_math = clean_ocr_text(math_text, remove_latex=clean_latex, filter_contamination=filter_contamination)
        if cleaned_math.strip():
            result_parts.append(cleaned_math)
    
    # Join with single newlines, no headers
    return "\n".join(result_parts).strip() if result_parts else ""

def create_output_path(pdf_path: Path, output_dir: Path, input_dir: Path, recursive: bool) -> Path:
    """Create appropriate output path maintaining directory structure for recursive processing"""
    if recursive:
        # Maintain relative directory structure
        relative_path = pdf_path.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        return output_subdir / f"{pdf_path.stem}.txt"
    else:
        return output_dir / f"{pdf_path.stem}.txt"

def process_pdf(pdf_path: Path, output_dir: Path, input_dir: Path, ocr_model=None, 
                math_ocr: Optional[MathOCR] = None, use_ocr: bool = True, 
                ocr_type: str = "nougat", clean_latex: bool = False, 
                recursive: bool = False, filter_contamination: bool = True) -> bool:
    """Process a single PDF file"""
    try:
        # Display relative path for better readability
        display_path = pdf_path.relative_to(input_dir) if recursive else pdf_path.name
        print(f"Processing: {display_path}")
        
        # Open PDF with compatibility across PyMuPDF versions
        try:
            # Convert Path to string for better compatibility
            pdf_str = str(pdf_path)
            
            # Try different opening methods based on PyMuPDF version
            doc = None
            
            # Method 1: Try with filetype parameter (newer versions)
            try:
                doc = fitz.open(pdf_str, filetype="pdf")
            except (TypeError, AttributeError):
                # Method 2: Basic open (older versions)
                doc = fitz.open(pdf_str)
            
            if doc is None:
                raise Exception("Failed to open PDF with all methods")
                
        except Exception as e:
            print(f"  ✗ Failed to open PDF: {e}")
            # Additional debugging info
            print(f"    File exists: {pdf_path.exists()}")
            print(f"    File size: {pdf_path.stat().st_size if pdf_path.exists() else 'N/A'} bytes")
            return False
        
        output_file = create_output_path(pdf_path, output_dir, input_dir, recursive)
        
        # Collect all content first, then write clean output
        all_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"  Page {page_num + 1}/{len(doc)}", end='\r')
            
            content = extract_page_content(page, ocr_model, math_ocr, use_ocr, ocr_type, clean_latex, filter_contamination)
            
            if content.strip():
                all_content.append(content.strip())
        
        # Write clean, concatenated content
        with open(output_file, 'w', encoding='utf-8') as f:
            if all_content:
                # Join pages with double newlines for readability
                f.write('\n\n'.join(all_content))
            else:
                f.write("")  # Empty file if no content extracted
        
        doc.close()
        output_relative = output_file.relative_to(output_dir) if recursive else output_file.name
        print(f"  ✓ Completed: {output_relative}          ")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdf_path.name}: {e}")
        return False

# --- Main Execution Block ---

def main():
    print("=== Advanced PDF OCR Processor ===")
    
    # Check PyMuPDF version and capabilities
    pymupdf_info = check_pymupdf_version()
    
    parser = argparse.ArgumentParser(
        description='Process PDFs with advanced OCR models (Phi-3 Vision, Nougat, TrOCR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDFs in current directory only
  python script.py ./pdfs ./output --ocr-model phi3-vision
  
  # Process PDFs recursively through all subdirectories
  python script.py ./pdfs ./output --recursive --ocr-model nougat
  
  # Process with user confirmation and LaTeX cleaning
  python script.py ./pdfs ./output --recursive --clean-latex --confirm
  
  # Disable contamination filtering (keep all characters including pinyin/CJK)
  python script.py ./pdfs ./output --no-filter-contamination
        """
    )
    
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('output_dir', help='Directory to save extracted text files')
    parser.add_argument('--ocr-model', choices=['nougat', 'trocr', 'phi3-vision', 'none'], default='phi3-vision',
                        help='OCR model to use (phi3-vision=multimodal, nougat=academic papers, trocr=general text)')
    parser.add_argument('--recursive', '-r', action='store_true', 
                        help='Process PDFs recursively in all subdirectories')
    parser.add_argument('--confirm', action='store_true', 
                        help='Prompt user for confirmation before processing (recommended for recursive mode)')
    parser.add_argument('--no-math', action='store_true', help='Disable math LaTeX OCR')
    parser.add_argument('--clean-latex', action='store_true', help='Convert LaTeX markup to plain text')
    parser.add_argument('--no-filter-contamination', action='store_true', 
                        help='Disable filtering of OCR contamination (pinyin, CJK characters, etc.)')
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
    print(f"Recursive: {'Yes' if args.recursive else 'No'}")
    print(f"Contamination Filtering: {'Enabled' if not args.no_filter_contamination else 'Disabled'}")
    print(f"LaTeX Cleaning: {'Enabled' if args.clean_latex else 'Disabled'}")
    
    # Set device with GPU selection
    if args.device == 'auto' and torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        print(f"GPU Available: True (using GPU {args.gpu_id})")
        print(f"VRAM Available: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.1f}GB")
    else:
        device = args.device
        print(f"GPU Available: {torch.cuda.is_available()}")
    
    # Find PDF files
    pdf_files = find_pdf_files(input_dir, args.recursive)
    if not pdf_files:
        search_type = "recursively" if args.recursive else "in root directory"
        print(f"✗ No PDF files found {search_type} in {input_dir}")
        sys.exit(1)
    
    # User confirmation
    if args.confirm:
        if not prompt_user_confirmation(pdf_files, args.recursive):
            print("Operation cancelled.")
            sys.exit(0)
    elif args.recursive and len(pdf_files) > 20:
        print(f"\nWarning: Found {len(pdf_files)} PDF files in recursive mode.")
        print("Consider using --confirm flag for large batches.")
        print("Continue? (y/N): ", end="")
        try:
            if input().strip().lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)
    
    print(f"\nFound {len(pdf_files)} PDF files")
    
    # Initialize OCR model
    ocr_model = None
    if args.ocr_model != 'none':
        try:
            if args.ocr_model == 'nougat':
                ocr_model = NougatOCR(device=device)
            elif args.ocr_model == 'trocr':
                ocr_model = TrOCR(device=device)
            elif args.ocr_model == 'phi3-vision':
                ocr_model = Phi3VisionOCR(device=device)
        except Exception as e:
            print(f"Warning: OCR model disabled due to error: {e}")
    
    # Initialize math OCR
    math_ocr = None
    if not args.no_math:
        try:
            math_ocr = MathOCR()
        except Exception as e:
            print(f"Warning: Math OCR disabled due to error: {e}")
    
    # Process files
    successful = 0
    failed = 0
    
    print(f"\n=== Processing Files ===")
    
    for i, pdf_file in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}]", end=" ")
        if process_pdf(pdf_file, output_dir, input_dir, ocr_model, math_ocr, 
                       args.ocr_model != 'none', args.ocr_model, args.clean_latex, 
                       args.recursive, not args.no_filter_contamination):
            successful += 1
        else:
            failed += 1
    
    print(f"\n\n=== Processing Complete ===")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Output directory: {output_dir}")
    
    if args.recursive and successful > 0:
        print("\nNote: Directory structure preserved in output folder")

if __name__ == '__main__':
    main()
