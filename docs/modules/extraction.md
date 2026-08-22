# Extraction Module (`src/extraction/`)

The extraction module is responsible for reading raw Nepali PDF files and converting them into structured, hierarchical JSON datasets.

---

## 1. `extractor.py`

**Purpose**: Handles the heavy lifting of reading PDFs and performing Optical Character Recognition (OCR) on the pages.

**Key Components**:
- **`PdfExtractor` Class**: A context manager (`__enter__` / `__exit__`) that wraps the `PyMuPDF` document object.
- **`_is_valid_nepali(text)`**: Validates native-extracted text using Devanagari character ratio and known legacy-font garbage patterns. If the text is invalid, the page is sent to Surya OCR.
- **`_load_surya_models()`**: Lazily loads Surya detection and recognition models into GPU on the first OCR page. Digital PDFs never pay the GPU warm-up cost.
- **`_ocr_page(page)`**: Renders the page at 300 DPI, crops the header URL strip, and runs **Surya OCR** with Nepali (`ne`) language. Surya uses deep-learning-based layout detection + LSTM recognition — significantly more accurate than Tesseract on scanned Devanagari.
- **`close()`**: Deletes Surya model references and calls `torch.cuda.empty_cache()` to free VRAM after extraction.

**VRAM tuning** (RTX 3050 4 GB) is controlled by environment variables set in the Dockerfile and `compose.yaml`:
- `RECOGNITION_BATCH_SIZE=2`
- `DETECTOR_BATCH_SIZE=2`
- `TORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Dependencies**: `pymupdf` (fitz), `surya-ocr`, `torch` (CUDA 12.1), `pillow`.

---

## 2. `cleaner.py`

**Purpose**: Normalizes the raw text output from Surya OCR.

**Key Components**:
- **`TextCleaner` Class**: Contains regex patterns to fix common OCR mistakes.
- **`clean(text)`**: 
  - Removes repeating whitespace and newlines.
  - Fixes broken Devanagari Unicode sequences (e.g., dangling matras).
  - Normalizes different types of hyphens and quotes.

---

## 3. `formatter.py`

**Purpose**: Parses the cleaned, flat text line-by-line to build a nested hierarchy (Chapter -> Section -> Sub-section -> Clause).

**Key Components**:
- **`RegexFormatter` Class**: Maintains state variables (e.g., `current_ch`, `current_sec`) as it iterates over the text.
- **Hierarchy Detection**: Uses specific Devanagari regex patterns to identify:
  - `परिच्छेद` (Chapter)
  - `दफा` (Section)
  - `उपदफा` (Sub-section)
  - `खण्ड` (Clause)
- **Proviso & Explanation Splitting**: 
  - Uses regex to safely split `तर` (Provisos) and `स्पष्टीकरणः` (Explanations) from the main clause text. It handles multiple provisos per clause.
- **Token Estimation**: Estimates token count for Devanagari using a character-length heuristic (`len // 3`) since standard word counts are inaccurate for this script.
