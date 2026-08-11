# Extraction Module (`src/extraction/`)

The extraction module is responsible for reading raw Nepali PDF files and converting them into structured, hierarchical JSON datasets.

---

## 1. `extractor.py`

**Purpose**: Handles the heavy lifting of reading PDFs and performing Optical Character Recognition (OCR) on the pages.

**Key Components**:
- **`PdfExtractor` Class**: A context manager (`__enter__` / `__exit__`) that wraps the `PyMuPDF` document object.
- **`_preprocess_for_ocr(page)`**: The most critical function for OCR accuracy. It:
  - Renders the PDF page at 300 DPI for high resolution.
  - Converts the image to grayscale.
  - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to make the text stand out.
  - Applies Otsu's Binarization to make the image pure black and white.
  - Applies a Median Blur to remove background noise.
- **`_ocr_page(page)`**: Passes the preprocessed image to Tesseract OCR using the specific `nep` (Nepali) language pack.

**Dependencies**: `pymupdf` (fitz), `pytesseract`, `cv2` (OpenCV), `numpy`.

---

## 2. `cleaner.py`

**Purpose**: Normalizes the raw text output from Tesseract.

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
