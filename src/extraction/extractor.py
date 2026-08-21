import os
import re
import logging
import fitz
import pytesseract
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# Devanagari Unicode range: U+0900 – U+097F
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Garbage patterns commonly seen when PyMuPDF extracts text from
# Nepali PDFs that use non-Unicode legacy fonts or glyph-mapped fonts.
# These produce runs of random Latin chars, PUA codepoints, or CID numbers.
_GARBAGE_PATTERNS = [
    re.compile(r'[\uE000-\uF8FF]{3,}'),          # Private Use Area runs
    re.compile(r'(?:\d{3,}\s*){3,}'),              # runs of bare numbers
    re.compile(r'[A-Za-z]{10,}'),                  # long Latin runs (not Nepali)
    re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]'),  # control characters
]


# Legacy font (Preeti/Kantipur etc) scrambled words commonly extracted via PyMuPDF
# These look like Devanagari but are mapped incorrectly by the PDF's internal font table.
_LEGACY_FONT_SCRAMBLED_WORDS = [
    'उद्योि',       # उद्योग
    'दतान',         # दर्ता
    'बमोशजम',       # बमोजिम
    'िन्नाले',      # भन्नाले
    'गनयम',         # नियम
    'पछन',          # पर्छ
    'लिानी',        # लगानी
    'प्रगतस्पर्धी',   # प्रतिस्पर्धी
    'तोद्धकएको',      # तोकिएको
    'लेशखएको',      # लेखिएको
    'प्रारम्ि',     # प्रारम्भ
    'सम्बन्त्र्धी',   # सम्बन्धी
]


def _is_valid_nepali(text: str, min_devanagari_ratio: float = 0.3) -> bool:
    """
    Check whether extracted text is valid Nepali (Devanagari) content.

    Returns False if:
      - Text is too short (< 20 non-whitespace chars)
      - Devanagari characters make up less than `min_devanagari_ratio` of
        the non-whitespace content (indicates legacy-font garbage)
      - Known garbage patterns are dominant
      - Legacy font scrambled words are present
    """
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    # Count non-whitespace characters
    non_ws = re.sub(r'\s', '', stripped)
    if len(non_ws) < 20:
        return False

    # Ratio check: what fraction of non-whitespace chars are Devanagari?
    devanagari_count = len(_DEVANAGARI_RE.findall(non_ws))
    ratio = devanagari_count / len(non_ws)

    if ratio < min_devanagari_ratio:
        return False

    # Check for garbage patterns — if any pattern has many matches,
    # the text is likely from a legacy-font PDF
    for pattern in _GARBAGE_PATTERNS:
        matches = pattern.findall(non_ws)
        if len(matches) > 5:
            return False

    # Check for legacy font scrambled words
    scrambled_count = sum(1 for word in _LEGACY_FONT_SCRAMBLED_WORDS if word in text)
    if scrambled_count >= 2:  # If at least 2 scrambled words are found, it's a corrupted PDF
        logger.debug("Legacy font scrambled text detected (found %d corrupted words), forcing OCR.", scrambled_count)
        return False

    return True


class PdfExtractor:
    def __init__(self, pdf_path, tesseract_cmd=None):
        self.pdf_path = pdf_path
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.name == 'nt' and os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.doc = fitz.open(pdf_path)

        # Track extraction method stats
        self._stats = {"digital": 0, "ocr": 0, "total": 0}

    def extract_all_pages(self):
        """
        Yields (page_number, text) for each page.

        Smart extraction strategy:
          1. Try PyMuPDF native text extraction (fast, no OCR needed)
          2. Validate the result is actually Nepali/Devanagari text
          3. If garbage (legacy fonts, glyph-mapped, empty), fall back to OCR
        """
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            self._stats["total"] += 1

            text = self._extract_page(page, page_num + 1)
            yield page_num + 1, text

        # Log extraction method stats
        logger.info(
            "Extraction complete: %d pages total — %d digital, %d OCR",
            self._stats["total"], self._stats["digital"], self._stats["ocr"]
        )

    def _extract_page(self, page, page_num: int) -> str:
        """
        Smart per-page extraction: try native text first, fall back to OCR.
        """
        # Attempt 1: Native text extraction (works for digital/text-layer PDFs)
        native_text = page.get_text("text")

        if _is_valid_nepali(native_text):
            self._stats["digital"] += 1
            logger.debug("Page %d: native text extraction (digital)", page_num)
            return native_text

        # Attempt 2: Fall back to OCR for scanned pages or garbage-Unicode pages
        logger.debug("Page %d: native text invalid/empty, falling back to OCR", page_num)
        self._stats["ocr"] += 1
        return self._ocr_page(page)

    def _preprocess_for_ocr(self, img):
        """
        Preprocess a scanned page image for optimal Devanagari OCR.
        Steps:
          1. Convert to grayscale
          2. Enhance contrast (autocontrast normalizes histogram)
          3. Slight sharpening to restore edges lost in scanning
          4. Binarize with a threshold to produce clean black text on white
          5. Median filter to remove salt-and-pepper scan noise
        """
        # 1. Grayscale
        gray = img.convert('L')

        # 2. Autocontrast — stretches the histogram to use the full 0-255 range,
        #    which helps with faded scans or uneven lighting.
        gray = ImageOps.autocontrast(gray, cutoff=1)

        # 3. Sharpen to recover edge detail (Devanagari has fine horizontal
        #    headline strokes that scanners often blur).
        gray = gray.filter(ImageFilter.SHARPEN)

        # 4. Binarize — Otsu-style: compute a threshold from the image histogram.
        #    This is critical for scanned images with varying background intensity.
        histogram = gray.histogram()
        total_pixels = sum(histogram)
        current_sum = 0
        weight_bg = 0
        sum_bg = 0
        max_variance = 0
        threshold = 128  # fallback

        total_intensity = sum(i * histogram[i] for i in range(256))

        for t in range(256):
            weight_bg += histogram[t]
            if weight_bg == 0:
                continue
            weight_fg = total_pixels - weight_bg
            if weight_fg == 0:
                break

            sum_bg += t * histogram[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (total_intensity - sum_bg) / weight_fg

            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if variance > max_variance:
                max_variance = variance
                threshold = t

        binary = gray.point(lambda x: 255 if x > threshold else 0, '1')

        # 5. Median filter to remove small scan noise specks
        binary = binary.filter(ImageFilter.MedianFilter(size=3))

        return binary

    
    def _ocr_page(self, page):
        """Renders the page to an image, crops the header region, preprocesses, and performs OCR.
        """
        # High DPI (300) for better OCR accuracy on scanned documents
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Crop top header (URL strip) — consistent across all pages/PDFs
        HEADER_CROP_PCT = 0.07  # top 7% of page height
        crop_y = int(img.height * HEADER_CROP_PCT)
        img = img.crop((0, crop_y, img.width, img.height))

        # Preprocess the scanned image for cleaner Devanagari recognition
        processed = self._preprocess_for_ocr(img)

        text = pytesseract.image_to_string(
            processed,
            lang='nep',
            config='--psm 4 --oem 1'
        )
        return text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.doc.close()
