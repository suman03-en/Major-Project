"""
PDF Extractor — Surya OCR
=========================
Extracts text from every page of a Nepali legal PDF using Surya OCR.
All pages are rendered at 300 DPI and passed through the Surya detection +
recognition pipeline (GPU-accelerated when CUDA is available).

Surya OCR models are loaded lazily on the first page, and VRAM is released
after extraction via ``torch.cuda.empty_cache()``.

VRAM tuning (RTX 3050 4 GB) is controlled by environment variables set in
the Dockerfile / compose.yaml:
    RECOGNITION_BATCH_SIZE=2
    DETECTOR_BATCH_SIZE=2
    TORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import logging
import re
from typing import Literal

import fitz
import torch
from PIL import Image

logger = logging.getLogger(__name__)

PageRoute = Literal["unicode", "legacy", "ocr"]

LEGACY_NEPALI_FONTS: tuple[str, ...] = (
    "preeti",
    "kantipur",
    "kalimati",
    "fontasy",
    "sagarmatha",
)

_CID_RE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)

# Preeti / legacy ASCII Nepali font -> Unicode Devanagari mapping
_PREETI_DIGRAPH_MAP: dict[str, str] = {
    "cf]": "ओ",
    "cf}": "औ",
    "cf": "आ",
    "O{": "ई",
    "P]": "ऐ",
    "pm": "ऊ",
    "c+": "अं",
    "cF": "अँ",
    "km": "फ",
    "Qm": "क्त",
}

_PREETI_CHAR_MAP: dict[str, str] = {
    "a": "ब",
    "b": "द",
    "c": "अ",
    "d": "म",
    "e": "भ",
    "f": "ा",
    "g": "न",
    "h": "ज",
    "i": "ष",
    "j": "व",
    "k": "प",
    "l": "ि",
    "m": "ः",
    "n": "ल",
    "o": "य",
    "p": "उ",
    "q": "त्र",
    "r": "च",
    "s": "क",
    "t": "त",
    "u": "ग",
    "v": "ख",
    "w": "ध",
    "x": "ह",
    "y": "थ",
    "z": "श",
    "A": "ब्",
    "B": "ध्",
    "C": "ऋ",
    "D": "म्",
    "E": "भ्",
    "F": "ँ",
    "G": "न्",
    "H": "ज्",
    "I": "क्ष्",
    "J": "व्",
    "K": "प्",
    "L": "ी",
    "M": "ः",
    "N": "ल्",
    "O": "इ",
    "P": "ए",
    "Q": "त्त",
    "R": "च्",
    "S": "क्",
    "T": "त्",
    "U": "ग्",
    "V": "ख्",
    "W": "ध्",
    "X": "ह्",
    "Y": "थ्",
    "Z": "श्",
    "1": "ज्ञ",
    "2": "द्द",
    "3": "घ",
    "4": "द्ध",
    "5": "छ",
    "6": "ट",
    "7": "ठ",
    "8": "ड",
    "9": "ढ",
    "0": "ण",
    "!": "१",
    "@": "२",
    "#": "३",
    "$": "४",
    "%": "५",
    "^": "६",
    "&": "७",
    "*": "८",
    "(": "९",
    ")": "०",
    "`": "ञ",
    "~": "ञ्",
    "[": "ृ",
    "]": "े",
    "}": "ै",
    "\\": "्",
    "|": "्र",
    ";": "स",
    ":": "स्",
    "'": "ु",
    '"': "ू",
    "<": "?",
    ".": "।",
    ">": "श्र",
    "/": "र",
    "?": "रु",
    "+": "ं",
    "=": ".",
    "-": "(",
    "_": ")",
    "{": "\uE000",  # placeholder for reph (र्) reordered before syllable
}

_PREETI_IMATRA_RE = re.compile(
    r"ि((?:[\u0915-\u0939]\u094D)*[\u0915-\u0939])"
)
_PREETI_REPH_RE = re.compile(
    r"((?:[\u0915-\u0939]\u094D)*[\u0915-\u0939][\u093E-\u094C\u0901-\u0903]*)\uE000"
)


def is_legacy_nepali_font(font_name: str) -> bool:
    """Return True if *font_name* matches a known legacy ASCII Nepali font."""
    lowered = (font_name or "").lower()
    return any(font in lowered for font in LEGACY_NEPALI_FONTS)


def convert_legacy_nepali(text: str) -> str:
    """Deterministically convert Preeti-encoded ASCII text to Unicode Devanagari."""
    if not text:
        return ""

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for length in (3, 2):
            chunk = text[i : i + length]
            if chunk in _PREETI_DIGRAPH_MAP:
                out.append(_PREETI_DIGRAPH_MAP[chunk])
                i += length
                matched = True
                break
        if matched:
            continue

        ch = text[i]
        out.append(_PREETI_CHAR_MAP.get(ch, ch))
        i += 1

    converted = "".join(out)
    # Move visual pre-base i-matra (ि) after its consonant/conjunct
    converted = _PREETI_IMATRA_RE.sub(r"\1ि", converted)
    # Move visual post-syllable reph placeholder before its syllable as र्
    converted = _PREETI_REPH_RE.sub(r"र्\1", converted)
    return converted.replace("\uE000", "र्")


def extract_legacy_spans(page_dict: dict) -> str:
    """Extract text from a page dict, converting legacy Nepali font spans only."""
    lines_out: list[str] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            span_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if not span_text:
                    continue
                if is_legacy_nepali_font(span.get("font", "")):
                    span_parts.append(convert_legacy_nepali(span_text))
                else:
                    span_parts.append(span_text)
            if span_parts:
                lines_out.append("".join(span_parts))
    return "\n".join(lines_out)


def classify_page(page) -> PageRoute:
    """
    Classify a PDF page into one of three extraction routes:
      - ``"unicode"``: True Unicode Devanagari text (direct PyMuPDF extraction)
      - ``"legacy"``: Legacy ASCII Nepali fonts (Preeti/Kantipur/Kalimati/Fontasy/Sagarmatha)
      - ``"ocr"``: Scanned image, broken CMap ``(cid:``, or high garbage ratio
    """
    page_dict = page.get_text("dict")
    spans: list[tuple[str, str]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    spans.append((span.get("font", ""), text))

    if not spans:
        return "ocr"

    combined_text = " ".join(text for _, text in spans)
    if _CID_RE.search(combined_text):
        return "ocr"

    non_ws_chars = [ch for ch in combined_text if not ch.isspace()]
    if not non_ws_chars:
        return "ocr"

    garbage_count = sum(
        1
        for ch in non_ws_chars
        if ("\uE000" <= ch <= "\uF8FF") or ch == "\uFFFD"
    )
    if (garbage_count / len(non_ws_chars)) > 0.15:
        return "ocr"

    if any(is_legacy_nepali_font(font) for font, _ in spans):
        return "legacy"

    devanagari_count = sum(1 for ch in non_ws_chars if "\u0900" <= ch <= "\u097F")
    if (devanagari_count / len(non_ws_chars)) >= 0.30:
        return "unicode"

    return "ocr"


class PdfExtractor:
    """
    Extracts text from every page of a PDF using Surya OCR.

    Surya models are loaded lazily on the first call to ``_ocr_page()``.
    Call ``close()`` (or use as a context manager) to free VRAM afterwards.
    """

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

        # Track stats
        self._stats = {"unicode": 0, "legacy": 0, "ocr": 0, "total": 0}

        # Surya OCR model handles — loaded lazily on first page
        self._det_model = None
        self._det_processor = None
        self._rec_model = None
        self._rec_processor = None

        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("CUDA available: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
            print(f"  🟢 CUDA available: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        else:
            self.device = "cpu"
            logger.info("CUDA not available — Surya OCR will run on CPU (slow)")
            print("  🟡 CUDA not available — Surya OCR will run on CPU (this will be slow)")

    # ------------------------------------------------------------------
    # Lazy Surya model loader
    # ------------------------------------------------------------------

    def _load_surya_models(self) -> None:
        """Load Surya detection + recognition models (once)."""
        if self._det_model is not None:
            return  # already loaded

        logger.info("Loading Surya OCR models...")
        print("  Loading Surya OCR models (first page)...")

        from surya.model.detection.model import (
            load_model as load_det_model,
        )
        from surya.model.detection.model import (
            load_processor as load_det_processor,
        )
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import (
            load_processor as load_rec_processor,
        )

        self._det_processor = load_det_processor()
        self._det_model = load_det_model()
        self._rec_processor = load_rec_processor()
        self._rec_model = load_rec_model()

        logger.info("Surya OCR models loaded on %s.", self.device)
        print(f"  Surya OCR models loaded on {self.device.upper()}.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all_pages(self):
        """
        Yields (page_number, text) for each page using the 3-way page router:
        direct Unicode text, span-level legacy-font conversion, or 300 DPI Surya OCR.
        """
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            route = classify_page(page)
            self._stats["total"] += 1
            self._stats[route] += 1

            if route == "unicode":
                text = page.get_text("text")
            elif route == "legacy":
                text = extract_legacy_spans(page.get_text("dict"))
            else:
                try:
                    text = self._ocr_page(page)
                except RuntimeError as exc:
                    logger.warning(
                        "OCR failed on page %d (%s): %s. Skipping page.",
                        page_num + 1,
                        self.pdf_path,
                        exc,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    text = ""

            yield page_num + 1, text

        logger.info(
            "Extraction complete: %d total (%d unicode, %d legacy, %d ocr).",
            self._stats["total"],
            self._stats["unicode"],
            self._stats["legacy"],
            self._stats["ocr"],
        )

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_page(self, page) -> str:
        """Render the page to an image and run Surya OCR on it."""
        # 300 DPI — Devanagari's complex conjuncts need high resolution
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Crop the top header strip (law commission URL) — consistent across all pages
        HEADER_CROP_PCT = 0.07
        crop_y = int(img.height * HEADER_CROP_PCT)
        img = img.crop((0, crop_y, img.width, img.height))

        # Lazy-load models
        self._load_surya_models()

        from surya.ocr import run_ocr

        predictions = run_ocr(
            [img],
            [["ne"]],
            self._det_model,
            self._det_processor,
            self._rec_model,
            self._rec_processor,
        )

        if not predictions or not predictions[0].text_lines:
            return ""

        lines = [line.text for line in predictions[0].text_lines]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """Close the PDF and free Surya model VRAM."""
        self.doc.close()

        if self._det_model is not None:
            try:
                del self._det_model, self._det_processor
                del self._rec_model, self._rec_processor
                self._det_model = None
                torch.cuda.empty_cache()
                logger.info("Cleared Surya OCR models from VRAM.")
            except (RuntimeError, AttributeError) as e:
                logger.debug("VRAM cleanup error (non-fatal): %s", e)
