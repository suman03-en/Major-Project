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
import torch
import fitz
from PIL import Image

logger = logging.getLogger(__name__)


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
        self._stats = {"ocr": 0, "total": 0}

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
            load_processor as load_det_processor,
        )
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

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
        Yields (page_number, text) for each page.

        Every page is rendered at 300 DPI and passed through Surya OCR.
        """
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            self._stats["total"] += 1
            self._stats["ocr"] += 1

            text = self._ocr_page(page)
            yield page_num + 1, text

        logger.info(
            "Extraction complete: %d pages processed via Surya OCR.",
            self._stats["total"],
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
            except Exception as e:
                logger.debug("VRAM cleanup error (non-fatal): %s", e)
