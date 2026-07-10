import fitz
import pytesseract
from PIL import Image, ImageFilter, ImageOps

class PdfExtractor:
    def __init__(self, pdf_path, tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        self.pdf_path = pdf_path
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.doc = fitz.open(pdf_path)

    def extract_all_pages(self):
        """Yields the OCR extracted text for each page in the document."""
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            yield page_num + 1, self._ocr_page(page)

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
        """Renders the page to an image, preprocesses it, and performs OCR."""
        # High DPI (300) for better OCR accuracy on scanned documents
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Preprocess the scanned image for cleaner Devanagari recognition
        processed = self._preprocess_for_ocr(img)

        # Use nep (Nepali) language data with:
        #   --psm 4: Assume a single column of text of variable sizes
        #   --oem 1: LSTM neural network engine (best for complex scripts like Devanagari)
        text = pytesseract.image_to_string(
            processed,
            lang='nep',
            config='--psm 4 --oem 1'
        )
        return text

    def close(self):
        self.doc.close()
