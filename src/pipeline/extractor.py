import fitz
import pytesseract
from PIL import Image

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

    def _ocr_page(self, page):
        """Renders the page to an image and performs OCR using Tesseract."""
        # High DPI (e.g. 300) for better OCR accuracy
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Use nepali language data with PSM 4 (Assume a single column of text of variable sizes)
        # to prevent Tesseract from splitting bullet points into a separate column.
        text = pytesseract.image_to_string(img, lang='nep', config='--psm 4')
        return text

    def close(self):
        self.doc.close()
