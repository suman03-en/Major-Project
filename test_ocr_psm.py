import fitz
import pytesseract
from PIL import Image

def test_psm():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    doc = fitz.open('input_pdfs/comapy_act.pdf')
    page = doc.load_page(2) # Page 3, where definitions are
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    with open('test_ocr_output.txt', 'w', encoding='utf-8') as f:
        f.write("--- PSM 3 (Default) ---\n")
        text_3 = pytesseract.image_to_string(img, lang='nep', config='--psm 3')
        f.write(text_3[:500] + "\n")
        
        f.write("\n--- PSM 6 (Assume single uniform block of text) ---\n")
        text_6 = pytesseract.image_to_string(img, lang='nep', config='--psm 6')
        f.write(text_6[:500] + "\n")
        
        f.write("\n--- PSM 4 (Assume single column of text) ---\n")
        text_4 = pytesseract.image_to_string(img, lang='nep', config='--psm 4')
        f.write(text_4[:500] + "\n")

if __name__ == '__main__':
    test_psm()
