import re

class TextCleaner:
    def clean(self, text):
        """
        Cleans the raw OCR text by removing unwanted artifacts like 
        headers, footers, page numbers, and excessive whitespace.
        """
        if not text:
            return ""

        # Remove the lawcommission header/footer URL
        text = re.sub(r'www\.lawcommission\.gov\.np', '', text, flags=re.IGNORECASE)
        
        # Remove standalone page numbers (e.g., lines with just a number)
        # Nepali numbers range is \u0966-\u096F, English is 0-9
        text = re.sub(r'^\s*[\d०-९]+\s*$', '', text, flags=re.MULTILINE)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Replace 3 or more newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
