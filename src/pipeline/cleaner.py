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
        text = re.sub(r"www\.lawcommission\.gov\.np", "", text, flags=re.IGNORECASE)

        # Remove recurring amendment footnotes (may wrap across lines)
        # e.g. "...सम्बन्धी केही नेपाल ऐनलाई संशोधन गर्ने ऐन, २०८१ द्वारा थप।"
        text = re.sub(
            r"(?:लगानी सहजीकरण|आर्थिक तथा व्यावसायिक)[^।]{0,150}सम्[बव]न्धी\s*केही\s*नेपाल\s*ऐनलाई\s*संशोधन\s*गर्न[े]?\s*ऐन\s*,?\s*[०-९\d]{4}\s*द्वारा\s*[^।]{0,20}।",
            "",
            text,
        )

        # Remove standalone page numbers (e.g., lines with just a number)
        # Nepali numbers range is \u0966-\u096F, English is 0-9
        text = re.sub(r"^\s*[\d०-९]+\s*$", "", text, flags=re.MULTILINE)

        # Replace multiple spaces with a single space
        text = re.sub(r" +", " ", text)

        # Replace 3 or more newlines with double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
