import re


class TextCleaner:
    def clean(self, text):
        """
        Cleans the raw extracted text by removing unwanted artifacts like
        headers, footers, page numbers, garbage Unicode, and excessive whitespace.

        Handles both OCR output and native PDF text extraction artifacts.
        """
        if not text:
            return ""

        # ── Garbage Unicode Cleanup ──────────────────────────────────────
        # Remove Private Use Area characters (U+E000–U+F8FF) — common in
        # legacy-font Nepali PDFs where glyphs are mapped to PUA codepoints
        text = re.sub(r'[\uE000-\uF8FF]+', '', text)

        # Remove zero-width and invisible Unicode characters that break
        # Devanagari text processing
        text = re.sub(r'[\u200B-\u200F\u202A-\u202E\uFEFF\u00AD]+', '', text)

        # Remove control characters (except newline \n and tab \t)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Remove isolated runs of Latin characters that are clearly not
        # part of Nepali legal text (but keep short English abbreviations
        # like "PDF", "OCR", section letters like "A", "B")
        text = re.sub(r'(?<!\w)[A-Za-z]{8,}(?!\w)', '', text)

        # Remove bare CID/glyph-ID patterns from broken font extraction
        # e.g., "(cid:123)" or "CID+45"
        text = re.sub(r'\(cid:\d+\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'CID\+\d+', '', text, flags=re.IGNORECASE)

        # ── Standard Cleanup (OCR + digital) ─────────────────────────────
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

        # Remove lines that are only punctuation, symbols, or whitespace
        # (residual artifacts from cleaned garbage)
        text = re.sub(r"^\s*[^\u0900-\u097F\dA-Za-z]{1,5}\s*$", "", text, flags=re.MULTILINE)

        # Replace multiple spaces with a single space
        text = re.sub(r" +", " ", text)

        # Replace 3 or more newlines with double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
