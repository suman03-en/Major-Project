import unittest
from unittest.mock import MagicMock, patch

from src.extraction.extractor import (
    PdfExtractor,
    classify_page,
    convert_legacy_nepali,
    extract_legacy_spans,
)


def _make_page(spans_by_line, plain_text=""):
    """Build a minimal fake PyMuPDF page returning get_text('dict') and get_text('text')."""
    page = MagicMock()
    blocks = [
        {
            "type": 0,
            "lines": [{"spans": line_spans} for line_spans in spans_by_line],
        }
    ]
    page_dict = {"blocks": blocks}

    def get_text(mode="text"):
        if mode == "dict":
            return page_dict
        return plain_text

    page.get_text.side_effect = get_text
    return page


class TestClassifyPage(unittest.TestCase):
    def test_unicode_devanagari_routes_to_unicode(self):
        page = _make_page(
            [[{"font": "Mangal", "text": "कम्पनी ऐन, २०६३ को दफा १"}]],
            plain_text="कम्पनी ऐन, २०६३ को दफा १",
        )
        self.assertEqual(classify_page(page), "unicode")

    def test_legacy_nepali_fonts_route_to_legacy(self):
        for font_name in ["Preeti", "Kantipur", "Kalimati", "Fontasy Himali", "Sagarmatha"]:
            with self.subTest(font_name=font_name):
                page = _make_page([[{"font": font_name, "text": "g]kfn sDkgL P]g"}]])
                self.assertEqual(classify_page(page), "legacy")

    def test_cid_artifact_routes_to_ocr(self):
        page = _make_page([[{"font": "Preeti", "text": "(cid:123)(cid:45) g]kfn"}]])
        self.assertEqual(classify_page(page), "ocr")

    def test_high_garbage_ratio_routes_to_ocr(self):
        page = _make_page([[{"font": "Unknown", "text": "\uE001\uE002\uE003\uFFFD\uFFFDक"}]])
        self.assertEqual(classify_page(page), "ocr")

    def test_empty_image_page_routes_to_ocr(self):
        page = _make_page([])
        self.assertEqual(classify_page(page), "ocr")


class TestLegacyConversion(unittest.TestCase):
    def test_convert_legacy_nepali_basic_words(self):
        self.assertEqual(convert_legacy_nepali("g]kfn"), "नेपाल")
        self.assertEqual(convert_legacy_nepali("bkmf"), "दफा")
        self.assertEqual(convert_legacy_nepali("sDkgL"), "कम्पनी")
        self.assertEqual(convert_legacy_nepali("ljsf;"), "विकास")

    def test_extract_legacy_spans_preserves_english_spans(self):
        page = _make_page(
            [
                [
                    {"font": "Preeti", "text": "bkmf "},
                    {"font": "Times-Roman", "text": "Section 2(a) "},
                    {"font": "Kantipur", "text": "g]kfn"},
                ]
            ]
        )
        result = extract_legacy_spans(page.get_text("dict"))
        self.assertEqual(result, "दफा Section 2(a) नेपाल")


class TestPdfExtractorRouting(unittest.TestCase):
    @patch("src.extraction.extractor.torch.cuda.is_available", return_value=False)
    @patch("src.extraction.extractor.fitz.open")
    def test_extract_all_pages_routes_three_ways(self, mock_fitz_open, _mock_cuda):
        unicode_page = _make_page(
            [[{"font": "Mangal", "text": "कम्पनी ऐन, २०६३"}]],
            plain_text="कम्पनी ऐन, २०६३",
        )
        legacy_page = _make_page(
            [
                [
                    {"font": "Preeti", "text": "g]kfn "},
                    {"font": "Arial", "text": "Act 2063"},
                ]
            ]
        )
        scanned_page = _make_page([])

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 3
        fake_doc.load_page.side_effect = [unicode_page, legacy_page, scanned_page]
        mock_fitz_open.return_value = fake_doc

        with PdfExtractor("dummy.pdf") as extractor:
            with patch.object(extractor, "_ocr_page", return_value="OCR टेक्स्ट") as mock_ocr:
                pages = list(extractor.extract_all_pages())

        self.assertEqual(
            pages,
            [
                (1, "कम्पनी ऐन, २०६३"),
                (2, "नेपाल Act 2063"),
                (3, "OCR टेक्स्ट"),
            ],
        )
        mock_ocr.assert_called_once_with(scanned_page)
        self.assertEqual(
            extractor._stats,
            {"unicode": 1, "legacy": 1, "ocr": 1, "total": 3},
        )


if __name__ == "__main__":
    unittest.main()
