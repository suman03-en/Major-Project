import unicodedata
import unittest

from src.extraction.cleaner import TextCleaner


class TestTextCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()

    def test_empty_input_returns_empty_string(self):
        self.assertEqual(self.cleaner.clean(""), "")
        self.assertEqual(self.cleaner.clean(None), "")

    def test_unicode_nfc_normalization(self):
        # Non-NFC input (both Devanagari composition exclusion \u0958 and NFD Latin e+\u0301)
        non_nfc = "\u0958ानून र e\u0301"
        expected_nfc = unicodedata.normalize("NFC", non_nfc)
        self.assertNotEqual(non_nfc, expected_nfc)
        cleaned = self.cleaner.clean(non_nfc)
        self.assertEqual(cleaned, expected_nfc)

    def test_nepali_legal_ocr_confusion_pairs(self):
        cases = [
            ("कम्पनी भुन्नाले दफा २ बमोजिम", "कम्पनी भन्नाले दफा २ बमोजिम"),
            ("कम्त्रीमा तीन जना संस्थापक", "कम्तीमा तीन जना संस्थापक"),
            ("रजिष्ट्रार समक्ष पेश गर्नु पनिछ।", "रजिष्ट्रार समक्ष पेश गर्नु पर्नेछ।"),
            ("प्राइभेट कम्पर्नी दर्ता गर्न", "प्राइभेट कम्पनी दर्ता गर्न"),
            ("प्रचेलित नेपाल कानून बमोजिम", "प्रचलित नेपाल कानून बमोजिम"),
            ("दफा ५ मा उल्लेख भए सरह सम्झन् पर्छ।", "दफा ५ मा उल्लेख भए सरह सम्झनु पर्छ।"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(self.cleaner.clean(raw), expected)

    def test_removes_pua_cid_zero_width_and_page_numbers(self):
        raw = "www.lawcommission.gov.np\n\uE001\u200Bकम्पनी ऐन (cid:123)\n१२\nदफा १"
        cleaned = self.cleaner.clean(raw)
        self.assertNotIn("lawcommission", cleaned)
        self.assertNotIn("\uE001", cleaned)
        self.assertNotIn("\u200B", cleaned)
        self.assertNotIn("cid:123", cleaned)
        self.assertIn("कम्पनी ऐन", cleaned)
        self.assertIn("दफा १", cleaned)


if __name__ == "__main__":
    unittest.main()
