import unittest

from src.extraction.formatter import RegexFormatter


class TestRegexFormatter(unittest.TestCase):
    def setUp(self):
        self.act_metadata = {
            "title": "कम्पनी ऐन, २०६३",
            "bs_year": 2063,
            "date_commenced": "२०६३/०६/२०",
        }
        self.act_slug = "company-act"

    def test_midline_markers_split_into_separate_provisions_preserving_inline_refs(self):
        text = (
            "परिच्छेद १\n"
            "प्रारम्भिक\n"
            "१. संक्षिप्त नाम र प्रारम्भः\n"
            "(१) यस ऐनको नाम कम्पनी ऐन, २०६३ रहेको छ । (२) उपदफा (१) बमोजिम यो ऐन तुरुन्त प्रारम्भ हुनेछ । "
            "(क) पहिलो खण्ड लागू हुनेछ । (ख) खण्ड (क) अनुसार दोस्रो खण्ड लागू हुनेछ ।"
        )
        formatter = RegexFormatter(self.act_metadata, self.act_slug)
        result = formatter.process_text(text)
        chunks = result["chunks"]

        sub_chunks = [c for c in chunks if c["type"] == "subsection"]
        clause_chunks = [c for c in chunks if c["type"] == "clause"]

        self.assertEqual(len(sub_chunks), 2)
        self.assertEqual(sub_chunks[0]["hierarchy"]["sub"], 1)
        self.assertIn("कम्पनी ऐन, २०६३ रहेको छ", sub_chunks[0]["text"])
        self.assertEqual(sub_chunks[1]["hierarchy"]["sub"], 2)
        # Inline reference "उपदफा (१)" must stay intact inside sub 2
        self.assertIn("उपदफा (१) बमोजिम", sub_chunks[1]["text"])

        self.assertEqual(len(clause_chunks), 2)
        self.assertEqual(clause_chunks[0]["hierarchy"]["clause"], "क")
        self.assertEqual(clause_chunks[1]["hierarchy"]["clause"], "ख")
        # Inline reference "खण्ड (क)" must stay intact inside clause ख
        self.assertIn("खण्ड (क) अनुसार", clause_chunks[1]["text"])

    def test_section_with_same_line_subsection_splits_into_sub1(self):
        text = (
            "परिच्छेद २\n"
            "कम्पनीको संस्थापना\n"
            "३. कम्पनीको संस्थापनाः (१) कुनै व्यक्तिले एकल शेयरधनी भएको प्राइभेट कम्पनी स्थापना गर्न सक्नेछ ।\n"
            "४. कम्पनीको प्रकार (१) प्रचलित कानून बमोजिम पब्लिक वा प्राइभेट कम्पनी दर्ता हुनेछ ।"
        )
        formatter = RegexFormatter(self.act_metadata, self.act_slug)
        result = formatter.process_text(text)
        chunks = result["chunks"]

        sub_chunks = [c for c in chunks if c["type"] == "subsection"]
        sec_chunks = [c for c in chunks if c["type"] == "section"]

        self.assertEqual(len(sec_chunks), 0)
        self.assertEqual(len(sub_chunks), 2)
        self.assertEqual(sub_chunks[0]["hierarchy"]["sec"], 3)
        self.assertEqual(sub_chunks[0]["hierarchy"]["sec_title"], "कम्पनीको संस्थापना")
        self.assertEqual(sub_chunks[0]["hierarchy"]["sub"], 1)
        self.assertIn("प्राइभेट कम्पनी स्थापना", sub_chunks[0]["text"])

        self.assertEqual(sub_chunks[1]["hierarchy"]["sec"], 4)
        self.assertEqual(sub_chunks[1]["hierarchy"]["sec_title"], "कम्पनीको प्रकार")
        self.assertEqual(sub_chunks[1]["hierarchy"]["sub"], 1)
        self.assertIn("पब्लिक वा प्राइभेट कम्पनी", sub_chunks[1]["text"])

    def test_widened_clause_markers_and_missing_dot_after_section_number(self):
        text = (
            "परिच्छेद २\n"
            "कम्पनीको नाम\n"
            "५ कम्पनीको नामः\n"
            "(१) प्रत्येक कम्पनीको नाम देहाय बमोजिम हुनेछः\n"
            "(क१) थप गरिएको उपखण्डको व्यवस्था लागू हुनेछ ।\n"
            "( खख ) अर्को विशेष खण्डको व्यवस्था लागू हुनेछ ।\n"
            "२०६३ सालको ऐन बमोजिम अन्य व्यवस्था यथावत रहनेछ ।"
        )
        formatter = RegexFormatter(self.act_metadata, self.act_slug)
        result = formatter.process_text(text)
        chunks = result["chunks"]

        sub_chunks = [c for c in chunks if c["type"] == "subsection"]
        clause_chunks = [c for c in chunks if c["type"] == "clause"]

        self.assertEqual(len(sub_chunks), 1)
        self.assertEqual(sub_chunks[0]["hierarchy"]["sec"], 5)
        self.assertEqual(sub_chunks[0]["hierarchy"]["sec_title"], "कम्पनीको नाम")

        self.assertEqual(len(clause_chunks), 2)
        self.assertEqual(clause_chunks[0]["hierarchy"]["clause"], "क१")
        self.assertTrue(clause_chunks[0]["text"].startswith("थप गरिएको"))
        self.assertEqual(clause_chunks[1]["hierarchy"]["clause"], "खख")
        self.assertIn("२०६३ सालको ऐन", clause_chunks[1]["text"])

    def test_oversized_provision_splits_at_sentence_boundary(self):
        sentence = "यो एक लामो कानूनी वाक्य हो जसमा कम्पनी दर्ता सम्बन्धी विस्तृत नियम र प्रक्रिया उल्लेख गरिएको छ ।"
        # Repeat sentence enough times to exceed 1200 chars (~90 chars * 20 = ~1800 chars)
        long_body = " ".join([sentence] * 20)
        text = (
            "परिच्छेद २\n"
            "कम्पनी दर्ता\n"
            "६. दर्ता सम्बन्धी व्यवस्थाः\n"
            f"(१) {long_body}"
        )
        formatter = RegexFormatter(self.act_metadata, self.act_slug)
        result = formatter.process_text(text)
        sub_chunks = [c for c in result["chunks"] if c["type"] == "subsection"]

        self.assertGreaterEqual(len(sub_chunks), 2)
        ids = [c["id"] for c in sub_chunks]
        self.assertEqual(len(ids), len(set(ids)), "Split chunks must have unique IDs")
        for c in sub_chunks:
            self.assertLessEqual(len(c["text"]), 1200)
            self.assertTrue(c["text"].endswith("।"))

    def test_parent_id_and_temporal_metadata_on_provisions(self):
        text = (
            "परिच्छेद २\n"
            "कम्पनीको संस्थापना\n"
            "७. दायित्वः\n"
            "कम्पनीको दायित्व सीमित हुनेछ ।\n"
            "(१) शेयरधनीको दायित्व देहाय बमोजिम हुनेछः\n"
            "(क) निजले खरिद गरेको शेयरको रकमसम्म सीमित रहनेछ ।"
        )
        formatter = RegexFormatter(self.act_metadata, self.act_slug)
        result = formatter.process_text(text)
        chunks = result["chunks"]

        sec_chunk = next(c for c in chunks if c["type"] == "section")
        sub_chunk = next(c for c in chunks if c["type"] == "subsection")
        cl_chunk = next(c for c in chunks if c["type"] == "clause")

        self.assertIsNone(sec_chunk["parent_id"])
        self.assertEqual(sub_chunk["parent_id"], "company-act-2063-ch2-sec7")
        self.assertEqual(cl_chunk["parent_id"], "company-act-2063-ch2-sec7-sub1")

        for c in (sec_chunk, sub_chunk, cl_chunk):
            self.assertEqual(c["status"], "active")
            self.assertEqual(c["effective_date"], "२०६३/०६/२०")
            self.assertEqual(c["bs_year"], 2063)

        # Custom status from act_metadata should propagate
        repealed_meta = {**self.act_metadata, "status": "repealed", "effective_date": "2063-10-06"}
        repealed_res = RegexFormatter(repealed_meta, self.act_slug).process_text(text)
        for c in repealed_res["chunks"]:
            self.assertEqual(c["status"], "repealed")
            self.assertEqual(c["effective_date"], "2063-10-06")

    def test_section_summary_provision_generated_per_section(self):
        text = (
            "परिच्छेद २\n"
            "कम्पनीको संस्थापना\n"
            "८. कम्पनीको प्रकारः\n"
            "(१) प्राइभेट कम्पनी दर्ता गर्न सकिनेछ ।\n"
            "(क) शेयरधनी संख्या सीमित हुनेछ ।\n"
            "(२) पब्लिक कम्पनी दर्ता गर्न सकिनेछ ।\n"
            "९. उद्देश्यः\n"
            "(१) प्रबन्धपत्रमा उद्देश्य उल्लेख हुनेछ ।"
        )
        formatter = RegexFormatter(self.act_metadata, self.act_slug)
        result = formatter.process_text(text)
        summaries = [c for c in result["chunks"] if c["type"] == "section_summary"]

        self.assertEqual(len(summaries), 2)

        sec8 = summaries[0]
        self.assertEqual(sec8["id"], "company-act-2063-ch2-sec8-summary")
        self.assertEqual(sec8["hierarchy"]["sec"], 8)
        self.assertEqual(sec8["hierarchy"]["sec_title"], "कम्पनीको प्रकार")
        self.assertIsNone(sec8["parent_id"])
        self.assertEqual(sec8["status"], "active")
        self.assertEqual(sec8["bs_year"], 2063)
        self.assertIn("प्राइभेट कम्पनी दर्ता गर्न सकिनेछ", sec8["text"])
        self.assertIn("शेयरधनी संख्या सीमित हुनेछ", sec8["text"])
        self.assertIn("पब्लिक कम्पनी दर्ता गर्न सकिनेछ", sec8["text"])

        sec9 = summaries[1]
        self.assertEqual(sec9["id"], "company-act-2063-ch2-sec9-summary")
        self.assertEqual(sec9["hierarchy"]["sec"], 9)
        self.assertIn("प्रबन्धपत्रमा उद्देश्य उल्लेख हुनेछ", sec9["text"])


if __name__ == "__main__":
    unittest.main()
