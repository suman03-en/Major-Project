"""Tests for zero-LLM ProcedureFormatter (Citizen Charters & procedural guides)."""

import unittest

from src.extraction.procedure_formatter import ProcedureFormatter
from src.knowledge_base.schemas import NERPipelineResult


class TestProcedureFormatter(unittest.TestCase):
    """Behavior tests at the ProcedureFormatter.process_text seam."""

    def test_parses_numbered_procedural_guide_without_dafa(self) -> None:
        text = (
            "कम्पनी दर्ता गर्ने प्रक्रिया\n"
            "१. अनलाइन पोर्टलमा खाता खोल्ने।\n"
            "२. कम्पनीको नाम स्वीकृतिका लागि निवेदन पेश गर्ने।\n"
            "३. दर्ता प्रमाणपत्र प्राप्त गर्ने।\n"
        )
        formatter = ProcedureFormatter(
            act_metadata={"title": "कम्पनी दर्ता मार्गदर्शन", "bs_year": "2080"},
            act_slug="company_guide",
        )

        result = formatter.process_text(text)

        self.assertEqual(len(result["procedures"]), 1)
        proc = result["procedures"][0]
        self.assertEqual(proc["process_name"], "कम्पनी दर्ता गर्ने प्रक्रिया")
        self.assertEqual(len(proc["steps"]), 3)
        self.assertEqual(proc["steps"][0]["step_number"], 1)
        self.assertIn("अनलाइन पोर्टलमा खाता खोल्ने", proc["steps"][0]["action"])
        self.assertEqual(proc["steps"][2]["step_number"], 3)

        # Qdrant-compatible Provision chunks
        self.assertGreaterEqual(len(result["chunks"]), 3)
        for chunk in result["chunks"]:
            self.assertIn("id", chunk)
            self.assertIn("type", chunk)
            self.assertIn("hierarchy", chunk)
            self.assertIn("text", chunk)
            self.assertIn("stats", chunk)
            self.assertGreater(chunk["stats"]["characters"], 0)

        # Neo4j-compatible ner_result
        ner = result["ner_result"]
        self.assertEqual(ner["act_slug"], "company_guide")
        self.assertEqual(len(ner["entities"]), 1)
        self.assertEqual(ner["entities"][0]["process_name"], "कम्पनी दर्ता गर्ने प्रक्रिया")
        self.assertEqual(len(ner["entities"][0]["steps"]), 3)

    def test_parses_citizen_charter_documents_office_forms_fees_and_duration(self) -> None:
        text = (
            "नागरिक बडापत्र\n"
            "सेवाको नाम: प्राइभेट लिमिटेड कम्पनी दर्ता\n"
            "आवश्यक कागजातहरू:\n"
            "- निवेदकको नेपाली नागरिकताको प्रमाणपत्रको प्रतिलिपि\n"
            "- प्रबन्धपत्र र नियमावली २ प्रति\n"
            "- पासपोर्ट साइजको फोटो २ थान\n"
            "आवेदन फाराम: अनुसूची–१ (कम्पनी दर्ता फाराम)\n"
            "जिम्मेवार कार्यालय: कम्पनी रजिष्ट्रारको कार्यालय\n"
            "लाग्ने शुल्क: रु. १,०००/-\n"
            "लाग्ने समय: ३ कार्यदिन\n"
            "प्रक्रिया:\n"
            "१. अनलाइन प्रणालीमा नाम स्वीकृति र कागजात अपलोड गर्ने।\n"
            "२. तोकिएको दस्तुर बैंकमा जम्मा गरी प्रमाणपत्र लिने।\n"
        )
        formatter = ProcedureFormatter(
            act_metadata={"title": "कम्पनी रजिष्ट्रार नागरिक बडापत्र", "bs_year": "2081"},
            act_slug="ocr_charter",
        )

        result = formatter.process_text(text)

        self.assertEqual(len(result["procedures"]), 1)
        proc = result["procedures"][0]
        self.assertEqual(proc["process_name"], "प्राइभेट लिमिटेड कम्पनी दर्ता")
        self.assertEqual(
            proc["documents_required"],
            [
                "निवेदकको नेपाली नागरिकताको प्रमाणपत्रको प्रतिलिपि",
                "प्रबन्धपत्र र नियमावली २ प्रति",
                "पासपोर्ट साइजको फोटो २ थान",
            ],
        )
        self.assertEqual(proc["forms"], ["अनुसूची–१ (कम्पनी दर्ता फाराम)"])
        self.assertEqual(proc["offices"], [{"name": "कम्पनी रजिष्ट्रारको कार्यालय", "level": "केन्द्र"}])
        self.assertEqual(len(proc["price_fees"]), 1)
        self.assertEqual(proc["price_fees"][0]["amount_npr"], 1000)
        self.assertEqual(proc["duration"], "३ कार्यदिन")

        # Procedure-level metadata propagates to steps for Neo4j graph ingestion
        self.assertEqual(len(proc["steps"]), 2)
        self.assertEqual(proc["steps"][0]["office"]["name"], "कम्पनी रजिष्ट्रारको कार्यालय")
        self.assertEqual(proc["steps"][0]["documents_required"], proc["documents_required"])
        self.assertEqual(proc["steps"][0]["duration"], "३ कार्यदिन")
        self.assertEqual(proc["steps"][0]["price_fees"][0]["amount_npr"], 1000)
        self.assertEqual(proc["steps"][0]["forms"], ["अनुसूची–१ (कम्पनी दर्ता फाराम)"])

        # Also produces a procedure overview chunk for Qdrant alongside step chunks
        chunk_types = [c["type"] for c in result["chunks"]]
        self.assertIn("procedure", chunk_types)
        self.assertIn("procedure_step", chunk_types)

    def test_extracts_inline_step_metadata_for_bulleted_guide(self) -> None:
        text = (
            "व्यवसाय दर्ता मार्गदर्शन\n"
            "• वडा कार्यालयमा नागरिकताको प्रतिलिपि र फाराम नं. १ सहित निवेदन पेश गर्ने।\n"
            "• दर्ता दस्तुर रु. ५०० बुझाई ३ दिनभित्र प्रमाणपत्र प्राप्त गर्ने।\n"
        )
        formatter = ProcedureFormatter(act_metadata={"title": "व्यवसाय दर्ता"}, act_slug="biz_guide")

        result = formatter.process_text(text)

        self.assertEqual(len(result["procedures"]), 1)
        steps = result["procedures"][0]["steps"]
        self.assertEqual(len(steps), 2)

        self.assertEqual(steps[0]["office"]["name"], "वडा कार्यालय")
        self.assertEqual(steps[0]["office"]["level"], "स्थानीय")
        self.assertIn("फाराम नं. १", steps[0]["forms"])
        self.assertTrue(any("नागरिकताको प्रतिलिपि" in d for d in steps[0]["documents_required"]))

        self.assertEqual(steps[1]["price_fees"][0]["amount_npr"], 500)
        self.assertEqual(steps[1]["duration"], "३ दिन")

    def test_splits_multiple_procedures_in_one_citizen_charter(self) -> None:
        text = (
            "सेवाको नाम: प्यान दर्ता\n"
            "जिम्मेवार कार्यालय: करदाता सेवा कार्यालय\n"
            "लाग्ने शुल्क: निःशुल्क\n"
            "लाग्ने समय: सोही दिन\n"
            "१. अनलाइन फाराम भरी प्रमाणपत्र लिने।\n"
            "\n"
            "सेवाको नाम: उद्योग दर्ता\n"
            "जिम्मेवार कार्यालय: उद्योग विभाग\n"
            "लाग्ने शुल्क: रु. ५,०००\n"
            "लाग्ने समय: ७ कार्यदिन\n"
            "१. निवेदन र प्रबन्धपत्र पेश गर्ने।\n"
        )
        formatter = ProcedureFormatter(act_metadata={"title": "एकीकृत नागरिक बडापत्र"}, act_slug="integrated_charter")

        result = formatter.process_text(text)

        self.assertEqual(len(result["procedures"]), 2)
        self.assertEqual(result["procedures"][0]["process_name"], "प्यान दर्ता")
        self.assertEqual(result["procedures"][0]["price_fees"][0]["amount_npr"], 0)
        self.assertEqual(result["procedures"][0]["duration"], "सोही दिन")

        self.assertEqual(result["procedures"][1]["process_name"], "उद्योग दर्ता")
        self.assertEqual(result["procedures"][1]["price_fees"][0]["amount_npr"], 5000)
        self.assertEqual(result["procedures"][1]["offices"][0]["name"], "उद्योग विभाग")

        # Unique chunk IDs across all procedures
        chunk_ids = [c["id"] for c in result["chunks"]]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual(len(result["ner_result"]["entities"]), 2)

        # Validate against Neo4j ingestion Pydantic schema
        validated = NERPipelineResult.model_validate(result["ner_result"])
        self.assertEqual(len(validated.entities), 2)

    def test_empty_text_returns_empty_records(self) -> None:
        formatter = ProcedureFormatter(act_metadata={"title": "Empty"}, act_slug="empty")
        result = formatter.process_text("   \n\n  ")
        self.assertEqual(result["procedures"], [])
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["ner_result"]["entities"], [])


if __name__ == "__main__":
    unittest.main()
