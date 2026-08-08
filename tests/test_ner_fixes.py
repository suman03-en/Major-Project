"""
Verification tests for NER extraction post-processing fixes.

Tests the _validate_and_clean() method and schema validators against
all 7 error categories identified in the dataset:
1. Word-level fragmentation in steps
2. Nested office objects (dict → string)
3. Category confusion (services as documents)
4. Legal cross-references as documents
5. Single-character documents (Nepali list markers)
6. Invalid duration values
7. Penalty amounts in price

Usage:
    python -m tests.test_ner_fixes
"""

import sys
import os

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_base.schemas import ClauseNEROutput
from src.knowledge_base.ner_extractor import NERExtractor, parse_office

# We need a minimal NERExtractor to call _validate_and_clean
# Mock the provider check to avoid needing Ollama/Mistral
class MockNERExtractor:
    """Minimal mock that only exposes _validate_and_clean."""
    def _validate_and_clean(self, ner_output):
        return NERExtractor._validate_and_clean(self, ner_output)


extractor = MockNERExtractor()

passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ PASS: {name}")
        passed += 1
    else:
        print(f"  ❌ FAIL: {name}")
        failed += 1


print("\n" + "=" * 60)
print("  NER POST-PROCESSING VERIFICATION TESTS")
print("=" * 60)


# ═══════════════════════════════════════
# ERROR 1: Word-level fragmentation in steps
# ═══════════════════════════════════════
print("\n--- Error 1: Word-level fragmentation in steps ---")

output = ClauseNEROutput(
    steps=["पेश", "सम्बन्धमा", "हुने", "कार्यसूचीको", "बोर्डको", "कुनै", "सदस्यको", "निजी", "सरोकार", "बा", "स्वार्थ", "रहेको", "भएमा", "त्यस्तो", "कार्यसूचीका", "सम्बन्धमा", "हुने", "निर्णय"]
)
cleaned = extractor._validate_and_clean(output)
test("Single-word steps are removed", len(cleaned.steps) == 0)

output2 = ClauseNEROutput(
    steps=["अनुगमन गरिए", "सिफारिस गरे", "उद्योग दर्ता गर्ने निकाय समक्ष निवेदन दिनुपर्छ"]
)
cleaned2 = extractor._validate_and_clean(output2)
test("Two-word step fragments removed", "अनुगमन गरिए" not in cleaned2.steps)
test("Valid multi-word step preserved", any("निवेदन दिनुपर्छ" in s for s in cleaned2.steps))


# ═══════════════════════════════════════
# ERROR 2: Nested office objects (dict → string)
# ═══════════════════════════════════════
print("\n--- Error 2: Nested office objects ---")

# Test Pydantic validator (dict input)
output_dict_office = ClauseNEROutput(**{
    "office": {"name": "उद्योग दर्ता गर्ने निकाय", "level": "केन्द्र"},
    "documents_required": [],
    "steps": [],
    "price": None,
    "duration_days": None,
    "prerequisites": None,
})
test("Dict office normalized to string", output_dict_office.office == "उद्योग दर्ता गर्ने निकाय")
test("Office is a string type", isinstance(output_dict_office.office, str))


# ═══════════════════════════════════════
# ERROR 3: Category confusion (services as documents)
# ═══════════════════════════════════════
print("\n--- Error 3: Service types as documents ---")

output3 = ClauseNEROutput(
    documents_required=["उद्योग दर्ता", "नवीकरण", "नामसारी", "नाम परिवर्तन", "स्थानान्तरण", "क्षमता वृद्धि", "पुँजी वृद्धि", "उद्योग दर्ता प्रमाणपत्र"]
)
cleaned3 = extractor._validate_and_clean(output3)
test("Service type 'उद्योग दर्ता' removed from documents", "उद्योग दर्ता" not in cleaned3.documents_required)
test("Service type 'नवीकरण' removed", "नवीकरण" not in cleaned3.documents_required)
test("Valid document 'उद्योग दर्ता प्रमाणपत्र' preserved", "उद्योग दर्ता प्रमाणपत्र" in cleaned3.documents_required)


# ═══════════════════════════════════════
# ERROR 4: Legal cross-references as documents
# ═══════════════════════════════════════
print("\n--- Error 4: Legal cross-references as documents ---")

output4 = ClauseNEROutput(
    documents_required=["उपदफा (४)", "उपदफा (१)", "बमोजिम", "दफा १३", "बमोजिम प्रतिलिपि", "वातावरणीय प्रभाव मूल्याङ्कन प्रतिवेदन"]
)
cleaned4 = extractor._validate_and_clean(output4)
test("'उपदफा (४)' removed", "उपदफा (४)" not in cleaned4.documents_required)
test("'उपदफा (१)' removed", "उपदफा (१)" not in cleaned4.documents_required)
test("'बमोजिम' standalone removed", "बमोजिम" not in cleaned4.documents_required)
test("Valid doc 'वातावरणीय प्रभाव मूल्याङ्कन प्रतिवेदन' preserved", "वातावरणीय प्रभाव मूल्याङ्कन प्रतिवेदन" in cleaned4.documents_required)
test("'बमोजिम प्रतिलिपि' kept (multi-word)", "बमोजिम प्रतिलिपि" in cleaned4.documents_required)


# ═══════════════════════════════════════
# ERROR 5: Single-character documents (Nepali list markers)
# ═══════════════════════════════════════
print("\n--- Error 5: Single-character documents ---")

output5 = ClauseNEROutput(
    documents_required=["ढ", "ण", "त", "थ", "द", "न", "उद्योग दर्ता प्रमाणपत्र"]
)
cleaned5 = extractor._validate_and_clean(output5)
test("Single-char 'ढ' removed", "ढ" not in cleaned5.documents_required)
test("Single-char 'ण' removed", "ण" not in cleaned5.documents_required)
test("All single chars removed (only valid doc remains)", len(cleaned5.documents_required) == 1)
test("Valid document preserved", "उद्योग दर्ता प्रमाणपत्र" in cleaned5.documents_required)


# ═══════════════════════════════════════
# ERROR 6: Invalid duration values
# ═══════════════════════════════════════
print("\n--- Error 6: Invalid duration values ---")

output6_invalid = ClauseNEROutput(duration_days="अद्यावधि")
cleaned6_invalid = extractor._validate_and_clean(output6_invalid)
test("'अद्यावधि' rejected as duration", cleaned6_invalid.duration_days is None)

output6_short = ClauseNEROutput(duration_days="तीन")
cleaned6_short = extractor._validate_and_clean(output6_short)
test("'तीन' without time unit rejected", cleaned6_short.duration_days is None)

output6_valid = ClauseNEROutput(duration_days="तीस दिनभित्र")
cleaned6_valid = extractor._validate_and_clean(output6_valid)
test("'तीस दिनभित्र' preserved", cleaned6_valid.duration_days == "तीस दिनभित्र")

output6_valid2 = ClauseNEROutput(duration_days="एक वर्ष")
cleaned6_valid2 = extractor._validate_and_clean(output6_valid2)
test("'एक वर्ष' preserved", cleaned6_valid2.duration_days == "एक वर्ष")

output6_valid3 = ClauseNEROutput(duration_days="सात कार्य दिनभित्र")
cleaned6_valid3 = extractor._validate_and_clean(output6_valid3)
test("'सात कार्य दिनभित्र' preserved", cleaned6_valid3.duration_days == "सात कार्य दिनभित्र")


# ═══════════════════════════════════════
# ERROR 7: Invalid office strings
# ═══════════════════════════════════════
print("\n--- Error 7: Invalid office strings ---")

output7_invalid = ClauseNEROutput(office="बमोजिम")
cleaned7_invalid = extractor._validate_and_clean(output7_invalid)
test("'बमोजिम' rejected as office", cleaned7_invalid.office is None)

output7_invalid2 = ClauseNEROutput(office="बमोजिम कार्यालय")
cleaned7_invalid2 = extractor._validate_and_clean(output7_invalid2)
test("'बमोजिम कार्यालय' rejected as office", cleaned7_invalid2.office is None)

output7_invalid3 = ClauseNEROutput(office="निकायको सिफारिसमा मन्त्रालय")
cleaned7_invalid3 = extractor._validate_and_clean(output7_invalid3)
test("'निकायको सिफारिसमा मन्त्रालय' rejected", cleaned7_invalid3.office is None)

output7_valid = ClauseNEROutput(office="मन्त्रालय")
cleaned7_valid = extractor._validate_and_clean(output7_valid)
test("'मन्त्रालय' preserved as valid office", cleaned7_valid.office == "मन्त्रालय")

output7_valid2 = ClauseNEROutput(office="उद्योग दर्ता गर्ने निकाय")
cleaned7_valid2 = extractor._validate_and_clean(output7_valid2)
test("'उद्योग दर्ता गर्ने निकाय' preserved", cleaned7_valid2.office == "उद्योग दर्ता गर्ने निकाय")


# ═══════════════════════════════════════
# BONUS: Price/penalty filtering
# ═══════════════════════════════════════
print("\n--- Bonus: Price/penalty filtering ---")

output_penalty = ClauseNEROutput(price="जरिवाना रु. ५,०००")
cleaned_penalty = extractor._validate_and_clean(output_penalty)
test("Penalty price rejected", cleaned_penalty.price is None)

output_valid_price = ClauseNEROutput(price="दस्तुर रु. ५,०००")
cleaned_valid_price = extractor._validate_and_clean(output_valid_price)
test("Valid fee price preserved", cleaned_valid_price.price == "दस्तुर रु. ५,०००")

output_no_amount = ClauseNEROutput(price="पैंतीस प्रतिशत")
cleaned_no_amount = extractor._validate_and_clean(output_no_amount)
test("Percentage without fee indicator rejected", cleaned_no_amount.price is None)


# ═══════════════════════════════════════
# BONUS: Short prerequisites filtering
# ═══════════════════════════════════════
print("\n--- Bonus: Prerequisites filtering ---")

output_short_prereq = ClauseNEROutput(prerequisites="हुने")
cleaned_short_prereq = extractor._validate_and_clean(output_short_prereq)
test("Very short prerequisite rejected", cleaned_short_prereq.prerequisites is None)


# ═══════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    print("  All tests passed! ✅")
    sys.exit(0)
