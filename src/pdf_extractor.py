import sys
import os

# Add project root to python path to allow running directly from src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import re
from src.pipeline.extractor import PdfExtractor
from src.pipeline.cleaner import TextCleaner
from src.pipeline.formatter import RegexFormatter


def extract_metadata_from_text(front_matter_text):
    """
    Dynamically extract metadata from the first few pages of the act.
    """
    from src.pipeline.formatter import nepali_to_int

    metadata = {
        "title": "Unknown Act",
        "number": "",
        "bs_year": None,
        "date_published": None,
        "date_commenced": None,
        "type": "act",
        "source": "",
        "lang": "ne",
        "amended_refs": [],
    }

    lines = [line.strip() for line in front_matter_text.split("\n") if line.strip()]

    # Heuristics for Title (Usually the first large text or near the top ending with year)
    for line in lines[:10]:
        if "ऐन" in line and re.search(r"[०-९\d]{4}", line):
            metadata["title"] = line
            # Extract year
            year_match = re.search(r"([०-९\d]{4})", line)
            if year_match:
                metadata["bs_year"] = nepali_to_int(year_match.group(1))
            break

    # Fallback: if the ऐन-based heuristic didn't find a title,
    # scan first page lines for common Nepali legal document type keywords.
    if metadata["title"] == "Unknown Act":
        # Common document types found on title pages of Nepali legal PDFs
        doc_type_keywords = [
            "ऐन",
            "निर्देशिका",
            "नियमावली",
            "विनियमावली",
            "कार्यविधि",
            "आदेश",
            "नीति",
            "सन्धि",
            "सम्झौता",
            "अधिनियम",
            "संहिता",
            "मापदण्ड",
            "सूचना",
            "विज्ञप्ति",
            "अधिसूचना",
        ]
        # Map keywords to document type values
        doc_type_map = {
            "ऐन": "act",
            "निर्देशिका": "directive",
            "नियमावली": "regulation",
            "विनियमावली": "bylaw",
            "कार्यविधि": "procedure",
            "आदेश": "order",
            "नीति": "policy",
            "सन्धि": "treaty",
            "सम्झौता": "agreement",
            "अधिनियम": "statute",
            "संहिता": "code",
            "मापदण्ड": "standard",
            "सूचना": "notice",
            "विज्ञप्ति": "communique",
            "अधिसूचना": "notification",
        }

        for i, line in enumerate(lines[:15]):
            matched_keyword = None
            for keyword in doc_type_keywords:
                if keyword in line:
                    matched_keyword = keyword
                    break

            if matched_keyword:
                metadata["title"] = line
                metadata["type"] = doc_type_map.get(matched_keyword, "act")

                # Check if the year is in the same line
                year_match = re.search(r"([०-९\d]{4})", line)
                if year_match:
                    metadata["bs_year"] = nepali_to_int(year_match.group(1))
                else:
                    # Look at the next few lines for a standalone year
                    # (Note: the cleaner may strip standalone number lines,
                    # so this may not find it — the date fallback below covers that)
                    for next_line in lines[i + 1 : i + 4]:
                        year_match = re.search(
                            r"^[^\d०-९]*([०-९\d]{4})[^\d०-९]*$", next_line
                        )
                        if year_match:
                            metadata["bs_year"] = nepali_to_int(year_match.group(1))
                            break
                break

    # --- Date extraction (runs unconditionally, regardless of which title
    # heuristic matched above — previously this was nested inside the
    # "Unknown Act" fallback only, so it silently never ran whenever the
    # primary title heuristic succeeded). Handles स्वीकृत/प्रकाशित/प्रमाणीकरण
    # मिति labels, and dates that appear on the line AFTER the label line
    # (common when OCR/layout splits label and value across lines). ---
    date_label_fields = [
        (r"स्वीक[ृत]+\s*मिति", "date_commenced"),
        (r"प्रमाणीकरण\s*मिति", "date_commenced"),
        (r"प्रकाशित\s*मिति", "date_published"),
    ]
    date_value_re = re.compile(r"([०-९\d]+\s*[।/]\s*[०-९\d]+\s*[।/]\s*[०-९\d]+)")

    for i, line in enumerate(lines[:20]):
        for label_re, field in date_label_fields:
            if metadata[field]:
                continue
            if re.search(label_re, line):
                m = date_value_re.search(line)
                if not m:
                    for next_line in lines[i + 1 : i + 3]:
                        m = date_value_re.search(next_line)
                        if m:
                            break
                if m:
                    metadata[field] = m.group(1).replace(" ", "")

    # If year still not found, extract from the date strings
    if metadata["bs_year"] is None:
        date_str = metadata["date_commenced"] or metadata["date_published"]
        if date_str:
            year_from_date = re.search(r"([०-९\d]{4})", date_str)
            if year_from_date:
                metadata["bs_year"] = nepali_to_int(year_from_date.group(1))

    # --- Amendment list extraction (merges wrapped continuation lines,
    # since act names often wrap across 2 lines with the date on the
    # second line) ---
    amendments_started = False
    current_entry_lines = []

    def flush_entry(entry_lines):
        if not entry_lines:
            return
        full_text = " ".join(entry_lines)
        date_match = date_value_re.search(full_text)
        date_val = date_match.group(1).replace(" ", "") if date_match else ""
        ref_name = re.sub(r"^[०-९\d]+\.\s*", "", full_text)
        if date_match:
            ref_name = full_text[: date_match.start()]
            ref_name = re.sub(r"^[०-९\d]+\.\s*", "", ref_name).strip()
        else:
            ref_name = ref_name.strip()
        metadata["amended_refs"].append({"ref": ref_name, "date": date_val})

    for line in lines:
        if "संशोधन गर्ने ऐन" in line:
            amendments_started = True
            continue

        if amendments_started:
            if re.match(r"^[०-९\d]+\.\s", line):
                flush_entry(current_entry_lines)
                current_entry_lines = [line]
            elif (
                line.startswith("परिच्छेद")
                or "प्रमाणीकरण" in line
                or re.match(r"^संवत्", line)
            ):
                break
            else:
                current_entry_lines.append(line)

    flush_entry(current_entry_lines)

    return metadata


def slugify(filename):
    # e.g. "comapy_act.pdf" -> "comapy_act"
    base = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "input_pdfs")
    output_dir = os.path.join(project_root, "output_jsons")

    os.makedirs(output_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    cleaner = TextCleaner()

    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path}...")
        extractor = PdfExtractor(pdf_path)

        # 1. Read first 2 pages for metadata
        front_matter = []
        for page_num in range(min(2, len(extractor.doc))):
            page = extractor.doc.load_page(page_num)
            raw_text = extractor._ocr_page(page)
            front_matter.append(cleaner.clean(raw_text))

        metadata = extract_metadata_from_text("\n".join(front_matter))
        act_slug = slugify(pdf_path)
        print(
            f"Extracted Metadata successfully. Found {len(metadata['amended_refs'])} amendments."
        )

        formatter = RegexFormatter(metadata, act_slug)

        full_cleaned_text = []
        # Process all pages
        for page_num, raw_text in extractor.extract_all_pages():
            print(f"  OCR Page {page_num}/{len(extractor.doc)}...")
            cleaned_text = cleaner.clean(raw_text)
            full_cleaned_text.append(cleaned_text)

        extractor.close()

        print("  Formatting text into JSON structure...")
        combined_text = "\n".join(full_cleaned_text)
        dataset = formatter.process_text(combined_text)

        output_filename = f"{act_slug}_dataset.json"
        output_path = os.path.join(output_dir, output_filename)

        print(f"  Writing output to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(
            f"Successfully processed {act_slug}: {len(dataset['chunks'])} chunks generated."
        )


if __name__ == "__main__":
    main()
