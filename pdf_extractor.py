import os
import glob
import json
import re
from dataset.pipeline.extractor import PdfExtractor
from dataset.pipeline.cleaner import TextCleaner
from dataset.pipeline.formatter import RegexFormatter

def extract_metadata_from_text(front_matter_text):
    """
    Dynamically extract metadata from the first few pages of the act.
    """
    metadata = {
        "title": "Unknown Act",
        "number": "",
        "bs_year": None,
        "date_published": "",
        "date_commenced": "",
        "type": "act",
        "source": "",
        "lang": "ne",
        "amended_refs": []
    }
    
    lines = [line.strip() for line in front_matter_text.split('\n') if line.strip()]
    
    # Heuristics for Title (Usually the first large text or near the top ending with year)
    for line in lines[:10]:
        if 'ऐन' in line and re.search(r'[०-९\d]{4}', line):
            metadata["title"] = line
            # Extract year
            year_match = re.search(r'([०-९\d]{4})', line)
            if year_match:
                from dataset.pipeline.formatter import nepali_to_int
                metadata["bs_year"] = nepali_to_int(year_match.group(1))
            break
            
    # Extract amendments
    amendments_started = False
    for line in lines:
        if 'संशोधन गर्ने ऐन' in line:
            amendments_started = True
            continue
        
        if amendments_started:
            # Usually format is "१. केही नेपाल ऐनलाई संशोधन गर्ने ऐन, २०६४    २०६४।०५।०९"
            if re.match(r'^[०-९\d]+\.\s', line):
                # Clean up the date part at the end
                act_name = re.sub(r'[०-९\d]+\s*[।/]\s*[०-९\d]+\s*[।/]\s*[०-९\d]+.*$', '', line).strip()
                metadata["amended_refs"].append(act_name)
            elif line.startswith('परिच्छेद') or 'प्रमाणीकरण' in line:
                pass # Maybe end of amendments
            
    return metadata

def slugify(filename):
    # e.g. "comapy_act.pdf" -> "comapy_act"
    base = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'[^a-zA-Z0-9]+', '_', base).strip('_').lower()

def main():
    input_dir = 'input_pdfs'
    output_dir = 'output_jsons'
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    
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
        print(f"Extracted Metadata successfully. Found {len(metadata['amended_refs'])} amendments.")
        
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
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully processed {act_slug}: {len(dataset['chunks'])} chunks generated.")

if __name__ == "__main__":
    main()
