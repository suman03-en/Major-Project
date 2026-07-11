# RAG for Business Domain in Nepal
 - It handles query related to business registration process and related questions.

# Nepali Legal PDF Extraction & Structuring Pipeline

This project provides an automated pipeline to extract text from Nepali legal PDFs (which often contain broken Unicode mappings or Preeti-like legacy fonts) and cleanly structure them into a highly nested, hierarchical JSON format optimized for RAG (Retrieval-Augmented Generation) systems.

The extraction bypasses broken PDF digital text by rendering pages as high-resolution images using `PyMuPDF` and extracting flawless Unicode Devanagari using `Tesseract OCR` configured specifically for Nepali document layouts.

## Prerequisites

1. **Python 3.10+** installed on your system.
2. **Tesseract OCR**:
   - Download and install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
   - During installation, make sure to check the box to install the **Nepali (`nep`) language pack**.
   - By default, the script looks for Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you install it elsewhere, update the path in `src/pipeline/extractor.py`.

## Installation Setup

1. Create a virtual environment (recommended):
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install the required Python packages:
   ```cmd
   pip install -r requirements.txt
   ```

## Project Directory Structure

```text
Major project/
│── pdf_extractor.py           # Main orchestration script
│── requirements.txt           # Python dependencies
│── README.md                  # This file
│── .gitignore
│── input_pdfs/                # 📂 Place your raw Nepali PDFs here!
│── output_jsons/              # 📂 Generated JSON datasets will appear here
└── src/
    ├── __init__.py
    ├── pipeline/
    │   ├── __init__.py
    │   ├── extractor.py       # PyMuPDF + Tesseract OCR hybrid logic
    │   ├── cleaner.py         # Regex-based text artifact cleaner
    │   └── formatter.py       # Regex-based structural JSON parser
    └── prompt/
        └── master_prompt_v1.md
```

## Usage Instructions

1. Place one or more Nepali legal PDF files (e.g., `company_act.pdf`) into the `input_pdfs/` directory.
2. Run the main extraction script:
   ```cmd
   python pdf_extractor.py
   ```
3. The script will automatically:
   - Parse the first few pages of each PDF to extract metadata and amendment references (`sansodhan`).
   - OCR all pages sequentially using Tesseract (PSM 4 mode to preserve legal column structure).
   - Clean URLs, floating page numbers, and excess spacing.
   - Parse the text hierarchically into Chapters (`परिच्छेद`), Sections (`दफा`), Subsections `(१)`, and Clauses `(क)`, mapping cross-references automatically.
4. Check the `output_jsons/` folder for your processed datasets, named dynamically (e.g., `company_act_dataset.json`).

## Architecture Details

- **Extractor**: Renders PDFs to 300 DPI images and leverages Tesseract's `psm 4` (Assume a single column of text of variable sizes) to ensure bulleted clauses like `(क)` remain attached to their definitions instead of splitting into distinct columns.
- **Cleaner**: Standard Regex heuristic passes to remove things like `www.lawcommission.gov.np`.
- **Formatter**: Eliminates title duplication from body texts, generates stats (characters & tokens), extracts `तर` (Provisos) and `स्पष्टीकरण` (Explanations), and builds an exact hierarchical schema ready for LLM vector embeddings.