Act as an elite Legal Data Engineer and Expert Dataset Generator with 20 years of experience specializing in the Nepalese legal and corporate compliance domains. 

Your objective is to extract data from the attached Nepalese legal PDF and convert it into a highly precise, production-ready JSON dataset optimized for LLM training and advanced hybrid retrieval systems (RAG).

### CRITICAL INSTRUCTIONS FOR LEGAL DOMAIN ACCURACY:
1. STRICT VERBATIM RULE: You must NOT paraphrase, summarize, adapt, or alter a single word or punctuation mark in the text. Legal definitions change meaning with slight modifications. The text MUST be extracted exactly as it appears in the law.
2. HANDLING FONTS (PREETI/UNICODE): If the input text uses legacy fonts (like Preeti, Kantipur, etc.) or is a scanned document, you must programmatically decode or map it to standard Unicode Devanagari accurately (e.g., 'sDkgL' to 'कम्पनी'). 
3. CHUNK COMPRESSION RULE: To keep the dataset optimized, if an optional field is empty (such as an empty array [] or a null value), simply OMIT that key from that specific chunk entirely. Do not include empty fields.
4. TOKEN ESTIMATION: For the "stats.tokens" field, estimate the token count by assuming roughly 1.5 tokens per Nepali word. 

### JSON SCHEMA SPECIFICATIONS:
You must output the final data strictly matching this enhanced JSON structure:

{
  "act_metadata": {
    "title": "Full name of the Act/Law (e.g., 'कम्पनी ऐन, २०६३')",
    "number": "Official Law Number (e.g., '२०६३ को ऐन नं. ३४')",
    "bs_year": Bikram Sambat year as an integer (e.g., 2063),
    "date_published": "Date of publication in YYYY-MM-DD format",
    "date_commenced": "Date of commencement in YYYY-MM-DD format",
    "type": "Type of legal document ('act', 'niyamawali', 'nirdeshika', 'suchana', 'karyabidhi')",
    "source": "Source body (e.g., 'Nepal Law Commission' or 'कम्पनी रजिष्ट्रारको कार्यालय')",
    "lang": "ne"
    "keywords": [
        "Generate 3-6 highly relevant legal keywords from this specific text block for hybrid search/BM25 retrieval (e.g., 'कम्पनी', 'संस्थापक', 'दर्ता')."
      ]
  },
  "chunks": [
    {
      "id": "A unique, descriptive slug. Format: {act-slug}-{year}-ch{X}-sec{Y}-sub{Z}-cl{A}",
      "type": "The granularity of this specific chunk ('section', 'subsection', 'clause', or 'sub_clause')",
      "hierarchy": {
        "ch": "Chapter/Parichhed number as an INTEGER (e.g., 1 instead of '१')",
        "ch_title": "Name of the chapter",
        "sec": "Section/Dafa number as an INTEGER (e.g., 2 instead of '२')",
        "sec_title": "Title/heading of the section",
        "sub": "Subsection/Upadafa number as an INTEGER (Optional). Omit if none.",
        "clause": "Clause/Khanda letter in Nepali (e.g., 'क'). (Optional). Omit if none.",
        "sub_clause": "Sub-clause/Upa-khanda value (Optional). Omit if none."
      },
      "text": "The exact legal text in pure, flawless Unicode Devanagari.",
      "provisos": ["List of strings containing 'तर' (Provided that / exceptions) clauses verbatim. Omit key if none."],
      "explanations": ["List of strings containing 'स्पष्टीकरण' (Explanations) verbatim. Omit key if none."],
      "refs": [
        {
          "type": "Type of reference ('section', 'chapter', 'act', etc.)",
          "id": "Standardized ID of the referenced target (e.g., 'company-act-2063-sec23')",
          "text": "The exact text phrase used inside the chunk (e.g., 'दफा २३')"
        }
      ],
      "amended_refs": ["List of amendment references if modified (e.g., ['कम्पनी (पहिलो संशोधन) ऐन, २०७४']). Omit key if not amended."],
      "stats": {
        "characters": "Integer representing the total character count of the 'text' field",
        "tokens": "Integer representing the estimated token count of the 'text' field"
      }
    }
  ]
}

### BATCH EXECUTION & TOKEN LIMIT MANAGEMENT:
Because legal documents are long, do not try to process the entire document at once to prevent context truncation or half-written JSON blocks. 

For this turn, process and generate the dataset ONLY for [INSERT PAGE RANGE HERE, e.g., Pages 1 to 5]. Ensure that you close the JSON formatting tags properly at the end of your response.

Begin processing now with maximum legal precision.