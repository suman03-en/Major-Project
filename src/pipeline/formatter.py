import re
import json

NEPALI_DIGITS = '०१२३४५६७८९'
ENGLISH_DIGITS = '0123456789'
trans_table = str.maketrans(NEPALI_DIGITS, ENGLISH_DIGITS)

def nepali_to_int(nepali_str):
    try:
        clean_str = re.sub(r'[^\d०-९]', '', nepali_str)
        if not clean_str:
            return None
        return int(clean_str.translate(trans_table))
    except ValueError:
        return None

def extract_references(text, slug_prefix):
    """
    Extract internal references like 'दफा २३', 'उपदफा (१)', 'परिच्छेद ५'.
    Returns a list of ref objects.
    """
    refs = []
    # Pattern to match reference type and the following number or (number/letter)
    pattern = r'(दफा|उपदफा|परिच्छेद|खण्ड)\s+([०-९\d]+|\([०-९\dक-ज्ञ]+\))'
    
    for match in re.finditer(pattern, text):
        ref_type_ne = match.group(1)
        ref_val_ne = match.group(2)
        exact_text = match.group(0)
        
        # Map to english types for the ID
        type_map = {
            'दफा': 'sec',
            'उपदफा': 'sub',
            'परिच्छेद': 'ch',
            'खण्ड': 'cl'
        }
        ref_type = type_map.get(ref_type_ne, 'sec')
        
        # Clean value
        val_clean = re.sub(r'[()\s]', '', ref_val_ne)
        if ref_type in ['sec', 'sub', 'ch']:
            val_int = nepali_to_int(val_clean)
            ref_id_part = f"{ref_type}{val_int}" if val_int is not None else f"{ref_type}{val_clean}"
        else:
            ref_id_part = f"{ref_type}{val_clean}"
            
        ref_id = f"{slug_prefix}-{ref_id_part}"
        
        refs.append({
            "type": "section" if ref_type == 'sec' else ("chapter" if ref_type == 'ch' else "subsection"),
            "id": ref_id,
            "text": exact_text
        })
    return refs

class RegexFormatter:
    def __init__(self, act_metadata, act_slug):
        self.act_metadata = act_metadata
        self.act_slug = act_slug
        self.slug_prefix = f"{act_slug}-{act_metadata.get('bs_year', '')}"
        self.chunks = []
        
        # State tracking
        self.current_ch = None
        self.current_ch_title = None
        self.current_sec = None
        self.current_sec_title = None
        self.current_sub = None
        self.current_clause = None
        self.is_anusuchi = False

    def process_text(self, full_text):
        lines = full_text.split('\n')
        current_text_block = []
        
        def save_chunk(text_content):
            if not text_content.strip():
                return
                
            text = text_content.strip()
            
            # Title Deduplication
            if self.current_sec_title:
                pattern = r'^([०-९\d]+)\.\s*' + re.escape(self.current_sec_title) + r'[:ः]?\s*'
                text = re.sub(pattern, '', text).strip()
                
            if self.current_sub is not None:
                # Strip the (१) part
                text = re.sub(r'^\([०-९\d]+\)\s*', '', text)
                
            if self.current_clause is not None:
                # Strip the (क) part
                text = re.sub(r'^\([क-ज्ञ]\)\s*', '', text)
                
            if not text.strip():
                return

            provisos = []
            explanations = []
            
            if 'तर ' in text:
                parts = text.split('तर ')
                text = parts[0].strip()
                provisos.append('तर ' + parts[1].strip())
                
            if 'स्पष्टीकरणः' in text:
                parts = text.split('स्पष्टीकरणः')
                text = parts[0].strip()
                explanations.append('स्पष्टीकरणः ' + parts[1].strip())

            chunk_type = 'section'
            if self.current_sub is not None:
                chunk_type = 'subsection'
            if self.current_clause is not None:
                chunk_type = 'clause'
                
            if self.is_anusuchi:
                if self.current_sec is None and self.current_sub is None and self.current_clause is None:
                    chunk_type = 'anusuchi'
                else:
                    chunk_type = f"anusuchi_{chunk_type}"
                
            slug_parts = [self.slug_prefix]
            if self.is_anusuchi:
                if self.current_ch is not None: slug_parts.append(f"anusuchi{self.current_ch}")
                else: slug_parts.append("anusuchi")
            else:
                if self.current_ch is not None: slug_parts.append(f"ch{self.current_ch}")
                
            if self.current_sec is not None: slug_parts.append(f"sec{self.current_sec}")
            if self.current_sub is not None: slug_parts.append(f"sub{self.current_sub}")
            if self.current_clause is not None: slug_parts.append(f"cl{self.current_clause}")
            
            chunk_id = "-".join(slug_parts)

            hierarchy = {}
            if self.current_ch is not None: hierarchy['ch'] = self.current_ch
            if self.current_ch_title: hierarchy['ch_title'] = self.current_ch_title
            if self.current_sec is not None: hierarchy['sec'] = self.current_sec
            if self.current_sec_title: hierarchy['sec_title'] = self.current_sec_title
            if self.current_sub is not None: hierarchy['sub'] = self.current_sub
            if self.current_clause is not None: hierarchy['clause'] = self.current_clause
            
            chunk = {
                "id": chunk_id,
                "type": chunk_type,
                "hierarchy": hierarchy,
                "text": text,
            }
            if provisos: chunk["provisos"] = provisos
            if explanations: chunk["explanations"] = explanations
            
            # Extract refs
            refs = extract_references(text, self.slug_prefix)
            if refs:
                chunk["refs"] = refs
            
            chunk["stats"] = {
                "characters": len(text),
                "tokens": int(len(text.split()) * 1.5)
            }
            self.chunks.append(chunk)

        in_front_matter = True
        non_empty_count = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if in_front_matter:
                non_empty_count += 1
                # Original act markers
                if (line.startswith('प्रस्तावना') or line.startswith('परिच्छेद')
                        or 'संक्षिप्त नाम' in line):
                    in_front_matter = False
                # Gazette / notification markers (scanned image PDFs)
                elif line in ('सूचना', 'आदेश', 'निर्देशिका', 'विज्ञप्ति',
                              'अधिसूचना', 'नियमावली'):
                    in_front_matter = False
                    current_text_block.append(line)
                    continue
                # Numbered section or subsection at the start means content has begun
                elif re.match(r'^[०-९\d]+\.', line) or re.match(r'^\([०-९\d]+\)', line):
                    in_front_matter = False
                # Safety fallback: if no marker found in first 20 non-empty lines,
                # assume there is no front matter and process everything.
                elif non_empty_count > 20:
                    in_front_matter = False
                else:
                    continue
                
            anusuchi_match = re.match(r'^अनुसूची\s*[-–]?\s*([०-९\d]*)(?:\s*\(([क-ज्ञ])\))?', line)
            if anusuchi_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.is_anusuchi = True
                
                anusuchi_val = anusuchi_match.group(1)
                self.current_ch = nepali_to_int(anusuchi_val) if anusuchi_val else 1
                self.current_ch_title = line
                self.current_sec = None
                self.current_sec_title = None
                self.current_sub = None
                
                clause_val = anusuchi_match.group(2)
                self.current_clause = clause_val if clause_val else None
                current_text_block.append(line)
                continue
                
            ch_match = re.match(r'^परिच्छेद\s*[-–]?\s*([०-९\d]+)', line)
            if ch_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.is_anusuchi = False
                self.current_ch = nepali_to_int(ch_match.group(1))
                self.current_ch_title = None 
                self.current_sec = None
                self.current_sec_title = None
                self.current_sub = None
                self.current_clause = None
                continue

            if self.current_ch is not None and self.current_ch_title is None and not re.match(r'^[०-९\d]+\.', line):
                self.current_ch_title = line
                continue
                
            sec_match = re.match(r'^([०-९\d]+)\.\s*(.+?)(?:[:ः]|$)', line)
            if sec_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.current_sec = nepali_to_int(sec_match.group(1))
                self.current_sec_title = sec_match.group(2).strip()
                self.current_sub = None
                self.current_clause = None
                current_text_block.append(line)
                continue
                
            sub_match = re.match(r'^\(([०-९\d]+)\)\s*(.*)', line)
            if sub_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.current_sub = nepali_to_int(sub_match.group(1))
                self.current_clause = None
                current_text_block.append(line)
                continue
                
            clause_match = re.match(r'^\(([क-ज्ञ])\)\s*(.*)', line)
            if clause_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.current_clause = clause_match.group(1)
                current_text_block.append(line)
                continue
                
            current_text_block.append(line)
            
        save_chunk(" ".join(current_text_block))
        
        return {
            "act_metadata": self.act_metadata,
            "chunks": self.chunks
        }
