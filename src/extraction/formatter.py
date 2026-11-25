import re

from src.utils import nepali_to_int


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

MAX_CHUNK_CHARS = 1200


def _split_oversized_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text exceeding max_chars at Devanagari sentence boundaries ('।')."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=।)\s+', text)
    if len(sentences) <= 1:
        return [text]

    segments: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            segments.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        segments.append(current)
    return segments or [text]


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
        # Pre-split mid-line subsection/clause markers that follow a boundary punctuation mark
        # (।, :, ः, ;) so OCR-merged lines split cleanly without breaking inline references.
        full_text = re.sub(
            r'([।:ः;])\s*(\(\s*(?:[०-९\d]+|[क-ज्ञ]{1,2}[०-९\d]?)\s*\))',
            r'\1\n\2',
            full_text,
        )
        lines = full_text.split('\n')
        current_text_block = []
        
        def save_chunk(text_content):
            if not text_content.strip():
                return
                
            text = text_content.strip()
            
            # Title Deduplication
            if self.current_sec_title:
                pattern = r'^([०-९\d]+)\.?\s*' + re.escape(self.current_sec_title) + r'[:ः]?\s*'
                text = re.sub(pattern, '', text).strip()
                
            if self.current_sub is not None:
                # Strip the (१) part
                text = re.sub(r'^\(\s*[०-९\d]+\s*\)\s*', '', text)
                
            if self.current_clause is not None:
                # Strip the (क) / (क१) / (खख) part
                text = re.sub(r'^\(\s*[क-ज्ञ]{1,2}[०-९\d]?\s*\)\s*', '', text)
                
            if not text.strip():
                return

            provisos = []
            explanations = []

            # Split explanations (स्पष्टीकरणः …) first so they aren't swallowed into trailing provisos
            explanation_parts = re.split(r'स्पष्टीकरणः\s*', text)
            if len(explanation_parts) > 1:
                text = explanation_parts[0].strip()
                for part in explanation_parts[1:]:
                    if part.strip():
                        explanations.append('स्पष्टीकरणः ' + part.strip())

            # Split provisos (तर …) — handles multiple provisos in one clause
            proviso_parts = re.split(r'(?<=।)\s*तर\s+', text)
            if len(proviso_parts) > 1:
                text = proviso_parts[0].strip()
                for part in proviso_parts[1:]:
                    provisos.append('तर ' + part.strip())

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

            has_parent = (
                (self.current_clause is not None and (self.current_sub is not None or self.current_sec is not None))
                or (self.current_sub is not None and self.current_sec is not None)
            )
            parent_id = "-".join(slug_parts[:-1]) if has_parent else None

            status = self.act_metadata.get("status", "active")
            effective_date = (
                self.act_metadata.get("effective_date")
                or self.act_metadata.get("date_commenced")
                or self.act_metadata.get("date_published")
            )
            bs_year = self.act_metadata.get("bs_year")

            hierarchy = {}
            if self.current_ch is not None: hierarchy['ch'] = self.current_ch
            if self.current_ch_title: hierarchy['ch_title'] = self.current_ch_title
            if self.current_sec is not None: hierarchy['sec'] = self.current_sec
            if self.current_sec_title: hierarchy['sec_title'] = self.current_sec_title
            if self.current_sub is not None: hierarchy['sub'] = self.current_sub
            if self.current_clause is not None: hierarchy['clause'] = self.current_clause
            
            segments = _split_oversized_text(text)
            for idx, seg_text in enumerate(segments, 1):
                part_id = chunk_id if len(segments) == 1 else f"{chunk_id}-p{idx}"
                chunk = {
                    "id": part_id,
                    "type": chunk_type,
                    "hierarchy": hierarchy,
                    "parent_id": parent_id,
                    "status": status,
                    "effective_date": effective_date,
                    "bs_year": bs_year,
                    "text": seg_text,
                }
                if idx == len(segments):
                    if provisos:
                        chunk["provisos"] = provisos
                    if explanations:
                        chunk["explanations"] = explanations

                refs = extract_references(seg_text, self.slug_prefix)
                if refs:
                    chunk["refs"] = refs

                chunk["stats"] = {
                    "characters": len(seg_text),
                    "tokens": max(1, len(seg_text) // 3),  # Devanagari-aware estimate
                }
                self.chunks.append(chunk)

        in_front_matter = True
        seen_amendment_header = False
        non_empty_count = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if in_front_matter:
                non_empty_count += 1

                if line.startswith('संशोधन गर्ने ऐन'):      
                    seen_amendment_header = True
                    continue

                # Original act markers
                if (line.startswith(('प्रस्तावना', 'परिच्छेद')) or 'संक्षिप्त नाम' in line):
                    in_front_matter = False
                # Gazette / notification markers (scanned image PDFs)
                elif line in ('सूचना', 'आदेश', 'निर्देशिका', 'विज्ञप्ति',
                              'अधिसूचना', 'नियमावली'):
                    in_front_matter = False
                    current_text_block.append(line)
                    continue

                # Numbered section or subsection at the start means content has begun
                elif not seen_amendment_header and (
                    re.match(r'^[०-९\d]+\.', line)
                    or re.match(r'^[०-९\d]{1,3}\s+[^।]+[:ः]', line)
                    or re.match(r'^\(\s*[०-९\d]+\s*\)', line)
                ) or non_empty_count > 20:
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

            if (
                self.current_ch is not None
                and self.current_ch_title is None
                and not re.match(r'^[०-९\d]+\.', line)
                and not re.match(r'^[०-९\d]{1,3}\s+[^।]+[:ः]', line)
            ):
                self.current_ch_title = line
                continue
                
            sec_match = (
                re.match(r'^([०-९\d]+)\.\s*(.+?)(?:[:ः]|$)', line)
                or re.match(r'^([०-९\d]{1,3})\s+([^।]+?)[:ः]', line)
            )
            if sec_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.current_sec = nepali_to_int(sec_match.group(1))
                sec_title = sec_match.group(2).strip()
                remainder = line[sec_match.end():].strip()

                # Handle same-line "(१) ..." immediately following section title (with or without colon)
                title_sub_match = re.search(r'\(\s*([०-९\d]+)\s*\)\s*.*$', sec_title)
                if title_sub_match and title_sub_match.start() > 0:
                    self.current_sec_title = sec_title[:title_sub_match.start()].strip()
                    self.current_sub = nepali_to_int(title_sub_match.group(1))
                    self.current_clause = None
                    sub_text = sec_title[title_sub_match.start():]
                    if remainder:
                        sub_text = f"{sub_text} {remainder}"
                    current_text_block.append(sub_text)
                elif remainder and re.match(r'^\(\s*([०-९\d]+)\s*\)', remainder):
                    rem_sub_match = re.match(r'^\(\s*([०-९\d]+)\s*\)', remainder)
                    self.current_sec_title = sec_title
                    self.current_sub = nepali_to_int(rem_sub_match.group(1))
                    self.current_clause = None
                    current_text_block.append(remainder)
                else:
                    self.current_sec_title = sec_title
                    self.current_sub = None
                    self.current_clause = None
                    current_text_block.append(line)
                continue
                
            sub_match = re.match(r'^\(\s*([०-९\d]+)\s*\)\s*(.*)', line)
            if sub_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.current_sub = nepali_to_int(sub_match.group(1))
                self.current_clause = None
                current_text_block.append(line)
                continue
                
            clause_match = re.match(r'^\(\s*([क-ज्ञ]{1,2}[०-९\d]?)\s*\)\s*(.*)', line)
            if clause_match:
                save_chunk(" ".join(current_text_block))
                current_text_block = []
                self.current_clause = clause_match.group(1)
                current_text_block.append(line)
                continue
                
            current_text_block.append(line)
            
        save_chunk(" ".join(current_text_block))
        self.chunks.extend(self._build_section_summaries())

        return {
            "act_metadata": self.act_metadata,
            "chunks": self.chunks
        }

    def _build_section_summaries(self):
        """Build one deterministic section_summary provision per Section (दफा)."""
        groups = {}
        for chunk in self.chunks:
            if chunk["type"].startswith("anusuchi") or chunk["type"] == "section_summary":
                continue
            hierarchy = chunk.get("hierarchy", {})
            sec = hierarchy.get("sec")
            if sec is None:
                continue
            ch = hierarchy.get("ch")
            key = (ch, sec)
            groups.setdefault(key, []).append(chunk)

        status = self.act_metadata.get("status", "active")
        effective_date = (
            self.act_metadata.get("effective_date")
            or self.act_metadata.get("date_commenced")
            or self.act_metadata.get("date_published")
        )
        bs_year = self.act_metadata.get("bs_year")

        summaries = []
        for (ch, sec), group_chunks in groups.items():
            first_h = group_chunks[0]["hierarchy"]
            ch_title = first_h.get("ch_title")
            sec_title = first_h.get("sec_title")

            id_parts = [self.slug_prefix]
            if ch is not None:
                id_parts.append(f"ch{ch}")
            id_parts.append(f"sec{sec}")
            summary_id = f"{'-'.join(id_parts)}-summary"

            summary_hierarchy = {}
            if ch is not None:
                summary_hierarchy["ch"] = ch
            if ch_title:
                summary_hierarchy["ch_title"] = ch_title
            summary_hierarchy["sec"] = sec
            if sec_title:
                summary_hierarchy["sec_title"] = sec_title

            header = f"{sec}. {sec_title}" if sec_title else f"दफा {sec}"
            body = "\n".join(c["text"] for c in group_chunks)
            summary_text = f"{header}\n{body}"

            summary_chunk = {
                "id": summary_id,
                "type": "section_summary",
                "hierarchy": summary_hierarchy,
                "parent_id": None,
                "status": status,
                "effective_date": effective_date,
                "bs_year": bs_year,
                "text": summary_text,
            }
            refs = extract_references(summary_text, self.slug_prefix)
            if refs:
                summary_chunk["refs"] = refs
            summary_chunk["stats"] = {
                "characters": len(summary_text),
                "tokens": max(1, len(summary_text) // 3),
            }
            summaries.append(summary_chunk)
        return summaries
