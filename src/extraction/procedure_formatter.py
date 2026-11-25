"""Zero-LLM Procedure and Citizen Charter (नागरिक बडापत्र) formatter."""

from __future__ import annotations

import re
from typing import Any

from src.utils import nepali_to_int

_NUMBERED_STEP_RE = re.compile(r"^\s*([०-९\d]+)\s*[.)।]\s+(.+)$")
_BULLET_STEP_RE = re.compile(r"^\s*(?:[-•*▪◦‣]|(?:\([०-९\dक-ज्ञ]+\)))\s+(.+)$")
_GENERIC_CHARTER_HEADERS = {"नागरिक बडापत्र", "citizen charter", "बडापत्र"}

_SERVICE_RE = re.compile(
    r"^(?:सेवाको नाम|सेवा|प्रक्रियाको नाम|Service(?:\s+Name)?)\s*[:ः]\s*(.+)$",
    re.IGNORECASE,
)
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "documents",
        re.compile(
            r"^(?:आवश्यक\s+कागजात(?:हरू)?|चाहिने\s+कागजात(?:हरू)?|संलग्न\s+गर्नुपर्ने\s+कागजात(?:हरू)?|Required\s+Documents?)\s*[:ः]\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "forms",
        re.compile(
            r"^(?:आवेदन\s*फाराम(?:हरू)?|फाराम(?:हरू)?|Form(?:s)?)\s*[:ः]\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "office",
        re.compile(
            r"^(?:जिम्मेवार\s*(?:कार्यालय|निकाय|शाखा|अधिकारी)|सेवा\s*(?:प्रदान गर्ने\s*)?कार्यालय|लाग्ने\s*स्थान|कार्यालय|Responsible\s+Office|Office)\s*[:ः]\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "fees",
        re.compile(
            r"^(?:(?:लाग्ने\s*)?(?:आवेदन\s*)?(?:दस्तुर|शुल्क)|Fee(?:s)?)\s*[:ः]\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "duration",
        re.compile(
            r"^(?:(?:लाग्ने\s*)?(?:समयावधि|समय|अवधि)|सेवा\s+प्रवाह\s+हुने\s+समय|Duration|Wait\s+Time|Processing\s+Time)\s*[:ः]\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "steps",
        re.compile(
            r"^(?:(?:अपनाउनुपर्ने\s*)?प्रक्रिया(?:हरू)?|चरण(?:हरू)?|Steps?)\s*[:ः]\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
]

_FEE_RE = re.compile(
    r"(?:(?:रु\.?|रू\.?|Rs\.?|NPR)\s*([०-९\d,]+)(?:/-)?)|(निःशुल्क|शुल्क\s*नलाग्ने|\bfree\b)",
    re.IGNORECASE,
)
_OFFICE_RE = re.compile(
    r"((?:[^\s,।]+(?:\s+[^\s,।]+){0,3}\s+)?(?:कार्यालय|विभाग|मन्त्रालय|महाशाखा|शाखा|समिति|आयोग|बोर्ड))(?:मा|बाट|लाई|को|माथि)?(?=\s|,|।|$)"
)
_FORM_RE = re.compile(
    r"(?:फाराम(?:\s*नं\.?)?\s*[०-९\dक-ज्ञA-Za-z-]+|अनुसूची\s*[-–]?\s*[०-९\d]+(?:\s*\([^)]+\))?|Form\s+[A-Za-z0-9-]+)"
)
_DURATION_RE = re.compile(
    r"(?:[०-९\d]+\s*(?:कार्य\s*दिन|कार्यदिन|दिन|घण्टा|हप्ता|महिना|वर्ष)|सोही\s+दिन|तत्काल|\d+\s*(?:working\s+days?|days?|hours?|weeks?|months?))",
    re.IGNORECASE,
)
_DOC_RE = re.compile(
    r"(?:नेपाली\s+)?नागरिकता(?:को\s+(?:प्रमाणपत्रको\s+)?प्रतिलिपि)?|प्रबन्धपत्र|नियमावली|कर\s+चुक्ता\s+प्रमाणपत्र|दर्ता\s+प्रमाणपत्र|सिफारिस\s+पत्र|पासपोर्ट\s+साइजको\s+फोटो|फोटो|रसिद|प्रमाणपत्र"
)


def _infer_office_level(name: str) -> str | None:
    if any(k in name for k in ("मन्त्रालय", "विभाग", "कम्पनी रजिष्ट्रार", "केन्द्रीय", "राष्ट्रिय", "आयोग", "बोर्ड")):
        return "केन्द्र"
    if any(k in name for k in ("प्रदेश", "प्रादेशिक")):
        return "प्रदेश"
    if any(k in name for k in ("वडा", "गाउँपालिका", "नगरपालिका", "महानगरपालिका", "उपमहानगरपालिका", "स्थानीय")):
        return "स्थानीय"
    return None


def _parse_fees(text: str, fee_type: str | None = None) -> list[dict[str, Any]]:
    fees: list[dict[str, Any]] = []
    for match in _FEE_RE.finditer(text):
        raw = match.group(0).strip()
        if match.group(2):
            fees.append({"type": fee_type, "amount_raw": raw, "amount_npr": 0})
        else:
            amount_npr = nepali_to_int(match.group(1))
            fees.append({"type": fee_type, "amount_raw": raw, "amount_npr": amount_npr})
    return fees


def _split_items(value: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,;]", value) if p.strip()]
    return [p for p in parts if len(p) > 2]


def _extract_inline_office(text: str) -> dict[str, Any] | None:
    match = _OFFICE_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    return {"name": name, "level": _infer_office_level(name)}


def _extract_inline_forms(text: str) -> list[str]:
    return [m.group(0).strip() for m in _FORM_RE.finditer(text)]


def _extract_inline_duration(text: str) -> str | None:
    match = _DURATION_RE.search(text)
    return match.group(0).strip() if match else None


def _extract_inline_documents(text: str) -> list[str]:
    seen: set[str] = set()
    docs: list[str] = []
    for match in _DOC_RE.finditer(text):
        doc = match.group(0).strip()
        if len(doc) > 2 and doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def _parse_pipe_row(line: str) -> list[str] | None:
    """Parse a pipe-delimited table row into stripped cell strings, or None if not a table row."""
    stripped = line.strip()
    if stripped.count("|") < 2:
        return None
    stripped = stripped.removeprefix("|").removesuffix("|")
    cells = [c.strip() for c in stripped.split("|")]
    return cells if len(cells) >= 2 else None


def _is_separator_row(cells: list[str]) -> bool:
    """Return True if all non-empty cells are Markdown table separator dashes."""
    non_empty = [c for c in cells if c]
    return bool(non_empty) and all(re.fullmatch(r"[-:\s]+", c) for c in non_empty)


def _classify_table_column(header: str) -> str | None:
    """Map a table column header to a canonical Procedure field name."""
    h = header.strip().lower()
    if not h:
        return None
    if any(k in h for k in ("सेवाको नाम", "प्रक्रियाको नाम", "सेवा", "service")):
        return "service"
    if any(k in h for k in ("कागजात", "document")):
        return "documents"
    if any(k in h for k in ("फाराम", "form")):
        return "forms"
    if any(k in h for k in ("कार्यालय", "निकाय", "शाखा", "office")):
        return "office"
    if any(k in h for k in ("दस्तुर", "शुल्क", "fee")):
        return "fees"
    if any(k in h for k in ("समयावधि", "समय", "अवधि", "duration", "wait time", "processing time")):
        return "duration"
    if any(k in h for k in ("प्रक्रिया", "चरण", "step", "action")):
        return "steps"
    return None


class ProcedureFormatter:
    """Format non-statutory procedural documents and Citizen Charters into Procedure and Provision chunks."""

    def __init__(
        self,
        act_metadata: dict[str, Any] | None = None,
        act_slug: str = "procedure",
    ) -> None:
        self.act_metadata = act_metadata or {}
        self.act_slug = act_slug
        bs_year = self.act_metadata.get("bs_year", "")
        self.slug_prefix = f"{act_slug}-{bs_year}" if bs_year else act_slug

    def process_text(self, full_text: str) -> dict[str, Any]:
        """Parse procedural text into Qdrant-compatible Provision chunks and Neo4j-compatible Procedure items."""
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        default_title = self.act_metadata.get("title") or self.act_slug

        procedures: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        entities: list[dict[str, Any]] = []

        state = self._new_state(default_title)
        table_headers: list[str | None] | None = None

        def flush_procedure() -> None:
            if not self._has_content(state):
                return
            proc_idx = len(procedures) + 1
            proc_id = f"{self.slug_prefix}-proc{proc_idx}"
            steps: list[dict[str, Any]] = state["steps"]
            documents: list[str] = state["documents"]
            forms: list[str] = state["forms"]
            offices: list[dict[str, Any]] = state["offices"]
            fees: list[dict[str, Any]] = state["fees"]
            duration: str | None = state["duration"]
            process_name: str = state["process_name"]

            primary_office = offices[0] if offices else None
            for step in steps:
                if step["office"] is None and primary_office is not None:
                    step["office"] = dict(primary_office)
                if not step["documents_required"] and documents:
                    step["documents_required"] = list(documents)
                if not step["forms"] and forms:
                    step["forms"] = list(forms)
                if not step["price_fees"] and fees:
                    step["price_fees"] = [dict(f) for f in fees]
                if step["duration"] is None and duration is not None:
                    step["duration"] = duration

            procedures.append(
                {
                    "id": proc_id,
                    "process_name": process_name,
                    "steps": steps,
                    "documents_required": documents,
                    "offices": offices,
                    "forms": forms,
                    "price_fees": fees,
                    "duration": duration,
                }
            )
            overview_text = self._build_overview_text(
                process_name, offices, documents, forms, fees, duration, steps
            )
            chunks.append(
                {
                    "id": proc_id,
                    "type": "procedure",
                    "hierarchy": {
                        "sec_title": process_name,
                        "process_name": process_name,
                    },
                    "text": overview_text,
                    "status": "active",
                    "stats": {
                        "characters": len(overview_text),
                        "tokens": max(1, len(overview_text) // 3),
                    },
                }
            )
            for step in steps:
                chunk_id = f"{proc_id}-step{step['step_number']}"
                text = step["action"]
                chunks.append(
                    {
                        "id": chunk_id,
                        "type": "procedure_step",
                        "hierarchy": {
                            "sec_title": process_name,
                            "process_name": process_name,
                            "step_number": step["step_number"],
                        },
                        "text": text,
                        "status": "active",
                        "stats": {
                            "characters": len(text),
                            "tokens": max(1, len(text) // 3),
                        },
                    }
                )
            entities.append(
                {
                    "chunk_id": proc_id,
                    "clause_ref": f"proc{proc_idx} ({process_name})",
                    "process_name": process_name,
                    "steps": steps,
                }
            )

        for line in lines:
            if line.lower() in _GENERIC_CHARTER_HEADERS:
                table_headers = None
                continue

            pipe_cells = _parse_pipe_row(line)
            if pipe_cells is not None:
                if _is_separator_row(pipe_cells):
                    continue
                classified = [_classify_table_column(c) for c in pipe_cells]
                if table_headers is None and sum(1 for k in classified if k is not None) >= 2:
                    table_headers = classified
                    continue
                if table_headers is not None:
                    svc_idx = next((i for i, k in enumerate(table_headers) if k == "service"), None)
                    if svc_idx is not None and svc_idx < len(pipe_cells):
                        svc_val = pipe_cells[svc_idx].strip()
                        if svc_val and svc_val not in {"-", "—"}:
                            if self._has_content(state) and svc_val != state["process_name"]:
                                flush_procedure()
                                state = self._new_state(default_title)
                            state["process_name"] = svc_val
                            state["has_explicit_title"] = True
                    for col_kind, cell in zip(table_headers, pipe_cells):
                        if col_kind and col_kind != "service" and cell and cell not in {"-", "—", "N/A"}:
                            self._apply_section_value(state, col_kind, cell)
                    continue
                if len(pipe_cells) == 2:
                    col_kind = _classify_table_column(pipe_cells[0])
                    if col_kind == "service" and pipe_cells[1]:
                        if self._has_content(state) and pipe_cells[1] != state["process_name"]:
                            flush_procedure()
                            state = self._new_state(default_title)
                        state["process_name"] = pipe_cells[1]
                        state["has_explicit_title"] = True
                        continue
                    if col_kind and pipe_cells[1]:
                        self._apply_section_value(state, col_kind, pipe_cells[1])
                        continue
                for cell in pipe_cells:
                    if not cell or cell in {"-", "—", "N/A"}:
                        continue
                    state["fees"].extend(_parse_fees(cell))
                    state["documents"].extend(d for d in _extract_inline_documents(cell) if d not in state["documents"])
                    if state["duration"] is None:
                        dur = _extract_inline_duration(cell)
                        if dur:
                            state["duration"] = dur
                continue

            table_headers = None

            svc_match = _SERVICE_RE.match(line)
            if svc_match:
                if self._has_content(state):
                    flush_procedure()
                    state = self._new_state(default_title)
                state["process_name"] = svc_match.group(1).strip()
                state["has_explicit_title"] = True
                state["mode"] = "steps"
                continue

            matched_section = False
            for section_name, pattern in _SECTION_PATTERNS:
                sec_match = pattern.match(line)
                if not sec_match:
                    continue
                state["mode"] = section_name
                inline_val = sec_match.group(1).strip()
                if inline_val:
                    self._apply_section_value(state, section_name, inline_val)
                matched_section = True
                break

            if matched_section:
                continue

            num_match = _NUMBERED_STEP_RE.match(line)
            bullet_match = _BULLET_STEP_RE.match(line)
            item_text = (
                num_match.group(2).strip()
                if num_match
                else (bullet_match.group(1).strip() if bullet_match else None)
            )

            if item_text is not None:
                if state["mode"] == "documents":
                    if len(item_text) > 2:
                        state["documents"].append(item_text)
                elif state["mode"] == "forms":
                    if len(item_text) > 2:
                        state["forms"].append(item_text)
                else:
                    step_num = (nepali_to_int(num_match.group(1)) if num_match else None) or (
                        len(state["steps"]) + 1
                    )
                    state["steps"].append(self._build_step(step_num, item_text))
                continue

            if not state["has_explicit_title"] and not self._has_content(state):
                state["process_name"] = line
                state["has_explicit_title"] = True

        flush_procedure()

        return {
            "act_metadata": self.act_metadata,
            "chunks": chunks,
            "procedures": procedures,
            "ner_result": {
                "act_title": self.act_metadata.get("title", default_title),
                "act_slug": self.act_slug,
                "total_chunks_processed": len(chunks),
                "total_chunks_filtered": 0,
                "entities": entities,
            },
        }

    @classmethod
    def _apply_section_value(cls, state: dict[str, Any], section_name: str, value: str) -> None:
        """Apply a parsed section or table-cell value onto *state*."""
        if section_name == "documents":
            state["documents"].extend(_split_items(value))
        elif section_name == "forms":
            state["forms"].append(value)
        elif section_name == "office":
            state["offices"].append({"name": value, "level": _infer_office_level(value)})
        elif section_name == "fees":
            state["fees"].extend(_parse_fees(value, fee_type="आवेदन दस्तुर"))
        elif section_name == "duration":
            state["duration"] = value
        elif section_name == "steps":
            parts = [p.strip() for p in re.split(r";\s*", value) if p.strip()]
            for part in parts:
                num_match = _NUMBERED_STEP_RE.match(part)
                if num_match:
                    step_num = nepali_to_int(num_match.group(1)) or (len(state["steps"]) + 1)
                    action = num_match.group(2).strip()
                else:
                    step_num = len(state["steps"]) + 1
                    action = part
                state["steps"].append(cls._build_step(step_num, action))

    @staticmethod
    def _new_state(default_title: str) -> dict[str, Any]:
        return {
            "process_name": default_title,
            "has_explicit_title": False,
            "mode": "steps",
            "documents": [],
            "forms": [],
            "offices": [],
            "fees": [],
            "duration": None,
            "steps": [],
        }

    @staticmethod
    def _has_content(state: dict[str, Any]) -> bool:
        return bool(
            state["steps"]
            or state["documents"]
            or state["offices"]
            or state["fees"]
            or state["duration"]
            or state["forms"]
        )

    @staticmethod
    def _build_step(step_number: int, action: str) -> dict[str, Any]:
        return {
            "step_number": step_number,
            "action": action,
            "office": _extract_inline_office(action),
            "documents_required": _extract_inline_documents(action),
            "forms": _extract_inline_forms(action),
            "price_fees": _parse_fees(action),
            "duration": _extract_inline_duration(action),
            "prerequisite": None,
        }

    @staticmethod
    def _build_overview_text(
        process_name: str,
        offices: list[dict[str, Any]],
        documents: list[str],
        forms: list[str],
        fees: list[dict[str, Any]],
        duration: str | None,
        steps: list[dict[str, Any]],
    ) -> str:
        parts = [f"प्रक्रिया: {process_name}"]
        if offices:
            parts.append("कार्यालय: " + ", ".join(o["name"] for o in offices))
        if documents:
            parts.append("आवश्यक कागजात: " + ", ".join(documents))
        if forms:
            parts.append("फाराम: " + ", ".join(forms))
        if fees:
            parts.append("शुल्क: " + ", ".join(f["amount_raw"] or "" for f in fees if f.get("amount_raw")))
        if duration:
            parts.append(f"समय: {duration}")
        if steps:
            parts.append("चरणहरू: " + " ".join(f"{s['step_number']}. {s['action']}" for s in steps))
        return "\n".join(parts)
