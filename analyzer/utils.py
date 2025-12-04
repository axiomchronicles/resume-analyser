from __future__ import annotations

import io
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyPDF2 import PdfReader
import fitz  

from helpers import starts_with_action_verb, contains_metric

def _detect_text_encoding(data: bytes, fallback: str = "latin-1") -> str:
    for enc in ("utf-8", "utf-16", fallback):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    return data.decode(fallback, errors="ignore")


def read_pdf(file_bytes: bytes, max_pages: Optional[int] = None) -> str:
    if not file_bytes:
        return ""

    pages: List[str] = []

    try:
        reader = PdfReader(io.BytesIO(file_bytes))

        total_pages = len(reader.pages)
        limit = total_pages if max_pages is None else min(max_pages, total_pages)

        for idx in range(limit):
            page = reader.pages[idx]
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(text)

    except Exception:
        # If PyPDF2 fails completely, fall back to PyMuPDF
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                total_pages = doc.page_count
                limit = total_pages if max_pages is None else min(max_pages, total_pages)
                for i in range(limit):
                    page = doc.load_page(i)
                    text = page.get_text("text") or ""
                    text = text.strip()
                    if text:
                        pages.append(text)
        except Exception:
            return ""

    return "\n\n".join(pages)


def read_text(file_bytes: bytes) -> str:
    if not file_bytes:
        return ""

    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    return _detect_text_encoding(file_bytes)


def extract_texts(path: str) -> str:
    if not path:
        raise ValueError("Path must be a non-empty string.")

    ext = path.lower().strip()

    with open(path, "rb") as f:
        data = f.read()

    if ext.endswith(".pdf"):
        return read_pdf(data)

    if ext.endswith(".txt"):
        return read_text(data)

    raise ValueError(f"File type not supported for path: {path!r}")

class HighlightSeverity(int, Enum):
    METRIC_MISSING = 1   # cyan
    ACTION_MISSING = 2   # yellow
    WEAK_PHRASE = 3      # red


@dataclass(frozen=True)
class HighlightRule:
    phrase: str
    severity: HighlightSeverity


COLOR_MAP: Dict[HighlightSeverity, Tuple[float, float, float]] = {
    HighlightSeverity.WEAK_PHRASE: (1.0, 0.0, 0.0),      # red
    HighlightSeverity.ACTION_MISSING: (1.0, 1.0, 0.0),   # yellow
    HighlightSeverity.METRIC_MISSING: (0.0, 1.0, 1.0),   # cyan
}


def _normalize_phrase(phrase: str) -> str:
    return " ".join((phrase or "").split())


def _build_highlight_rules(
    weak_phrases: Sequence[Dict[str, Any]],
    bullets: Sequence[str],
) -> List[HighlightRule]:

    phrase_to_severity: Dict[str, HighlightSeverity] = {}

    for w in weak_phrases or []:
        phrase = _normalize_phrase(str(w.get("phrase", "")))
        if len(phrase) < 3:
            continue
        existing = phrase_to_severity.get(phrase)
        new_severity = HighlightSeverity.WEAK_PHRASE
        if existing is None or new_severity > existing:
            phrase_to_severity[phrase] = new_severity

    for b in bullets or []:
        b_norm = _normalize_phrase(b)
        if len(b_norm) < 3:
            continue
        if not starts_with_action_verb(b):
            existing = phrase_to_severity.get(b_norm)
            new_severity = HighlightSeverity.ACTION_MISSING
            if existing is None or new_severity > existing:
                phrase_to_severity[b_norm] = new_severity

    for b in bullets or []:
        b_norm = _normalize_phrase(b)
        if len(b_norm) < 3:
            continue
        if not contains_metric(b):
            existing = phrase_to_severity.get(b_norm)
            new_severity = HighlightSeverity.METRIC_MISSING
            if existing is None or new_severity > existing:
                phrase_to_severity[b_norm] = new_severity

    rules: List[HighlightRule] = [
        HighlightRule(phrase=p, severity=s) for p, s in phrase_to_severity.items()
    ]
    return rules


def highlight_pdf(
    input_path: str,
    output_path: str,
    weak_phrases: Sequence[Dict[str, Any]],
    bullets: Sequence[str],
) -> str:
    if not input_path.lower().endswith(".pdf"):
        raise ValueError(f"highlight_pdf only supports PDF files, got: {input_path!r}")

    rules = _build_highlight_rules(weak_phrases, bullets)

    if not rules:
        return input_path

    doc = fitz.open(input_path)

    try:
        for page in doc:
            for rule in rules:
                phrase = rule.phrase
                if not phrase:
                    continue

                rects = page.search_for(
                    phrase,
                    flags=fitz.TEXT_DEHYPHENATE,
                )

                if not rects:
                    continue

                color = COLOR_MAP.get(rule.severity, (1.0, 0.0, 0.0))  

                for rect in rects:
                    highlight = page.add_highlight_annot(rect)
                    highlight.set_colors({"stroke": color})
                    highlight.update()

        doc.save(output_path, incremental=False)
    finally:
        doc.close()

    return output_path
