from __future__ import annotations

import re
import typing as t

from config import ACTION_VERBS, EXPECTED_SECTIONS, WEAK_PHRASES

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BULLET_CHARS = "•‣▪●◦–—·*+-"
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001F\u007F]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_METRIC_RE = re.compile(
    r"""
    (
        \b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|pts?|x)?\b      
        |
        \b\d+(?:\.\d+)?\s*(?:k|m|b)\b                            
        |
        \$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?                          
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PASSIVE_VOICE_RE = re.compile(
    r"\b(?:was|were|is|are|been|be|being)\s+\w+ed\b", re.IGNORECASE
)
_FIRST_PERSON_RE = re.compile(r"\b(I|me|my|we|our|us)\b", re.IGNORECASE)

def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = (
        text.replace("\u2013", "-")   
            .replace("\u2014", "-")   
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
    )

    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)

    return text.strip()


def split_into_sentences(text: str) -> t.List[str]:
    text = clean_text(text)
    if not text:
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def extract_bullets(text: str) -> t.List[str]:
    pattern = re.compile(
        r"""
        (?:^| )
        (?:
            ^[ \t]*  
            |
            (?<=\s)
        )            
        [•\u2022\u2023\u25CF\u25AA\u25E6\u00B7] 
        \s*
        (?P<item>[^•\u2022\u2023\u25CF\u25AA\u25E6\u00B7]+)
        """,
        re.VERBOSE | re.MULTILINE
    )

    bullets: t.List[str] = []
    for match in pattern.finditer(text):
        item = match.group("item").strip()
        item = _WHITESPACE_RE.sub(" ", item)
        if item:
            bullets.append(item)

    if not bullets:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and stripped[0] in BULLET_CHARS:
                bullets.append(stripped.lstrip(BULLET_CHARS).strip())

    return bullets


def contains_metric(text: str) -> bool:
    if not text:
        return False
    return bool(_METRIC_RE.search(text))


def starts_with_action_verb(text: str) -> bool:
    if not text:
        return False

    stripped = text.lstrip(" " + BULLET_CHARS)
    stripped = stripped.strip()

    if not stripped:
        return False

    first_token = stripped.split()[0]
    first_token = re.sub(r"[^\w']", "", first_token).lower()

    return first_token in {v.lower() for v in ACTION_VERBS}


def coverage_score(text: str) -> t.Tuple[float, t.Dict[str, bool]]:
    if not text:
        found = {s: False for s in EXPECTED_SECTIONS}
        return 0.0, found

    lower_lines = [l.strip().lower() for l in text.splitlines() if l.strip()]
    found: t.Dict[str, bool] = {}

    for section in EXPECTED_SECTIONS:
        sec = section.lower()
        pattern = re.compile(rf"^{re.escape(sec)}\b[:\-]?", re.IGNORECASE)
        is_present = any(pattern.search(line) for line in lower_lines)
        found[section] = is_present

    score = sum(found.values()) / len(found) if found else 0.0
    return float(score), found


def keyword_match_score(resume: str, jd: str) -> float:
    resume = clean_text(resume)
    jd = clean_text(jd)

    if not jd or len(jd.split()) < 5 or not resume:
        return 0.0

    try:
        vector = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        tf = vector.fit_transform([resume, jd])
        sim = cosine_similarity(tf[0:1], tf[1:2])[0][0]
        return float(sim)
    except Exception:
        resume_tokens = set(resume.lower().split())
        jd_tokens = set(jd.lower().split())
        if not jd_tokens:
            return 0.0
        overlap = len(resume_tokens & jd_tokens) / len(jd_tokens)
        return float(overlap)

def compile_phrase_patterns(phrases: t.Iterable[str]) -> t.List[t.Tuple[str, re.Pattern]]:
    patterns: t.List[t.Tuple[str, re.Pattern]] = []

    for phrase in phrases:
        phrase = (phrase or "").strip()
        if not phrase:
            continue

        escaped = re.escape(phrase)
        flexible = re.sub(r"\\ ", r"\\s+", escaped)  

        pattern = re.compile(
            rf"\b({flexible})\b",
            re.IGNORECASE,
        )
        patterns.append((phrase, pattern))

    return patterns


def weak_phrases(text: str) -> t.List[t.Dict[str, t.Any]]:
    if not text:
        return []

    patterns = compile_phrase_patterns(WEAK_PHRASES)
    out: t.List[t.Dict[str, t.Any]] = []
    seen = set()  

    for raw_phrase, pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()

            key = (raw_phrase, start, end)
            if key in seen:
                continue
            seen.add(key)

            snippet = text[max(0, start - 40): min(len(text), end + 40)].strip()

            out.append(
                {
                    "phrase": raw_phrase,
                    "start": start,
                    "end": end,
                    "snippet": snippet,
                }
            )

    return out

def bullet_quality_stats(bullets: t.List[str]) -> t.Dict[str, float]:
    if not bullets:
        return {
            "avg_length_words": 0.0,
            "pct_with_action_verb": 0.0,
            "pct_with_metric": 0.0,
            "pct_too_long": 0.0,
            "pct_too_short": 0.0,
        }

    lengths = [len(b.split()) for b in bullets]
    total = len(bullets)

    with_action = sum(starts_with_action_verb(b) for b in bullets)
    with_metric = sum(contains_metric(b) for b in bullets)
    too_long = sum(l > 40 for l in lengths)
    too_short = sum(l < 5 for l in lengths)

    return {
        "avg_length_words": sum(lengths) / total,
        "pct_with_action_verb": with_action / total,
        "pct_with_metric": with_metric / total,
        "pct_too_long": too_long / total,
        "pct_too_short": too_short / total,
    }


def _count_syllables(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False

    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def readability_scores(text: str) -> t.Dict[str, float]:
    sentences = split_into_sentences(text)
    words = clean_text(text).split()

    if not sentences or not words:
        return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(_count_syllables(w) for w in words)

    fre = (
        206.835
        - 1.015 * (num_words / num_sentences)
        - 84.6 * (num_syllables / num_words)
    )

    fkg = (
        0.39 * (num_words / num_sentences)
        + 11.8 * (num_syllables / num_words)
        - 15.59
    )

    return {
        "flesch_reading_ease": float(round(fre, 2)),
        "flesch_kincaid_grade": float(round(fkg, 2)),
    }


def passive_voice_ratio(text: str) -> float:
    sentences = split_into_sentences(text)
    if not sentences:
        return 0.0

    passive_count = sum(bool(_PASSIVE_VOICE_RE.search(s)) for s in sentences)
    return passive_count / len(sentences)


def first_person_ratio(text: str) -> float:
    sentences = split_into_sentences(text)
    if not sentences:
        return 0.0

    fp_count = sum(bool(_FIRST_PERSON_RE.search(s)) for s in sentences)
    return fp_count / len(sentences)


def estimate_experience_years(text: str) -> float:
    years = [int(y) for y in re.findall(r"\b(19[8-9]\d|20[0-4]\d)\b", text)]
    if len(years) < 2:
        return 0.0

    min_year = min(years)
    max_year = max(years)

    if max_year <= min_year or max_year - min_year > 45:
        return 0.0

    return float(max_year - min_year)


def skill_coverage_score(resume: str, required_skills: t.Iterable[str]) -> float:
    resume_lower = resume.lower()
    skills = [s.strip().lower() for s in required_skills if s.strip()]

    if not skills:
        return 0.0

    found = 0
    for skill in skills:
        escaped = re.escape(skill)
        flexible = re.sub(r"\\ ", r"\\s+", escaped)
        pattern = re.compile(flexible, re.IGNORECASE)
        if pattern.search(resume_lower):
            found += 1

    return found / len(skills)
