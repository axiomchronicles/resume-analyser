import re
import typing

from config import ACTION_VERBS, EXPECTED_SECTIONS, WEAK_PHRASES

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BULLET_CHARS = "•‣▪●◦–—·*+-"


def clean_text(text: str) -> str:
    # print(type(text))
    # print(type(text))
    return re.sub(r"\s+", " ", text).strip()

def split_into_sentences(text: str) -> typing.List[str]:
    sents = re.split(r"(?<=[\.\.!\?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def extract_bullets(text: str) -> typing.List[str]:
    pattern = re.compile(
        r"""
        (?:^| )                # start or space
        [•\u2022\u2023\u25CF\u25AA\u25E6\u00B7]   # bullet-like characters
        \s*
        (?P<item>[^•\u2022\u2023\u25CF\u25AA\u25E6\u00B7]+)
        """,
        re.VERBOSE
    )

    bullets = []
    for m in pattern.finditer(text):
        bullets.append(m.group("item").strip())

    return bullets

def contains_metric(text: str) -> bool:
    return bool(re.search(r"(\d+[%]?)|(\$\d+)", text))

def starts_with_action_verb(text: str) -> bool:
    if not text:
        return False
    
    return text.split()[0].lower() in ACTION_VERBS

def coverage_score(text: str) -> typing.Tuple[float, typing.Dict[str, bool]]:
    lower = text.lower()
    found = {s: (s in lower) for s in EXPECTED_SECTIONS}
    score = sum(found.values()) / len(found)
    return score, found


def keyword_match_score(resume: str, jd: str) -> float:
    resume = clean_text(resume)
    jd = clean_text(jd)

    if not jd or len(jd.split()) < 5:
        return 0.0
    
    vector = TfidfVectorizer(stop_words = "english")
    tf = vector.fit_transform([resume, jd])

    return float(cosine_similarity(tf[0:1], tf[1:2])[0][0])


def compile_phrase_patterns(phrases):
    patterns = []

    for phrase in phrases:
        escaped = re.escape(phrase)
        flexible = re.sub(r"\\ ", r"\\s+", escaped)  # allow variable whitespace

        pattern = re.compile(
            rf"\b({flexible})\b",
            re.IGNORECASE
        )
        patterns.append((phrase, pattern))

    return patterns


def weak_phrases(text: str) -> typing.List[typing.Dict]:
    patterns = compile_phrase_patterns(WEAK_PHRASES)
    out: typing.List[typing.Dict] = []

    for raw_phrase, pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()

            # You can adjust snippet window if you want more context
            snippet = text[max(0, start - 30): min(len(text), end + 30)]

            out.append({
                "phrase": raw_phrase,
                "start": start,
                "end": end,
                "snippet": snippet
            })

    return out