import re
import typing

from analyzer.config import ACTION_VERBS, EXPECTED_SECTIONS, WEAK_PHRASES

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def split_into_sentences(text: str) -> typing.List[str]:
    sents = re.split(r"(?<=[\.\.!\?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def extract_bullets(text: str) -> typing.List[str]:
    bullets = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("-", "•", "*")):
            bullets.append(s.lstrip("-•*").strip())

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


def weak_phrases(text: str) -> typing.List[typing.Dict]:
    low = text.lower()
    out =[]

    for phrase in WEAK_PHRASES:
        start = 0
        while True:
            idx = low.find(phrase, start)
            if idx == -1:
                break

            out.append(
                {"phrase": phrase, "start": idx, "end": idx + len(phrase)}
            )
            start = idx + len(phrase)

    return out