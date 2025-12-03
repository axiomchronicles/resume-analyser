from utils import extract_texts
from helpers import (
    clean_text,
    contains_metric,
    cosine_similarity,
    coverage_score,
    extract_bullets,
    starts_with_action_verb,
    split_into_sentences,
    keyword_match_score,
    weak_phrases
)

from pprint import pprint

def main():
    data = extract_texts("../PawanKumar_Resume.pdf")

    jd_text = "AI&ML Engineer"

    cleaned_text = clean_text(data)
    bullets = extract_bullets(cleaned_text)

    section_score, section_found = coverage_score(clean_text)
    kw_score = keyword_match_score(clean_text, jd_text)

    action_score = (
        sum(starts_with_action_verb(b) for b in bullets) / len(bullets)
        if bullets else 0.3
    )

    metric_score = (
        sum(contains_metric(b) for b in bullets) / len(bullets)
        if bullets else 0.2
    )

    wc = len(cleaned_text.split())
    if wc < 200:
        length_score = 0.3
    elif wc <= 800:
        length_score = 1.0
    elif wc <= 1200:
        length_score = 0.7
    else:
        length_score = 0.4

    final = (
        section_score * 0.2 +
        kw_score * 0.3 +
        action_score * 0.2 +
        metric_score * 0.15 +
        length_score * 0.15
    ) * 100

    return {
        "final_score": round(final, 1),
        "section_score": round(section_score * 100, 1),
        "keyword_score": round(kw_score * 100, 1),
        "action_score": round(action_score * 100, 1),
        "metric_score": round(metric_score * 100, 1),
        "length_score": round(length_score * 100, 1),
        "word_count": wc,
        "bullets_count": len(bullets),
        "section_found": section_found,
        "bullets": bullets,
    }

if __name__ == "__main__":
    pprint(main())