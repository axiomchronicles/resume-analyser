from helpers import (
    clean_text,
    extract_bullets,
    coverage_score,
    keyword_match_score,
    starts_with_action_verb,
    contains_metric
)

from typing import Dict

def compute_ats_scores(resume_text: str, jd_text: str = "") -> Dict:
    resume_text = clean_text(resume_text)
    bullets = extract_bullets(resume_text)

    section_score, section_found = coverage_score(resume_text)
    kw_score = keyword_match_score(resume_text, jd_text)

    action_score = (
        sum(starts_with_action_verb(b) for b in bullets) / len(bullets)
        if bullets else 0.3
    )

    metric_score = (
        sum(contains_metric(b) for b in bullets) / len(bullets)
        if bullets else 0.2
    )

    wc = len(resume_text.split())
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