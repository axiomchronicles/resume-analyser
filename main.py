# app_cli.py
import io
import re
import argparse
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PyPDF2 import PdfReader
# import docx


# ==============
# CONFIG / CONSTANTS
# ==============

WEAK_PHRASES = [
    "responsible for", "worked on", "helped with", "assisted with",
    "participated in", "various tasks", "duties included", "hard-working",
    "team player", "result-oriented", "fast learner", "self-motivated",
    "detail-oriented"
]

ACTION_VERBS = [
    "achieved", "analyzed", "built", "created", "designed", "developed",
    "implemented", "led", "managed", "optimized", "reduced", "improved",
    "increased", "delivered", "launched", "owned", "resolved", "conducted",
    "orchestrated", "shipped", "enhanced", "automated", "deployed"
]

EXPECTED_SECTIONS = [
    "summary", "objective", "experience", "work experience", "professional experience",
    "education", "skills", "projects", "certifications", "achievements"
]


# ==============
# FILE READING
# ==============

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages)


# def read_docx(file_bytes: bytes) -> str:
#     doc = docx.Document(io.BytesIO(file_bytes))
#     return "\n".join([p.text for p in doc.paragraphs])


def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except:
        return file_bytes.decode("latin-1", errors="ignore")


def extract_text_from_file(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()

    if path.lower().endswith(".pdf"):
        return read_pdf(data)
    # elif path.lower().endswith(".docx"):
    #     return read_docx(data)
    elif path.lower().endswith(".txt"):
        return read_txt(data)
    else:
        return read_txt(data)


# ==============
# NLP HELPERS
# ==============

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[\.\!\?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def extract_bullets(text: str) -> List[str]:
    bullets = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("-", "•", "*")):
            bullets.append(s.lstrip("-•* ").strip())
    return bullets


def contains_metric(text: str) -> bool:
    return bool(re.search(r"(\d+[%]?)|(\$\d+)", text))


def starts_with_action_verb(text: str) -> bool:
    if not text:
        return False
    return text.split()[0].lower() in ACTION_VERBS


def section_coverage_score(text: str) -> Tuple[float, Dict[str, bool]]:
    lower = text.lower()
    found = {s: (s in lower) for s in EXPECTED_SECTIONS}
    score = sum(found.values()) / len(found)
    return score, found


def keyword_match_score(resume: str, jd: str) -> float:
    resume = clean_text(resume)
    jd = clean_text(jd)

    if not jd or len(jd.split()) < 5:
        return 0.0

    vec = TfidfVectorizer(stop_words="english")
    tf = vec.fit_transform([resume, jd])
    return float(cosine_similarity(tf[0:1], tf[1:2])[0][0])


def detect_weak_phrases(text: str) -> List[Dict]:
    low = text.lower()
    out = []
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


def compute_ats_scores(resume_text: str, jd_text: str = "") -> Dict:
    resume_text = clean_text(resume_text)
    bullets = extract_bullets(resume_text)

    section_score, section_found = section_coverage_score(resume_text)
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


def generate_suggestions(analysis: Dict, weak_phrases, has_jd: bool):
    out = []

    missing = [s for s, p in analysis["section_found"].items() if not p]
    if missing:
        out.append(f"Missing important sections: {', '.join(missing)}")

    if has_jd and analysis["keyword_score"] < 50:
        out.append("Low keyword match — tailor resume more closely to the job description.")

    if analysis["action_score"] < 60:
        out.append("More bullet points should start with action verbs.")

    if analysis["metric_score"] < 40:
        out.append("Add more measurable achievements (%, $, numbers).")

    if analysis["length_score"] < 60:
        if analysis["word_count"] < 200:
            out.append("Resume is too short — add more detail.")
        elif analysis["word_count"] > 1200:
            out.append("Resume too long — reduce irrelevant content.")

    if weak_phrases:
        wp = sorted(set(w["phrase"] for w in weak_phrases))
        out.append(f"Weak phrases detected: {', '.join(wp)}")

    return out


# ==============
# CLI MAIN
# ==============

def main():
    parser = argparse.ArgumentParser(description="CLI ATS Resume Analyzer")
    parser.add_argument("resume", help="Path to resume file (.pdf/.docx/.txt)")
    parser.add_argument("--jd", help="Optional job description text file")
    args = parser.parse_args()

    print("== Reading Resume ==")
    resume_text = extract_text_from_file(args.resume)

    jd_text = ""
    has_jd = False

    if args.jd:
        print("== Reading Job Description ==")
        jd_text = extract_text_from_file(args.jd)
        has_jd = True

    print("\n== Analyzing Resume... ==")
    analysis = compute_ats_scores(resume_text, jd_text if has_jd else "")

    weak = detect_weak_phrases(resume_text)
    suggestions = generate_suggestions(analysis, weak, has_jd)

    print("\n--- ATS SCORE ---")
    print(f"Final Score: {analysis['final_score']}/100")

    print("\n--- SCORE BREAKDOWN ---")
    for k in ["keyword_score", "section_score", "action_score", "metric_score", "length_score"]:
        print(f"{k.replace('_', ' ').title()}: {analysis[k]}%")

    print("\nWord Count:", analysis["word_count"])
    print("Bullets:", analysis["bullets_count"])

    print("\n--- SECTIONS DETECTED ---")
    for sec, present in analysis["section_found"].items():
        print(f"{sec.title():25}: {'YES' if present else 'NO'}")

    print("\n--- WEAK PHRASES ---")
    if not weak:
        print("No weak/generic phrases detected.")
    else:
        print(", ".join(sorted(set(w['phrase'] for w in weak))))

    print("\n--- SUGGESTIONS ---")
    for s in suggestions:
        print("-", s)

    print("\nDone.")


if __name__ == "__main__":
    main()
