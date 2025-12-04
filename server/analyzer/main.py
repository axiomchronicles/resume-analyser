from .utils import extract_texts, highlight_pdf
from .helpers import (
    clean_text,
    weak_phrases,
    extract_bullets
)
from .suggestions import generate_suggestions
from .compute import compute_ats_scores
from .predict import model as Model, mlb

from pprint import pprint

def main():
    path = "./functionalsample.pdf"
    output_path = "./_higlighted.pdf"

    data = extract_texts(path)
    # print(type(data))

    jd_text = "AI&ML Engineer"

    cleaned_text = clean_text(data)
    compute = compute_ats_scores(cleaned_text, jd_text)
    weak_phrase = weak_phrases(cleaned_text)
    bullets = extract_bullets(cleaned_text)
    # print(compute)
    classified = generate_suggestions(analysis=compute, weak_phrases=weak_phrase, has_jd= True if jd_text else False,
                                      model=Model, mlb=mlb)
    
    pprint(compute)
    print("\n")
    for item in classified:
        print(" → ", item["suggestion"])
        # print(" → ", item["categories"])

    highlight_pdf(input_path=path, output_path=output_path, weak_phrases=weak_phrase, bullets=bullets)

    return compute

if __name__ == "__main__":
    main()