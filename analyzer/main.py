from utils import extract_texts, highlight_pdf
from helpers import (
    clean_text,
)
from compute import compute_ats_scores

from pprint import pprint

def main():
    data = extract_texts("./functionalsample.pdf")
    # print(type(data))

    jd_text = "AI&ML Engineer"

    cleaned_text = clean_text(data)
    compute = compute_ats_scores(cleaned_text, jd_text)

    # print(compute)
    return compute

if __name__ == "__main__":
    pprint(main())