from PyPDF2 import PdfReader

import io
import fitz

from helpers import starts_with_action_verb, contains_metric

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def read_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except:
        return file_bytes.decode("latin-1", errors="ignore")
    

def extract_texts(path: str):
    with open(path, "rb") as f:
        data = f.read()

    if path.lower().endswith(".pdf"):
        return read_pdf(data)
    
    if path.lower().endswith(".txt"):
        return read_text(data)

    else:
        raise Exception("File type: %s not supported" %(path.lower()))
    

def highlight_pdf(input_path: str, output_path: str, weak_phrases, bullets):
    doc = fitz.open(input_path)

    all_targets = set()

    for w in weak_phrases:
        all_targets.add(("red", w["phrase"]))

    for b in bullets:
        if not starts_with_action_verb(b):
            all_targets.add(("yellow", b))

    for b in bullets:
        if not contains_metric(b):
            all_targets.add(("cyan", b))

    COLOR_MAP = {
        "red": (1, 0, 0),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
    }

    for page in doc:
        text = page.get_text("text").lower()

        for color_key, phrase in all_targets:
            search = page.search_for(phrase)  # find all matches

            for rect in search:
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors({"stroke": COLOR_MAP[color_key]})
                highlight.update()

    doc.save(output_path, incremental=False)
    doc.close()

    return output_path
