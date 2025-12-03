from PyPDF2 import PdfReader

import io

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
    
