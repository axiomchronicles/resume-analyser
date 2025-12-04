# 🧠 Resume e‑Filing 2.0 – AI‑Powered Resume Analyzer

An end‑to‑end **AI Resume Analyzer** that:

- Extracts text from **PDF resumes** or plain text  
- Cleans and analyzes the content using **NLP**  
- Computes an **ATS‑style score** (0–100)  
- Detects **weak phrases** and **bullet points**  
- Uses **ML** to improve ATS scoring & suggestions  
- Highlights weak phrases directly in the **PDF**  
- Exposes a clean **API** used by an interactive **web UI** (Tailwind + Framer Motion)

---

## 📁 Project Structure

```bash
.
├── analyzer/
│   ├── compute.py          # compute_ats_scores() – ATS scoring engine
│   ├── helpers.py          # clean_text, extract_bullets, weak_phrases, etc.
│   ├── suggestions.py      # generate_suggestions() – rule + ML hybrid
│   ├── utils.py            # extract_texts(), highlight_pdf(), helpers
│   ├── predict.py          # ML model & multi-label binarizer (mlb, model)
│   └── synthetic_data.py   # Synthetic ATS dataset generator (optional)
│
├── api/
│   ├── views.py            # ApiResponsev1 – /api/analyze implementation
│   ├── db.py               # collection (if using a DB)
│   └── exceptions.py       # ApiResponseError wrapper
│
├── templates/
│   ├── base.html           # Layout with Tailwind + React + Framer Motion CDNs
│   └── index.html          # Resume analyzer UI (Jinja2)
│
├── static/                 # (optional) Custom JS / CSS if you extract them
│
├── README.md               # ← this file
└── requirements.txt
```

> The exact filenames may differ slightly in your project – adjust paths as needed.

---

## 🚀 Features

### ✅ Backend

- Built around an async API class: `ApiResponsev1`
- Accepts either:
  - `resume_text` (plain text), or  
  - `resume_file` (PDF upload)
- Optionally accepts `jd_text` (job description) for keyword‑aware scoring
- Uses:
  - `extract_texts()` to read PDF
  - `clean_text()` to normalize content
  - `extract_bullets()` to detect bullet points
  - `weak_phrases()` to find vague / weak wording
  - `compute_ats_scores()` to compute ATS metrics & overall score
  - `generate_suggestions()` to produce human‑readable guidance
  - `highlight_pdf()` to create a highlighted PDF (weak phrases + bullets)
- Automatically **deletes temp files** after each request

### ✅ ATS Scoring

`compute_ats_scores(resume_text, jd_text="")` returns a dictionary with:

- `final_score` – overall ATS score (0–100)
- `section_score` – coverage of key sections (Experience, Education, Skills, Summary, etc.)
- `keyword_score` – overlap between resume and job description
- `action_score` – share of bullets starting with strong action verbs
- `metric_score` – share of bullets containing measurable numbers / %, / $
- `length_score` – effectiveness of resume length & density
- Additional metadata:
  - `word_count`
  - `bullets_count`
  - `section_found`
  - `bullets`

### ✅ Suggestions (Rule + ML Hybrid)

`generate_suggestions(analysis, weak_phrases, has_jd)` uses:

- ATS scores from `analysis`
- List of weak phrases found
- Info on whether a JD was provided
- Optionally ML classifier (if wired in) to prioritise suggestion categories

Example suggestions:

- “Missing important sections: Experience, Skills”
- “Low keyword match — tailor resume more closely to the job description.”
- “More bullet points should start with action verbs.”
- “Add more measurable achievements (%, $, numbers).”
- “Weak phrases detected: responsible for, worked on, assisted with”

### ✅ ML Engine

The project is designed to support an ML‑based ATS score & suggestion engine using scikit‑learn:

- `synthetic_data.py` (or similar) can generate thousands of synthetic resumes with labels
- `predict.py` holds:
  - `model` – scikit‑learn pipeline (e.g. TF‑IDF + Ridge/Logistic Regression)
  - `mlb` – MultiLabelBinarizer for suggestion categories
- ML models can:
  - Predict ATS score directly
  - Predict suggestion categories (e.g. missing sections, weak metrics, etc.)
  - Be retrained using synthetic or real datasets

---

## 🌐 API Design

### Endpoint: `POST /api/analyze`

**Request (multipart/form-data):**

- `resume_text` (optional, string)
- `resume_file` (optional, file – PDF)
- `jd_text` (optional, string)

At least one of `resume_text` or `resume_file` **must** be present.

**Response (JSON):**

```jsonc
{
  "compute": {
    "final_score": 78.5,
    "section_score": 90.0,
    "keyword_score": 65.0,
    "action_score": 70.0,
    "metric_score": 40.0,
    "length_score": 85.0,
    "word_count": 546,
    "bullets_count": 12,
    "section_found": {
      "Summary": true,
      "Experience": true,
      "Education": true,
      "Skills": false
    },
    "bullets": [
      "Led migration of legacy system to microservices architecture...",
      "Improved query performance by 45% by optimising indexes..."
    ]
  },
  "suggestions": [
    "Missing important sections: Skills",
    "Add more measurable achievements (%, $, numbers).",
    "Low keyword match — tailor resume more closely to the job description."
  ],
  "weak_phrases": [
    { "phrase": "responsible for", "start": 123, "end": 137, "snippet": "..." }
  ],
  "bullets": [
    "Led migration of legacy system...",
    "Improved performance by 45%..."
  ],
  "file_out": "abcd-1234_highlighted.pdf"
}
```

---

## 🧩 Backend: `ApiResponsev1` Flow

Simplified logic:

```python
class ApiResponsev1:
    async def analyse(self, request: Request) -> Response:
        temp_files = []

        try:
            if request.method != "POST":
                raise ApiResponseError(details="Method Not Allowed", status=404)

            form = await request.form()
            resume_text = form.get("resume_text") or ""
            jd_text = form.get("jd_text") or ""
            file = form.get("resume_file")

            # CASE 1: file upload
            if file and hasattr(file, "filename") and file.filename:
                file_id, file_path, file_name = await self._process_file(file)
                temp_files.append(file_path)

                extracted_text = extract_texts(file_path)
                resume_text = extracted_text

                output = self._build_result(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    file_path=file_path,
                    file_name=file_name,
                )

                if output.get("file_out"):
                    temp_files.append(os.path.join(self.UPLOAD_DIR, output["file_out"]))

            # CASE 2: text only
            elif resume_text.strip():
                output = self._build_result(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    file_path=None,
                    file_name=None,
                )

            else:
                raise ApiResponseError(
                    details="No resume text or file provided", status=400
                )

            return JsonResponse(content=output, status=200)

        except ApiResponseError as e:
            return JsonResponse(content={"error": e.details}, status=e.status, headers=e.headers)

        except Exception:
            import traceback
            traceback.print_exc()
            return JsonResponse(content={"error": "Internal Server Error"}, status=500)

        finally:
            # Delete temp files
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    traceback.print_exc()
```

---

## 🎨 Frontend Overview

The frontend is implemented with:

- **Jinja2** templates (compatible with Flask/FastAPI/Aquilify)
- **Tailwind CSS (CDN)** for styling
- **React + Framer Motion (CDN)** for animated score ring
- **Meta‑/portal‑style UI** inspired by official government / ITS portals
- Light‑mode only, responsive layout

### Key UX Features

- **Drag & drop** PDF upload
- Live **file name preview**
- API call via `fetch("/api/analyze")` with `FormData`
- **Loading spinner overlay** while the backend works
- **Score widget** with animated circular gauge
- Tabbed panels:
  - Scores
  - Suggestions
  - Sections & Bullets
  - ML Details

Example submission (using JS):

```js
const formData = new FormData();
formData.append("resume_text", resumeTextArea.value);
formData.append("jd_text", jdTextArea.value);
if (fileInput.files[0]) {
  formData.append("resume_file", fileInput.files[0]);
}

const res = await fetch("/api/analyze", {
  method: "POST",
  body: formData
});
const data = await res.json();
renderResults(data);
```

---

## 🔬 Core Analysis Functions

### `clean_text(text: str) -> str`

- Lowercases
- Normalizes whitespace
- Strips control characters & noise
- Optionally handles special unicode bullets / ligatures

### `extract_bullets(text: str) -> List[str>`

- Detects bullets even when PDF is flattened into a **single line**
- Handles Unicode bullets such as `•`, `▪`, `–`, etc.
- Supports:
  - Inline bullets (`… • Managed 25 volunteers. • Maintained records …`)
  - Multiline bullets with indentation

Example approach for inline bullets:

```python
pattern = re.compile(
    r"(?:^| )"
    r"[•\u2022\u2023\u25CF\u25AA\u25E6\u00B7]"
    r"\s*(?P<item>[^•\u2022\u2023\u25CF\u25AA\u25E6\u00B7]+)"
)
```

### `weak_phrases(text: str) -> List[dict]`

- Scans the resume for weak phrases such as:
  - “responsible for”
  - “worked on”
  - “helped with”
  - “participated in”
  - “various tasks”
- Uses regex with word boundaries and flexible whitespace
- Returns:
  - `phrase`
  - `start` / `end` indices
  - `snippet` for context

---

## 🧪 Synthetic Dataset (Optional)

The project can include a script like `synthetic_data.py` that:

- Generates ~5,000 synthetic resumes
- Varies:
  - strength of action verbs
  - presence of metrics
  - coverage of sections
  - keyword alignment
- Computes rule‑based ATS scores as labels
- Adds noise to simulate human scoring
- Saves to `synthetic_ats_dataset.csv`

Used to train:

- ATS scoring regression model
- Suggestion multi‑label classifier

---

## ⚙️ Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/axiomchronicles/resume-analyzer.git
cd resume-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Typical `requirements.txt` could include:

```txt
aquilify
fastapi
uvicorn
pymupdf
scikit-learn
pandas
numpy
python-docx
jinja2
reportlab
```

Adjust based on your actual project.

### 4. Run the backend

Depending on how Aquilify is wired (or if using FastAPI / Starlette):

```bash
uvicorn main:app --reload
```

or (if you use Aquilify’s CLI):

```bash
aquilify run
```

### 5. Open the UI

- Navigate to `http://localhost:8000/` (or the port you configured)
- Paste or upload a resume
- Optionally paste a job description
- Click **Run Analysis**

You should see:

- ATS score ring animate
- Detailed sub‑scores
- Suggestions
- Extracted bullets
- Weak phrases badges
- Optionally a link to download the highlighted PDF (if you expose it)

---

## 📚 Possible Extensions

- Add **authentication** and store user analyses
- Export full **PDF report** summarising the analysis
- Add **multi‑language** support (English + others)
- Integrate a **large language model** for richer, natural‑language feedback
- Deploy the app using:
  - Docker
  - Render / Railway / Fly.io / AWS / Azure

---

## 🙋 FAQ

**Q: Does this exactly match real ATS systems?**  
A: No. It simulates common ATS heuristics (sections, keywords, metrics), and uses ML to improve scoring, but it’s a learning / demo project, not tied to any proprietary ATS.

**Q: Can I plug in my own ML model?**  
A: Yes. As long as your model exposes a `predict()` or `predict_proba()` interface (and you adapt `predict.py`), you can swap out the core model.

**Q: Can I use only rule‑based scoring without ML?**  
A: Absolutely. `compute_ats_scores()` is fully rule‑based. ML is optional and layered on top.

---

## 💡 Credits & Acknowledgements

This project blends concepts from:

- Natural Language Processing
- Applied Machine Learning
- Web backend APIs
- Modern frontend UIs

It’s ideal as a **college project**, **portfolio piece**, or **internal tool** to understand how real‑world resume screening tools work.

---

Happy hacking & resume‑optimizing! 🚀
