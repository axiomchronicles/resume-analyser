from aquilify.wrappers import Request, Response
from aquilify.shortcuts import render
from aquilify.responses import JsonResponse

from .db import collection
from .exceptions import ApiResponseError

from analyzer.utils import extract_texts, highlight_pdf
from analyzer.helpers import clean_text, extract_bullets, weak_phrases
from analyzer.compute import compute_ats_scores
from analyzer.suggestions import generate_suggestions

import pathlib
import uuid
import os


class ApiResponsev1:
    def __init__(self):
        self.output_path: pathlib.Path = pathlib.Path("temp")
        self.UPLOAD_DIR = "temp"

    async def _process_file(self, file):
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        save_name = file_id + ext
        save_path = os.path.join(self.UPLOAD_DIR, save_name)

        with open(save_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        return file_id, save_path, save_name

    def _build_result(self, resume_text: str, jd_text: str, file_path: str = None, file_name: str = None):
        resume_text_clean = clean_text(resume_text)
        bullets = extract_bullets(resume_text_clean)
        weak_phrase = weak_phrases(resume_text_clean)

        compute = compute_ats_scores(
            resume_text=resume_text_clean,
            jd_text=jd_text or ""
        )

        classified = generate_suggestions(
            analysis=compute,
            weak_phrases=weak_phrase,
            has_jd=True if jd_text else False,
        )

        file_out = None
        if file_path and file_name and file_name.lower().endswith(".pdf"):
            ext = pathlib.Path(file_name).suffix  # ".pdf"
            file_out = f"{pathlib.Path(file_name).stem}_highlighted{ext}"

            highlight_pdf(
                input_path=file_path,
                output_path=os.path.join(self.UPLOAD_DIR, file_out),
                weak_phrases=weak_phrase,
                bullets=bullets
            )

        return {
            "compute": compute,
            "suggestions": classified,
            "weak_phrases": weak_phrase,
            "bullets": bullets,
            "file_out": file_out,
        }

    async def analyse(self, request: Request) -> Response:
        try:
            if request.method != "POST":
                raise ApiResponseError(details="Method Not Allowed", status=404)

            form = await request.form()

            resume_text = form.get("resume_text") or ""
            jd_text = form.get("jd_text") or ""
            file = form.get("resume_file")

            if file and hasattr(file, "filename") and file.filename:
                file_id, file_path, file_name = await self._process_file(file)
                extracted_text = extract_texts(file_path)
                resume_text = extracted_text

                output = self._build_result(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    file_path=file_path,
                    file_name=file_name
                )

            elif resume_text.strip():
                output = self._build_result(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    file_path=None,
                    file_name=None
                )

            else:
                raise ApiResponseError(
                    details="No resume text or file provided",
                    status=400
                )

            return JsonResponse(content=output, status=200)

        except ApiResponseError as e:
            return JsonResponse(
                content={"error": e.details},
                status=e.status,
                headers=e.headers
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return JsonResponse(
                content={"error": "Internal Server Error"},
                status=500
            )

apiresponse = ApiResponsev1()