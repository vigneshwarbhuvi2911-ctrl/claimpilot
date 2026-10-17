"""FastAPI service.

POST /claims                      run a claim through the agent workflow (JSON documents)
POST /claims/upload               same, with PDF or text files as multipart form fields
GET  /claims                      recent runs
GET  /claims/{run_id}             full result: recommendation, findings, breakdown, trace, usage
POST /claims/{run_id}/decision    the handler's final decision (human in the loop)
GET  /samples, /samples/{ref}     the synthetic demo claims
GET  /                            handler web page
GET  /health
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Literal, Optional, get_args

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logging_setup import configure_logging
from app.models import ClaimDocument, ClaimSubmission, DocType
from app.service import ClaimService

DOC_TYPES = set(get_args(DocType))


class HandlerDecision(BaseModel):
    decision: Literal["APPROVE", "PARTIAL", "REJECT", "REQUEST_DOCUMENTS"]
    approved_amount: Optional[float] = Field(default=None, ge=0)
    note: str = Field(default="", max_length=2000)


def pdf_or_text(raw: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text.strip()) < 20:   # scanned page: OCR it
                    try:
                        import pytesseract
                        text = pytesseract.image_to_string(page.to_image(resolution=300).original)
                    except Exception:
                        text = ""
                pages.append(text)
        return "\n".join(pages)
    return raw.decode("utf-8", errors="replace")


def create_app(service: ClaimService | None = None) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    svc = service or ClaimService(settings)
    app = FastAPI(title="ClaimPilot", version="1.0.0",
                  description="Agentic claims processing assistant (demo with synthetic data)")
    app.state.svc = svc

    samples_dir = Path(svc.settings.data_dir) / "claims"

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/samples")
    def samples():
        out = []
        for p in sorted(samples_dir.glob("*.json")):
            c = json.loads(p.read_text())
            out.append({"claim_ref": c["claim_ref"], "note": c.get("note"), "documents": [d["doc_type"] for d in c["documents"]]})
        return out

    @app.get("/samples/{claim_ref}")
    def sample(claim_ref: str):
        p = samples_dir / f"{claim_ref}.json"
        if not p.resolve().parent == samples_dir.resolve() or not p.exists():
            raise HTTPException(404, "Unknown sample")
        return json.loads(p.read_text())

    @app.get("/health")
    def health():
        return {"status": "ok", "llm_provider": svc.llm.provider, "model": svc.llm.model,
                "embedder": svc.index.embedder.name}

    @app.post("/claims")
    def submit(claim: ClaimSubmission):
        return svc.process(claim)

    @app.post("/claims/upload")
    async def upload(request: Request):
        form = await request.form()
        docs = []
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                text = pdf_or_text(await value.read(), value.filename or "")
                if text.strip():
                    docs.append(ClaimDocument(doc_type=key if key in DOC_TYPES else "other", text=text))
        if not docs:
            raise HTTPException(400, "Upload at least one document as a form field named by type, e.g. claim_form")
        return svc.process(ClaimSubmission(claim_ref=form.get("claim_ref"), documents=docs))

    @app.get("/claims")
    def list_claims(limit: int = 50):
        return svc.db.list_runs(min(max(limit, 1), 200))

    @app.get("/claims/{run_id}")
    def get_claim(run_id: str):
        run = svc.db.get_run(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run_id")
        return run

    @app.post("/claims/{run_id}/decision")
    def decide(run_id: str, body: HandlerDecision):
        if body.decision in ("APPROVE", "PARTIAL") and body.approved_amount is None:
            raise HTTPException(422, "approved_amount is required to approve")
        if not svc.db.record_handler_decision(run_id, body.decision, body.note, body.approved_amount):
            raise HTTPException(409, "Run not found or already decided")
        return svc.db.get_run(run_id)

    return app


def get_app() -> FastAPI:  # uvicorn --factory entry point
    return create_app()
