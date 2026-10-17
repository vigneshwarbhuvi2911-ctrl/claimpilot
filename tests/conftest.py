from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

try:
    import langgraph.graph  # noqa: F401
    REAL_LANGGRAPH = True
except ImportError:
    from tests import langgraph_shim
    langgraph_shim.install()
    REAL_LANGGRAPH = False

from app.config import ROOT, Settings
from app.llm.client import LLMClient
from app.models import ClaimSubmission
from app.retrieval.clauses import HashingEmbedder
from app.service import ClaimService

DATA = ROOT / "data"


def load_claim(ref: str) -> ClaimSubmission:
    return ClaimSubmission.model_validate(json.loads((DATA / "claims" / f"{ref}.json").read_text()))


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(data_dir=DATA, db_path=tmp_path / "claimpilot.db", llm_provider="mock")


@pytest.fixture()
def service(settings) -> ClaimService:
    return ClaimService(settings, llm=LLMClient("mock", "mock"), embedder=HashingEmbedder())


class ScriptedLLM(LLMClient):
    """Pretends to be a real model: returns scripted JSON per agent and records every prompt."""

    def __init__(self, responses: dict):
        super().__init__("openai", "gpt-4.1-mini")
        self.responses = responses
        self.calls: list[dict] = []

    def complete_json(self, system, user, *, agent, usage, run_id, max_tokens=1200):
        self.calls.append({"agent": agent, "system": system, "user": user})
        usage.add(agent, self.model, len(user) // 4, 50)
        r = self.responses.get(agent)
        if callable(r):
            r = r(user)
        if isinstance(r, Exception):
            raise r
        return r
