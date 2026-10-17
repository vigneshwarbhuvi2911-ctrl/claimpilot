"""LLM access for the agents: OpenAI (default), Amazon Bedrock (Nova Pro via the Converse API) or an
offline mock. Every call returns parsed JSON and records tokens, latency and an estimated cost.

The mock provider makes no network calls; each agent then falls back to its deterministic heuristic
(see app/agents/*). This keeps tests, CI and the evaluation runnable without an API key.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

import requests

from app.logging_setup import log_event

log = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Approximate list prices, used only for the cost estimate in logs.
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "amazon.nova-pro-v1:0": (0.80, 3.20),
    "amazon.nova-lite-v1:0": (0.06, 0.24),
}


class LLMError(Exception):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_agent: dict = field(default_factory=dict)

    def add(self, agent: str, model: str, tin: int, tout: int) -> None:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = (tin * pin + tout * pout) / 1_000_000
        self.calls += 1
        self.input_tokens += tin
        self.output_tokens += tout
        self.cost_usd += cost
        a = self.by_agent.setdefault(agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        a["calls"] += 1
        a["input_tokens"] += tin
        a["output_tokens"] += tout

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6), "by_agent": self.by_agent}


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise LLMError("model did not return JSON")


class LLMClient:
    def __init__(self, provider: str, model: str, timeout_s: float = 30.0, region: str = "us-east-1"):
        self.provider = provider
        self.model = model
        self.timeout_s = timeout_s
        self.region = region
        self._session = requests.Session()
        self._bedrock = None

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def complete_json(self, system: str, user: str, *, agent: str, usage: Usage, run_id: str,
                      max_tokens: int = 1200) -> dict:
        if self.is_mock:
            raise LLMError("mock provider has no model; agents use their heuristic instead")
        t0 = time.time()
        text, tin, tout = (self._openai if self.provider == "openai" else self._bedrock_converse)(system, user, max_tokens)
        usage.add(agent, self.model, tin, tout)
        log_event(log, "llm_call", run_id=run_id, agent=agent, provider=self.provider, model=self.model,
                  input_tokens=tin, output_tokens=tout, latency_ms=int((time.time() - t0) * 1000))
        try:
            return extract_json(text)
        except (LLMError, json.JSONDecodeError) as exc:
            raise LLMError(f"{agent}: invalid JSON from model") from exc

    # ------------------------------------------------------------- OpenAI
    def _openai(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            r = self._session.post(f"{base}/v1/chat/completions", json=body, timeout=self.timeout_s,
                                   headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        except requests.RequestException as exc:
            raise LLMError(f"openai unreachable: {type(exc).__name__}") from exc
        if r.status_code >= 400:
            raise LLMError(f"openai HTTP {r.status_code}")
        data = r.json()
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    # ------------------------------------------------------------ Bedrock
    def _bedrock_converse(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        if self._bedrock is None:
            try:
                import boto3  # optional dependency, see requirements-bedrock.txt
            except ImportError as exc:
                raise LLMError("LLM_PROVIDER=bedrock needs boto3 (pip install -r requirements-bedrock.txt)") from exc
            self._bedrock = boto3.client("bedrock-runtime", region_name=self.region)
        try:
            resp = self._bedrock.converse(
                modelId=self.model,
                system=[{"text": system + "\nReturn only a JSON object."}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"temperature": 0, "maxTokens": max_tokens})
        except Exception as exc:  # botocore raises many exception types
            raise LLMError(f"bedrock error: {type(exc).__name__}") from exc
        text = "".join(p.get("text", "") for p in resp["output"]["message"]["content"])
        u = resp.get("usage", {})
        return text, u.get("inputTokens", 0), u.get("outputTokens", 0)


def make_client(settings) -> LLMClient:
    provider = settings.resolved_provider()
    model = {"openai": settings.openai_model, "bedrock": settings.bedrock_model}.get(provider, "mock")
    return LLMClient(provider, model, settings.llm_timeout_s, settings.aws_region)
