"""FastAPI endpoints, including the human-in-the-loop decision."""
import json

import pytest
from fastapi.testclient import TestClient

from app.config import ROOT
from app.main import create_app


@pytest.fixture()
def client(service):
    return TestClient(create_app(service))


def claim_json(ref):
    return json.loads((ROOT / "data/claims" / f"{ref}.json").read_text())


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_submit_and_fetch(client):
    r = client.post("/claims", json=claim_json("CLM-H02")).json()
    assert r["recommendation"] == "PARTIAL" and r["payable"] == 45000
    got = client.get(f"/claims/{r['run_id']}").json()
    assert got["status"] == "awaiting_handler" and got["breakdown"]
    assert client.get("/claims").json()[0]["run_id"] == r["run_id"]


def test_validation_and_unknown_run(client):
    assert client.post("/claims", json={"documents": []}).status_code == 422
    assert client.get("/claims/nope").status_code == 404


def test_handler_decision_is_final_and_updates_history(client, service):
    # HLT-2004 has Rs 50,000 left. Approve 50,000 now -> a second claim finds nothing left.
    r = client.post("/claims", json=claim_json("CLM-H08")).json()
    assert r["payable"] == 50000
    assert client.post(f"/claims/{r['run_id']}/decision", json={"decision": "APPROVE"}).status_code == 422
    done = client.post(f"/claims/{r['run_id']}/decision",
                       json={"decision": "PARTIAL", "approved_amount": 50000, "note": "checked bills"}).json()
    assert done["status"] == "closed" and done["handler_decision"] == "PARTIAL"
    assert client.post(f"/claims/{r['run_id']}/decision",
                       json={"decision": "REJECT"}).status_code == 409          # cannot decide twice
    again = client.post("/claims", json=claim_json("CLM-H08")).json()
    assert again["payable"] == 0 and again["recommendation"] == "PARTIAL"


def test_upload_text_files(client):
    c = claim_json("CLM-H01")
    files = [(d["doc_type"], (f"{d['doc_type']}.txt", d["text"].encode(), "text/plain")) for d in c["documents"]]
    r = client.post("/claims/upload", files=files, data={"claim_ref": "UP-1"}).json()
    assert r["recommendation"] == "APPROVE" and r["payable"] == 38500 and r["claim_ref"] == "UP-1"


def test_ui_and_samples(client):
    assert "ClaimPilot" in client.get("/").text
    s = client.get("/samples").json()
    assert len(s) == 23 and s[0]["claim_ref"] == "CLM-H01"
    assert client.get("/samples/CLM-M01").json()["claim_ref"] == "CLM-M01"
    assert client.get("/samples/..%2Fseed").status_code == 404
