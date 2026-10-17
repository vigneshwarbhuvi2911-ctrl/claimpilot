"""Evaluation: run every claim in eval/cases.json through the full workflow and score it.

    python -m eval.run                  # provider from LLM_PROVIDER (mock when no key)
    LLM_PROVIDER=openai python -m eval.run --out eval/report_openai.md

Scores: decision accuracy, payable-amount accuracy (exact to the rupee), and a confusion list of
mismatches. Also reports LLM calls, tokens, estimated cost and latency per claim. Expected answers
were worked out by hand from the policy wordings, not produced by this code.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app.config import ROOT, Settings
from app.models import ClaimSubmission
from app.service import ClaimService


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write a markdown report to this file")
    ap.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 if decision accuracy is below this")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    svc = ClaimService(Settings(db_path=tmp / "eval.db"))
    cases = json.loads((ROOT / "eval" / "cases.json").read_text())
    rows, dec_ok, amt_ok, cost, calls, lat = [], 0, 0, 0.0, 0, []
    for c in cases:
        sub = ClaimSubmission.model_validate(json.loads((ROOT / "data" / "claims" / f"{c['claim_ref']}.json").read_text()))
        r = svc.process(sub)
        d_ok = r["recommendation"] == c["decision"]
        a_ok = (c["payable"] is None and r["payable"] is None) or (
            c["payable"] is not None and r["payable"] is not None and abs(r["payable"] - c["payable"]) < 0.5)
        dec_ok += d_ok
        amt_ok += a_ok
        cost += r["usage"]["cost_usd"]
        calls += r["usage"]["calls"]
        lat.append(r["latency_ms"])
        rows.append((c["claim_ref"], c["decision"], r["recommendation"], c["payable"], r["payable"], d_ok and a_ok,
                     [f["flag"] for f in r["flags"]]))

    n = len(cases)
    lines = [f"# ClaimPilot evaluation ({svc.llm.provider}: {svc.llm.model})", "",
             f"- Decision accuracy: **{dec_ok}/{n} ({dec_ok / n:.0%})**",
             f"- Payable amount exact: **{amt_ok}/{n} ({amt_ok / n:.0%})**",
             f"- LLM calls: {calls}, estimated cost: ${cost:.4f}, "
             f"median latency: {sorted(lat)[len(lat) // 2]} ms", "",
             "| Claim | Expected | Got | Expected Rs | Got Rs | OK | Flags |", "|---|---|---|---|---|---|---|"]
    for ref, ed, gd, ep, gp, ok, flags in rows:
        lines.append(f"| {ref} | {ed} | {gd} | {ep if ep is not None else '-'} | {gp if gp is not None else '-'} | "
                     f"{'yes' if ok else '**NO**'} | {', '.join(flags) or '-'} |")
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
    return 0 if dec_ok / n >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
