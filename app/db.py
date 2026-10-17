"""SQLite data access: customers, policies, claims history (the 'core insurance system') and the
pipeline's own run records. The schema is created and seeded from data/seed.json on first start."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
  customer_id TEXT PRIMARY KEY, full_name TEXT, dob TEXT, phone TEXT, email TEXT, city TEXT);
CREATE TABLE IF NOT EXISTS policies (
  policy_no TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id), product_code TEXT,
  start_date TEXT, end_date TEXT, first_start_date TEXT, sum_insured REAL, status TEXT,
  vehicle TEXT, registration_no TEXT, vehicle_reg_date TEXT, ped_declared TEXT);
CREATE TABLE IF NOT EXISTS claims_history (
  claim_id TEXT PRIMARY KEY, policy_no TEXT REFERENCES policies(policy_no), loss_date TEXT,
  amount_paid REAL, status TEXT, fraud_flag INTEGER);
CREATE TABLE IF NOT EXISTS claim_runs (
  run_id TEXT PRIMARY KEY, claim_ref TEXT, policy_no TEXT, created_at TEXT, status TEXT,
  recommendation TEXT, payable REAL, result_json TEXT, handler_decision TEXT, handler_note TEXT,
  decided_at TEXT);
CREATE INDEX IF NOT EXISTS idx_history_policy ON claims_history(policy_no);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self, seed_file: Path) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.connect() as c:
            c.executescript(SCHEMA)
            if c.execute("SELECT COUNT(*) FROM policies").fetchone()[0]:
                return
            seed = json.loads(Path(seed_file).read_text())
            for table in ("customers", "policies", "claims_history"):
                rows = seed[table]
                cols = list(rows[0])
                c.executemany(
                    f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [tuple(r[k] for k in cols) for r in rows])

    # -------------------------------------------------------------- tools
    def get_policy(self, policy_no: str) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                """SELECT p.*, c.full_name, c.dob, c.phone, c.email, c.city
                   FROM policies p JOIN customers c USING (customer_id) WHERE p.policy_no = ?""",
                (policy_no,)).fetchone()
        return dict(row) if row else None

    def claims_history(self, policy_no: str) -> list[dict]:
        with self.connect() as c:
            rows = c.execute("SELECT * FROM claims_history WHERE policy_no = ? ORDER BY loss_date",
                             (policy_no,)).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- runs
    def save_run(self, run: dict) -> None:
        with self._lock, self.connect() as c:
            c.execute(
                """INSERT OR REPLACE INTO claim_runs
                   (run_id, claim_ref, policy_no, created_at, status, recommendation, payable, result_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run["run_id"], run.get("claim_ref"), run.get("policy_no"), run["created_at"], run["status"],
                 run.get("recommendation"), run.get("payable"), json.dumps(run, default=str)))

    def get_run(self, run_id: str) -> dict | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM claim_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        out = json.loads(row["result_json"])
        out.update(status=row["status"], handler_decision=row["handler_decision"],
                   handler_note=row["handler_note"], decided_at=row["decided_at"])
        return out

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                """SELECT run_id, claim_ref, policy_no, created_at, status, recommendation, payable,
                          handler_decision FROM claim_runs ORDER BY created_at DESC LIMIT ?""",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def record_handler_decision(self, run_id: str, decision: str, note: str, approved_amount: float | None) -> bool:
        """Human-in-the-loop: the handler's decision is final. An approved payout is added to the
        claims history so that later claims see the reduced sum insured and the claim count."""
        with self._lock, self.connect() as c:
            row = c.execute("SELECT status, policy_no, claim_ref FROM claim_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None or row["status"] != "awaiting_handler":
                return False
            today = date.today().isoformat()
            c.execute("""UPDATE claim_runs SET status = 'closed', handler_decision = ?, handler_note = ?,
                         decided_at = ? WHERE run_id = ?""", (decision, note, today, run_id))
            run = json.loads(c.execute("SELECT result_json FROM claim_runs WHERE run_id = ?",
                                       (run_id,)).fetchone()[0])
            loss_date = (run.get("extracted") or {}).get("incident_date") or today
            if decision in ("APPROVE", "PARTIAL") and approved_amount:
                c.execute("INSERT INTO claims_history VALUES (?,?,?,?,?,?)",
                          (f"R-{run_id[:8]}", row["policy_no"], loss_date, approved_amount, "settled", 0))
        return True
