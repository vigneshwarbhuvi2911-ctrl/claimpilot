from app.config import ROOT
from app.guardrails import Masker, is_grounded, scrub_for_logs
from app.retrieval.clauses import ClauseIndex, HashingEmbedder, parse_wording
from app.service import load_products


def test_masking_known_labelled_and_pattern_pii_roundtrip():
    m = Masker()
    m.add_known("PERSON", "Arjun Mehta")
    text = ("Insured name: Arjun Mehta\nName: Karthik Mehta\nContact: +91 98400 11201\n"
            "Mail arjun.mehta@example.com, PAN ABCDE1234F, Aadhaar 1234 5678 9012\n"
            "Licence number: TN09 2008 0012345\nARJUN MEHTA signed. Amount Rs 69,000 on 2026-09-18")
    masked = m.mask(text)
    for secret in ("Arjun", "Karthik", "98400", "example.com", "ABCDE1234F", "5678", "0012345"):
        assert secret.lower() not in masked.lower()
    assert "Rs 69,000" in masked and "2026-09-18" in masked     # facts survive
    assert m.unmask(masked).count("Arjun Mehta") == 2


def test_grounding_check():
    src = ["Breath analyser test conducted at the spot showed 62 mg/100 ml of alcohol."]
    assert is_grounded("breath analyser test conducted at the spot", src)
    assert not is_grounded("driver admitted drinking", src)
    assert not is_grounded("alcohol", src)                       # too short to count as evidence


def test_log_scrubbing():
    assert "98400" not in scrub_for_logs('{"x": "+91 98400 11201 a@b.com"}')


def test_wording_parsed_into_clauses():
    clauses = parse_wording("MOTOR_PC", (ROOT / "data/policies/motor_private_car.md").read_text())
    assert [c.clause_id for c in clauses] == [f"M{i}" for i in range(1, 9)]
    assert "under the influence of alcohol" in clauses[5].text


def test_hybrid_search_finds_paraphrase_and_stays_in_product():
    idx = ClauseIndex.from_dir(ROOT / "data/policies", load_products(ROOT / "data"), HashingEmbedder())
    top = idx.search("driver was drunk alcohol", "MOTOR_PC", k=2)
    assert top[0].clause_id == "M6"
    assert all(c.product_code == "HEALTH_IND" for c in idx.search("room rent limit nursing", "HEALTH_IND"))
    assert idx.search("room rent limit", "HEALTH_IND", k=1)[0].clause_id == "H4"
    assert idx.get("H3(b)").clause_id == "H3"
