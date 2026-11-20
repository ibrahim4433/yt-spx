#!/usr/bin/env python3
"""Offline verification of the Phase-A golden-result fixes on real backfilled GT data.

Checks (PASS/FAIL):
  V1. _enforce_trajectory_integrity: repairs DIALOGUE_TURN_TEXT_SHIFTED (text_en := GT) and
      appends missing GT INTERLOCUTOR turns (MISSING_GT_TURNS fix) on real scenarios.
  V2. _extend_quote_to_sentence_end: repairs an INTERLOCUTOR_QUOTE_TRUNCATION tail (owPIQBEjtK8_002).
  V3. extract_video_m6 hard-reject: FUNCTIONAL_DIARIZATION_COLLAPSE recovery is vetoed (hiixMRVFlfk).
  V4. Website sanity (FastAPI TestClient): /api/search, /api/videos/{id}/scenario, /api/stats return data.
      (Requires web deps installed: fastapi, uvicorn.)

Usage: python tools/verify_golden_fixes.py [--skip-web]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.scenario_extractor import (
    _enforce_trajectory_integrity,
    _extend_quote_to_sentence_end,
    _canon_speaker_role,
)
from tools.scenario_backfill import backfill  # noqa

WS = REPO / "reports" / "regen_ws"

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _canon_role(raw):
    r = (raw or "").upper()
    if "PEDESTRIAN" in r or "عابرة" in (raw or ""):
        return "PEDESTRIAN"
    if "CO_DAEE" in r or "CO-DAEE" in r:
        return "CO_DAEE"
    if "INTERLOCUTOR" in r or "GUEST" in r or "السائل" in (raw or ""):
        return "INTERLOCUTOR"
    if "DAEE" in r or "الداعية" in (raw or "") or "محاجج" in (raw or ""):
        return "MAIN_DAEE"
    return (raw or "").strip() or "INTERLOCUTOR"


def build_lookups(scen):
    turns = scen.get("turns", [])
    raw_by_id = {}
    for t in turns:
        tid = str(t.get("turn_id") or "").strip()
        if tid:
            raw_by_id[tid] = t
    return turns, raw_by_id


def main():
    backfill(["jt4BisLcSoI", "owPIQBEjtK8", "hiixMRVFlfk"])

    print("== V1: trajectory integrity on jt4BisLcSoI (real GT) ==")
    scen = json.loads((WS / "scenarios" / "jt4BisLcSoI.json").read_text(encoding="utf-8"))
    turns, raw_by_id = build_lookups(scen)
    ts_sec = 1042.0
    # Simulate R1/R2 trajectory with the known defects: T0158 shifted, T0160 duplicated, missing T0161
    def ex(tid, role, text_en):
        return {"turn_id": tid, "role": role, "text_en": text_en, "timestamp_seconds": 1042}
    exchange = [
        ex("jt4BisLcSoI_T0157", "INTERLOCUTOR", "Hmm... this is hard to accept."),
        ex("jt4BisLcSoI_T0158", "INTERLOCUTOR", "And when you come back... ...from If you treat the body well, Peace."),
        ex("jt4BisLcSoI_T0160", "INTERLOCUTOR", "out, you will"),
        ex("jt4BisLcSoI_T0161", "INTERLOCUTOR", "out, you will"),
    ]
    repaired = _enforce_trajectory_integrity({"dialogue_exchange": exchange}, exchange, raw_by_id, anchor_ts_ms=ts_sec * 1000)
    by_id = {str(e["turn_id"]): e for e in repaired}
    gt_T0158 = raw_by_id["jt4BisLcSoI_T0158"]["text_original"]
    ok = by_id.get("jt4BisLcSoI_T0158", {}).get("text_en") == gt_T0158
    check("V1.a DIALOGUE_TURN_TEXT_SHIFTED repaired to GT", ok,
          f"card='...If you treat the body well, Peace.' -> GT len={len(gt_T0158)}")
    ok2 = any(str(e["turn_id"]) == "jt4BisLcSoI_T0161" and str(e["text_en"]) == raw_by_id["jt4BisLcSoI_T0161"]["text_original"] for e in repaired)
    check("V1.b MISSING_GT_TURNS: GT T0161 appended to trajectory", ok2)

    print("== V2: interlocator quote extension on owPIQBEjtK8_002 ==")
    scen2 = json.loads((WS / "scenarios" / "owPIQBEjtK8.json").read_text(encoding="utf-8"))
    _, raw_by_id2 = build_lookups(scen2)
    gt_quote = raw_by_id2["owPIQBEjtK8_T0125"]["text_original"]
    card_quote = gt_quote.rstrip()
    while card_quote and card_quote[-1] not in ".!?؟":
        card_quote = card_quote[:-1].rstrip()
    extended = _extend_quote_to_sentence_end(card_quote, raw_by_id2["owPIQBEjtK8_T0125"]["text_original"].lower())
    check("V2 INTERLOCUTOR_QUOTE_TRUNCATION extended to sentence end",
          extended.lower().rstrip(".") == gt_quote.lower().rstrip(".") or extended.lower() == gt_quote.lower(),
          f"card='{card_quote[-30:]}...' -> extended='{extended[-30:]}...'")

    print("== V3: hiixMRVFlfk functional-collapse hard-reject path ==")
    # extract_video_m6 pre-gate: classify_diarization_health of the backfilled GT scenario
    from core.debug_tracer import classify_diarization_health
    scen3 = json.loads((WS / "scenarios" / "hiixMRVFlfk.json").read_text(encoding="utf-8"))
    total_words = sum(len((t.get("text_original") or "").split()) for t in scen3.get("turns", []))
    inter_words = sum(len((t.get("text_original") or "").split())
                      for t in scen3.get("turns", [])
                      if _canon_role(t.get("speaker_role")) == "INTERLOCUTOR")
    health = classify_diarization_health(scen3["duration_sec"], len(scen3.get("speakers") or []), total_words, inter_words, len(scen3.get("turns") or []))
    check("V3 classify_diarization_health(hiixMRVFlfk) == FUNCTIONAL_DIARIZATION_COLLAPSE",
          health["status"] == "COLLAPSED" and str(health["reason"]).startswith("FUNCTIONAL_DIARIZATION_COLLAPSE"),
          f"status={health['status']} reason={health['reason'][:60]}")

    print("== V4: website sanity ==")
    if not Path("fastapi").exists():
        pass
    try:
        import importlib
        web_app = importlib.import_module("web.app")
        from fastapi.testclient import TestClient
        client = TestClient(web_app.app)
        r = client.get("/api/stats")
        check("V4.1 GET /api/stats returns 200", r.status_code == 200, f"status={r.status_code}")
        check("V4.2 /api/stats has total_qa >= 1", r.json().get("total_qa", 0) >= 1)
        r2 = client.get("/api/videos/0Y7cFAUlil4/scenario")
        check("V4.3 GET /api/videos/{id}/scenario returns 200", r2.status_code == 200, f"status={r2.status_code}")
        check("V4.4 scenario has speakers+turns",
              r2.status_code == 200 and len(r2.json().get("speakers", [])) >= 1 and len(r2.json().get("turns", [])) >= 1)
        r3 = client.get("/api/search", params={"q": "spirit"})
        check("V4.5 GET /api/search?q=spirit returns results", r3.status_code == 200 and r3.json().get("total", 0) >= 1,
              f"total={r3.json().get('total', 0) if r3.status_code == 200 else 'N/A'}")
    except Exception as e:
        check("V4 website sanity (import/TestClient)", False, f"error: {e}")

    print("")
    print(f"[+] VERIFICATION: {len(PASS)} PASS, {len(FAIL)} FAIL")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
