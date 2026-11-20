#!/usr/bin/env python3
"""Summarize stage2_m6 trace data for the 11 traced videos (thinking steps, R2 repairs, outcomes)."""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "reports" / "traces_m6_summary.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
_buf = []

def log(*a):
    _buf.append(" ".join(str(x) for x in a))

def flush():
    OUT_PATH.write_text("\n".join(_buf), encoding="utf-8")

traces = sorted((REPO / "debug_traces").glob("*.trace.json"))
log(f"[+] Found {len(traces)} trace files")

_dump = lambda: flush()

for tf in traces:
    try:
        d = json.loads(tf.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"!! {tf.name}: {e}")
        continue
    vid = d.get("video_id", "?")
    s2 = d.get("stage2_m6") or {}
    s5 = d.get("stage1_m5") or {}
    log(f"\n=== {vid} | title_ar={d.get('title_ar','')[:70]}")
    log(f"    stage1_m5: speakers={s5.get('speakers_count')} turns={s5.get('turns_count')} words={s5.get('total_words')} inter_words={s5.get('interlocutor_words')} health={(s5.get('diarization_health') or {}).get('status')}")
    log(f"    stage2_m6: kept={s2.get('kept_cards_count')} r1_cands={s2.get('r1_candidates_total')} windows={len(s2.get('windows', []))}")
    for w in s2.get("windows", []):
        for u in w.get("units", []):
            think = (u.get("r2_think_excerpt") or "").replace("\n", " ")
            log(f"      unit {u.get('unit_index')}: outcome={u.get('outcome')} final_score={u.get('final_score')} tier={u.get('final_tier','')[:40]}")
            log(f"        gates={u.get('final_gates')} defects={u.get('final_defect_codes')}")
            if u.get("r2_en_repairs"):
                log(f"        r2_en_repairs={u['r2_en_repairs']}")
            if think:
                log(f"        r2_think: {think[:220]}")

log("\n[+] Done")
flush()
