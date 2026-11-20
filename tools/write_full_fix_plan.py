#!/usr/bin/env python3
"""Synthesize the FULL FIX PLAN from manual audit reports + baseline audit + traces.

Produces reports/full_fix_plan.md — a concrete, prioritized, root-cause-driven fix plan.
Read-only w.r.t. data/dawah_knowledge.db, debug_traces/.
"""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent


def write_report(name: str, text: str):
    p = REPO / "reports" / name
    p.write_text(text, encoding="utf-8")
    print(f"[+] Wrote {p}")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_manual():
    d = REPO / "reports" / "manual_audit"
    return {f.stem: load_json(f) for f in sorted(d.glob("*.json"))} if d.is_dir() else {}


def load_baseline():
    p = REPO / "reports" / "audit_baseline_v2.json"
    if not p.exists():
        p = REPO / "reports" / "audit_baseline.json"
    return load_json(p).get("cards", [])


STAGE_LABELS = {
    "Stage 1 M3 Diarization": "Stage 1 — Acoustic diarization (core/overlap_resolver.py, core/audio_diarizer.py)",
    "Stage 1 M5 Turn Assembly": "Stage 1 — ASR & scenario assembly (core/scenario_extractor.py M5, core/audio_diarizer.py)",
    "Stage 2 R1 Prompt/Extraction": "Stage 2 — R1 LLM extraction prompt (core/scenario_extractor.py, Stage2ScenarioExtractor.extract_video_m6)",
    "Stage 2 R2 Arabic Localization": "Stage 2 — R2 Arabic localization (merge_round1_and_round2_units + R2 compact prompt in extract_video_m6)",
    "merge_round1_and_round2_units": "Stage 2 — R1/R2 zipper merge (core/scenario_extractor.py merge_round1_and_round2_units)",
    "anchor_and_enrich_extracted_unit": "Stage 2 — Pre-Stage-2 Grounding (core/scenario_extractor.py anchor_and_enrich_extracted_unit, _sync_role_quotes_in_item)",
    "Post-Processing insert_qa_unit": "Post-processing — DB ingestion guards (db/database.py _sanitize_and_validate_qa_unit, _insert_qa_unit_in_tx)",
    "Website/DB Hydration": "Website/DB hydration (db/database.py _hydrate_qa_row, web/app.py endpoints, core/smart_bot.py)",
    "R2 English repairs": "Stage 2 — R2 English repair (merge_round1_and_round2_units conditional EN fixes)",
}


def main():
    manual = load_manual()
    baseline = {c["card_id"]: c for c in load_baseline()}

    L = []
    L.append("# FULL FIX PLAN — Dawah Dialectics (yt-spx) Deep Rating & Investigation Outcomes")
    L.append("")
    L.append(f"- Generated: (see reports/) — from 11 manually-audited cards (reports/manual_audit/), automated baseline audit (reports/audit_baseline_v2.json), LLM thinking trails (debug_traces/*.trace.json).")
    L.append(f"- Cards audited: **{len(manual)}**")
    tiers = defaultdict(int)
    for m in manual.values():
        tiers[(m.get("tier") or "?").split(":")[0]] += 1
    L.append(f"- Tier distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())) + "")
    L.append("")

    L.append("## 0. Method")
    L.append("")
    L.append("- Deep manual rating of every current card (one subagent per card) using the strict process: tools/card_rating_system/RUBRIC.md v3.2 (5 dimensions, 100 pts) + 7 hard-fail gates + AGENT_PROMPT.md v3.2 workflow + enhanced audit_cards.py checks (new timestamp-fidelity & website-presentation gates, audit_cards.py v3.3).")
    L.append("- Cross-checked against the diarized ground truth (scenario_turns, video_speakers) for quote accuracy, role swaps, timestamps, and perspective integrity.")
    L.append("- Cross-checked LLM reasoning trails in debug_traces/*.trace.json (r2_think_excerpt / r2_en_repairs / r2_retries) to confirm Round-2 behavior.")
    L.append("- Core goal check: cards must be production-ready dashboards of seeker-facing objections vs. Islamic truth — accurate, grounded in spoken transcript, perspective-correct, bilingual, deployable to the website. The website displays qa_units via /api/search + /api/videos/{id}/scenario; timestamps drive seek links (timestamp_url).")
    L.append("")

    L.append("## 1. Root-Cause Problem Register (by pipeline stage, consolidated from all manual reports)")
    L.append("")
    stage_problems = defaultdict(dict)
    for card_id, m in sorted(manual.items()):
        for d in (m.get("defects") or []):
            stage = d.get("pipeline_root_cause") or "Website/DB Hydration"
            code = d.get("code") or "UNKNOWN"
            p = stage_problems[stage].setdefault(code, {
                "code": code, "severity": d.get("severity", "MEDIUM"), "stage": stage,
                "cards": set(), "field": d.get("field", ""),
                "card_level_fix": [], "pipeline_fix": (m.get("pipeline_fix_recommendation") or [""])[:1],
            })
            p["cards"].add(card_id)
            cr = (m.get("card_level_rewrite_recommendation") or [""])[0]
            if cr:
                p["card_level_fix"].append(f"`{card_id}`: {cr[:200]}")

    problems = []
    for stage, codes in sorted(stage_problems.items()):
        for code, p in sorted(codes.items(), key=lambda kv: -len(kv[1]["cards"])):
            problems.append(p)

    for i, p in enumerate(problems, 1):
        sev = p["severity"]
        L.append(f"### 1.{i} `{p['code']}` — {sev} — `{p['stage']}`")
        L.append(f"- **Affected cards**: {', '.join('`'+c+'`' for c in sorted(p['cards']))}")
        L.append(f"- **Fields touched**: {p['field'] or '—'}")
        L.append("- **Card-level rewrites (per-card)**:")
        for c in p["card_level_fix"][:6]:
            L.append(f"  - {c}")
        L.append(f"- **Pipeline-level root cause + fix**: {STAGE_LABELS.get(p['stage'], p['stage'])}")
        for fix in (p["pipeline_fix"] or []):
            if fix:
                L.append(f"  - {fix[:250]}")
        L.append("")

    L.append("## 2. Per-Card Detail + Recommendations (Deep)")
    L.append("")
    for card_id in sorted(manual):
        m = manual[card_id]
        L.append(f"### {card_id} — {m.get('tier', '?')} | {m.get('status', '?')}")
        L.append("")
        for d in (m.get("defects") or []):
            L.append(f"- [{d.get('severity', '?')}] `{d.get('code')}`: {d.get('description', '')[:180]}")
        for r in (m.get("card_level_rewrite_recommendation") or [])[:4]:
            L.append(f"- Card fix: {r[:220]}")
        for r in (m.get("pipeline_fix_recommendation") or [])[:4]:
            L.append(f"- Pipeline fix: {r[:220]}")
        L.append("")

    L.append("## 3. Automated vs Manual Cross-Check Summary")
    L.append("")
    for card_id in sorted(manual):
        m = manual[card_id]
        b = baseline.get(card_id, {})
        mismatch = "OK" if (b.get("status") == "PASS") == (m.get("status") in ("PASS", None)) else "DIFF"
        L.append(f"- `{card_id}`: auto={b.get('status','?')}/{b.get('tier','?')} (score {b.get('total_score','?')}) | manual={m.get('status','?')}/{m.get('tier','?')} (score {m.get('total_score','?')}) | {mismatch}")
    L.append("")

    L.append("## 4. Recommended Multi-Stage Fix Sequence")
    L.append("")
    L.append("1. **Stage 1 M3 Diarization fixes** (core/overlap_resolver.py): raise min-speaker separation for multi-speaker debates (>180s); re-diarize `hiixMRVFlfk` (COLLAPSED, 0.8% inter words) and `7SS0KGNFwVE`-class videos.")
    L.append("2. **Stage 2 R1 extraction hardening** (core/scenario_extractor.py R1 prompts): block offensive cross-exam as question (DAEE_CROSS_EXAM_IN_QUESTION), remove perspective inversion (GATE-PERSPECTIVE).")
    L.append("3. **Stage 2 R2 Arabic localization**: eliminate TRUNCATED_FRAGMENT_QUOTE, DAEE_DOMINATED_DIALOGUE_EXCHANGE, MISSING_INTERLOCUTOR_SPEAKER_ID, MISSING_DIALOGUE_TEXT_AR.")
    L.append("4. **merge_round1_and_round2_units / anchor_and_enrich_extracted_unit**: strengthen role-swap + turn-ID re-anchoring + timestamp reset (TIMESTAMP_ORDER_VIOLATION, QUESTION_TS_MISMATCH, ANSWER_TS_MISMATCH).")
    L.append("5. **Post-processing (db/database.py)**: enrich diacritics, validate arabic quotes non-empty, re-check theological bias with stricter gates.")
    L.append("6. **Website/DB hydration**: verify timestamps drive `timestamp_url` correctly, parallel EN/AR rendering, scenario studio sync.")
    L.append("")

    L.append("## 5. Verification Plan")
    L.append("")
    L.append("- Re-run `python tools/card_rating_system/audit_cards.py --out reports/audit_after.json` after fixes — goal: 100% Tier 1-2 PASS.")
    L.append("- Run the team-audit again on affected cards (updated card_packets).")
    L.append("- Manually re-verify: (a) zero swapped quotes (VERBATIM_QUOTE_ROLE_SWAPPED_INTERLOCUTOR/DAEE), (b) all question_en seeker-perspective (GATE-PERSPECTIVE), (c) all cards >= 90 Tier 1.")
    L.append("- Run the website: `python run_app.py` -> http://127.0.0.1:8000 — verify scenario studio + parallel mode render without errors.")
    L.append("")

    write_report("full_fix_plan.md", "\n".join(L))


if __name__ == "__main__":
    main()
