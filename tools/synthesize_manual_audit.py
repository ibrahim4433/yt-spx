#!/usr/bin/env python3
"""Synthesize the master manual-audit synthesis from reports/manual_audit/*.json + baseline audit + trace summaries.

Produces:
- reports/master_synthesis.md — per-card verdicts + consolidated problems & deep root causes by pipeline stage.
- reports/manual_audit_summary.json — machine-readable synthesis.

Read-only w.r.t. data/dawah_knowledge.db, debug_traces/.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
MAYBE_UTF8 = lambda p: p.read_text(encoding="utf-8")

def write_report(name: str, text: str):
    p = REPO / "reports" / name
    p.write_text(text, encoding="utf-8")
    print(f"[+] Wrote {p}")

def load_manual_reports():
    d = REPO / "reports" / "manual_audit"
    out = {}
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                out[f.stem] = json.loads(MAYBE_UTF8(f))
            except Exception as e:
                print(f"[!] Could not parse {f}: {e}")
    return out

def load_baseline():
    p = REPO / "reports" / "audit_baseline_v2.json"
    if not p.exists():
        p = REPO / "reports" / "audit_baseline.json"
    return json.loads(MAYBE_UTF8(p)) if p.exists() else {"cards": []}

def load_traces_summary():
    p = REPO / "reports" / "traces_m6_summary.txt"
    return p.read_text(encoding="utf-8") if p.exists() else "(no trace summary)"

def main():
    manual = load_manual_reports()
    baseline = load_baseline()
    traces = load_traces_summary()

    def _as_str(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, indent=1)
        return str(v or "")

    lines = ["# Manual Manual Audit — Master Synthesis & Root-Cause Extraction", ""]
    lines.append(f"- Cards manually audited: **{len(manual)}**")
    tier_counts = Counter((m.get("tier") or "?").split(":")[0] for m in manual.values())
    lines.append(f"- Tier distribution: " + ", ".join(f"{k}: {v}" for k, v in sorted(tier_counts.items())))
    lines.append("")

    lines.append("## 1. Per-Card Verdicts (Manual)")
    lines.append("")
    lines.append("| Card | Tier | Status | Score | Top Defects | Main Root Cause |")
    lines.append("|---|---|---|---:|---|---|")
    for card_id, m in sorted(manual.items()):
        defects = (m.get("defects") or [])[:3]
        codes = ", ".join(d.get("code", "?") for d in defects) or "—"
        rc = (m.get("root_causes") or ["—"])[0] if m.get("root_causes") else (m.get("root_cause") or "—")
        lines.append(
            f"| `{card_id}` | {m.get('tier','?')} | {m.get('status','?')} | {m.get('total_score','?')} | "
            f"{codes} | {rc[:90]} |"
        )
    lines.append("")

    lines.append("## 2. Consolidated Problems & Defect Patterns (by pipeline stage)")
    lines.append("")
    stage_map = defaultdict(list)
    for card_id, m in manual.items():
        for d in (m.get("defects") or []):
            stage = d.get("pipeline_root_cause") or "—"
            stage_map[stage].append({"card_id": card_id, "code": d.get("code"), "severity": d.get("severity"),
                                     "desc": d.get("description", "")[:160], "field": d.get("field", "")})
    for stage in sorted(stage_map):
        lines.append(f"### 2.x `{stage}` — {len(stage_map[stage])} defect(s)")
        lines.append("")
        seen = set()
        for it in stage_map[stage]:
            key = (it["code"], it["desc"][:60])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- **[{it['severity']}] `{it['code']}`** on `{it['card_id']}` — {it['desc']}")
        lines.append("")

    lines.append("## 3. Extraction-Correctness Findings (comparative: manual vs. automated)")
    lines.append("")
    base_by_id = {c["card_id"]: c for c in baseline.get("cards", [])}
    for card_id, m in sorted(manual.items()):
        b = base_by_id.get(card_id, {})
        ext = _as_str(m.get("extraction_correctness")).strip()
        if ext:
            lines.append(f"### {card_id}")
            lines.append("")
            lines.append(ext)
            lines.append("")

    lines.append("## 4. Website Presentation Findings")
    lines.append("")
    for card_id, m in sorted(manual.items()):
        wp = _as_str(m.get("website_presentation_check")).strip()
        if wp:
            lines.append(f"### {card_id}")
            lines.append("")
            lines.append(wp)
            lines.append("")

    write_report("master_synthesis.md", "\n".join(lines))

    summary = {
        "manual_cards": {cid: {"tier": m.get("tier"), "status": m.get("status"), "score": m.get("total_score"),
                                "defects": [d.get("code") for d in (m.get("defects") or [])]}
                         for cid, m in manual.items()},
        "stage_defect_counts": {s: len(v) for s, v in stage_map.items()},
    }
    write_report("manual_audit_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
