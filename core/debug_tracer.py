#!/usr/bin/env python3
"""
core/debug_tracer.py
End-to-end Forensic Pipeline Debugger & Session Report Generator for Dawah Dialectics V2 (yt-spx).

Persists compact, structured per-video execution traces (`debug_traces/{video_id}.trace.json`)
across Stage 1 M3 (diarization + retries), Stage 1 M5 (turn assembly & speaker balance), and
Stage 2 M6 (pre-extraction gates, Round 1 candidates, pre-Round-2 diagnostics, Round 2 <think>
excerpts & repair diffs, retry loops, exact anchor drop sub-reasons, and final v3.2 rubric audit),
and generates a human-readable `debug_traces/session_debug_report.md` after every batch.
"""

import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _trace_path(workspace: Any, video_id: str) -> Path:
    traces_dir = getattr(workspace, "debug_traces_dir", None)
    if traces_dir is None:
        traces_dir = Path(workspace.root) / "debug_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    return traces_dir / f"{video_id}.trace.json"


def load_video_trace(workspace: Any, video_id: str) -> Dict[str, Any]:
    p = _trace_path(workspace, video_id)
    data = workspace.read_json(p, default=None) if hasattr(workspace, "read_json") else None
    if isinstance(data, dict) and data.get("video_id") == video_id:
        return data
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "updated_at": time.time(),
        "stage1_m3": {},
        "stage1_m5": {},
        "stage2_m6": {},
    }


def record_video_trace_section(
    workspace: Any,
    video_id: str,
    section: str,
    payload: Dict[str, Any],
    title_ar: str = "",
) -> Path:
    """Atomically updates one pipeline stage section (`stage1_m3`, `stage1_m5`, `stage2_m6`)."""
    trace = load_video_trace(workspace, video_id)
    if title_ar:
        trace["title_ar"] = title_ar
    trace[section] = payload
    trace["updated_at"] = time.time()
    trace["updated_at_iso"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p = _trace_path(workspace, video_id)
    workspace.atomic_write_json(p, trace)
    return p


def extract_think_excerpt(raw_llm_output: str, max_chars: int = 1200) -> str:
    """Extracts the model's <think>...</think> reasoning block (truncated safely for diagnostics)."""
    if not raw_llm_output or not isinstance(raw_llm_output, str):
        return ""
    low = raw_llm_output.lower()
    s_idx = low.find("<think>")
    if s_idx == -1:
        return ""
    e_idx = low.find("</think>", s_idx + 7)
    inner = raw_llm_output[s_idx + 7 : e_idx] if e_idx != -1 else raw_llm_output[s_idx + 7 :]
    inner = " ".join(inner.split()).strip()
    if len(inner) > max_chars:
        return inner[:max_chars] + "... [truncated]"
    return inner


def classify_diarization_health(
    duration_sec: float,
    speakers_count: int,
    total_words: int,
    interlocutor_words: int,
    turns_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Classifies Stage 1 speaker separation quality for both diagnostics and Pre-Stage-2 gating."""
    inter_ratio = (interlocutor_words / total_words) if total_words > 0 else 0.0
    if duration_sec > 180.0 and total_words >= 200:
        if speakers_count <= 1:
            return {
                "status": "COLLAPSED",
                "reason": f"SINGLE_SPEAKER_COLLAPSE (speakers_count={speakers_count}, dur={duration_sec:.0f}s, words={total_words})",
                "interlocutor_word_ratio": round(inter_ratio, 4),
            }
        if (turns_count is None or turns_count >= 10) and (interlocutor_words < 40 or inter_ratio < 0.035):
            return {
                "status": "COLLAPSED",
                "reason": f"FUNCTIONAL_DIARIZATION_COLLAPSE (inter_words={interlocutor_words}/{total_words}={inter_ratio:.1%}, dur={duration_sec:.0f}s)",
                "interlocutor_word_ratio": round(inter_ratio, 4),
            }
        if (turns_count is None or turns_count >= 10) and inter_ratio < 0.12:
            return {
                "status": "BORDERLINE",
                "reason": f"LOW_INTERLOCUTOR_SHARE (inter_words={interlocutor_words}/{total_words}={inter_ratio:.1%})",
                "interlocutor_word_ratio": round(inter_ratio, 4),
            }
    return {
        "status": "HEALTHY",
        "reason": "OK",
        "interlocutor_word_ratio": round(inter_ratio, 4),
    }


def generate_session_debug_report(workspace: Any) -> Path:
    """
    Aggregates all `debug_traces/*.trace.json` files into a concise, actionable Markdown
    forensic report (`debug_traces/session_debug_report.md`) showing Stage 1 diarization health,
    Stage 2 extraction funnel, exact drop sub-reasons, Round 2 repair effectiveness, and top defects.
    """
    traces_dir = getattr(workspace, "debug_traces_dir", None) or (Path(workspace.root) / "debug_traces")
    traces_dir.mkdir(parents=True, exist_ok=True)
    report_path = traces_dir / "session_debug_report.md"

    traces: List[Dict[str, Any]] = []
    for tf in sorted(traces_dir.glob("*.trace.json")):
        data = workspace.read_json(tf, default=None) if hasattr(workspace, "read_json") else None
        if isinstance(data, dict) and data.get("video_id"):
            traces.append(data)

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: List[str] = [
        "# YT-SPX V2 Pipeline Forensic Debug Report",
        "",
        f"- **Generated:** {now_iso}",
        f"- **Traced Videos:** {len(traces)}",
        "",
    ]

    if not traces:
        lines.append("_No per-video traces recorded yet._")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    # 1. Stage 1 Diarization Health Overview
    s1_health_counts: Counter = Counter()
    m3_retry_videos = 0
    m3_recovered_videos = 0
    s1_rows: List[str] = []
    for tr in traces:
        vid = tr["video_id"]
        m3 = tr.get("stage1_m3") or {}
        m5 = tr.get("stage1_m5") or {}
        if m3.get("collapse_triggered"):
            m3_retry_videos += 1
            if m3.get("recovered"):
                m3_recovered_videos += 1
        if m5 or m3:
            health = (m5.get("diarization_health") or {}).get("status") or ("OK" if m3 else "UNKNOWN")
            s1_health_counts[health] += 1
            dur = m5.get("duration_sec") or round((m3.get("total_ms") or 0) / 1000.0, 1)
            spks = m5.get("speakers_count", m3.get("final_speakers_count", "?"))
            turns = m5.get("turns_count", "?")
            tot_w = m5.get("total_words", "?")
            int_w = m5.get("interlocutor_words", "?")
            int_pct = (m5.get("diarization_health") or {}).get("interlocutor_word_ratio")
            int_str = f"{int_w}/{tot_w} ({int_pct:.1%})" if isinstance(int_pct, (int, float)) else "-"
            m3_note = (
                f"retried({len(m3.get('retry_attempts') or [])})->"
                f"{'RECOVERED:' + str(m3.get('accepted_strategy') or '') if m3.get('recovered') else 'KEPT_INITIAL'}"
                if m3.get("collapse_triggered")
                else "direct"
            )
            flickers = m5.get("flicker_smoothings", 0)
            s1_rows.append(
                f"| `{vid}` | {dur}s | {spks} | {turns} | {int_str} | {flickers} | {m3_note} | **{health}** |"
            )

    lines.extend([
        "## 1. Stage 1 Acoustic Diarization & Turn Assembly Health",
        "",
        f"- **Health Breakdown:** {dict(s1_health_counts)}",
        f"- **M3 Anti-Collapse Triggered:** {m3_retry_videos} video(s) | **Recovered:** {m3_recovered_videos}",
        "",
        "| Video | Duration | Speakers | Turns | Interlocutor Words | Flickers Smoothed | M3 Strategy | Health |",
        "| :--- | ---: | ---: | ---: | :--- | ---: | :--- | :--- |",
    ])
    lines.extend(s1_rows or ["| _None_ | - | - | - | - | - | - | - |"])
    lines.append("")

    # 2. Stage 2 Extraction Funnel & Drop Attribution
    s2_traces = [tr for tr in traces if tr.get("stage2_m6")]
    s2_total = len(s2_traces)
    s2_diar_skipped = 0
    s2_zero_card = 0
    total_r1_cands = 0
    total_kept = 0
    total_r2_flagged = 0
    total_r2_repaired = 0
    total_r2_retries = 0
    drop_counter: Counter = Counter()
    defect_counter: Counter = Counter()
    s2_rows: List[str] = []
    drilldowns: List[str] = []

    for tr in s2_traces:
        vid = tr["video_id"]
        s2 = tr["stage2_m6"]
        kept = int(s2.get("kept_cards_count") or 0)
        r1_c = int(s2.get("r1_candidates_total") or 0)
        total_r1_cands += r1_c
        total_kept += kept
        if s2.get("skipped_pre_stage2_diarization"):
            s2_diar_skipped += 1
        if kept == 0:
            s2_zero_card += 1

        drops = s2.get("dropped_gate_counts") or {}
        for dk, dv in drops.items():
            drop_counter[dk] += int(dv)

        vid_defects: List[str] = []
        vid_elapsed = float(s2.get("total_elapsed_sec") or 0.0)
        for win in (s2.get("windows") or []):
            for u_tr in (win.get("units") or []):
                if u_tr.get("pre_r2_flagged"):
                    total_r2_flagged += 1
                    if u_tr.get("r2_en_repairs"):
                        total_r2_repaired += 1
                total_r2_retries += len(u_tr.get("r2_retries") or [])
                for code in (u_tr.get("final_defect_codes") or []):
                    defect_counter[code] += 1
                    vid_defects.append(code)
                for code in (u_tr.get("pre_r2_defect_codes") or []):
                    if u_tr.get("outcome") != "KEPT":
                        defect_counter[f"PRE_R2:{code}"] += 1

        drops_str = ", ".join(f"`{k}:{v}`" for k, v in sorted(drops.items())) if drops else "none"
        status_str = "SKIP_DIAR" if s2.get("skipped_pre_stage2_diarization") else (f"KEPT({kept})" if kept > 0 else "ZERO_CARDS")
        s2_rows.append(
            f"| `{vid}` | {vid_elapsed:.1f}s | {r1_c} | {kept} | {status_str} | {drops_str} |"
        )

        # Build detailed forensic drilldown for zero-card videos, retries, or cards with defects
        needs_drill = (
            kept == 0
            or bool(drops)
            or bool(vid_defects)
            or any((u_tr.get("r2_retries") or u_tr.get("pre_r2_flagged")) for w in (s2.get("windows") or []) for u_tr in (w.get("units") or []))
        )
        if needs_drill:
            d_lines = [f"### `{vid}` — {status_str} ({vid_elapsed:.1f}s)"]
            if tr.get("title_ar"):
                d_lines.append(f"- **Title:** {tr['title_ar']}")
            if s2.get("skipped_pre_stage2_diarization"):
                d_lines.append(f"- **Pre-Stage-2 Diarization Skip:** `{s2.get('pre_stage2_skip_reason')}`")
            for win in (s2.get("windows") or []):
                w_idx = win.get("window_index", 1)
                if win.get("gatekeeper_skipped"):
                    d_lines.append(f"- **Window {w_idx} Gatekeeper Skip:** `{win.get('gatekeeper_reason')}`")
                    continue
                d_lines.append(
                    f"- **Window {w_idx}:** R1={win.get('r1_elapsed_sec', 0)}s ({win.get('r1_parsed_count', 0)} parsed -> {win.get('r1_accepted_count', 0)} pre-grounded), "
                    f"R2={win.get('r2_elapsed_sec', 0)}s"
                )
                for u_tr in (win.get("units") or []):
                    uid = u_tr.get("unit_index", "?")
                    outcome = u_tr.get("outcome", "UNKNOWN")
                    drop_r = u_tr.get("drop_reason") or ""
                    d_lines.append(f"  - **Unit {uid} -> `{outcome}`** {('(' + drop_r + ')') if drop_r else ''}")
                    d_lines.append(f"    - R1 `question_en`: `{str(u_tr.get('r1_question_en') or '')[:180]}`")
                    if u_tr.get("pre_r2_defect_codes"):
                        d_lines.append(f"    - Pre-R2 Flags: `{', '.join(u_tr['pre_r2_defect_codes'])}` (thinking={u_tr.get('r2_thinking_enabled')})")
                    if u_tr.get("r2_en_repairs"):
                        for rk, rdiff in u_tr["r2_en_repairs"].items():
                            d_lines.append(f"    - R2 Repair `{rk}`: `{str(rdiff.get('before', ''))[:100]}` -> `{str(rdiff.get('after', ''))[:120]}`")
                    if u_tr.get("r2_think_excerpt"):
                        d_lines.append(f"    - R2 `<think>` excerpt: _{u_tr['r2_think_excerpt'][:320]}_")
                    if u_tr.get("r2_retries"):
                        for rt in u_tr["r2_retries"]:
                            d_lines.append(
                                f"    - Retry {rt.get('attempt')}/3 ({rt.get('mode')}): missing=`{rt.get('missing_before')}` -> ok={rt.get('passed')}"
                            )
                    if u_tr.get("final_score") is not None:
                        d_lines.append(
                            f"    - Final Audit: **{u_tr.get('final_score')}/100** ({u_tr.get('final_tier')}) | gates=`{u_tr.get('final_gates')}` | defects=`{u_tr.get('final_defect_codes')}`"
                        )
            drilldowns.append("\n".join(d_lines))

    lines.extend([
        "## 2. Stage 2 Extraction Funnel & Yield",
        "",
        f"- **Stage 2 Videos Traced:** {s2_total} | **Pre-Skipped (Diarization Collapse):** {s2_diar_skipped} | **Zero-Card Videos:** {s2_zero_card} ({(s2_zero_card / max(1, s2_total)):.1%})",
        f"- **Round 1 Candidates:** {total_r1_cands} -> **Final Kept Cards:** {total_kept} ({(total_kept / max(1, total_r1_cands)):.1%} survival)",
        f"- **Pre-R2 Flagged Units:** {total_r2_flagged} | **Units with Applied R2 English Repairs:** {total_r2_repaired} | **Total R2 Arabic Retries:** {total_r2_retries}",
        f"- **Drop Breakdown:** {dict(drop_counter.most_common()) if drop_counter else 'none'}",
        f"- **Top Recurring Defect Codes:** {dict(defect_counter.most_common(12)) if defect_counter else 'none'}",
        "",
        "| Video | Stage 2 Time | R1 Cands | Kept | Verdict | Drop Tags |",
        "| :--- | ---: | ---: | ---: | :--- | :--- |",
    ])
    lines.extend(s2_rows or ["| _None_ | - | - | - | - | - |"])
    lines.append("")

    if drilldowns:
        lines.extend(["## 3. Per-Video Forensic Drilldown (Drops, Repairs & Warnings)", ""])
        lines.extend(drilldowns)
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
