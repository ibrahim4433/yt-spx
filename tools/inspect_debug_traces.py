#!/usr/bin/env python3
"""
tools/inspect_debug_traces.py
CLI inspector for `debug_traces/*.trace.json` and `session_debug_report.md`.

Usage:
    python tools/inspect_debug_traces.py
    python tools/inspect_debug_traces.py --dir data/debug_traces --failures-only
    python tools/inspect_debug_traces.py --video-id benbST7nrfc
    python tools/inspect_debug_traces.py --regen-report
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.debug_tracer import generate_session_debug_report
from core.drive_sync import SharedWorkspace


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect YT-SPX per-video forensic pipeline traces.")
    parser.add_argument("--dir", type=str, default="", help="Path to workspace root or debug_traces directory")
    parser.add_argument("--video-id", type=str, default="", help="Inspect a single video_id trace in detail")
    parser.add_argument("--failures-only", action="store_true", help="Show only videos with 0 cards or collapsed diarization")
    parser.add_argument("--regen-report", action="store_true", help="Regenerate session_debug_report.md and print path")
    args = parser.parse_args(argv)

    if args.dir:
        p = Path(args.dir)
        ws_root = p.parent if p.name == "debug_traces" else p
        ws = SharedWorkspace(root_dir=ws_root)
    else:
        ws = SharedWorkspace()

    traces_dir = ws.debug_traces_dir
    if args.regen_report:
        rep = generate_session_debug_report(ws)
        print(f"[OK] Regenerated forensic report: {rep}")
        return 0

    if args.video_id:
        tf = traces_dir / f"{args.video_id}.trace.json"
        if not tf.exists():
            print(f"[-] Trace not found: {tf}")
            return 1
        print(json.dumps(json.loads(tf.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
        return 0

    trace_files = sorted(traces_dir.glob("*.trace.json"))
    if not trace_files:
        print(f"[*] No trace files found in {traces_dir}")
        return 0

    print(f"[*] Inspecting {len(trace_files)} trace file(s) in {traces_dir}:\n")
    for tf in trace_files:
        tr = json.loads(tf.read_text(encoding="utf-8"))
        vid = tr.get("video_id", tf.stem)
        m5 = tr.get("stage1_m5") or {}
        s2 = tr.get("stage2_m6") or {}
        health = (m5.get("diarization_health") or {}).get("status", "UNKNOWN")
        kept = s2.get("kept_cards_count")
        drops = s2.get("dropped_gate_counts") or {}
        if args.failures_only and health == "HEALTHY" and (kept is None or kept > 0) and not drops:
            continue
        print(
            f"[{vid}] S1={health} (spk={m5.get('speakers_count', '?')}, "
            f"inter={m5.get('interlocutor_words', '?')}/{m5.get('total_words', '?')}) | "
            f"S2: kept={kept} r1={s2.get('r1_candidates_total', '?')} drops={drops}"
        )
        for w in (s2.get("windows") or []):
            for u in (w.get("units") or []):
                if u.get("outcome") != "KEPT" or u.get("r2_retries") or u.get("final_defect_codes"):
                    print(
                        f"    W{w.get('window_index')} U{u.get('unit_index')}: {u.get('outcome')} "
                        f"reason={u.get('drop_reason')} pre_r2={u.get('pre_r2_defect_codes')} "
                        f"repairs={list((u.get('r2_en_repairs') or {}).keys())} "
                        f"final_score={u.get('final_score')} defects={u.get('final_defect_codes')}"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
