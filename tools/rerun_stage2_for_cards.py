#!/usr/bin/env python3
"""Re-run the hardened Stage-2 extraction (M6) for the audited videos with the golden-result pipeline.

Uses a scratch workspace rooted at reports/regen_ws/ (scenarios backfilled from the DB clone).
Does NOT touch data/dawah_knowledge.db or the Drive workspace: new extraction shards are written to
reports/regen/{video_id}.qa.json for manual review.

Requires: tools/scenario_backfill.py first (or it backfills automatically).

Usage:
    python tools/rerun_stage2_for_cards.py [--ws PATH] [--force]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.scenario_extractor import Stage2ScenarioExtractor
from core.drive_sync import SharedWorkspace

REGEN_DIR = REPO / "reports" / "regen"
DEFAULT_WS = REPO / "reports" / "regen_ws"


def affected_video_ids() -> list:
    try:
        idx = json.loads((REPO / "reports" / "manual_audit_summary.json").read_text(encoding="utf-8"))
        vids = sorted({c.split("_")[0] for c in idx.get("manual_cards", {})})
        return [v for v in vids if v]
    except Exception:
        return []


def _parse_args(args):
    video_ids = [a for a in args if not a.startswith("-")]
    ws_arg = None
    force = "--force" in args
    if "--ws" in args:
        i = args.index("--ws")
        ws_arg = Path(args[i + 1]) if i + 1 < len(args) else None
    return video_ids, ws_arg, force


def main():
    args = sys.argv[1:]
    video_ids, ws_arg, force = _parse_args(args) or ([], None, False)
    video_ids = video_ids or affected_video_ids()
    ws = SharedWorkspace(root_dir=ws_arg) if ws_arg else SharedWorkspace(root_dir=DEFAULT_WS)

    # Backfill scenarios if missing (read-only from DB)
    scen_missing = [v for v in video_ids if not (ws.scenarios_dir / f"{v}.json").exists()]
    if scen_missing:
        print(f"[*] Scenarios missing for {len(scen_missing)} video(s); running scenario_backfill --ws {ws.root} ...")
        import subprocess
        subprocess.run([sys.executable, str(REPO / "tools" / "scenario_backfill.py"), "--ws", str(ws.root), *scen_missing], check=False)

    extractor = Stage2ScenarioExtractor(workspace=ws)

    REGEN_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for vid in video_ids:
        scen = ws.scenarios_dir / f"{vid}.json"
        if not scen.exists():
            results[vid] = {"status": "NO_SCENARIO", "units": []}
            continue
        if force:
            shard = ws.extractions_dir / f"{vid}.qa.json"
            if shard.exists():
                shard.unlink()
        try:
            units = extractor.extract_video_m6(vid, verbose=False)
        except Exception as e:
            results[vid] = {"status": f"ERROR: {e}", "units": []}
            continue
        # Write review copy to local REGEN_DIR (and the workspace shard is written by extract_video_m6)
        payload = {"schema_version": "2.0", "video_id": vid, "units": units or []}
        (REGEN_DIR / f"{vid}.qa.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        results[vid] = {"status": "OK", "units": len(units or [])}

    n = 0
    for vid, r in sorted(results.items()):
        print(f"  - {vid}: {r['status']} | units={r.get('units', '-')}")
        n += int(r.get("units") or 0)
    print(f"[+] Stage-2 re-extraction complete: {n} unit(s) written to {REGEN_DIR}")


if __name__ == "__main__":
    main()
