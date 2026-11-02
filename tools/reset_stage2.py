#!/usr/bin/env python3
"""
tools/reset_stage2.py
Surgical Reset Tool for Stage 2 LLM Extractions (M6 -> M5).

CRITICAL SAFETY DESIGN:
- PRESERVES 100% of Stage 1 Audio, Vocal Isolations, Diarizations, and Scenarios!
  (scenarios/*.json, scenarios/*.diarized.srt, audio_clean/*.opus, video_speakers, scenario_turns).
- ONLY cleans Stage 2 LLM Q&A cards (data/extractions/*.qa.json, partial windows, and qa_units in SQLite).
- Resets video states from M6_LLM_EXTRACTED back to M5_SCENARIO_BUILT so Qwen 2.5 14B can
  re-extract high-quality seeker-focused dossiers using the updated prompt in ~1-2 min per video.
"""

import os
import sys
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.drive_sync import SharedWorkspace, STEP_RANK
from db.database import get_db_connection, DB_PATH


def reset_stage2_extractions(workspace: SharedWorkspace) -> None:
    print("=" * 72)
    print("  [STAGE 2 RESET] DAWAH DIALECTICS: SURGICAL LLM Q&A RESET")
    print("=" * 72)
    print(f"Workspace: {workspace.root}")

    # 1. Clean extractions/ directory
    ext_dir = workspace.extractions_dir
    partial_dir = workspace.partial_extractions_dir
    removed_shards = 0
    removed_partials = 0

    if ext_dir.exists():
        for p in ext_dir.glob("*.qa.json"):
            try:
                p.unlink()
                removed_shards += 1
            except Exception as e:
                print(f"   [-] Failed to unlink {p.name}: {e}")

    if partial_dir.exists():
        for p in partial_dir.glob("*.json"):
            try:
                p.unlink()
                removed_partials += 1
            except Exception as e:
                print(f"   [-] Failed to unlink {p.name}: {e}")

    print(f"   -> Removed {removed_shards} completed .qa.json shard(s)")
    print(f"   -> Removed {removed_partials} partial episode window(s)")

    # 2. Reset state files from M6_LLM_EXTRACTED back to M5_SCENARIO_BUILT
    reset_states = 0
    state_dir = workspace.state_dir
    if state_dir.exists():
        for sf in state_dir.glob("*.json"):
            st = workspace.read_json(sf, default={}) or {}
            if st.get("micro_step") == "M6_LLM_EXTRACTED" or st.get("step_rank", 0) >= STEP_RANK["M6_LLM_EXTRACTED"]:
                st["micro_step"] = "M5_SCENARIO_BUILT"
                st["step_rank"] = STEP_RANK["M5_SCENARIO_BUILT"]
                st["updated_at"] = __import__("time").time()
                workspace.atomic_write_json(sf, st)
                reset_states += 1

    print(f"   -> Reset {reset_states} video state file(s) back to M5_SCENARIO_BUILT")

    # 3. Clean SQLite qa_units and qa_fts tables, leaving scenario_turns & video_speakers untouched!
    db_file = DB_PATH
    if db_file.exists():
        conn = get_db_connection(db_file)
        try:
            count_qa = conn.execute("SELECT COUNT(*) FROM qa_units").fetchone()[0]
            conn.execute("DELETE FROM qa_units")
            try:
                conn.execute("DELETE FROM qa_fts")
            except Exception:
                pass
            conn.execute("UPDATE videos SET stage2_status = 'pending'")
            conn.commit()
            print(f"   -> Purged {count_qa} old Q&A row(s) from SQLite qa_units & qa_fts")
        finally:
            conn.close()

    # Verify Stage 1 Scenarios remain completely intact!
    scen_count = len(list(workspace.scenarios_dir.glob("*.json"))) if workspace.scenarios_dir.exists() else 0
    print("=" * 72)
    print("  [OK] STAGE 2 RESET COMPLETE!")
    print(f"  [OK] STAGE 1 SCENARIOS PRESERVED : {scen_count} complete multi-speaker scenario(s) intact!")
    print("  [OK] READY FOR STAGE 2 RE-EXTRACTION : Run Stage 2 worker or Auto mode to generate new cards.")
    print("=" * 72)


if __name__ == "__main__":
    ws = SharedWorkspace()
    reset_stage2_extractions(ws)
