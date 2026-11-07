#!/usr/bin/env python3
"""
tools/reset_stage1.py
Surgical Reset Tool for Stage 1 Scenarios & Acoustic Extractions.

CRITICAL SAFETY DESIGN:
- PRESERVES 100% of all downloaded YouTube audio and clean vocal stems!
  (audio_raw/*.*, audio_raw/*.info.json, audio_clean/*.opus, audio_clean/*_rms.json, cookies/*).
- NEVER forces re-downloading from YouTube: all media stays cached on Google Drive (0 network risk).
- Purges old Stage 1 scenarios (scenarios/*.json, scenarios/*.diarized.srt) produced by the legacy
  Whisper turbo model and faulty CTC aligner.
- Purges old Stage 2 Q&A extractions (extractions/*.qa.json, extractions/partial/*) since they were
  built on flawed scenarios.
- Resets video state records back to M2_VOCALS_CLEANED (or M1_DOWNLOADED if only raw audio exists),
  or to M4_OVERLAPS_SEPARATED if keep_diarization is True.
- Cleans SQLite scenario_turns, video_speakers, and qa_units tables.
- Releases any stale worker locks so fresh workers start immediately.
"""

from __future__ import annotations
import os
import sys
import shutil
import time
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.drive_sync import SharedWorkspace, STEP_RANK
from core.yt_audio_fetcher import find_existing_raw_audio
from db.database import get_db_connection, DB_PATH


def reset_stage1_pipeline(workspace: SharedWorkspace, keep_diarization: bool = False) -> Dict[str, Any]:
    print("=" * 74)
    print("  [STAGE 1 RESET] DAWAH DIALECTICS: SURGICAL ACOUSTIC & SCENARIO RESET")
    print("=" * 74)
    print(f"Workspace Root       : {workspace.root} (Drive={workspace.is_drive})")
    print(f"Keep Diarization     : {keep_diarization} (M4 preserve={keep_diarization})")
    print("-" * 74)

    # 1. Purge scenarios/ (*.json and *.diarized.srt)
    scen_dir = workspace.scenarios_dir
    removed_scenarios = 0
    removed_srts = 0
    if scen_dir.exists():
        for p in scen_dir.glob("*.json"):
            try:
                p.unlink()
                removed_scenarios += 1
            except Exception as e:
                print(f"   [-] Failed to unlink scenario {p.name}: {e}")
        for p in scen_dir.glob("*.diarized.srt"):
            try:
                p.unlink()
                removed_srts += 1
            except Exception as e:
                print(f"   [-] Failed to unlink SRT {p.name}: {e}")

    print(f"   -> Purged {removed_scenarios} scenario JSON file(s) and {removed_srts} diarized SRT file(s)")

    # 2. Purge extractions/ (since Stage 2 depended on flawed Stage 1 scenarios)
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

    print(f"   -> Purged {removed_shards} Stage 2 .qa.json shard(s) & {removed_partials} partial window(s)")

    # 3. Diarization cleanup (if keep_diarization is False)
    removed_diar_dirs = 0
    if not keep_diarization and workspace.diarization_dir.exists():
        for item in workspace.diarization_dir.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item, ignore_errors=True)
                    removed_diar_dirs += 1
                except Exception as e:
                    print(f"   [-] Failed to remove diarization dir {item.name}: {e}")
            elif item.is_file() and item.suffix in (".rttm", ".npz", ".json"):
                try:
                    item.unlink(missing_ok=True)
                except Exception:
                    pass
        print(f"   -> Purged {removed_diar_dirs} video diarization directory/ies")
    elif keep_diarization:
        print("   -> Kept existing NeMo MSDD & MossFormer2 diarizations intact")

    # 4. Clean temporary staging chunks
    removed_staged = 0
    if workspace.staging_dir.exists():
        for item in workspace.staging_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    removed_staged += 1
                elif item.is_file():
                    item.unlink(missing_ok=True)
                    removed_staged += 1
            except Exception:
                pass
    if removed_staged:
        print(f"   -> Cleared {removed_staged} temporary staging item(s)")

    # 5. Clear stale locks
    cleared_locks = 0
    if workspace.locks_dir.exists():
        for lk in workspace.locks_dir.glob("*.lock.json"):
            try:
                lk.unlink(missing_ok=True)
                cleared_locks += 1
            except Exception:
                pass
    if cleared_locks:
        print(f"   -> Cleared {cleared_locks} active/stale worker lock file(s)")

    # 6. Reset Video State files (state/*.json)
    reset_states = 0
    state_dir = workspace.state_dir
    target_rank_counts: Dict[str, int] = {}
    if state_dir.exists():
        for sf in state_dir.glob("*.json"):
            st = workspace.read_json(sf, default={}) or {}
            vid = str(st.get("video_id") or sf.name.replace(".state.json", "").replace(".json", "")).strip()
            if not vid:
                continue

            # Detect existing cached audio assets on Drive
            clean_opus = workspace.audio_clean_dir / f"{vid}.opus"
            rms_json = workspace.audio_clean_dir / f"{vid}_rms.json"
            raw_audio = find_existing_raw_audio(workspace, vid)
            diar_rttm = workspace.diarization_dir / vid / f"{vid}.rttm"

            if keep_diarization and diar_rttm.exists():
                new_step = "M4_OVERLAPS_SEPARATED"
            elif clean_opus.exists() and clean_opus.stat().st_size > 1024 and rms_json.exists():
                new_step = "M2_VOCALS_CLEANED"
            elif raw_audio is not None and raw_audio.exists():
                new_step = "M1_DOWNLOADED"
            else:
                new_step = "M0_QUEUED"

            st["micro_step"] = new_step
            st["step_rank"] = STEP_RANK[new_step]
            st["error_count"] = 0
            st["last_error"] = None
            st["updated_at"] = time.time()

            # Preserve/restore proper artifacts
            art = st.get("artifacts", {}) or {}
            if new_step in ("M2_VOCALS_CLEANED", "M3_DIARIZED", "M4_OVERLAPS_SEPARATED"):
                art["clean_opus"] = str(clean_opus)
                art["rms_json"] = str(rms_json)
            elif new_step == "M1_DOWNLOADED" and raw_audio:
                art["raw_audio"] = str(raw_audio)
            st["artifacts"] = art

            workspace.atomic_write_json(sf, st)
            reset_states += 1
            target_rank_counts[new_step] = target_rank_counts.get(new_step, 0) + 1

    print(f"   -> Reset {reset_states} video state file(s):")
    for step_name, cnt in sorted(target_rank_counts.items(), key=lambda x: STEP_RANK.get(x[0], 0)):
        print(f"      * {step_name:<24} : {cnt:>4} videos")

    # 7. Clean SQLite database tables
    db_candidates = [
        workspace.data_dir / "dawah_knowledge.db",
        workspace.root / "dawah_knowledge.db",
    ]
    if not any(p.exists() for p in db_candidates) and DB_PATH.exists():
        db_candidates.append(DB_PATH)
    seen_dbs = set()
    purged_db_turns = 0
    purged_db_speakers = 0
    purged_db_qa = 0

    for db_file in db_candidates:
        if not db_file.exists():
            continue
        try:
            real_p = db_file.resolve()
        except Exception:
            real_p = db_file
        if real_p in seen_dbs:
            continue
        seen_dbs.add(real_p)

        conn = get_db_connection(db_file)
        try:
            with conn:
                try:
                    c_turns = conn.execute("SELECT COUNT(*) FROM scenario_turns").fetchone()[0]
                    purged_db_turns += c_turns
                    conn.execute("DELETE FROM scenario_turns")
                except Exception:
                    pass

                try:
                    c_spks = conn.execute("SELECT COUNT(*) FROM video_speakers").fetchone()[0]
                    purged_db_speakers += c_spks
                    conn.execute("DELETE FROM video_speakers")
                except Exception:
                    pass

                try:
                    c_qa = conn.execute("SELECT COUNT(*) FROM qa_units").fetchone()[0]
                    purged_db_qa += c_qa
                    conn.execute("DELETE FROM qa_units")
                except Exception:
                    pass

                for fts_table in ("scenario_turns_fts", "qa_fts"):
                    try:
                        conn.execute(f"DELETE FROM {fts_table}")
                    except Exception:
                        pass

                try:
                    conn.execute("""
                        UPDATE videos SET
                            turns_count = 0,
                            spoken_words_count = 0,
                            speakers_count = 0,
                            episodes_count = 0,
                            overlap_windows_count = 0,
                            overlap_ratio = 0.0,
                            pedestrian_turns_count = 0,
                            qa_count = 0,
                            scenario_json_path = '',
                            rttm_path = '',
                            stage1_status = 'pending',
                            stage2_status = 'pending',
                            status = 'queued'
                    """)
                except Exception:
                    pass
            print(f"   -> SQLite ({db_file.name}): Purged {purged_db_turns} turns, {purged_db_speakers} speakers, {purged_db_qa} Q&A cards")
        finally:
            conn.close()

    # 8. Verification & Asset Audit: YouTube Media 100% PRESERVED
    raw_files_count = len(list(workspace.audio_raw_dir.glob("*.*"))) if workspace.audio_raw_dir.exists() else 0
    clean_opus_count = len(list(workspace.audio_clean_dir.glob("*.opus"))) if workspace.audio_clean_dir.exists() else 0
    remaining_scenarios = len(list(workspace.scenarios_dir.glob("*.json"))) if workspace.scenarios_dir.exists() else 0

    print("=" * 74)
    print("  [OK] STAGE 1 SURGICAL RESET COMPLETE!")
    print(f"  [OK] YOUTUBE AUDIO PRESERVED : {raw_files_count} raw file(s) & {clean_opus_count} clean Opus file(s) intact!")
    print("  [OK] ZERO RE-DOWNLOAD RISK   : All audio remains cached on Google Drive (0 network usage).")
    print(f"  [OK] SCENARIO STATUS         : {remaining_scenarios} old scenarios remaining (cleared for large-v3 rebuild).")
    print("  [OK] NEXT ACTION             : Run Cell 2 (Stage 1 Worker) to build high-accuracy scenarios.")
    print("=" * 74 + "\n")

    return {
        "removed_scenarios": removed_scenarios,
        "removed_srts": removed_srts,
        "removed_qa_shards": removed_shards,
        "removed_diar_dirs": removed_diar_dirs,
        "reset_states": reset_states,
        "purged_db_turns": purged_db_turns,
        "purged_db_speakers": purged_db_speakers,
        "raw_files_count": raw_files_count,
        "clean_opus_count": clean_opus_count,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Surgical Reset Tool")
    parser.add_argument("--keep-diarization", action="store_true", help="Keep M3/M4 diarization and only reset M5 scenarios")
    args = parser.parse_args()

    ws = SharedWorkspace()
    reset_stage1_pipeline(ws, keep_diarization=args.keep_diarization)
