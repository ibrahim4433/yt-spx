#!/usr/bin/env python3
"""
core/pipeline_orchestrator.py
Smart-Batch, Anti-Ban, Multi-Account Orchestrator for Dawah Dialectics V2 (yt-spx).

Engineered for:
  1. Free-Tier Google Colab T4 sessions (2-4 hours/day):
     - Graceful time-budget stop (--max-session-minutes 165) before abrupt runtime eviction.
     - Resumes mid-video micro-steps (M1..M5) FIRST before touching YouTube for new M0 downloads.
     - Small anti-ban batches (--batch-videos 6) + automatic offline fallback if YouTube rate-limits.
  2. Multi-Account Shared 5 TB Google Drive (/content/drive/MyDrive/yt-spx):
     - Multiple Colab accounts can run simultaneously in 'auto', 'stage1', or 'stage2' mode.
     - Optimistic per-video locks prevent duplicate GPU work across accounts.
     - Single-writer '_sqlite_compactor' merges completed JSON shards into SQLite safely.
  3. Subprocess CUDA Isolation:
     - Stage 1 (Demucs + NeMo MSDD + MossFormer2 + Whisper + CTC ~8.8 GB VRAM) and
       Stage 2 (Qwen 2.5 14B Q5_K_M GGUF 12,288 ctx ~13.2 GB VRAM) run in separate child
       processes so CUDA memory is 100% reclaimed between stages.
"""

import os
import sys
import json
import time
import random
import argparse
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.drive_sync import SharedWorkspace, MICRO_STEPS, STEP_RANK
from core.metadata_linker import build_manifest
from core.yt_audio_fetcher import AntiBanCircuitBreakerError, PermanentVideoUnavailableError


def load_or_create_manifest(workspace: SharedWorkspace) -> List[Dict[str, Any]]:
    """Loads the canonical 2,640-video manifest from Drive data_dir or local repo."""
    drive_manifest = workspace.data_dir / "video_manifest.json"
    repo_manifest = BASE_DIR / "data" / "video_manifest.json"

    if drive_manifest.exists():
        data = workspace.read_json(drive_manifest)
        if isinstance(data, list) and data:
            return data

    if repo_manifest.exists():
        data = workspace.read_json(repo_manifest)
        if isinstance(data, list) and data:
            if workspace.is_drive:
                workspace.atomic_write_json(drive_manifest, data)
            return data

    manifest = build_manifest(output_path=drive_manifest)
    return manifest


def collect_workspace_telemetry(workspace: SharedWorkspace, manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scans micro-step states, active locks, voiceprints, and YouTube rate-limit health."""
    step_counts = {step: 0 for step in MICRO_STEPS}
    permanently_unavailable = 0
    unsynced_scenarios = 0
    unsynced_qa = 0

    # Fast scan of state_dir files first instead of 2,640 individual stat calls when empty
    known_states: Dict[str, Dict[str, Any]] = {}
    for st_file in workspace.state_dir.glob("*.state.json"):
        vid = st_file.name.replace(".state.json", "")
        st = workspace.read_json(st_file, default={}) or {}
        if isinstance(st, dict):
            known_states[vid] = st

    # Also check scenarios/ and extractions/ in case state files were reconciled implicitly
    scen_ids = {p.stem for p in workspace.scenarios_dir.glob("*.json")}
    qa_ids = {p.name.replace(".qa.json", "") for p in workspace.extractions_dir.glob("*.qa.json")}

    for entry in manifest:
        vid = entry["video_id"]
        st = known_states.get(vid)
        if st is None:
            if vid in qa_ids:
                ms = "M6_LLM_EXTRACTED"
                unsynced_qa += 1
            elif vid in scen_ids:
                ms = "M5_SCENARIO_BUILT"
                unsynced_scenarios += 1
            else:
                ms = "M0_QUEUED"
            step_counts[ms] = step_counts.get(ms, 0) + 1
            continue

        if st.get("permanent_unavailable"):
            permanently_unavailable += 1
            continue

        ms = st.get("micro_step", "M0_QUEUED")
        if ms not in step_counts:
            ms = "M0_QUEUED"
        step_counts[ms] += 1

        rank = STEP_RANK.get(ms, 0)
        if rank >= STEP_RANK["M5_SCENARIO_BUILT"] and not st.get("db_scenario_synced"):
            unsynced_scenarios += 1
        if rank >= STEP_RANK["M6_LLM_EXTRACTED"] and not st.get("db_qa_synced"):
            unsynced_qa += 1

    now = time.time()
    active_locks = []
    for lp in workspace.locks_dir.glob("*.lock.json"):
        ld = workspace.read_json(lp, default={}) or {}
        if float(ld.get("expires_at", 0.0)) > now:
            active_locks.append({
                "lock": ld.get("lock_name", lp.stem),
                "worker": ld.get("worker_id", "unknown"),
                "ttl_remaining_sec": int(float(ld["expires_at"]) - now),
            })

    rl = workspace.read_json(workspace.locks_dir / "_youtube_ratelimit.json", default={}) or {}
    cb_until = float(rl.get("circuit_breaker_until", 0.0))
    cb_active = cb_until > now

    vp_reg = workspace.read_json(workspace.profiles_dir / "voiceprint_registry.json", default={}) or {}
    profiles_count = len(vp_reg.get("profiles", []))

    return {
        "workspace_root": str(workspace.root),
        "is_google_drive": workspace.is_drive,
        "worker_id": workspace.worker_id,
        "manifest_total": len(manifest),
        "step_counts": step_counts,
        "stage1_ready_or_done": step_counts["M5_SCENARIO_BUILT"] + step_counts["M6_LLM_EXTRACTED"],
        "stage2_done": step_counts["M6_LLM_EXTRACTED"],
        "in_progress_stage1": (
            step_counts["M1_DOWNLOADED"]
            + step_counts["M2_VOCALS_CLEANED"]
            + step_counts["M3_DIARIZED"]
            + step_counts["M4_OVERLAPS_SEPARATED"]
        ),
        "permanently_unavailable": permanently_unavailable,
        "unsynced_scenarios": unsynced_scenarios,
        "unsynced_qa": unsynced_qa,
        "active_locks": active_locks,
        "voiceprint_profiles_enrolled": profiles_count,
        "youtube_circuit_breaker": {
            "active": cb_active,
            "remaining_minutes": round((cb_until - now) / 60.0, 1) if cb_active else 0.0,
            "reason": rl.get("reason", "") if cb_active else "Healthy",
        },
    }


def select_stage1_candidates(
    workspace: SharedWorkspace,
    manifest: List[Dict[str, Any]],
    include_shorts: bool = True,
) -> List[Dict[str, Any]]:
    """
    Orders Stage 1 candidates so partially completed videos (M4 -> M3 -> M2 -> M1)
    on Drive are ALWAYS resumed first before downloading any new M0 video from YouTube!
    """
    in_progress = []
    unstarted = []

    for v in manifest:
        if not include_shorts and v.get("is_short"):
            continue
        vid = v["video_id"]
        st_path = workspace.state_path(vid)
        if st_path.exists():
            st = workspace.get_video_state(vid)
        elif (workspace.scenarios_dir / f"{vid}.json").exists():
            continue
        else:
            st = {"step_rank": 0, "permanent_unavailable": False, "error_count": 0}

        if st.get("permanent_unavailable"):
            continue
        if int(st.get("error_count", 0)) >= 5:
            continue
        rank = int(st.get("step_rank", 0))
        if rank >= STEP_RANK["M5_SCENARIO_BUILT"]:
            continue

        item = {**v, "_step_rank": rank}
        if rank > 0:
            in_progress.append(item)
        else:
            unstarted.append(item)

    # Highest micro-step rank first (finish M4 before M3 before M2 before M1)
    in_progress.sort(key=lambda x: (x["_step_rank"], int(x.get("priority_score", 0))), reverse=True)

    # Small deterministic+random tie-breaker so multiple Colab accounts spread cleanly across top priority videos
    unstarted.sort(
        key=lambda x: (int(x.get("priority_score", 0)), random.random()),
        reverse=True,
    )
    return in_progress + unstarted


def select_stage2_candidates(
    workspace: SharedWorkspace,
    manifest: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Finds all videos with completed M5_SCENARIO_BUILT scenarios that have not yet finished M6_LLM_EXTRACTED.
    Prioritizes videos with partial episode checkpoints in extractions/partial/ first!
    """
    manifest_by_id = {v["video_id"]: v for v in manifest}
    partial_vids = {
        p.name.split("_win")[0]
        for p in workspace.partial_extractions_dir.glob("*_win*.json")
    }

    candidates = []
    for scen_path in workspace.scenarios_dir.glob("*.json"):
        vid = scen_path.stem
        qa_path = workspace.extractions_dir / f"{vid}.qa.json"
        if qa_path.exists():
            continue
        st = workspace.get_video_state(vid)
        if st["step_rank"] >= STEP_RANK["M6_LLM_EXTRACTED"] and qa_path.exists():
            continue
        if int(st.get("error_count", 0)) >= 5:
            continue
        meta = manifest_by_id.get(vid, {"video_id": vid, "priority_score": 50})
        has_partial = 1 if vid in partial_vids else 0
        candidates.append({**meta, "_has_partial": has_partial})

    candidates.sort(
        key=lambda x: (x["_has_partial"], int(x.get("priority_score", 0)), random.random()),
        reverse=True,
    )
    return candidates


def run_stage1_worker(args: argparse.Namespace) -> int:
    """Runs inside an isolated Python process so Stage 1 audio models own the GPU cleanly."""
    import torch
    from core.audio_diarizer import Stage1AudioPipeline

    workspace = SharedWorkspace(worker_name=args.worker_id)
    manifest = load_or_create_manifest(workspace)
    candidates = select_stage1_candidates(
        workspace,
        manifest,
        include_shorts=not args.debates_only,
    )

    if not candidates:
        print("[Stage 1] No pending videos require Stage 1 processing.")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    budget_label = f"{args.max_session_minutes} min" if args.max_session_minutes > 0 else "unlimited (until disconnect)"
    print("=" * 72)
    print(f"[Stage 1 Audio & Diarization Worker] Worker: {workspace.worker_id}")
    print(f"  Device: {device} | Whisper: {args.whisper_model} | Max Batch: {args.batch_videos} videos")
    print(f"  Candidates Ready: {len(candidates)} | Session Budget: {budget_label}")
    print("=" * 72, flush=True)

    pipeline = Stage1AudioPipeline(
        workspace=workspace,
        whisper_model=args.whisper_model,
        device=device,
        batch_size=args.whisper_batch_size,
        keep_raw_audio=args.keep_raw_audio,
    )

    t_session_start = time.time()
    deadline = (t_session_start + (args.max_session_minutes * 60.0)) if args.max_session_minutes > 0 else None
    completed_count = 0
    consecutive_errors = 0
    last_error_sig = ""
    allow_downloads = not args.offline_only
    exit_code = 0

    # Auto-heal transient errors from previous environment bug iterations (e.g. metadata_errors, msdd_models, float4_e2m1fn, libcublas, cuDNN, fts5, rotated cookies)
    for cand in candidates:
        vid = cand["video_id"]
        st = workspace.get_video_state(vid)
        last_err = str(st.get("last_error") or "")
        if any(tok in last_err for tok in ("metadata_errors", "msdd_models", "float4_e2m1fn", "NeuralDiarizer", "libcublas", "libcudnn", "cudnn", "sublibrary", "fts5", "invalid", "cookie", "circuit breaker")):
            st["error_count"] = 0
            st["last_error"] = None
            workspace.atomic_write_json(workspace.state_path(vid), st)

    for cand in candidates:
        if args.batch_videos > 0 and completed_count >= args.batch_videos:
            print(f"\n[Stage 1] Reached target smart-batch size ({args.batch_videos} videos). Stopping cleanly.")
            break
        if deadline is not None and time.time() >= deadline:
            print(f"\n[Stage 1] Session time budget ({args.max_session_minutes}m) reached. Saving & exiting cleanly.")
            break

        vid = cand["video_id"]
        if not allow_downloads and cand.get("_step_rank", 0) < STEP_RANK["M1_DOWNLOADED"]:
            continue

        lock_name = f"{vid}.stage1"
        if not workspace.acquire_lock(lock_name, ttl_sec=2700, verify_delay_sec=0.3 if workspace.is_drive else 0.0):
            continue

        try:
            t0 = time.time()
            print(
                f"\n>>> [Stage 1] [{completed_count + 1}/{args.batch_videos}] {vid} "
                f"(Resume rank={cand.get('_step_rank', 0)}) | {cand.get('title_ar', '')[:55]}",
                flush=True,
            )
            scen = pipeline.process_single_video(cand, allow_youtube_download=allow_downloads)
            if scen is not None:
                completed_count += 1
                consecutive_errors = 0
                last_error_sig = ""
                elapsed = round(time.time() - t0, 1)
                m = scen.get("metrics", {})
                print(
                    f"    [OK] M5_SCENARIO_BUILT in {elapsed}s | Turns={m.get('turns_count') or m.get('total_turns') or len(scen.get('turns', []))} "
                    f"| Overlaps={m.get('overlap_windows_count', 0)} | Pedestrians={m.get('pedestrian_turns_count', 0)}",
                    flush=True,
                )
        except AntiBanCircuitBreakerError as cb_err:
            print(f"\n[!] ANTI-BAN CIRCUIT BREAKER: {cb_err}", flush=True)
            print("    -> Switching Stage 1 to OFFLINE-ONLY mode (finishing M1..M4 files already on Drive)!", flush=True)
            allow_downloads = False
        except PermanentVideoUnavailableError as perm_err:
            print(f"    [SKIP PERMANENT] {vid}: {perm_err}", flush=True)
            workspace.advance_micro_step(
                vid,
                "M0_QUEUED",
                extra_updates={"permanent_unavailable": True, "last_error": str(perm_err)[:300]},
            )
        except Exception as exc:
            err_sig = f"{type(exc).__name__}: {str(exc)[:140]}"
            print(f"    [ERROR] Stage 1 error on {vid} ({type(exc).__name__}): {exc}", flush=True)
            if consecutive_errors == 0:
                print(traceback.format_exc(limit=5), flush=True)

            if err_sig == last_error_sig:
                consecutive_errors += 1
            else:
                consecutive_errors = 1
                last_error_sig = err_sig

            # Do not penalize per-video error_count on repeated systemic environment/import errors
            if not (consecutive_errors >= 2 and isinstance(exc, (ImportError, ModuleNotFoundError, AttributeError, TypeError))):
                workspace.record_error(vid, err_sig)

            if consecutive_errors >= 3:
                print(
                    f"\n[!] ABORTING STAGE 1 BATCH: 3 consecutive identical errors ({err_sig}). "
                    "Stopping immediately so no further videos are touched until resolved.",
                    flush=True,
                )
                exit_code = 2
                break
        finally:
            workspace.release_lock(lock_name)

    # Sync newly built scenarios into SQLite immediately if compactor lock is free
    sync_stats = workspace.sync_completed_shards_to_sqlite()
    print(f"\n[Stage 1 Summary] Completed {completed_count} scenario(s). Compactor synced: {sync_stats}", flush=True)

    return exit_code


def run_stage2_worker(args: argparse.Namespace) -> int:
    """Runs inside an isolated Python process so Qwen 2.5 14B Q5_K_M gets full 15 GB T4 VRAM."""
    from core.scenario_extractor import Stage2ScenarioExtractor
    from core.colab_runner import log_extracted_unit_live

    workspace = SharedWorkspace(worker_name=args.worker_id)
    manifest = load_or_create_manifest(workspace)
    candidates = select_stage2_candidates(workspace, manifest)

    should_wait = getattr(args, "wait_for_scenarios", False) or args.mode in ("stage2", "stage2-worker")
    if not candidates and not should_wait:
        print("[Stage 2] No M5 scenarios waiting for LLM extraction.")
        return 0

    budget_label = f"{args.max_session_minutes} min" if args.max_session_minutes > 0 else "unlimited (until disconnect)"
    print("=" * 72)
    print(f"[Stage 2 Local LLM Extraction Worker] Worker: {workspace.worker_id}")
    print(f"  Model: {args.llm_model} (12,288 ctx) | Max Batch: {args.stage2_batch_videos} videos | Watcher: {'ON' if should_wait else 'OFF'}")
    print(f"  M5 Scenarios Ready: {len(candidates)} | Session Budget: {budget_label}")
    print("=" * 72, flush=True)

    try:
        extractor = Stage2ScenarioExtractor(
            workspace=workspace,
            model_name=args.llm_model,
            context_length=args.llm_context_length,
        )
    except Exception as init_exc:
        print(f"[!] Failed to initialize Stage 2 Local LLM ({type(init_exc).__name__}): {init_exc}", flush=True)
        print(traceback.format_exc(limit=5), flush=True)
        return 1

    deadline = (time.time() + (args.max_session_minutes * 60.0)) if args.max_session_minutes > 0 else None
    completed_count = 0
    consecutive_errors = 0
    last_error_sig = ""
    exit_code = 0

    while True:
        candidates = select_stage2_candidates(workspace, manifest)
        if not candidates:
            if not should_wait:
                print("\n[Stage 2] No further M5 scenarios waiting. Stopping cleanly.")
                break
            print("\n" + "=" * 72)
            print("  ⏳ [Profile 2 Standby] All ready scenarios have been extracted!")
            print("     Waiting for Profile 1 (Stage 1 worker) to finish next debate...")
            print("     Listening on Drive (checking every 20s). Press stop to exit.")
            print("=" * 72, flush=True)

            found_new = False
            wait_t0 = time.time()
            while True:
                if deadline is not None and time.time() >= deadline:
                    break
                time.sleep(20)
                elapsed_wait = int(time.time() - wait_t0)
                candidates = select_stage2_candidates(workspace, manifest)
                if candidates:
                    found_new = True
                    print(f"\n[+] Profile 1 completed {len(candidates)} new scenario(s)! Resuming LLM extraction...", flush=True)
                    break
                print(f"     [Standby] Profile 1 is processing audio... ({elapsed_wait}s elapsed)", flush=True)

            if not found_new:
                break

        for cand in candidates:
            if deadline is not None and time.time() >= deadline:
                print(f"\n[Stage 2] Session time budget ({args.max_session_minutes}m) reached. Exiting cleanly.")
                return 0
            if not should_wait and args.stage2_batch_videos > 0 and completed_count >= args.stage2_batch_videos:
                print(f"\n[Stage 2] Reached target batch size ({args.stage2_batch_videos} videos). Stopping cleanly.")
                return 0

            vid = cand["video_id"]
            lock_name = f"{vid}.stage2"
            if not workspace.acquire_lock(lock_name, ttl_sec=2700, verify_delay_sec=0.3 if workspace.is_drive else 0.0):
                continue

            try:
                units = extractor.extract_video_m6(vid, verbose=True)
                if units is not None:
                    completed_count += 1
                    consecutive_errors = 0
                    last_error_sig = ""
                    for u in units:
                        log_extracted_unit_live(u)
                    print(f"    [OK] M6_LLM_EXTRACTED [{vid}]: {len(units)} dialectical Q&A unit(s) saved.", flush=True)
            except Exception as exc:
                err_sig = f"{type(exc).__name__}: {str(exc)[:140]}"
                print(f"    [ERROR] Stage 2 error on {vid} ({type(exc).__name__}): {exc}", flush=True)
                if consecutive_errors == 0:
                    print(traceback.format_exc(limit=5), flush=True)

                if err_sig == last_error_sig:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 1
                    last_error_sig = err_sig

                if not (consecutive_errors >= 2 and isinstance(exc, (ImportError, ModuleNotFoundError, AttributeError, TypeError))):
                    workspace.record_error(vid, err_sig)

                if consecutive_errors >= 3:
                    print(
                        f"\n[!] ABORTING STAGE 2 BATCH: 3 consecutive identical errors ({err_sig}).",
                        flush=True,
                    )
                    exit_code = 2
                    break
            finally:
                workspace.release_lock(lock_name)

            if completed_count > 0 and completed_count % 3 == 0:
                workspace.sync_completed_shards_to_sqlite()

    sync_stats = workspace.sync_completed_shards_to_sqlite()
    print(f"\n[Stage 2 Summary] Extracted {completed_count} video(s). Compactor synced: {sync_stats}", flush=True)
    return exit_code


def print_status_report(workspace: SharedWorkspace) -> None:
    manifest = load_or_create_manifest(workspace)
    tel = collect_workspace_telemetry(workspace, manifest)

    from db.database import init_db, get_stats

    db_path = workspace.data_dir / "dawah_knowledge.db"
    init_db(db_path=db_path)
    db_stats = get_stats(db_path=db_path)

    print("\n" + "=" * 74)
    print("  YT-SPX V2 MULTI-ACCOUNT PIPELINE TELEMETRY (5 TB SHARED WORKSPACE)")
    print("=" * 74)
    print(f"  Workspace Path       : {tel['workspace_root']} (Google Drive={tel['is_google_drive']})")
    print(f"  Active Worker ID     : {tel['worker_id']}")
    print(f"  Total Catalog Videos : {tel['manifest_total']} (Unavailable/Private: {tel['permanently_unavailable']})")
    print(f"  Enrolled Voiceprints : {tel['voiceprint_profiles_enrolled']} recurring Da'ees")
    cb = tel["youtube_circuit_breaker"]
    cb_txt = f"COOLDOWN ({cb['remaining_minutes']}m left: {cb['reason']})" if cb["active"] else "HEALTHY (Anti-Ban Pacing Active)"
    print(f"  YouTube Rate-Limiter : {cb_txt}")
    print("-" * 74)
    print("  Micro-Step Checkpoints on Shared Storage:")
    for step in MICRO_STEPS:
        print(f"    {step:<24} : {tel['step_counts'].get(step, 0):>5}")
    print("-" * 74)
    print(
        f"  Compacted SQLite DB  : {db_stats['total_qa']} Q&A units | "
        f"{db_stats['stage1_completed']} Scenarios | {db_stats['overlaps_resolved']} Overlaps Resolved | "
        f"{db_stats['cleaned_hours']}h Clean Audio"
    )
    if tel["active_locks"]:
        print("-" * 74)
        print("  Active Parallel Worker Locks:")
        for lk in tel["active_locks"]:
            print(f"    - {lk['lock']} held by {lk['worker']} ({lk['ttl_remaining_sec']}s TTL)")
    print("=" * 74 + "\n")


def _resolve_runner_script() -> str:
    """Returns run_colab_all_in_one.py if available so worker subprocesses run verified environments."""
    all_in_one = BASE_DIR / "run_colab_all_in_one.py"
    if all_in_one.exists():
        return str(all_in_one.resolve())
    return str(Path(__file__).resolve())


def _build_stage1_subcmd(args: argparse.Namespace, budget_min: int) -> List[str]:
    cmd = [
        sys.executable,
        _resolve_runner_script(),
        "--mode", "stage1-worker",
        "--batch-videos", str(args.batch_videos),
        "--max-session-minutes", str(budget_min),
        "--whisper-model", args.whisper_model,
        "--whisper-batch-size", str(args.whisper_batch_size),
    ]
    if args.debates_only:
        cmd.append("--debates-only")
    if args.keep_raw_audio:
        cmd.append("--keep-raw-audio")
    if args.offline_only:
        cmd.append("--offline-only")
    if args.worker_id:
        cmd += ["--worker-id", args.worker_id]
    return cmd


def _build_stage2_subcmd(args: argparse.Namespace, budget_min: int) -> List[str]:
    cmd = [
        sys.executable,
        _resolve_runner_script(),
        "--mode", "stage2-worker",
        "--stage2-batch-videos", str(args.stage2_batch_videos),
        "--max-session-minutes", str(budget_min),
        "--llm-model", args.llm_model,
        "--llm-context-length", str(args.llm_context_length),
    ]
    if args.worker_id:
        cmd += ["--worker-id", args.worker_id]
    return cmd


def _get_subprocess_env() -> Dict[str, str]:
    """Prepares isolated subprocess environment with CUDA paths & TF-disabled guards."""
    sub_env = os.environ.copy()
    sub_env["USE_TF"] = "0"
    sub_env["TRANSFORMERS_NO_TF"] = "1"
    sub_env["TRANSFORMERS_NO_FLAX"] = "1"
    sub_env["USE_TORCH"] = "1"
    sub_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    sub_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        from core.audio_diarizer import get_cuda_library_dirs
        lib_dirs = get_cuda_library_dirs()
        curr_ld = sub_env.get("LD_LIBRARY_PATH", "")
        existing = [p for p in curr_ld.split(":") if p and "nvidia/cudnn" not in p and "nvidia_cudnn" not in p]
        new_parts = [d for d in lib_dirs if d not in existing]
        combined = new_parts + existing
        if combined:
            sub_env["LD_LIBRARY_PATH"] = ":".join(combined)
    except Exception:
        pass
    return sub_env


def run_auto_orchestrator(args: argparse.Namespace) -> int:
    """
    Coordinates Stage 1 (Audio/Diarization/Overlaps) and Stage 2 (Qwen 2.5 14B Extraction)
    in isolated GPU child processes so 100% of T4 VRAM is reclaimed between stages.
    - When --max-session-minutes <= 0 (default): loops continuously in smart rounds
      (Stage 1 batch -> Stage 2 batch -> SQLite shard compaction) until Colab disconnects
      or all candidate videos are finished.
    - When --max-session-minutes > 0: splits the explicit session budget (~62% Stage 1 / ~38% Stage 2).
    """
    t_start = time.time()
    total_budget_min = float(args.max_session_minutes)
    workspace = SharedWorkspace(worker_name=args.worker_id or None)
    exit_code = 0

    if total_budget_min <= 0:
        round_num = 0
        while True:
            round_num += 1
            manifest = load_or_create_manifest(workspace)
            s1_before = len(select_stage1_candidates(workspace, manifest, include_shorts=not args.debates_only))
            s2_before = len(select_stage2_candidates(workspace, manifest))
            if s1_before == 0 and s2_before == 0:
                print("\n[Orchestrator] All candidate videos have completed Stage 1 & Stage 2!", flush=True)
                break

            print(
                f"\n[Orchestrator] === Continuous Smart-Batch Round #{round_num} "
                f"(Pending Stage1={s1_before}, Stage2={s2_before}) ===",
                flush=True,
            )

            s1_failed = False
            if s1_before > 0:
                print(
                    f"[Orchestrator] Launching Isolated Stage 1 Subprocess "
                    f"(batch={args.batch_videos}, budget=unlimited)...",
                    flush=True,
                )
                s1_res = subprocess.run(_build_stage1_subcmd(args, 0), env=_get_subprocess_env(), check=False)
                if s1_res.returncode != 0:
                    s1_failed = True
                    exit_code = s1_res.returncode
                    print(f"[Orchestrator] Stage 1 note: exited with code {s1_res.returncode}; processing ready Stage 2 backlog...", flush=True)
                    if s2_before == 0:
                        break

            s2_ready = len(select_stage2_candidates(workspace, manifest))
            if s2_ready > 0:
                print(
                    f"\n[Orchestrator] Launching Isolated Stage 2 LLM Subprocess "
                    f"(batch={args.stage2_batch_videos}, budget=unlimited)...",
                    flush=True,
                )
                s2_res = subprocess.run(_build_stage2_subcmd(args, 0), env=_get_subprocess_env(), check=False)
                if s2_res.returncode != 0:
                    exit_code = s2_res.returncode
                    break

            synced = workspace.sync_completed_shards_to_sqlite()
            print(f"\n[Orchestrator] Round #{round_num} SQLite Compaction: {synced}", flush=True)

            if s1_failed:
                print(f"[Orchestrator] Stopping round loop cleanly due to Stage 1 exit code {s1_res.returncode}.", flush=True)
                break

            s1_after = len(select_stage1_candidates(workspace, manifest, include_shorts=not args.debates_only))
            s2_after = len(select_stage2_candidates(workspace, manifest))
            if s1_after >= s1_before and s2_after >= s2_before:
                print(
                    "[Orchestrator] No unlocked/downloadable candidates progressed in this round "
                    "(remaining items may be locked by another worker or offline-only). Stopping cleanly.",
                    flush=True,
                )
                break
    else:
        s1_budget_min = max(5, int(round(total_budget_min * 0.62)))
        print(f"[Orchestrator] Launching Isolated Stage 1 Subprocess (budget={s1_budget_min}m)...", flush=True)
        s1_res = subprocess.run(_build_stage1_subcmd(args, s1_budget_min), env=_get_subprocess_env(), check=False)
        if s1_res.returncode != 0:
            exit_code = s1_res.returncode
            print(f"[Orchestrator] Stage 1 note: exited with code {s1_res.returncode}; checking Stage 2 backlog...", flush=True)

        manifest = load_or_create_manifest(workspace)
        s2_ready = len(select_stage2_candidates(workspace, manifest))
        elapsed_min = (time.time() - t_start) / 60.0
        rem_min = max(5, int(total_budget_min - elapsed_min))
        if s2_ready > 0 and rem_min > 2:
            print(f"\n[Orchestrator] Launching Isolated Stage 2 LLM Subprocess (pending={s2_ready}, budget={rem_min}m)...", flush=True)
            s2_res = subprocess.run(_build_stage2_subcmd(args, rem_min), env=_get_subprocess_env(), check=False)
            if s2_res.returncode != 0:
                exit_code = s2_res.returncode

    synced = workspace.sync_completed_shards_to_sqlite()
    print(f"\n[Orchestrator] Final SQLite Shard Compaction: {synced}")
    print_status_report(workspace)
    return exit_code


def run_cookie_tests(args: argparse.Namespace) -> int:
    """
    Actively tests EVERY cookies file in the pool with one `yt-dlp --simulate` request each
    (spaced out to stay anti-ban safe) and updates the shared Drive health registry.
    """
    from core.cookie_pool import CookiePool

    workspace = SharedWorkspace(worker_name=args.worker_id or None)
    pool = CookiePool(workspace)
    cookies = pool.discover()
    print("=" * 74)
    print("  YT-SPX COOKIE POOL TEST")
    print("=" * 74)
    if not cookies:
        pool.print_summary(indent="  ")
        return 1

    # Reset existing quarantine timestamps so all discovered cookies get freshly evaluated
    pool.reset_quarantine()

    test_url = "https://www.youtube.com/watch?v=YC86BQwvHyk"
    try:
        manifest = load_or_create_manifest(workspace)
        if manifest:
            for entry in manifest:
                if not entry.get("is_short"):
                    test_url = entry.get("youtube_url") or f"https://www.youtube.com/watch?v={entry['video_id']}"
                    break
    except Exception:
        pass

    n_ok = 0
    for i, ck in enumerate(cookies):
        if i > 0:
            time.sleep(random.uniform(5.0, 10.0))
        ok, kind, msg = pool.test_cookie_online(ck, test_url)
        n_ok += int(ok)
        status = "OK  " if ok else f"FAIL ({kind})"
        print(f"  [{status}] {pool.cookie_key(ck)} -> {msg[:160]}", flush=True)

    print("-" * 74)
    pool.print_summary(indent="  ")
    print("=" * 74)
    return 0 if n_ok > 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YT-SPX V2 Smart-Batch & Multi-Account Pipeline Orchestrator")
    p.add_argument(
        "--mode",
        choices=["auto", "stage1", "stage2", "stage1-worker", "stage2-worker", "sync-db", "status", "test-cookies", "reset-stage2"],
        default="auto",
        help="Execution mode: 'auto' runs Stage 1 -> Stage 2 in isolated GPU subprocesses.",
    )
    p.add_argument("--batch-videos", type=int, default=6, help="Max Stage 1 videos per smart batch (anti-ban safe).")
    p.add_argument("--stage2-batch-videos", type=int, default=8, help="Max Stage 2 videos per LLM batch.")
    p.add_argument(
        "--max-session-minutes",
        type=int,
        default=0,
        help="Graceful stop timer in minutes (default 0 = continuous until disconnect; or e.g. 165 for 2h45m).",
    )
    p.add_argument("--whisper-model", default="large-v3-turbo", help="Faster-Whisper model size.")
    p.add_argument("--whisper-batch-size", type=int, default=24, help="BatchedInferencePipeline batch size.")
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-14B-Instruct", help="Stage 2 LLM model repo.")
    p.add_argument("--llm-context-length", type=int, default=14336, help="Stage 2 GGUF context window.")
    p.add_argument("--debates-only", action="store_true", help="Process full-length debates first, skipping #shorts.")
    p.add_argument("--wait-for-scenarios", action="store_true", help="Stage 2 standby watcher: keeps running and polls Drive for new scenarios from Profile 1.")
    p.add_argument("--keep-raw-audio", action="store_true", help="Keep original raw download alongside 64kbps Opus.")
    p.add_argument("--offline-only", action="store_true", help="Do not touch YouTube; only process M1..M5 files on Drive.")
    p.add_argument("--worker-id", default="", help="Optional friendly worker label (e.g. colab-acct-1).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "status":
        ws = SharedWorkspace(worker_name=args.worker_id or None)
        print_status_report(ws)
        return 0

    if args.mode == "test-cookies":
        return run_cookie_tests(args)

    if args.mode == "reset-stage2":
        from tools.reset_stage2 import reset_stage2_extractions
        ws = SharedWorkspace(worker_name=args.worker_id or None)
        reset_stage2_extractions(ws)
        return 0

    if args.mode == "sync-db":
        ws = SharedWorkspace(worker_name=args.worker_id or None)
        res = ws.sync_completed_shards_to_sqlite(force_reingest=False)
        print(f"[Sync-DB] Merged shards into SQLite: {res}")
        print_status_report(ws)
        return 0

    if args.mode == "stage1":
        total_budget_min = float(args.max_session_minutes)
        t_start = time.time()
        deadline = (t_start + (total_budget_min * 60.0)) if total_budget_min > 0 else None
        round_num = 0
        batch_size = args.batch_videos if args.batch_videos > 0 else 8
        exit_code = 0
        ws = SharedWorkspace(worker_name=args.worker_id or None)

        while True:
            round_num += 1
            if deadline is not None and time.time() >= deadline:
                print(f"\n[Stage 1 Dedicated Worker] Session time budget ({args.max_session_minutes}m) reached. Stopping cleanly.", flush=True)
                break

            manifest = load_or_create_manifest(ws)
            cands = select_stage1_candidates(ws, manifest, include_shorts=not args.debates_only)
            if not cands:
                print("\n[Stage 1 Dedicated Worker] All candidate videos have completed Stage 1 processing! Stopping cleanly.", flush=True)
                break

            budget_txt = f"{round((deadline - time.time()) / 60.0, 1)}m remaining" if deadline is not None else "continuous until disconnect"
            print(
                f"\n[Stage 1 Dedicated Worker] === Continuous Batch Round #{round_num} "
                f"({len(cands)} candidates ready | batch size={batch_size} | {budget_txt}) ===",
                flush=True,
            )

            rem_min = max(1, int((deadline - time.time()) / 60.0)) if deadline is not None else 0
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--mode", "stage1-worker",
                "--batch-videos", str(batch_size),
                "--max-session-minutes", str(rem_min),
                "--whisper-model", args.whisper_model,
                "--whisper-batch-size", str(args.whisper_batch_size),
            ]
            if args.debates_only:
                cmd.append("--debates-only")
            if args.keep_raw_audio:
                cmd.append("--keep-raw-audio")
            if args.offline_only:
                cmd.append("--offline-only")
            if args.worker_id:
                cmd += ["--worker-id", args.worker_id]

            res = subprocess.run(cmd, env=_get_subprocess_env(), check=False)
            if res.returncode != 0:
                exit_code = res.returncode
                print(f"[Stage 1 Dedicated Worker] Round #{round_num} exited with code {res.returncode}.", flush=True)
                if res.returncode == 2:  # Fatal abort (e.g. 3 consecutive identical crashes)
                    break
                time.sleep(10)

            # If user explicitly set max-session-minutes > 0 and deadline is reached, exit cleanly
            if total_budget_min > 0 and deadline is not None and time.time() >= deadline:
                break

            # Check if progress was made or if remaining candidates are locked
            manifest = load_or_create_manifest(ws)
            cands_after = select_stage1_candidates(ws, manifest, include_shorts=not args.debates_only)
            if len(cands_after) >= len(cands):
                active_locks = list(ws.locks_dir.glob("*.stage1.lock.json"))
                if active_locks:
                    print(f"[Stage 1 Standby] Remaining candidates currently locked by other workers. Waiting 30s...", flush=True)
                    time.sleep(30)
                else:
                    print(f"[Stage 1 Standby] No candidates progressed in Round #{round_num}. Waiting 20s before retry...", flush=True)
                    time.sleep(20)

        synced = ws.sync_completed_shards_to_sqlite()
        print(f"\n[Stage 1 Dedicated Worker] Final SQLite Sync: {synced}", flush=True)
        return exit_code

    if args.mode == "stage2":
        total_budget_min = float(args.max_session_minutes)
        t_start = time.time()
        deadline = (t_start + (total_budget_min * 60.0)) if total_budget_min > 0 else None
        exit_code = 0
        ws = SharedWorkspace(worker_name=args.worker_id or None)

        while True:
            if deadline is not None and time.time() >= deadline:
                print(f"\n[Stage 2 Dedicated Worker] Session time budget ({args.max_session_minutes}m) reached. Stopping cleanly.", flush=True)
                break

            rem_min = max(1, int((deadline - time.time()) / 60.0)) if deadline is not None else 0
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--mode", "stage2-worker",
                "--stage2-batch-videos", str(args.stage2_batch_videos),
                "--max-session-minutes", str(rem_min),
                "--llm-model", args.llm_model,
                "--llm-context-length", str(args.llm_context_length),
                "--wait-for-scenarios",
            ]
            if args.worker_id:
                cmd += ["--worker-id", args.worker_id]

            res = subprocess.run(cmd, env=_get_subprocess_env(), check=False)
            if res.returncode != 0:
                exit_code = res.returncode
                print(f"[Stage 2 Dedicated Worker] Process exited with code {res.returncode}.", flush=True)
                if res.returncode == 2:  # Fatal abort
                    break
                time.sleep(15)

            if total_budget_min > 0 and deadline is not None and time.time() >= deadline:
                break

            manifest = load_or_create_manifest(ws)
            cands = select_stage2_candidates(ws, manifest)
            if not cands and total_budget_min > 0:
                break
            time.sleep(5)

        return exit_code

    if args.mode == "stage1-worker":
        return run_stage1_worker(args)

    if args.mode == "stage2-worker":
        return run_stage2_worker(args)

    return run_auto_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
