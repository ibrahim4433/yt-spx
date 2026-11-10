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
from core.yt_audio_fetcher import (
    AntiBanCircuitBreakerError,
    PermanentVideoUnavailableError,
    patch_youtube_oauth2_plugin,
)


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
    spk_profs = workspace.read_json(workspace.profiles_dir / "speaker_profiles.json", default={}) or {}
    enrolled_from_profs = sum(
        1 for v in (spk_profs.values() if isinstance(spk_profs, dict) else [])
        if isinstance(v, dict) and int(v.get("enrolled_samples", 0)) > 0
    )
    profiles_count = max(len(vp_reg.get("profiles", [])), enrolled_from_profs)

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

    from core.scenario_extractor import has_valid_arabic_localization

    candidates = []
    for scen_path in workspace.scenarios_dir.glob("*.json"):
        vid = scen_path.stem
        qa_path = workspace.extractions_dir / f"{vid}.qa.json"
        if qa_path.exists():
            saved = workspace.read_json(qa_path, default={}) or {}
            saved_units = saved.get("units", []) if isinstance(saved, dict) else saved
            if isinstance(saved_units, list) and (
                not saved_units or all(has_valid_arabic_localization(u) for u in saved_units)
            ):
                continue
        st = workspace.get_video_state(vid)
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
    """
    Runs inside an isolated Python process so Stage 1 audio models own the GPU cleanly.
    Executes horizontal phase batching with immediate M5 priority:
      1. Phase C (M5: Whisper large-v3 + CTC Aligner + Punctuation) for videos already at M4
      2. Phase B (M3-M4: NeMo MSDD + TitaNet + MossFormer2) for videos at M2-M3 -> then Phase C
      3. Phase A (M1-M2: yt-dlp + Demucs htdemucs) for new/M1 videos -> then Phase B -> then Phase C
    Each heavy GPU model stack loads only ONCE per phase instead of reloading on every video,
    and per-video locks are acquired/released strictly per phase step so multi-account workers never block.
    """
    import gc
    try:
        import torch
    except ImportError:
        torch = None
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

    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
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

    # Auto-clear any stale circuit breaker lock from previous runs so fresh batch starts unblocked
    workspace.clear_youtube_circuit_breaker()

    # Auto-heal transient errors from previous environment bug iterations
    for cand in candidates:
        vid = cand["video_id"]
        st = workspace.get_video_state(vid)
        last_err = str(st.get("last_error") or "").lower()
        if any(tok in last_err for tok in ("metadata_errors", "msdd_models", "float4_e2m1fn", "neuraldiarizer", "libcublas", "libcudnn", "cudnn", "sublibrary", "fts5", "invalid", "cookie", "circuit breaker", "out of memory", "invalid device", "cuda failed", "input_manifest.json")):
            st["error_count"] = 0
            st["last_error"] = None
            workspace.atomic_write_json(workspace.state_path(vid), st)

    user_interrupted = False

    def _should_stop() -> bool:
        if user_interrupted or exit_code != 0:
            return True
        if args.batch_videos > 0 and completed_count >= args.batch_videos:
            return True
        if deadline is not None and time.time() >= deadline:
            return True
        return False

    def _handle_video_error(vid: str, exc: Exception) -> None:
        nonlocal consecutive_errors, last_error_sig, exit_code, allow_downloads
        if isinstance(exc, AntiBanCircuitBreakerError):
            print(f"\n[!] ANTI-BAN CIRCUIT BREAKER: {exc}", flush=True)
            print("    -> Switching Stage 1 to OFFLINE-ONLY mode (finishing M1..M4 files already on Drive)!", flush=True)
            allow_downloads = False
            return
        if isinstance(exc, PermanentVideoUnavailableError):
            print(f"    [SKIP PERMANENT] {vid}: {exc}", flush=True)
            workspace.advance_micro_step(
                vid,
                "M0_QUEUED",
                extra_updates={"permanent_unavailable": True, "last_error": str(exc)[:300]},
            )
            return

        err_sig = f"{type(exc).__name__}: {str(exc)[:140]}"
        print(f"    [ERROR] Stage 1 error on {vid} ({type(exc).__name__}): {exc}", flush=True)
        if consecutive_errors == 0:
            print(traceback.format_exc(limit=5), flush=True)

        # Evict models from GPU so a memory crash never poisons subsequent videos
        try:
            pipeline.shutdown()
        except Exception:
            pass
        gc.collect()

        if err_sig == last_error_sig:
            consecutive_errors += 1
        else:
            consecutive_errors = 1
            last_error_sig = err_sig

        if not (consecutive_errors >= 2 and isinstance(exc, (ImportError, ModuleNotFoundError, AttributeError, TypeError))):
            workspace.record_error(vid, err_sig)

        if consecutive_errors >= 3:
            print(
                f"\n[!] ABORTING STAGE 1 BATCH: 3 consecutive identical errors ({err_sig}). "
                "Stopping immediately so no further videos are touched until resolved.",
                flush=True,
            )
            exit_code = 2

    def _run_phase_c_batch(batch_items: List[Dict[str, Any]], label: str) -> None:
        nonlocal completed_count, consecutive_errors, last_error_sig, user_interrupted, exit_code
        if not batch_items or _should_stop():
            return
        # Ensure Demucs, NeMo, and MossFormer2 are unloaded so Whisper large-v3 + CTC get full VRAM
        pipeline.demucs.unload()
        pipeline.diarizer.unload()
        pipeline.separator.unload()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        for cand in batch_items:
            if _should_stop():
                break
            vid = cand["video_id"]
            st = workspace.get_video_state(vid)
            if st["step_rank"] >= STEP_RANK["M5_SCENARIO_BUILT"] or st["step_rank"] < STEP_RANK["M4_OVERLAPS_SEPARATED"]:
                continue
            lock_name = f"{vid}.stage1"
            if not workspace.acquire_lock(lock_name, ttl_sec=2700, verify_delay_sec=0.3 if workspace.is_drive else 0.0):
                continue
            try:
                t0 = time.time()
                print(
                    f"\n>>> [Stage 1 - {label} (M5 ASR)] [{completed_count + 1}/{args.batch_videos or '∞'}] {vid} "
                    f"| {cand.get('title_ar', '')[:55]}",
                    flush=True,
                )
                scen = pipeline.run_phase_c_m5(cand, t_start_vid=t0)
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
            except KeyboardInterrupt:
                user_interrupted = True
                exit_code = 130
                print(f"\n[Stage 1] ⏹️ Stop signal received during {vid} — releasing lock and saving progress...", flush=True)
                break
            except Exception as exc:
                _handle_video_error(vid, exc)
            finally:
                workspace.release_lock(lock_name)

        pipeline.asr.unload()

    def _run_phase_b_batch(batch_items: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
        nonlocal consecutive_errors, last_error_sig, user_interrupted, exit_code
        ready_for_c: List[Dict[str, Any]] = []
        if not batch_items or _should_stop():
            return ready_for_c
        pipeline.demucs.unload()
        pipeline.asr.unload()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        for cand in batch_items:
            if _should_stop():
                break
            vid = cand["video_id"]
            st = workspace.get_video_state(vid)
            if st["step_rank"] >= STEP_RANK["M4_OVERLAPS_SEPARATED"]:
                ready_for_c.append(cand)
                continue
            if st["step_rank"] < STEP_RANK["M2_VOCALS_CLEANED"]:
                continue
            lock_name = f"{vid}.stage1"
            if not workspace.acquire_lock(lock_name, ttl_sec=2700, verify_delay_sec=0.3 if workspace.is_drive else 0.0):
                continue
            try:
                print(
                    f"\n>>> [Stage 1 - {label} (M3-M4 Diarization & Overlaps)] {vid} "
                    f"(Resume rank={st['step_rank']}) | {cand.get('title_ar', '')[:55]}",
                    flush=True,
                )
                if pipeline.run_phase_b_m3_m4(cand):
                    consecutive_errors = 0
                    last_error_sig = ""
                    ready_for_c.append(cand)
            except KeyboardInterrupt:
                user_interrupted = True
                exit_code = 130
                print(f"\n[Stage 1] ⏹️ Stop signal received during {vid} — releasing lock and saving progress...", flush=True)
                break
            except Exception as exc:
                _handle_video_error(vid, exc)
            finally:
                workspace.release_lock(lock_name)

        pipeline.diarizer.unload()
        pipeline.separator.unload()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return ready_for_c

    def _run_phase_a_batch(batch_items: List[Dict[str, Any]], target_slots: int) -> List[Dict[str, Any]]:
        nonlocal consecutive_errors, last_error_sig, user_interrupted, exit_code
        ready_for_b: List[Dict[str, Any]] = []
        if not batch_items or _should_stop() or target_slots <= 0:
            return ready_for_b
        pipeline.diarizer.unload()
        pipeline.separator.unload()
        pipeline.asr.unload()

        for cand in batch_items:
            if _should_stop() or len(ready_for_b) >= target_slots:
                break
            vid = cand["video_id"]
            st = workspace.get_video_state(vid)
            if st["step_rank"] >= STEP_RANK["M2_VOCALS_CLEANED"]:
                ready_for_b.append(cand)
                continue
            if not allow_downloads and st["step_rank"] < STEP_RANK["M1_DOWNLOADED"]:
                continue
            lock_name = f"{vid}.stage1"
            if not workspace.acquire_lock(lock_name, ttl_sec=2700, verify_delay_sec=0.3 if workspace.is_drive else 0.0):
                continue
            try:
                print(
                    f"\n>>> [Stage 1 - Phase A (M1-M2 Vocal Isolation)] [{len(ready_for_b) + 1}/{target_slots}] {vid} "
                    f"(Resume rank={st['step_rank']}) | {cand.get('title_ar', '')[:55]}",
                    flush=True,
                )
                if pipeline.run_phase_a_m1_m2(cand, allow_youtube_download=allow_downloads):
                    consecutive_errors = 0
                    last_error_sig = ""
                    ready_for_b.append(cand)
            except KeyboardInterrupt:
                user_interrupted = True
                exit_code = 130
                print(f"\n[Stage 1] ⏹️ Stop signal received during {vid} — releasing lock and saving progress...", flush=True)
                break
            except Exception as exc:
                _handle_video_error(vid, exc)
            finally:
                workspace.release_lock(lock_name)

        pipeline.demucs.unload()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return ready_for_b

    # Partition candidates by current micro-step rank so M4 -> M5 finishes first!
    tier_m4 = [c for c in candidates if c.get("_step_rank", 0) == STEP_RANK["M4_OVERLAPS_SEPARATED"]]
    tier_m2_m3 = [c for c in candidates if c.get("_step_rank", 0) in (STEP_RANK["M2_VOCALS_CLEANED"], STEP_RANK["M3_DIARIZED"])]
    tier_m0_m1 = [c for c in candidates if c.get("_step_rank", 0) in (STEP_RANK["M0_QUEUED"], STEP_RANK["M1_DOWNLOADED"])]

    # Step 1: Immediate Phase C (M5) on any videos already at M4 -> pushes scenarios to Drive immediately!
    if tier_m4 and not _should_stop():
        _run_phase_c_batch(tier_m4, label="Priority Phase C")

    # Step 2: Phase B (M3-M4) on any videos at M2-M3, followed immediately by Phase C (M5) on them in agile waves!
    wave_limit = 4
    if tier_m2_m3 and not _should_stop():
        rem_slots = (args.batch_videos - completed_count) if args.batch_videos > 0 else len(tier_m2_m3)
        b_candidates = tier_m2_m3[: max(0, rem_slots)]
        for w_start in range(0, len(b_candidates), wave_limit):
            if _should_stop():
                break
            b_slice = b_candidates[w_start : w_start + wave_limit]
            b_done = _run_phase_b_batch(b_slice, label="Resume Phase B")
            if b_done and not _should_stop():
                _run_phase_c_batch(b_done, label="Resume Phase C")
                workspace.sync_completed_shards_to_sqlite()

    # Step 3: Horizontal Phase A -> Phase B -> Phase C on remaining M0/M1 candidates in agile 4-video waves!
    models_persisted = False
    if tier_m0_m1 and not _should_stop():
        m0_cursor = 0
        while m0_cursor < len(tier_m0_m1) and not _should_stop():
            rem_slots = (args.batch_videos - completed_count) if args.batch_videos > 0 else wave_limit
            wave_target = min(wave_limit, max(0, rem_slots))
            if wave_target <= 0:
                break
            a_done = _run_phase_a_batch(tier_m0_m1[m0_cursor:], target_slots=wave_target)
            # Advance cursor past examined items so we never re-scan skipped/locked items in the same pass
            m0_cursor += max(len(a_done), wave_target)
            if a_done and not _should_stop():
                b_done = _run_phase_b_batch(a_done, label="Batch Phase B")
                if b_done and not _should_stop():
                    _run_phase_c_batch(b_done, label="Batch Phase C")
                    workspace.sync_completed_shards_to_sqlite()
                    if completed_count > 0 and workspace.is_drive and not models_persisted:
                        try:
                            from run_colab_all_in_one import persist_stage1_models_to_drive
                            models_persisted = persist_stage1_models_to_drive(workspace=workspace)
                        except Exception:
                            pass

    if args.batch_videos > 0 and completed_count >= args.batch_videos:
        print(f"\n[Stage 1] Reached target smart-batch size ({args.batch_videos} videos). Stopping cleanly.")
    elif deadline is not None and time.time() >= deadline:
        print(f"\n[Stage 1] Session time budget ({args.max_session_minutes}m) reached. Saving & exiting cleanly.")

    pipeline.shutdown()

    # Persist Stage 1 model bundle (.tar) to Google Drive at the end of the Stage 1 batch if not cached yet
    if completed_count > 0 and workspace.is_drive and not models_persisted:
        try:
            from run_colab_all_in_one import persist_stage1_models_to_drive
            persist_stage1_models_to_drive(workspace=workspace)
        except Exception as cache_exc:
            print(f"[Stage 1 Cache] Note: {cache_exc}", flush=True)

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

    should_wait = bool(getattr(args, "wait_for_scenarios", False))
    if not candidates and not should_wait:
        print("[Stage 2] No M5 scenarios waiting for LLM extraction.")
        return 0

    budget_label = f"{args.max_session_minutes} min" if args.max_session_minutes > 0 else "unlimited (until disconnect)"
    print("=" * 72)
    print(f"[Stage 2 Local LLM Extraction Worker] Worker: {workspace.worker_id}")
    print(f"  Model: {args.llm_model} ({args.llm_context_length:,} ctx) | Max Batch: {args.stage2_batch_videos} videos | Watcher: {'ON' if should_wait else 'OFF'}")
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
    first_pass = True

    while True:
        if first_pass:
            first_pass = False
        else:
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
            try:
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
            except KeyboardInterrupt:
                print("\n[Stage 2] ⏹️ Stop signal received during standby — exiting cleanly.", flush=True)
                break

            if not found_new:
                break

        batch_target_reached = False
        for cand in candidates:
            if deadline is not None and time.time() >= deadline:
                print(f"\n[Stage 2] Session time budget ({args.max_session_minutes}m) reached. Exiting cleanly.")
                batch_target_reached = True
                break
            if not should_wait and args.stage2_batch_videos > 0 and completed_count >= args.stage2_batch_videos:
                print(f"\n[Stage 2] Reached target batch size ({args.stage2_batch_videos} videos). Stopping cleanly.")
                batch_target_reached = True
                break

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
            except KeyboardInterrupt:
                print(f"\n[Stage 2] ⏹️ Stop signal received during {vid} — releasing lock and saving progress...", flush=True)
                batch_target_reached = True
                break
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

        if batch_target_reached:
            break

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

    # Auto-clear any previous circuit breaker cooldown so new orchestrator session starts unblocked
    workspace.clear_youtube_circuit_breaker()

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
                    print(f"[Orchestrator] Stage 1 note: exited with code {s1_res.returncode}; checking Stage 2 backlog...", flush=True)

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
                    print(f"[Orchestrator] Stage 2 note: exited with code {s2_res.returncode}.", flush=True)

            synced = workspace.sync_completed_shards_to_sqlite()
            print(f"\n[Orchestrator] Round #{round_num} SQLite Compaction: {synced}", flush=True)

            if s1_failed:
                print(
                    f"[Orchestrator] Stage 1 exited with code {s1_res.returncode}. "
                    "Pausing 15s to let resources/locks settle before next round...",
                    flush=True,
                )
                time.sleep(15)

            s1_after = len(select_stage1_candidates(workspace, manifest, include_shorts=not args.debates_only))
            s2_after = len(select_stage2_candidates(workspace, manifest))
            if s1_after >= s1_before and s2_after >= s2_before:
                # Check if YouTube circuit breaker / rate-limit cooldown is active
                allowed, wait_sec, reason = workspace.check_youtube_download_slot()
                active_locks = list(workspace.locks_dir.glob("*.lock.json"))
                if not allowed and wait_sec > 0:
                    rem_m = int(wait_sec // 60)
                    rem_s = int(wait_sec % 60)
                    print(
                        f"\n[Orchestrator Standby] YouTube Anti-Ban Cooldown is active ({rem_m}m {rem_s}s remaining). "
                        f"Reason: {reason[:120]}...\n"
                        f"                      Sleeping 60s before re-checking (continuous pipeline will resume automatically)...",
                        flush=True,
                    )
                    time.sleep(min(wait_sec, 60.0))
                elif active_locks:
                    print(
                        f"[Orchestrator] Candidates currently locked by other workers ({len(active_locks)} active locks). "
                        "Waiting 30s before next round...",
                        flush=True,
                    )
                    time.sleep(30)
                else:
                    print(
                        f"[Orchestrator Standby] Round #{round_num} had no unlocked/offline candidates ready. "
                        "Waiting 30s before retry (continuous loop until disconnect)...",
                        flush=True,
                    )
                    time.sleep(30)
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


def run_auth_youtube(args: argparse.Namespace) -> int:
    """
    Guides the user through 1-time Smart TV OAuth2 pairing for YouTube directly inside Colab.
    Waits up to 15-30 minutes for user approval with live progress heartbeats.
    Saves persistent token into workspace.root / '.cache' / 'yt-dlp' on Google Drive.
    """
    import urllib.request
    import urllib.error
    import uuid
    import datetime

    CLIENT_ID = "861556708454-d6dlm3lh05idd8npek18k6be8ba3oc68.apps.googleusercontent.com"
    CLIENT_SECRET = "SboVhoG9s0rNafixCSGGKXAT"
    SCOPES = "https://gdata.youtube.com https://www.googleapis.com/auth/youtube"

    workspace = SharedWorkspace(worker_name=args.worker_id or None)
    oauth_cache_dir = workspace.root / ".cache" / "yt-dlp"
    oauth_cache_dir.mkdir(parents=True, exist_ok=True)
    token_file = oauth_cache_dir / "youtube-oauth2" / "token_data.json"

    # Make sure plugin is installed & patched against Google RFC 8628 parameter rejection
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp-youtube-oauth2"], check=False)
    patch_youtube_oauth2_plugin()

    print("=" * 74)
    print("  DAWAH DIALECTICS (YT-SPX): YOUTUBE SMART-TV OAUTH2 AUTHENTICATION")
    print("=" * 74)
    print(f"  Shared Token Cache Path : {oauth_cache_dir}")
    print(f"  Existing Token On Drive : {'FOUND (will refresh/overwrite)' if token_file.exists() else 'NOT FOUND (first-time pairing)'}")

    # Request device code directly from Google
    try:
        req_data = json.dumps({
            "client_id": CLIENT_ID,
            "scope": SCOPES,
            "device_id": uuid.uuid4().hex,
            "device_model": "ytlr::",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://www.youtube.com/o/oauth2/device/code",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            code_resp = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"\n[!] Failed to initialize OAuth2 device flow with Google: {exc}")
        return 1

    verification_url = code_resp.get("verification_url", "https://www.google.com/device")
    user_code = code_resp.get("user_code", "")
    device_code = code_resp.get("device_code", "")
    interval = max(5, int(code_resp.get("interval", 5)))
    expires_in = int(code_resp.get("expires_in", 1800))  # 30 minutes!

    print("\n" + "#" * 74)
    print("  >>> STEP 1: Open this link in your browser:")
    print(f"              {verification_url}")
    print("\n  >>> STEP 2: Enter this code:")
    print(f"              >>>  {user_code}  <<<")
    print("\n  >>> STEP 3: Choose your Google/YouTube account and click 'Allow'.")
    print("#" * 74)
    print(f"\n[*] The pipeline is waiting for you to enter the code (timeout: {expires_in // 60} minutes)...")
    print("[*] Waiting for approval: ", end="", flush=True)

    poll_payload = json.dumps({
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }).encode("utf-8")

    poll_url = "https://www.youtube.com/o/oauth2/token"
    start_time = time.time()
    token_data = None

    while time.time() - start_time < expires_in:
        time.sleep(interval)
        poll_req = urllib.request.Request(
            poll_url,
            data=poll_payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            try:
                resp_json = json.loads(err.read().decode("utf-8"))
            except Exception:
                resp_json = {"error": f"HTTP_{err.code}"}
        except Exception as ex:
            resp_json = {"error": str(ex)}

        err = resp_json.get("error")
        if not err and "access_token" in resp_json:
            token_data = resp_json
            break
        elif err == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif err == "slow_down":
            interval += 5
            print("s", end="", flush=True)
            continue
        elif err == "expired_token":
            print("\n\n[!] Device code expired. Please re-run Cell 0.")
            return 1
        elif err == "access_denied":
            print("\n\n[!] Access denied by user.")
            return 1
        else:
            print(f"\n[!] Polling status: {resp_json}")
            time.sleep(interval)

    if not token_data or "access_token" not in token_data:
        print("\n\n[!] Timed out waiting for authorization. Please re-run Cell 0.")
        return 1

    print(" [APPROVED!]", flush=True)

    token_record = {
        "access_token": token_data["access_token"],
        "expires": datetime.datetime.now(datetime.timezone.utc).timestamp() + token_data.get("expires_in", 3600),
        "token_type": token_data.get("token_type", "Bearer"),
        "refresh_token": token_data.get("refresh_token", ""),
    }

    try:
        import yt_dlp
        ydl = yt_dlp.YoutubeDL({"cachedir": str(oauth_cache_dir), "quiet": True})
        ydl.cache.store("youtube-oauth2", "token_data", token_record)
    except Exception:
        dest_dir = oauth_cache_dir / "youtube-oauth2"
        dest_dir.mkdir(parents=True, exist_ok=True)
        token_file.write_text(json.dumps({"yt-dlp_version": "2025.01.15", "data": token_record}), encoding="utf-8")

    # Clear any active cooldown tripped by previous unauthenticated attempts
    workspace.clear_youtube_circuit_breaker()

    print("\n" + "=" * 74)
    print("  [SUCCESS] YouTube Smart-TV OAuth2 token saved to Google Drive!")
    print(f"  Location: {token_file}")
    print("  Shared Drive cooldowns cleared.")
    print("  All future Stage 1 workers will now use this authenticated session!")
    print("=" * 74 + "\n", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YT-SPX V2 Smart-Batch & Multi-Account Pipeline Orchestrator")
    p.add_argument(
        "--mode",
        choices=["auto", "stage1", "stage2", "stage1-worker", "stage2-worker", "sync-db", "status", "test-cookies", "reset-stage1", "reset-stage2", "auth-youtube", "clean-drive"],
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
    p.add_argument("--whisper-model", default="large-v3", help="Faster-Whisper model size.")
    p.add_argument("--whisper-batch-size", type=int, default=16, help="BatchedInferencePipeline batch size (default 16 scales to ~12.2 GB T4 VRAM with automatic step-down).")
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-14B-Instruct", help="Stage 2 LLM model repo.")
    p.add_argument("--llm-context-length", type=int, default=20480, help="Stage 2 GGUF context window (20480 fits in 15 GB T4 VRAM with 8-bit KV cache).")
    p.add_argument("--debates-only", action="store_true", help="Process full-length debates first, skipping #shorts.")
    p.add_argument("--wait-for-scenarios", action="store_true", help="Stage 2 standby watcher: keeps running and polls Drive for new scenarios from Profile 1.")
    p.add_argument("--keep-raw-audio", action="store_true", help="Keep original raw download alongside 64kbps Opus.")
    p.add_argument("--keep-diarization", action="store_true", help="When resetting stage1, keep M3/M4 diarization and only reset M5 Whisper scenarios.")
    p.add_argument("--offline-only", action="store_true", help="Do not touch YouTube; only process M1..M5 files on Drive.")
    p.add_argument("--dry-run", action="store_true", help="Preview leaked files/actions without modifying anything.")
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

    if args.mode == "auth-youtube":
        return run_auth_youtube(args)

    if args.mode == "reset-stage1":
        from tools.reset_stage1 import reset_stage1_pipeline
        ws = SharedWorkspace(worker_name=args.worker_id or None)
        reset_stage1_pipeline(ws, keep_diarization=args.keep_diarization)
        return 0

    if args.mode == "reset-stage2":
        from tools.reset_stage2 import reset_stage2_extractions
        ws = SharedWorkspace(worker_name=args.worker_id or None)
        reset_stage2_extractions(ws)
        return 0

    if args.mode == "clean-drive":
        from tools.clean_drive_root import clean_drive_root
        return clean_drive_root(dry_run=args.dry_run)

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
        wave_size = args.batch_videos if (0 < args.batch_videos <= 4) else 4
        exit_code = 0
        ws = SharedWorkspace(worker_name=args.worker_id or None)

        try:
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
                    f"\n[Stage 1 Dedicated Worker] === In-Process Agile Wave #{round_num} "
                    f"({len(cands)} candidates ready | wave size={wave_size} | {budget_txt}) ===",
                    flush=True,
                )

                rem_min = max(1, int((deadline - time.time()) / 60.0)) if deadline is not None else 0
                wave_args = argparse.Namespace(
                    **{
                        **vars(args),
                        "batch_videos": wave_size,
                        "max_session_minutes": rem_min,
                    }
                )
                rc = run_stage1_worker(wave_args)
                if rc == 130:
                    exit_code = 0
                    break
                if rc != 0:
                    exit_code = rc
                    print(f"[Stage 1 Dedicated Worker] Wave #{round_num} exited with code {rc}.", flush=True)
                    if rc == 2:  # Fatal abort (e.g. 3 consecutive identical crashes)
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
                        print(f"[Stage 1 Standby] No candidates progressed in Wave #{round_num}. Waiting 20s before retry...", flush=True)
                        time.sleep(20)
        except KeyboardInterrupt:
            print("\n[Stage 1 Dedicated Worker] ⏹️ Interrupted by user — syncing completed shards and exiting cleanly.", flush=True)
            exit_code = 0

        synced = ws.sync_completed_shards_to_sqlite()
        print(f"\n[Stage 1 Dedicated Worker] Final SQLite Sync: {synced}", flush=True)
        return exit_code

    try:
        if args.mode == "stage2":
            args.wait_for_scenarios = True
            return run_stage2_worker(args)

        if args.mode == "stage1-worker":
            rc = run_stage1_worker(args)
            return 0 if rc == 130 else rc

        if args.mode == "stage2-worker":
            return run_stage2_worker(args)

        return run_auto_orchestrator(args)
    except KeyboardInterrupt:
        print("\n[YT-SPX] ⏹️ Session stopped by user. Exiting cleanly.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
