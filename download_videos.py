#!/usr/bin/env python3
"""
download_videos.py
V2 Anti-Ban Audio Staging CLI for Dawah Dialectics (yt-spx).

Stages M1_DOWNLOADED best-audio streams (Opus/M4A + .info.json) from the canonical
2,640-video manifest (all_channels_videos.csv + clean_urls.txt) into the shared
workspace (audio_raw/), enforcing global anti-ban pacing, isolated cookies.txt copies,
and automatic circuit-breaker protection.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.drive_sync import SharedWorkspace, STEP_RANK
from core.pipeline_orchestrator import load_or_create_manifest, select_stage1_candidates
from core.yt_audio_fetcher import (
    fetch_video_audio_m1,
    AntiBanCircuitBreakerError,
    PermanentVideoUnavailableError,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage M1 YouTube audio streams in safe anti-ban batches.")
    parser.add_argument("--batch-videos", type=int, default=8, help="Number of videos to stage in this batch.")
    parser.add_argument("--cookies", type=str, default="", help="Optional path to Netscape cookies.txt.")
    parser.add_argument("--debates-only", action="store_true", help="Stage full-length debates before #shorts.")
    parser.add_argument("--workspace-dir", type=str, default="", help="Optional custom workspace root path.")
    args = parser.parse_args(argv)

    ws_root = Path(args.workspace_dir) if args.workspace_dir else None
    cookies_override = Path(args.cookies) if args.cookies else None

    workspace = SharedWorkspace(root_dir=ws_root)
    manifest = load_or_create_manifest(workspace)
    candidates = [
        c
        for c in select_stage1_candidates(workspace, manifest, include_shorts=not args.debates_only)
        if c.get("_step_rank", 0) < STEP_RANK["M1_DOWNLOADED"]
    ]

    print("=" * 70)
    print(f"  YT-SPX V2 AUDIO STAGING (M0 -> M1_DOWNLOADED)")
    print(f"  Workspace : {workspace.root}")
    print(f"  Queued    : {len(candidates)} un-staged videos | Batch Limit: {args.batch_videos}")
    print("=" * 70)

    staged = 0
    for cand in candidates:
        if staged >= args.batch_videos:
            break
        vid = cand["video_id"]
        lock_name = f"{vid}.stage1"
        if not workspace.acquire_lock(lock_name, ttl_sec=1800, verify_delay_sec=0.25 if workspace.is_drive else 0.0):
            continue
        try:
            print(f"\n[{staged + 1}/{args.batch_videos}] Staging [{vid}] {cand.get('title_ar', '')[:55]}...")
            audio_path, _ = fetch_video_audio_m1(cand, workspace, cookies_override=cookies_override)
            staged += 1
            print(f"   -> Saved raw audio: {audio_path.name}")
        except AntiBanCircuitBreakerError as cb:
            print(f"\n[!] Anti-Ban Circuit Breaker tripped: {cb}")
            print("    Stopping download batch early to protect cookies and IP reputation.")
            break
        except PermanentVideoUnavailableError as perm:
            print(f"   -> Permanently unavailable ({perm}); marking skipped.")
            workspace.advance_micro_step(
                vid,
                "M0_QUEUED",
                extra_updates={"permanent_unavailable": True, "last_error": str(perm)[:300]},
            )
        except Exception as exc:
            print(f"   -> Error staging {vid}: {exc}")
            workspace.record_error(vid, str(exc))
        finally:
            workspace.release_lock(lock_name)

    print(f"\n[+] Batch finished: {staged} video(s) staged at M1_DOWNLOADED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
