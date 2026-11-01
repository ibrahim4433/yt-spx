#!/usr/bin/env python3
"""
core/yt_audio_fetcher.py
Micro-Step M1: Safe, Anti-Ban YouTube Audio Fetcher for Google Colab & Shared 5 TB Drive.

Key Safeguards:
  1. Never re-downloads if M1_DOWNLOADED (or M2_VOCALS_CLEANED+) already exists on Drive.
  2. Coordinates global download pacing across ALL shared Colab accounts via Drive rate-limiter.
  3. Copies cookies.txt to an isolated local /tmp jar so yt-dlp never corrupts shared Drive cookies.
  4. Uses Deno/Node JS runtimes + BgUtils PO-Token plugin (when installed) + safe audio stream selection.
  5. Trips a shared anti-ban circuit breaker immediately on HTTP 429 / 'Sign in to confirm you're not a bot'
     so cookies stay safe while GPU workers continue offline on M1..M5 queued items.
"""

import os
import re
import shutil
import random
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from core.drive_sync import SharedWorkspace, STEP_RANK
from core.cookie_pool import CookiePool, classify_ytdlp_output

BASE_DIR = Path(__file__).resolve().parent.parent

BOT_CHALLENGE_PATTERNS = [
    r"sign in to confirm you.?re not a bot",
    r"http error 429",
    r"too many requests",
    r"confirm your age",
    r"this content isn.?t available, try again later",
]

UNAVAILABLE_VIDEO_PATTERNS = [
    r"video unavailable",
    r"private video",
    r"this video has been removed",
    r"account associated with this video has been terminated",
]


class AntiBanCircuitBreakerError(RuntimeError):
    """Raised when YouTube challenges or rate-limits downloads so the batch pauses fetching safely."""


class PermanentVideoUnavailableError(RuntimeError):
    """Raised when a YouTube video was deleted or made private."""


def _find_cookies_source(workspace: SharedWorkspace) -> Optional[Path]:
    """Returns the best currently-healthy cookies file from the rotation pool (see core/cookie_pool.py)."""
    healthy = CookiePool(workspace).healthy_cookies()
    return healthy[0] if healthy else None


def find_existing_raw_audio(workspace: SharedWorkspace, video_id: str) -> Optional[Path]:
    """Returns existing complete raw audio file on Drive if already downloaded."""
    valid_exts = {".m4a", ".opus", ".webm", ".mp3", ".ogg", ".wav", ".aac", ".flac"}
    for p in workspace.audio_raw_dir.glob(f"{video_id}.*"):
        if p.suffix.lower() in valid_exts and p.stat().st_size > 10_000:
            return p
    return None


def fetch_video_audio_m1(
    video_record: Dict[str, Any],
    workspace: SharedWorkspace,
    min_global_gap_sec: float = 15.0,
    max_wait_for_slot_sec: float = 90.0,
    max_cookie_attempts: int = 4,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Executes Micro-Step M1 (YouTube Audio Fetch) with full resume & anti-ban protection.
    If the video already reached M2_VOCALS_CLEANED and clean Opus exists, returns (None, info)
    signaling downstream steps can skip straight to M2/M3!
    """
    vid = video_record["video_id"]
    url = video_record.get("youtube_url") or f"https://www.youtube.com/watch?v={vid}"
    state = workspace.get_video_state(vid)

    # Fast-skip 1: Already past M2 and clean Opus exists on Drive -> no download ever needed!
    clean_opus = workspace.audio_clean_dir / f"{vid}.opus"
    if state["step_rank"] >= STEP_RANK["M2_VOCALS_CLEANED"] and clean_opus.exists():
        info_cached = workspace.read_json(workspace.audio_raw_dir / f"{vid}.info.json", default={}) or {}
        return None, info_cached

    # Fast-skip 2: Raw audio already downloaded on Drive at M1 -> instant zero-network resume!
    existing_raw = find_existing_raw_audio(workspace, vid)
    if existing_raw is not None:
        info_cached = workspace.read_json(workspace.audio_raw_dir / f"{vid}.info.json", default={}) or {}
        if state["step_rank"] < STEP_RANK["M1_DOWNLOADED"]:
            workspace.advance_micro_step(
                vid,
                "M1_DOWNLOADED",
                artifacts={"raw_audio": str(existing_raw)},
                metrics={"duration_sec": float(info_cached.get("duration") or 0.0)},
            )
        return existing_raw, info_cached

    # Coordinate global download rate-limit across ALL shared Colab accounts
    waited = 0.0
    while True:
        allowed, wait_sec, reason = workspace.check_youtube_download_slot(min_global_gap_sec=min_global_gap_sec)
        if allowed:
            break
        if "Anti-ban cooldown" in reason or waited + wait_sec > max_wait_for_slot_sec:
            raise AntiBanCircuitBreakerError(f"YouTube anti-ban cooldown active ({reason}; {wait_sec:.0f}s remaining)")
        time.sleep(min(wait_sec, 15.0))
        waited += min(wait_sec, 15.0)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"yt_spx_dl_{vid}_"))
    try:
        out_tmpl = str(tmp_dir / f"{vid}.%(ext)s")
        base_cmd = [
            "yt-dlp",
            "--no-playlist",
            "--format", "bestaudio[ext=m4a][abr<=160]/bestaudio[ext=opus][abr<=160]/bestaudio/best",
            "--write-info-json",
            "--no-write-subs",
            "--no-progress",
            "--retries", "4",
            "--fragment-retries", "4",
            "--sleep-requests", "1",
            "-o", out_tmpl,
        ]

        # Detect available JS runtimes (Deno preferred in Colab for PO token / n-challenge)
        runtimes = []
        if shutil.which("deno"):
            runtimes.append("deno")
        if shutil.which("node"):
            runtimes.append("node")
        if runtimes:
            base_cmd.extend(["--js-runtimes", ",".join(runtimes)])

        # Multi-cookie rotation: try each healthy cookies file (best first); on a bot-check / 429 /
        # dead-cookie response, quarantine that cookie in the shared Drive registry and retry the
        # SAME video with the next one. Only when every cookie fails is the shared circuit breaker tripped.
        pool = CookiePool(workspace)
        discovered = pool.discover()
        attempts: List[Optional[Path]] = list(pool.healthy_cookies()[:max_cookie_attempts])
        if not attempts:
            if discovered:
                print(
                    f"    [cookies] All {len(discovered)} cookie file(s) are expired/quarantined "
                    "-> trying one anonymous attempt.",
                    flush=True,
                )
            attempts = [None]

        proc = None
        last_kind = ""
        last_tail = ""
        success = False
        for idx, cookie in enumerate(attempts):
            if idx > 0:
                time.sleep(random.uniform(6.0, 14.0))
            # Clean partial files from any previous failed attempt
            for f in tmp_dir.iterdir():
                if f.is_file():
                    f.unlink(missing_ok=True)

            cmd = list(base_cmd)
            cookie_label = "anonymous"
            if cookie is not None:
                # Isolated working copy so yt-dlp never clobbers shared Drive cookies
                cookies_tmp = tmp_dir / "cookies_working.txt"
                shutil.copy2(cookie, cookies_tmp)
                cmd.extend(["--cookies", str(cookies_tmp)])
                cookie_label = pool.cookie_key(cookie)
            cmd.append(url)

            if idx > 0 or len(attempts) > 1:
                print(f"    [cookies] Attempt {idx + 1}/{len(attempts)} with {cookie_label}", flush=True)

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            output = proc.stdout or ""
            output_lower = output.lower()
            kind = classify_ytdlp_output(output_lower)
            last_tail = output[-240:].strip()

            if proc.returncode == 0:
                if cookie is not None:
                    if kind == "invalid":
                        # Downloaded anonymously, but this cookie is dead -> quarantine so next video uses another
                        pool.mark_failed(cookie, "invalid", last_tail)
                        print(f"    [cookies] {cookie_label} is no longer valid (rotated/logged out) -> quarantined.", flush=True)
                    else:
                        pool.mark_ok(cookie)
                success = True
                break

            if kind in ("invalid", "challenged", "rate_limited", "age_gate"):
                last_kind = kind
                if cookie is not None and kind != "age_gate":
                    pool.mark_failed(cookie, kind, last_tail)
                    print(f"    [cookies] {cookie_label} failed ({kind}) -> quarantined, rotating to next cookie.", flush=True)
                continue

            for pat in UNAVAILABLE_VIDEO_PATTERNS:
                if re.search(pat, output_lower):
                    raise PermanentVideoUnavailableError(f"Video {vid} is permanently unavailable on YouTube.")

            raise RuntimeError(f"yt-dlp exited with code {proc.returncode}: {output[-300:].strip()}")

        if not success:
            if last_kind == "age_gate":
                raise RuntimeError(f"Video {vid} is age-restricted and no available cookie could access it.")
            workspace.trip_youtube_circuit_breaker(
                reason=f"All {len(attempts)} cookie attempt(s) failed ({last_kind}) on {vid}: {last_tail}",
                cooldown_minutes=45,
            )
            raise AntiBanCircuitBreakerError(
                f"YouTube {last_kind or 'bot-check'} on {vid} with all {len(attempts)} cookie option(s). "
                "Tripped 45-min shared Drive cooldown."
            )

        # Locate downloaded media file and .info.json
        downloaded_audio: Optional[Path] = None
        info_json_tmp: Optional[Path] = None
        for f in tmp_dir.iterdir():
            if f.name == "cookies_working.txt":
                continue
            if f.name.endswith(".info.json"):
                info_json_tmp = f
            elif f.is_file() and not f.name.endswith((".part", ".ytdl")):
                downloaded_audio = f

        if downloaded_audio is None or downloaded_audio.stat().st_size < 5_000:
            raise RuntimeError(f"yt-dlp completed for {vid} without producing a valid audio stream.")

        # Atomically move raw audio + compact metadata to shared 5 TB Drive (Micro-Step M1!)
        workspace.audio_raw_dir.mkdir(parents=True, exist_ok=True)
        dest_audio = workspace.audio_raw_dir / downloaded_audio.name
        tmp_dest = workspace.audio_raw_dir / f".{downloaded_audio.name}.tmp"
        shutil.copy2(downloaded_audio, tmp_dest)
        os.replace(tmp_dest, dest_audio)

        compact_info: Dict[str, Any] = {}
        if info_json_tmp and info_json_tmp.exists():
            raw_info = workspace.read_json(info_json_tmp, default={}) or {}
            compact_info = {
                "id": vid,
                "title": raw_info.get("title", video_record.get("title_ar", "")),
                "duration": float(raw_info.get("duration") or 0.0),
                "upload_date": raw_info.get("upload_date", ""),
                "channel": raw_info.get("channel") or video_record.get("channel", "SCDawah Arabic"),
                "chapters": raw_info.get("chapters") or [],
                "acodec": raw_info.get("acodec", ""),
                "abr": raw_info.get("abr", 0),
            }
            workspace.atomic_write_json(workspace.audio_raw_dir / f"{vid}.info.json", compact_info)

        workspace.advance_micro_step(
            vid,
            "M1_DOWNLOADED",
            artifacts={
                "raw_audio": str(dest_audio),
                "info_json": str(workspace.audio_raw_dir / f"{vid}.info.json"),
            },
            metrics={
                "duration_sec": float(compact_info.get("duration") or 0.0),
                "raw_size_mb": round(dest_audio.stat().st_size / (1024 * 1024), 2),
            },
        )

        # Gentle post-download jitter
        time.sleep(random.uniform(1.5, 4.0))
        return dest_audio, compact_info

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
