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
from core.cookie_pool import CookiePool, classify_ytdlp_output, sanitize_cookie_file

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
    info_cached = workspace.read_json(workspace.audio_raw_dir / f"{video_id}.info.json", default={}) or {}
    expected_dur = float(info_cached.get("duration") or 0.0)

    for p in workspace.audio_raw_dir.glob(f"{video_id}.*"):
        if p.suffix.lower() in valid_exts:
            size = p.stat().st_size
            if size < 50_000:
                continue
            # If duration is known and significant (>= 30s), ensure file has at least 2KB/sec of expected data
            if expected_dur >= 30.0 and size < expected_dur * 2000:
                continue
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
    if clean_opus.exists() and clean_opus.stat().st_size <= 1024:
        clean_opus.unlink(missing_ok=True)
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
            "--format", "ba/b/bestaudio[ext=m4a]/bestaudio[ext=opus]/bestaudio/best",
            "--extractor-args", "youtube:player_client=android,web",
            "--write-info-json",
            "--no-write-subs",
            "--no-progress",
            "--retries", "4",
            "--fragment-retries", "4",
            "--sleep-requests", "1",
            "-o", out_tmpl,
        ]

        # Detect available JS runtimes (Deno & Node in Colab)
        runtimes = []
        if shutil.which("deno"):
            runtimes.append("deno")
        if shutil.which("node"):
            runtimes.append("node")
        if runtimes:
            base_cmd.extend(["--js-runtimes", ",".join(runtimes)])

        # Multi-attempt strategy:
        # 1. Anonymous download FIRST by default!
        #    YouTube serves public debate videos cleanly without authentication via the android/web client.
        # 2. If anonymous encounters a restriction (bot-check / rate-limit / age-gate),
        #    only then fallback to the healthy cookie pool.
        pool = CookiePool(workspace)
        healthy = pool.healthy_cookies()[:max_cookie_attempts]

        oauth_cache_dir = workspace.root / ".cache" / "yt-dlp"
        oauth_token_file = oauth_cache_dir / "youtube-oauth2" / "token_data.json"
        has_oauth = oauth_token_file.exists() and oauth_token_file.stat().st_size > 20

        # Multi-attempt strategy:
        # 1. Anonymous download FIRST by default!
        #    YouTube serves public debate videos cleanly without authentication via the android/web client.
        # 2. If anonymous encounters a restriction (bot-check / rate-limit / age-gate):
        #    a) If Smart-TV OAuth2 token exists on Drive, use OAuth2!
        #    b) Otherwise, fallback to the healthy cookie pool.
        attempts: List[Any] = [None]
        if has_oauth:
            attempts.append("__oauth2__")
        attempts.extend(list(healthy))

        proc = None
        last_kind = ""
        last_tail = ""
        success = False
        for idx, cookie in enumerate(attempts):
            if idx > 0:
                time.sleep(random.uniform(4.0, 8.0))
            # Clean partial files from any previous failed attempt
            for f in tmp_dir.iterdir():
                if f.is_file():
                    f.unlink(missing_ok=True)

            cmd = list(base_cmd)
            cookie_label = "anonymous"
            if cookie == "__oauth2__":
                # Strip android client extractor args because OAuth2 operates on TV client
                cmd = [c for c in base_cmd if "youtube:player_client" not in c]
                cmd.extend([
                    "--username", "oauth2",
                    "--password", "",
                    "--cache-dir", str(oauth_cache_dir),
                ])
                cookie_label = "Smart-TV OAuth2 (Colab token)"
            elif cookie is not None:
                # Isolated working copy with Netscape header & tab repair so yt-dlp never fails syntax
                cookies_tmp = tmp_dir / "cookies_working.txt"
                sanitize_cookie_file(cookie, cookies_tmp)
                cmd.extend(["--cookies", str(cookies_tmp)])
                cookie_label = pool.cookie_key(cookie)
            cmd.append(url)

            if idx > 0:
                print(f"    [auth] Fallback attempt {idx + 1}/{len(attempts)} with {cookie_label}", flush=True)

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
                if cookie is not None and cookie != "__oauth2__":
                    if kind == "invalid":
                        # Downloaded anonymously, but this cookie is dead -> quarantine so next video uses another
                        pool.mark_failed(cookie, "invalid", last_tail)
                        print(f"    [cookies] {cookie_label} is no longer valid (rotated/logged out) -> quarantined.", flush=True)
                    else:
                        pool.mark_ok(cookie)
                elif cookie == "__oauth2__":
                    print("    [auth] Smart-TV OAuth2 download succeeded cleanly.", flush=True)
                success = True
                break

            for pat in UNAVAILABLE_VIDEO_PATTERNS:
                if re.search(pat, output_lower):
                    raise PermanentVideoUnavailableError(f"Video {vid} is permanently unavailable on YouTube.")

            # If this attempt failed while using OAuth2, rotate to cookie pool
            if cookie == "__oauth2__":
                fail_kind = kind or "oauth_restricted"
                last_kind = fail_kind
                print(f"    [auth] Smart-TV OAuth2 attempt failed ({fail_kind}) -> rotating to next attempt.", flush=True)
                continue

            # If this attempt failed while using a cookie, quarantine/record it and ROTATE!
            if cookie is not None:
                fail_kind = kind or "format_or_access_restricted"
                last_kind = fail_kind
                if fail_kind != "age_gate":
                    pool.mark_failed(cookie, fail_kind, last_tail)
                    print(f"    [cookies] {cookie_label} failed ({fail_kind}) -> quarantined, rotating to next attempt.", flush=True)
                continue

            # If anonymous attempt failed (idx == 0) and fallbacks are available, log fallback
            if idx == 0 and len(attempts) > 1:
                print(f"    [auth] Anonymous download encountered restriction ({kind or 'code ' + str(proc.returncode)}) -> trying authenticated fallback pool...", flush=True)
                continue

            # If any attempt failed but there are remaining fallback attempts
            if idx + 1 < len(attempts):
                continue

            if kind in ("invalid", "challenged", "rate_limited", "age_gate"):
                last_kind = kind
                break

            raise RuntimeError(f"yt-dlp exited with code {proc.returncode}: {output[-300:].strip()}")

        if not success:
            if last_kind == "age_gate":
                raise RuntimeError(f"Video {vid} is age-restricted and no available cookie could access it.")
            if last_kind in ("challenged", "rate_limited"):
                workspace.trip_youtube_circuit_breaker(
                    reason=f"YouTube challenge ({last_kind}) on {vid} with all {len(attempts)} option(s): {last_tail}",
                    cooldown_minutes=45,
                )
                raise AntiBanCircuitBreakerError(
                    f"YouTube {last_kind} on {vid} with all {len(attempts)} option(s). "
                    "Tripped 45-min shared Drive cooldown."
                )
            raise RuntimeError(
                f"Failed to fetch {vid} with all {len(attempts)} option(s) (last status: {last_kind or 'unknown'}): {last_tail}"
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
