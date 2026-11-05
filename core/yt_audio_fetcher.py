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
import sys
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


def patch_youtube_oauth2_plugin() -> bool:
    """
    Patches yt-dlp-youtube-oauth2 plugin in site-packages if installed:
    1. Google's OAuth2 device endpoint strictly requires parameter 'device_code'
       and rejects 'code' with HTTP 400 Bad Request. Upstream plugin used 'code'.
    2. Google's /v1/next and /watch endpoints reject TV OAuth tokens with
       'ERROR - Request contains an invalid argument'. OAuth should only be sent to player APIs.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("yt_dlp_plugins.extractor.youtubeoauth")
        if spec and spec.origin:
            p = Path(spec.origin)
            content = p.read_text(encoding="utf-8")
            modified = False

            # Fix 1: RFC 8628 device_code parameter
            target1 = "'code': code_response['device_code']"
            replacement1 = "'device_code': code_response['device_code']"
            if target1 in content:
                content = content.replace(target1, replacement1)
                modified = True

            # Fix 2: Do not attach TV OAuth token to non-player endpoints like /v1/next or /watch
            target2 = "if not urllib.parse.urlparse(request.url).netloc.endswith('youtube.com'):\n            return"
            replacement2 = (
                "_p_url = urllib.parse.urlparse(request.url)\n"
                "        if not _p_url.netloc.endswith('youtube.com') or '/v1/next' in _p_url.path or '/watch' in _p_url.path:\n"
                "            return"
            )
            if target2 in content:
                content = content.replace(target2, replacement2)
                modified = True

            if modified:
                p.write_text(content, encoding="utf-8")
                return True
    except Exception:
        pass
    return False


# Auto-patch plugin on import if present
patch_youtube_oauth2_plugin()


def _find_cookies_source(workspace: SharedWorkspace) -> Optional[Path]:
    """Returns the best currently-healthy cookies file from the rotation pool (see core/cookie_pool.py)."""
    healthy = CookiePool(workspace).healthy_cookies()
    return healthy[0] if healthy else None


def find_existing_raw_audio(workspace: SharedWorkspace, video_id: str) -> Optional[Path]:
    """Returns existing complete raw audio file on Drive if already downloaded."""
    valid_exts = {".mp4", ".m4a", ".opus", ".webm", ".mp3", ".ogg", ".wav", ".aac", ".flac", ".m4v", ".mkv"}
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
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--format", "234/233/140/18/ba/b/bestaudio/best",
            "--write-info-json",
            "--no-write-subs",
            "--no-progress",
            "--retries", "4",
            "--fragment-retries", "4",
            "--sleep-requests", "1",
            "-o", out_tmpl,
        ]

        if shutil.which("aria2c"):
            base_cmd.extend(["--downloader", "aria2c", "--downloader-args", "aria2c:-x 4 -s 4 -k 1M"])

        # Detect available JS runtimes (Deno & Node in Colab)
        for d in [Path.home() / ".deno" / "bin", Path("/root/.deno/bin")]:
            if d.exists() and str(d) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{d}:{os.environ.get('PATH', '')}"

        runtimes = []
        if shutil.which("deno"):
            runtimes.append("deno")
        if shutil.which("node"):
            runtimes.append("node")
        if runtimes:
            base_cmd.extend(["--js-runtimes", ",".join(runtimes)])

        # Multi-attempt strategy:
        # 1. Anonymous download FIRST by default!
        #    yt-dlp uses visionos/web HLS formats (234/233) which don't require PO tokens or auth.
        # 2. Healthy rotating cookie pool for restricted/age-gated videos.
        pool = CookiePool(workspace)
        healthy = pool.healthy_cookies()[:max_cookie_attempts]

        attempts: List[Any] = [None]
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
            cookie_label = "anonymous (visionos/hls)"
            cmd.extend([
                "--extractor-args", "youtube:player_skip=initial_data",
            ])
            if cookie is not None:
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
                if cookie is not None:
                    if kind == "invalid":
                        # Downloaded anonymously, but this cookie is dead -> quarantine so next video uses another
                        pool.mark_failed(cookie, "invalid", last_tail)
                        print(f"    [cookies] {cookie_label} is no longer valid (rotated/logged out) -> quarantined.", flush=True)
                    else:
                        pool.mark_ok(cookie)
                success = True
                break

            for pat in UNAVAILABLE_VIDEO_PATTERNS:
                if re.search(pat, output_lower):
                    raise PermanentVideoUnavailableError(f"Video {vid} is permanently unavailable on YouTube.")

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

            last_kind = kind or f"code_{proc.returncode}"
            break

        if not success:
            # PytubeFix fallback (WEB client uses Node.js PoToken generation)
            try:
                print(f"    [auth] Trying PytubeFix fallback for {vid}...", flush=True)
                from pytubefix import YouTube
                yt = YouTube(url, client="WEB")
                st = yt.streams.get_audio_only()
                if not st:
                    st = yt.streams.filter(progressive=True).first()
                if not st:
                    st = yt.streams.filter(only_audio=True).first()
                if st:
                    dl_name = f"{vid}.mp4"
                    st.download(output_path=str(tmp_dir), filename=dl_name)
                    dl_path = tmp_dir / dl_name
                    if dl_path.exists() and dl_path.stat().st_size > 5_000:
                        success = True
                        print(f"    [auth] PytubeFix fallback download succeeded ({dl_path.stat().st_size // 1024} KB).", flush=True)
            except Exception as pt_exc:
                print(f"    [auth] PytubeFix fallback note: {pt_exc}", flush=True)

        if not success:
            if last_kind == "age_gate":
                raise RuntimeError(f"Video {vid} is age-restricted and no available cookie could access it.")
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
            raise RuntimeError(f"Download pipeline completed for {vid} without producing a valid audio stream.")

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
        else:
            compact_info = {
                "id": vid,
                "title": video_record.get("title_ar", ""),
                "duration": float(video_record.get("duration") or 0.0),
                "upload_date": video_record.get("upload_date", ""),
                "channel": video_record.get("channel", "SCDawah Arabic"),
                "chapters": video_record.get("chapters") or [],
                "acodec": "aac",
                "abr": 128,
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
