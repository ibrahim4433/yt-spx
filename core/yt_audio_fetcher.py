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


class YouTubeBotChallengeError(RuntimeError):
    """Raised when fetch attempts fail due to YouTube bot-check, HTTP 429, or HTTP 403 challenge."""

    def __init__(self, message: str, status_kind: str = "challenged") -> None:
        super().__init__(message)
        self.status_kind = status_kind


class PermanentVideoUnavailableError(RuntimeError):
    """Raised when a YouTube video was deleted or made private."""


def _extract_concise_ytdlp_error(output: str, max_len: int = 240) -> str:
    """Returns the last ERROR line from yt-dlp output, or a trimmed tail fallback."""
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if "ERROR:" in stripped:
            return stripped[-max_len:]
    return output[-max_len:].strip()


def _attempt_pytubefix_download(url: str, vid: str, tmp_dir: Path, expected_dur: float = 0.0) -> bool:
    """Best-effort PytubeFix fallback trying ANDROID_VR (PO-token-free) then WEB PoToken client."""
    try:
        from pytubefix import YouTube
    except Exception as exc:
        print(f"    [auth] PytubeFix unavailable: {exc}", flush=True)
        return False

    min_bytes = int(expected_dur * 1500) if expected_dur >= 300.0 else 5_000
    dl_name = f"{vid}.mp4"
    for client_name, kwargs in (
        ("ANDROID_VR", {}),
        ("WEB", {"use_po_token": True}),
    ):
        dl_path = tmp_dir / dl_name
        dl_path.unlink(missing_ok=True)
        try:
            print(f"    [auth] Trying PytubeFix fallback ({client_name}) for {vid}...", flush=True)
            yt = YouTube(url, client=client_name, **kwargs)
            st = (
                yt.streams.get_audio_only()
                or yt.streams.filter(progressive=True).first()
                or yt.streams.filter(only_audio=True).first()
            )
            if not st:
                continue
            st.download(output_path=str(tmp_dir), filename=dl_name)
            if dl_path.exists() and dl_path.stat().st_size >= min_bytes:
                print(
                    f"    [auth] PytubeFix ({client_name}) succeeded ({dl_path.stat().st_size // 1024} KB).",
                    flush=True,
                )
                return True
        except Exception as pt_exc:
            print(f"    [auth] PytubeFix ({client_name}) note: {pt_exc}", flush=True)
    return False


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
    expected_dur = float(video_record.get("duration") or 0.0)

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
        base_cmd.extend(["--remote-components", "ejs:github"])

        # Multi-attempt strategy:
        # 1. Anonymous download FIRST (visionos/HLS avoids PO tokens when unflagged).
        #    If anonymous fails due to transient network/format issues, immediately try android_vr anonymously
        #    before touching authenticated cookies.
        # 2. Healthy rotating cookie pool for bot-challenged / restricted / age-gated videos.
        # 3. Anonymous android_vr retry if not yet tried, followed by PytubeFix fallback.
        pool = CookiePool(workspace)
        healthy = pool.healthy_cookies()[:max_cookie_attempts]

        plan: List[Tuple[Optional[Path], Optional[str], str]] = [
            (None, None, "anonymous (visionos/hls)"),
        ]
        for c_path in healthy:
            plan.append((c_path, None, pool.cookie_key(c_path)))

        last_kind = ""
        last_tail = ""
        saw_bot_challenge = False
        tried_android_vr = False
        success = False
        attempts_run = 0
        quarantine_kinds = {"invalid", "challenged", "rate_limited", "forbidden_403"}

        while plan:
            cookie, player_client, cookie_label = plan.pop(0)
            if attempts_run > 0:
                time.sleep(random.uniform(4.0, 8.0))
            attempts_run += 1

            for f in tmp_dir.iterdir():
                if f.is_file():
                    f.unlink(missing_ok=True)

            cmd = list(base_cmd)
            ext_arg = "youtube:player_skip=initial_data"
            if player_client:
                ext_arg = f"youtube:player_client={player_client};player_skip=initial_data"
                tried_android_vr = tried_android_vr or (player_client == "android_vr")
            cmd.extend(["--extractor-args", ext_arg])

            if cookie is not None:
                cookies_tmp = tmp_dir / "cookies_working.txt"
                sanitize_cookie_file(cookie, cookies_tmp)
                cmd.extend(["--cookies", str(cookies_tmp)])

            if attempts_run > 1:
                total_planned = attempts_run + len(plan)
                print(f"    [auth] Fallback attempt {attempts_run}/{total_planned} with {cookie_label}", flush=True)

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
            last_tail = _extract_concise_ytdlp_error(output)

            if proc.returncode == 0:
                cand_audio = None
                for f in tmp_dir.iterdir():
                    if f.is_file() and not f.name.endswith((".part", ".ytdl", ".info.json", "cookies_working.txt")):
                        cand_audio = f
                        break
                min_bytes = int(expected_dur * 1500) if expected_dur >= 300.0 else 5_000
                if cand_audio is None or cand_audio.stat().st_size < min_bytes:
                    cand_sz = cand_audio.stat().st_size if cand_audio else 0
                    print(f"    [download] Attempt {attempts_run} produced truncated file ({cand_sz // 1024} KB < {min_bytes // 1024} KB expected) -> rotating to next attempt.", flush=True)
                    last_kind = "network"
                    continue

                if cookie is not None:
                    if kind == "invalid":
                        pool.mark_failed(cookie, "invalid", last_tail)
                        print(f"    [cookies] {cookie_label} is no longer valid (rotated/logged out) -> quarantined.", flush=True)
                    else:
                        pool.mark_ok(cookie)
                success = True
                break

            for pat in UNAVAILABLE_VIDEO_PATTERNS:
                if re.search(pat, output_lower):
                    raise PermanentVideoUnavailableError(f"Video {vid} is permanently unavailable on YouTube.")

            last_kind = kind or f"code_{proc.returncode}"
            if kind in ("challenged", "rate_limited", "forbidden_403"):
                saw_bot_challenge = True

            if cookie is not None:
                if kind in quarantine_kinds:
                    pool.mark_failed(cookie, kind, last_tail)
                    print(f"    [cookies] {cookie_label} failed ({kind}) -> cooldown, rotating to next attempt.", flush=True)
                else:
                    print(
                        f"    [cookies] {cookie_label} failed ({last_kind}: {last_tail[:100]}) -> keeping cookie active.",
                        flush=True,
                    )
            elif attempts_run == 1:
                # If anonymous HLS failed for non-auth reasons, test android_vr before burning cookies
                if not tried_android_vr and kind in ("network", "format_unavailable", "forbidden_403"):
                    plan.insert(0, (None, "android_vr", "anonymous (android_vr)"))
                elif plan:
                    print(
                        f"    [auth] Anonymous download encountered restriction ({last_kind}) -> trying authenticated fallback pool...",
                        flush=True,
                    )

            # Before giving up on yt-dlp, give anonymous android_vr one clean shot unless age-gated
            if not plan and not tried_android_vr and last_kind != "age_gate":
                plan.append((None, "android_vr", "anonymous (android_vr)"))

        if not success:
            success = _attempt_pytubefix_download(url, vid, tmp_dir, expected_dur=expected_dur)

        if not success:
            if last_kind == "age_gate":
                raise RuntimeError(f"Video {vid} is age-restricted and no available cookie could access it.")
            err_msg = (
                f"Failed to fetch {vid} with all {attempts_run} option(s) "
                f"(last status: {last_kind or 'unknown'}): {last_tail}"
            )
            if saw_bot_challenge or any(re.search(pat, last_tail.lower()) for pat in BOT_CHALLENGE_PATTERNS):
                raise YouTubeBotChallengeError(err_msg, status_kind=last_kind or "challenged")
            raise RuntimeError(err_msg)

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

        min_bytes = int(expected_dur * 1500) if expected_dur >= 300.0 else 5_000
        if downloaded_audio is None or downloaded_audio.stat().st_size < min_bytes:
            cand_sz = downloaded_audio.stat().st_size if downloaded_audio else 0
            raise RuntimeError(
                f"Download pipeline completed for {vid} with incomplete file "
                f"({cand_sz // 1024} KB vs >={min_bytes // 1024} KB expected)."
            )

        # Save raw audio + compact metadata to shared 5 TB Drive (Micro-Step M1!)
        # Copy directly to final destination (avoid os.replace on Drive FUSE)
        workspace.audio_raw_dir.mkdir(parents=True, exist_ok=True)
        dest_audio = workspace.audio_raw_dir / downloaded_audio.name
        shutil.copy2(downloaded_audio, dest_audio)

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
