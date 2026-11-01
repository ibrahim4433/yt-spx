#!/usr/bin/env python3
"""
core/cookie_pool.py
Multi-Cookie Rotation Pool for YouTube downloads (shared across all Colab accounts via Drive).

Cookie sources (all Netscape-format .txt files, discovered in this order):
  1. /content/drive/MyDrive/yt-spx/cookies/*.txt   <- recommended: drop one file per Google account
  2. /content/drive/MyDrive/yt-spx/cookies.txt
  3. /content/drive/MyDrive/yt-spx/data/cookies/*.txt, data/cookies.txt
  4. <repo>/cookies/*.txt, <repo>/cookies.txt

Health tracking:
  - Offline test (free, zero network): valid Netscape lines, contains YouTube/Google auth cookies
    (SID / __Secure-1PSID / __Secure-3PSID / LOGIN_INFO / SAPISID), and those are not expired.
  - Online test (yt-dlp --simulate against one video) via `test_cookie_online()` / `--mode test-cookies`.
  - Shared registry at locks/_cookie_health.json records which cookie failed, why, and until when it is
    quarantined. If you REPLACE a cookie file (new mtime/size), its quarantine is cleared automatically.
"""

import os
import re
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.drive_sync import SharedWorkspace

BASE_DIR = Path(__file__).resolve().parent.parent

AUTH_COOKIE_NAMES = {
    "SID",
    "__Secure-1PSID",
    "__Secure-3PSID",
    "LOGIN_INFO",
    "SAPISID",
    "__Secure-1PAPISID",
    "__Secure-3PAPISID",
}

# yt-dlp output patterns that mean THIS cookie file is dead / logged-out / rotated.
COOKIE_INVALID_PATTERNS = [
    r"cookies are no longer valid",
    r"account cookies are no longer valid",
    r"have likely been rotated in the browser",
    r"failed to load cookies",
    r"invalid netscape format",
]

# Patterns where a different (healthy) cookie may succeed: bot-check / rate limit / age-gate.
COOKIE_CHALLENGED_PATTERNS = [
    r"sign in to confirm you.?re not a bot",
    r"use --cookies-from-browser or --cookies for the authentication",
]
RATE_LIMIT_PATTERNS = [
    r"http error 429",
    r"too many requests",
    r"this content isn.?t available, try again later",
    r"rate.?limit",
]
AGE_GATE_PATTERNS = [
    r"confirm your age",
    r"age.?restricted",
    r"inappropriate for some users",
]

# Quarantine durations (seconds) per failure kind
QUARANTINE_SEC = {
    "invalid": 7 * 24 * 3600,   # logged-out/rotated cookies: until the file is replaced
    "challenged": 6 * 3600,     # bot-check while using this cookie
    "rate_limited": 3600,       # HTTP 429 (often IP-level; short quarantine)
    "age_gate": 0,              # age-gate failure doesn't mean the cookie is bad
}


def classify_ytdlp_output(output_lower: str) -> str:
    """Returns one of: 'invalid', 'challenged', 'rate_limited', 'age_gate', '' (unknown/other)."""
    for kind, pats in (
        ("invalid", COOKIE_INVALID_PATTERNS),
        ("challenged", COOKIE_CHALLENGED_PATTERNS),
        ("rate_limited", RATE_LIMIT_PATTERNS),
        ("age_gate", AGE_GATE_PATTERNS),
    ):
        if any(re.search(p, output_lower) for p in pats):
            return kind
    return ""


def validate_cookie_file_offline(path: Path) -> Tuple[bool, str]:
    """
    Zero-network sanity test of a Netscape cookies.txt:
    must contain non-expired YouTube/Google authentication cookies.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"unreadable ({exc})"

    now = time.time()
    total_lines = 0
    yt_lines = 0
    live_auth: List[str] = []
    expired_auth: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        elif line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        total_lines += 1
        domain, _flag, _path, _secure, expiry, name = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        dom = domain.lower()
        if "youtube.com" not in dom and "google.com" not in dom:
            continue
        yt_lines += 1
        if name in AUTH_COOKIE_NAMES:
            try:
                exp = int(float(expiry or 0))
            except ValueError:
                exp = 0
            if exp == 0 or exp > now:
                live_auth.append(name)
            else:
                expired_auth.append(name)

    if total_lines == 0:
        return False, "not a Netscape cookies.txt (no tab-separated cookie lines)"
    if yt_lines == 0:
        return False, "contains no youtube.com/google.com cookies"
    if not live_auth:
        if expired_auth:
            return False, f"YouTube login cookies expired ({', '.join(sorted(set(expired_auth)))})"
        return False, "no YouTube login cookies (export while signed in to YouTube)"
    return True, f"ok ({len(set(live_auth))} live auth cookies)"


class CookiePool:
    """Discovers, health-checks, orders and quarantines multiple cookies.txt files."""

    REGISTRY_NAME = "_cookie_health.json"

    def __init__(self, workspace: SharedWorkspace, include_repo_dir: bool = True):
        self.ws = workspace
        self.include_repo_dir = include_repo_dir
        self.registry_path = workspace.locks_dir / self.REGISTRY_NAME

    # ------------------------------------------------------------------
    # Discovery & registry
    # ------------------------------------------------------------------
    def discover(self) -> List[Path]:
        dirs = [
            self.ws.root / "cookies",
            self.ws.root / "Cookies",
            self.ws.data_dir / "cookies",
        ]
        if self.ws.is_drive and self.ws.root.parent.exists():
            dirs.extend([
                self.ws.root.parent / "cookies",
                self.ws.root.parent / "Cookies",
            ])
        singles = [
            self.ws.root / "cookies.txt",
            self.ws.data_dir / "cookies.txt",
        ]
        if self.include_repo_dir:
            dirs.append(BASE_DIR / "cookies")
            dirs.append(BASE_DIR / "Cookies")
            singles.append(BASE_DIR / "cookies.txt")

        found: List[Path] = []
        seen = set()

        def _add(p: Path) -> None:
            try:
                if not p.is_file():
                    return
                if p.stat().st_size < 10:
                    return
                key = str(p.resolve())
            except OSError:
                return
            if key not in seen:
                seen.add(key)
                found.append(p)

        for d in dirs:
            if d.is_dir():
                try:
                    for p in sorted(d.iterdir()):
                        if p.is_file() and not p.name.startswith("."):
                            name_lower = p.name.lower()
                            if name_lower.endswith((".txt", ".cookie", ".cookies")) or "cookie" in name_lower:
                                _add(p)
                except OSError:
                    pass

        for s in singles:
            _add(s)

        return found

    def cookie_key(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.ws.root.resolve())).replace("\\", "/")
        except ValueError:
            return f"{path.parent.name}/{path.name}"

    @staticmethod
    def _fingerprint(path: Path) -> str:
        try:
            st = path.stat()
            return f"{int(st.st_mtime)}:{st.st_size}"
        except OSError:
            return ""

    def _load(self) -> Dict[str, Any]:
        data = self.ws.read_json(self.registry_path, default={}) or {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: Dict[str, Any]) -> None:
        self.ws.atomic_write_json(self.registry_path, data)

    def _entry_status(self, path: Path, entry: Dict[str, Any], now: float) -> Tuple[bool, str]:
        """Returns (usable, reason) based on registry quarantine + file fingerprint."""
        if not entry:
            return True, "never used"
        if entry.get("fingerprint") and entry["fingerprint"] != self._fingerprint(path):
            return True, "file replaced (quarantine cleared)"
        until = float(entry.get("quarantined_until", 0.0))
        if until > now:
            mins = int((until - now) / 60)
            return False, f"quarantined {mins}m ({entry.get('last_failure_kind', '?')}: {entry.get('last_failure', '')[:80]})"
        return True, "healthy"

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def healthy_cookies(self) -> List[Path]:
        """Returns usable cookie files ordered: most recently successful first, then least failures."""
        now = time.time()
        reg = self._load()
        usable: List[Tuple[float, int, str, Path]] = []
        for p in self.discover():
            ok_off, _ = validate_cookie_file_offline(p)
            if not ok_off:
                continue
            entry = reg.get(self.cookie_key(p), {})
            ok, _ = self._entry_status(p, entry, now)
            if not ok:
                continue
            fresh = entry.get("fingerprint") and entry["fingerprint"] != self._fingerprint(p)
            last_ok = 0.0 if fresh else float(entry.get("last_ok_at", 0.0))
            fails = 0 if fresh else int(entry.get("consecutive_failures", 0))
            usable.append((last_ok, -fails, self.cookie_key(p), p))
        usable.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [t[3] for t in usable]

    def mark_ok(self, path: Path) -> None:
        reg = self._load()
        key = self.cookie_key(path)
        entry = reg.get(key, {})
        entry.update({
            "last_ok_at": time.time(),
            "consecutive_failures": 0,
            "quarantined_until": 0.0,
            "fingerprint": self._fingerprint(path),
            "last_worker": self.ws.worker_id,
        })
        reg[key] = entry
        self._save(reg)

    def mark_failed(self, path: Path, kind: str, reason: str) -> None:
        reg = self._load()
        key = self.cookie_key(path)
        entry = reg.get(key, {})
        # A replaced file starts a fresh failure history
        if entry.get("fingerprint") and entry["fingerprint"] != self._fingerprint(path):
            entry = {}
        fails = int(entry.get("consecutive_failures", 0)) + 1
        base = QUARANTINE_SEC.get(kind, 1800)
        # Escalate repeat challenges (6h -> 12h -> 24h ...), capped at 7 days
        quarantine = min(base * (2 ** max(0, fails - 1)), QUARANTINE_SEC["invalid"]) if base > 0 else 0
        entry.update({
            "consecutive_failures": fails,
            "last_failure_at": time.time(),
            "last_failure_kind": kind,
            "last_failure": reason[:300],
            "quarantined_until": time.time() + quarantine,
            "fingerprint": self._fingerprint(path),
            "last_worker": self.ws.worker_id,
        })
        reg[key] = entry
        self._save(reg)

    def status_report(self) -> List[Dict[str, Any]]:
        now = time.time()
        reg = self._load()
        rows = []
        for p in self.discover():
            ok_off, off_reason = validate_cookie_file_offline(p)
            entry = reg.get(self.cookie_key(p), {})
            usable, reg_reason = self._entry_status(p, entry, now)
            rows.append({
                "cookie": self.cookie_key(p),
                "path": str(p),
                "usable": bool(ok_off and usable),
                "offline_check": off_reason,
                "registry": reg_reason,
            })
        return rows

    def print_summary(self, indent: str = "   -> ") -> None:
        rows = self.status_report()
        if not rows:
            target_dir = self.ws.root / "cookies"
            diag_notes = []
            if target_dir.is_dir():
                try:
                    entries = [e for e in target_dir.iterdir() if not e.name.startswith(".")]
                    if entries:
                        diag_notes.append(f"Folder exists ({target_dir}) with {len(entries)} item(s): {[e.name for e in entries]}")
                        for e in entries:
                            if e.is_file():
                                _, r = validate_cookie_file_offline(e)
                                diag_notes.append(f"  - {e.name} ({e.stat().st_size} bytes): {r}")
                    else:
                        diag_notes.append(
                            f"Folder exists on Drive ({target_dir}) but is currently reported as empty by Colab's Drive FUSE cache."
                        )
                except Exception as exc:
                    diag_notes.append(f"Could not read {target_dir}: {exc}")
            else:
                diag_notes.append(f"Folder {target_dir} is not yet visible on Drive.")

            print(f"{indent}Cookies pool        : none found.")
            for note in diag_notes:
                print(f"{indent}   {note}")
            print(
                f"{indent}   [Tip] In Colab's left sidebar file manager (folder icon), expand 'drive/MyDrive/yt-spx/cookies/' "
                "or drag & drop cookies into '/content/yt-spx/cookies/' for instant zero-sync-delay detection."
            )
            return
        n_ok = sum(1 for r in rows if r["usable"])
        print(f"{indent}Cookies pool        : {n_ok}/{len(rows)} usable (auto-rotates on failure)")
        for r in rows:
            flag = "OK " if r["usable"] else "BAD"
            offline_ok = r["offline_check"].startswith("ok")
            detail = r["registry"] if offline_ok else r["offline_check"]
            print(f"{' ' * len(indent)}  [{flag}] {r['cookie']}: {detail}")

    # ------------------------------------------------------------------
    # Active online test
    # ------------------------------------------------------------------
    def test_cookie_online(self, path: Path, test_url: str, timeout_sec: int = 90) -> Tuple[bool, str, str]:
        """
        Runs `yt-dlp --simulate` with this cookie (isolated /tmp copy) against test_url.
        Returns (ok, kind, message) and updates the shared health registry.
        """
        ok_off, off_reason = validate_cookie_file_offline(path)
        if not ok_off:
            self.mark_failed(path, "invalid", off_reason)
            return False, "invalid", off_reason

        tmp_dir = Path(tempfile.mkdtemp(prefix="yt_spx_cookie_test_"))
        try:
            jar = tmp_dir / "cookies_working.txt"
            shutil.copy2(path, jar)
            cmd = [
                "yt-dlp",
                "--simulate",
                "--no-playlist",
                "--no-progress",
                "--print", "id",
                "--format", "bestaudio/best",
                "--cookies", str(jar),
            ]
            runtimes = [r for r in ("deno", "node") if shutil.which(r)]
            if runtimes:
                cmd += ["--js-runtimes", ",".join(runtimes)]
            cmd.append(test_url)
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return False, "timeout", "yt-dlp test timed out"
            except FileNotFoundError:
                return False, "missing", "yt-dlp is not installed"

            out = proc.stdout or ""
            kind = classify_ytdlp_output(out.lower())
            if proc.returncode == 0 and kind != "invalid":
                self.mark_ok(path)
                return True, "ok", out.strip().splitlines()[-1] if out.strip() else "ok"
            if proc.returncode == 0 and kind == "invalid":
                # yt-dlp fell back to anonymous access: cookie itself is dead
                self.mark_failed(path, "invalid", out[-240:].strip())
                return False, "invalid", "cookies no longer valid (yt-dlp fell back to anonymous)"
            kind = kind or "error"
            if kind not in ("age_gate", "error"):
                self.mark_failed(path, kind, out[-240:].strip())
            return False, kind, out[-240:].strip()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
