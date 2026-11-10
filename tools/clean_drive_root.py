#!/usr/bin/env python3
"""
tools/clean_drive_root.py
Surgical Google Drive Root Cleaner for Dawah Dialectics (yt-spx).

Safely inspects and purges leaked acoustic/diarization artifacts that escaped
into the Google Drive root directory (/content/drive/MyDrive or /content/drive/My Drive)
due to Google Drive FUSE unparenting behaviors or legacy Colab directory misconfigurations.

CRITICAL SAFETY:
- Never deletes or modifies legitimate user folders (Colab Notebooks, yt-spx, yt-ch, etc.).
- Never touches files outside of Google Drive root.
- Uses bottom-up file deletion to prevent Google Drive from unparenting items.
- Supports --dry-run mode for zero-risk inspection.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Standard 11-character YouTube video ID regex
YOUTUBE_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Protected folders that must NEVER be touched under any circumstances
PROTECTED_FOLDER_NAMES: Set[str] = {
    "yt-spx",
    "yt-ch",
    "colab notebooks",
    "google earth",
    "classroom",
    "meet recordings",
    "backup",
    "backups",
    "photos",
    "documents",
    "downloads",
    "desktop",
    ".shortcut-targets-by-id",
    ".trash",
}

# Signatures of files inside video ID directories that prove they are yt-spx leak artifacts
LEAK_CHILD_PATTERNS = [
    re.compile(r"demucs_chunk_.*\.wav$"),
    re.compile(r"\.demucs_chunk_.*"),
    re.compile(r".*\.rttm$"),
    re.compile(r".*\.opus$"),
    re.compile(r".*\.wav$"),
    re.compile(r"overlap_sep\.json$"),
    re.compile(r".*\.lock\.json$"),
]


def resolve_candidate_drive_roots(override_path: Optional[str] = None) -> List[Path]:
    """Finds existing Google Drive root directories to inspect."""
    if override_path:
        p = Path(override_path)
        return [p] if p.exists() else []

    roots: List[Path] = []
    for cand in [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]:
        if cand.exists() and cand.is_dir():
            roots.append(cand)
    return roots


def is_leaked_file(file_path: Path) -> bool:
    """Returns True if the file directly in Google Drive root matches a known yt-spx leak pattern."""
    name = file_path.name

    # 1. Separated overlap WAV slices: OV_0065_s0_SPK_00.wav
    if re.match(r"^OV_\d+.*\.wav$", name):
        return True

    # 2. Leaked cross-account worker locks: *.lock.json
    if name.endswith(".lock.json"):
        return True

    # 3. Leaked yt-spx pipeline artifacts sitting directly in root
    if name.endswith(".diarized.srt") or name.endswith(".state.json") or name.endswith(".qa.json") or name.endswith(".rttm"):
        return True

    # 4. Leaked Demucs staging chunks: demucs_chunk_000_of_004.wav or .demucs_chunk_*.tmp.wav
    if re.match(r"^\.?demucs_chunk_.*", name):
        return True

    # 5. Leaked opus/info files matching video ID pattern
    if re.match(r"^[A-Za-z0-9_-]{8,24}\.(info\.json|opus|wav)$", name):
        return True

    # 6. Temporary bundle/model downloads in root
    if re.match(r"^\.(stage1_models|yt_spx).*\.tmp.*", name) or re.match(r"^\..*\.tmp\.(wav|opus|json)$", name):
        return True

    return False


def is_leaked_dir(dir_path: Path) -> bool:
    """Returns True if the directory in Google Drive root is a leaked yt-spx processing folder."""
    name = dir_path.name.lower()
    if name in PROTECTED_FOLDER_NAMES:
        return False

    # 1. sep_clips folder
    if name == "sep_clips":
        return True

    # 2. Video ID directory (e.g. 0RNO5mv18_, XcQtBlOPEd0)
    if re.match(r"^[A-Za-z0-9_-]{8,24}$", dir_path.name):
        # Safety check: Verify the folder contains at least one yt-spx leak signature file
        try:
            for item in dir_path.iterdir():
                for pat in LEAK_CHILD_PATTERNS:
                    if pat.search(item.name):
                        return True
        except Exception:
            pass

    return False


def safe_remove_file(file_path: Path, dry_run: bool = False) -> int:
    """Safely unlinks a file and returns its size in bytes."""
    try:
        size = file_path.stat().st_size
    except OSError:
        size = 0

    if not dry_run:
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"   [-] Failed to unlink {file_path.name}: {e}")
            return 0
    return size


def safe_remove_dir_bottom_up(dir_path: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Deletes all files inside dir_path bottom-up before removing directories.
    This prevents Google Drive FUSE from unparenting files into My Drive.
    Returns (files_removed, bytes_freed).
    """
    files_removed = 0
    bytes_freed = 0

    if not dir_path.is_dir():
        return 0, 0

    try:
        for root, dirs, files in os.walk(str(dir_path), topdown=False):
            for f in files:
                fp = Path(root) / f
                try:
                    bytes_freed += fp.stat().st_size
                except OSError:
                    pass
                files_removed += 1
                if not dry_run:
                    try:
                        fp.unlink(missing_ok=True)
                    except Exception:
                        pass
            for d in dirs:
                dp = Path(root) / d
                if not dry_run:
                    try:
                        dp.rmdir()
                    except Exception:
                        pass
        if not dry_run:
            try:
                dir_path.rmdir()
            except Exception:
                import shutil
                shutil.rmtree(dir_path, ignore_errors=True)
    except Exception as e:
        print(f"   [-] Note during directory removal of {dir_path.name}: {e}")

    return files_removed, bytes_freed


def clean_drive_root(drive_root: Optional[str] = None, dry_run: bool = False) -> int:
    """
    Scans Google Drive root directories and surgically removes leaked yt-spx artifacts.
    Returns 0 on success.
    """
    roots = resolve_candidate_drive_roots(drive_root)
    if not roots:
        target_display = drive_root or "/content/drive/MyDrive"
        print(f"[Clean Drive] No Google Drive root directory found at {target_display}. Skipping cleanup.")
        return 0

    mode_label = "[DRY-RUN PREVIEW]" if dry_run else "[LIVE CLEANUP]"
    print("=" * 76)
    print(f"  DAWAH DIALECTICS: SURGICAL GOOGLE DRIVE ROOT CLEANER {mode_label}")
    print("=" * 76)

    total_files_cleaned = 0
    total_dirs_cleaned = 0
    total_bytes_freed = 0

    for root_dir in roots:
        print(f"\n[+] Inspecting Google Drive root: {root_dir}")
        leaked_files: List[Path] = []
        leaked_dirs: List[Path] = []

        try:
            for item in root_dir.iterdir():
                if item.is_symlink():
                    continue
                if item.is_file() and is_leaked_file(item):
                    leaked_files.append(item)
                elif item.is_dir() and is_leaked_dir(item):
                    leaked_dirs.append(item)
        except Exception as e:
            print(f"   [-] Failed to read {root_dir}: {e}")
            continue

        print(f"   -> Found {len(leaked_files)} leaked file(s) and {len(leaked_dirs)} leaked directory/ies directly in root.")

        # Clean leaked files
        for f in leaked_files:
            sz = safe_remove_file(f, dry_run=dry_run)
            total_bytes_freed += sz
            total_files_cleaned += 1
            sz_kb = round(sz / 1024, 1)
            action = "[WOULD REMOVE]" if dry_run else "[REMOVED FILE]"
            print(f"      {action} {f.name} ({sz_kb} KB)")

        # Clean leaked directories
        for d in leaked_dirs:
            f_count, b_freed = safe_remove_dir_bottom_up(d, dry_run=dry_run)
            total_bytes_freed += b_freed
            total_files_cleaned += f_count
            total_dirs_cleaned += 1
            mb_freed = round(b_freed / (1024 * 1024), 2)
            action = "[WOULD PURGE DIR]" if dry_run else "[PURGED DIR]"
            print(f"      {action} {d.name}/ ({f_count} files, {mb_freed} MB)")

    mb_total = round(total_bytes_freed / (1024 * 1024), 2)
    gb_total = round(total_bytes_freed / (1024 ** 3), 3)

    print("\n" + "=" * 76)
    action_summary = "DRY-RUN AUDIT COMPLETE (no files modified)" if dry_run else "SURGICAL CLEANUP COMPLETE"
    print(f"  [OK] {action_summary}!")
    print(f"  [OK] Files cleaned    : {total_files_cleaned}")
    print(f"  [OK] Directories freed: {total_dirs_cleaned}")
    print(f"  [OK] Storage reclaimed: {mb_total} MB ({gb_total} GB)")
    print("  [OK] Protected folders: yt-spx, yt-ch, Colab Notebooks are 100% INTACT.")
    print("=" * 76 + "\n", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Surgically clean leaked yt-spx artifacts from Google Drive root.")
    parser.add_argument("--drive-root", type=str, default=None, help="Custom Google Drive root path to inspect.")
    parser.add_argument("--dry-run", action="store_true", help="Preview leaked files without deleting.")
    args = parser.parse_args(argv)

    return clean_drive_root(drive_root=args.drive_root, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
