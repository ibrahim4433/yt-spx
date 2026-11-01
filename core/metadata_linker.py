#!/usr/bin/env python3
"""
core/metadata_linker.py
Links SRT files in Downloads/ to metadata records in all_channels_videos.csv.
Produces a unified manifest with YouTube URLs, video IDs, titles, and processing status.
"""

import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "Downloads"
CSV_PATH = BASE_DIR / "all_channels_videos.csv"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = DATA_DIR / "video_manifest.json"


def extract_video_id(url: str) -> str:
    """Extracts 11-char YouTube video ID from watch or shorts URL."""
    m = re.search(r"(?:v=|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return url.strip()


def normalize_title(title: str) -> str:
    """
    Normalizes Arabic & English titles for robust matching across
    filesystem variations, yt-dlp sanitizations, and character differences.
    """
    t = re.sub(r"\.(?:en|ar|ja|en-US|en-GB)\.srt$", "", title, flags=re.IGNORECASE)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("｜", "|").replace("–", "-").replace("—", "-")
    # Remove all punctuation, whitespace, and symbols
    t = re.sub(r"[\s\W_]+", "", t.lower())
    return t


def build_manifest(force_rebuild: bool = False) -> List[Dict[str, Any]]:
    """
    Builds or loads the master video manifest.
    Maps every subtitle file to its channel, Arabic title, YouTube URL, and status.
    """
    if MANIFEST_PATH.exists() and not force_rebuild:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1. Read CSV records
    csv_records = []
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    channel, title, url = row[0].strip(), row[1].strip(), row[2].strip()
                    vid = extract_video_id(url)
                    csv_records.append({
                        "channel": channel,
                        "title": title,
                        "url": url,
                        "video_id": vid,
                        "norm_title": normalize_title(title)
                    })

    # Build lookup index by normalized title
    csv_lookup = {r["norm_title"]: r for r in csv_records}

    # 2. Read archive.txt if present
    archive_ids = set()
    archive_file = DOWNLOADS_DIR / "archive.txt"
    if archive_file.exists():
        with open(archive_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    archive_ids.add(parts[1])

    # 3. Scan Downloads/ directory for SRT files
    srt_files = list(DOWNLOADS_DIR.glob("*.srt"))
    manifest = []
    matched_count = 0

    for srt in srt_files:
        filename = srt.name
        norm = normalize_title(filename)
        
        match = csv_lookup.get(norm)
        
        # Substring prefix fallback if exact normalized match fails
        if not match:
            for k, candidate in csv_lookup.items():
                if len(norm) > 10 and (k.startswith(norm[:20]) or norm.startswith(k[:20])):
                    match = candidate
                    break

        if match:
            matched_count += 1
            manifest.append({
                "video_id": match["video_id"],
                "channel": match["channel"],
                "title_ar": match["title"],
                "youtube_url": match["url"],
                "srt_file": srt.relative_to(BASE_DIR).as_posix(),
                "srt_filename": filename,
                "file_size": srt.stat().st_size,
                "has_archive": match["video_id"] in archive_ids,
                "status": "ready"
            })
        else:
            # Fallback entry without CSV metadata
            clean_name = re.sub(r"\.(?:en|ar|ja|en-US)\.srt$", "", filename)
            manifest.append({
                "video_id": f"local_{hash(filename) % 1000000:06d}",
                "channel": "SCDawah Arabic",
                "title_ar": clean_name,
                "youtube_url": f"https://www.youtube.com/results?search_query={clean_name}",
                "srt_file": srt.relative_to(BASE_DIR).as_posix(),
                "srt_filename": filename,
                "file_size": srt.stat().st_size,
                "has_archive": False,
                "status": "unlinked_metadata"
            })

    # Deduplicate manifest by video_id
    dedup_dict = {}
    for item in manifest:
        vid = item["video_id"]
        if vid not in dedup_dict or item.get("file_size", 0) > dedup_dict[vid].get("file_size", 0):
            dedup_dict[vid] = item
    manifest = list(dedup_dict.values())

    # Save to JSON
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[+] Manifest built: {len(manifest)} SRT files ({matched_count} matched to CSV metadata).")
    print(f"[+] Saved to {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    build_manifest(force_rebuild=True)
