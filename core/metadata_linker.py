#!/usr/bin/env python3
"""
core/metadata_linker.py
Canonical V2 Video Inventory Builder for yt-spx.
Builds data/video_manifest.json directly from all_channels_videos.csv and clean_urls.txt
for all 2,640 YouTube videos (zero dependency on legacy .srt subtitle files).
Extracts video_id, channel, Arabic title, #shorts classification, smart batch priority,
and Da'ee speaker hints from Arabic titles to bootstrap TitaNet-Large voiceprint matching.
"""

import re
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "all_channels_videos.csv"
CLEAN_URLS_PATH = BASE_DIR / "clean_urls.txt"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = DATA_DIR / "video_manifest.json"

# Known recurring Speakers' Corner / Dawah speakers detectable in Arabic video titles
DAEE_TITLE_PATTERNS = [
    ("daee_uthman", "Sheikh Uthman Ibn Farooq", "الشيخ عثمان بن فاروق", [r"عثمان"]),
    ("daee_hamza", "Hamza Myatt", "حمزة", [r"حمزة", r"حمزه"]),
    ("daee_hijab", "Mohammed Hijab", "محمد حجاب", [r"حجاب"]),
    ("daee_mansur", "Mansur", "منصور", [r"منصور"]),
    ("daee_hashim", "Hashim", "هاشم", [r"هاشم"]),
    ("daee_adnan", "Adnan Rashid", "عدنان رشيد", [r"عدنان"]),
    ("daee_shamsi", "Shamsi", "شمس", [r"\bشمس\b", r"شمسي"]),
    ("daee_ali_dawah", "Ali Dawah", "علي دعوة", [r"علي دعوة", r"علي دعوه"]),
    ("daee_efthimios", "Efthimios", "إفثيميوس", [r"افثيميوس", r"إفثيميوس"]),
    ("daee_zakir", "Dr. Zakir Naik", "ذاكر نايك", [r"ذاكر نايك"]),
    ("daee_farid", "Farid Responds", "فريد", [r"\bفريد\b"]),
    ("daee_bilal", "Bilal", "بلال", [r"\bبلال\b"]),
]


def extract_video_id(url: str) -> str:
    """Extracts 11-char YouTube video ID from watch, youtu.be, or shorts URL."""
    m = re.search(r"(?:v=|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return url.strip()


def normalize_title(title: str) -> str:
    """Normalizes Arabic & English titles for deduplication and matching."""
    t = re.sub(r"\.(?:en|ar|ja|en-US|en-GB)\.srt$", "", title, flags=re.IGNORECASE)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("｜", "|").replace("–", "-").replace("—", "-")
    t = re.sub(r"[\s\W_]+", "", t.lower())
    return t


def detect_title_daee_hints(title_ar: str) -> List[Dict[str, str]]:
    """Extracts candidate Da'ee speaker profile hints mentioned in the Arabic title."""
    hints = []
    if not title_ar:
        return hints
    for profile_id, name_en, name_ar, patterns in DAEE_TITLE_PATTERNS:
        if any(re.search(pat, title_ar) for pat in patterns):
            hints.append({
                "profile_id": profile_id,
                "name_en": name_en,
                "name_ar": name_ar,
            })
    return hints


def compute_smart_priority(title_ar: str, is_short: bool, has_daee_hint: bool) -> int:
    """
    Computes batch scheduling priority for 2-4 hr/day Free Colab sessions:
      Tier 1 (score 90-100): Full street debates with named Da'ees or high-value debate keywords
      Tier 2 (score 75-89): Standard full-length conversations (non-shorts, non-promo)
      Tier 3 (score 60-74): #shorts clips (fast 20-second GPU wins)
      Tier 4 (score 20-40): Fundraising / channel announcement videos
    """
    t = title_ar or ""
    promo_markers = ["تبرع", "رمضان القادم", "ساهم في", "فرصتك التي لن تتكرر", "بناء مسجد"]
    if any(pm in t for pm in promo_markers):
        return 25

    if is_short:
        return 68 + (5 if has_daee_hint else 0)

    debate_markers = ["مناظرة", "نقاش", "حوار", "مواجهة", "ملحد", "مسيحي", "مبشر", "ثالوث", "يسوع", "القرآن", "الإسلام"]
    has_debate_kw = any(kw in t for kw in debate_markers)

    score = 80
    if has_daee_hint:
        score += 10
    if has_debate_kw:
        score += 8
    return min(score, 100)


def get_default_speaker_profiles() -> List[Dict[str, str]]:
    """Returns canonical seed records for the speaker_profiles table."""
    return [
        {
            "profile_id": pid,
            "name_en": name_en,
            "name_ar": name_ar,
            "role_default": "MAIN_DAEE",
        }
        for pid, name_en, name_ar, _ in DAEE_TITLE_PATTERNS
    ]


def build_manifest(
    force_rebuild: bool = False,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Builds or loads the canonical V2 master video manifest (all 2,640 YouTube URLs).
    """
    target_manifest = output_path or MANIFEST_PATH
    if target_manifest.exists() and not force_rebuild:
        with open(target_manifest, "r", encoding="utf-8") as f:
            return json.load(f)

    by_vid: Dict[str, Dict[str, Any]] = {}

    # 1. Primary rich metadata source: all_channels_videos.csv
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            _header = next(reader, None)
            for idx, row in enumerate(reader, start=1):
                if len(row) < 3:
                    continue
                channel = row[0].strip() or "SCDawah Arabic"
                title_ar = row[1].strip()
                url = row[2].strip()
                vid = extract_video_id(url)
                if not vid or len(vid) != 11:
                    continue

                is_short = ("#shorts" in title_ar.lower()) or ("/shorts/" in url.lower())
                daee_hints = detect_title_daee_hints(title_ar)
                priority = compute_smart_priority(title_ar, is_short, bool(daee_hints))

                by_vid[vid] = {
                    "video_id": vid,
                    "catalog_order": idx,
                    "channel": channel,
                    "title_ar": title_ar,
                    "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                    "is_short": is_short,
                    "daee_hints": daee_hints,
                    "primary_daee_hint": daee_hints[0]["profile_id"] if len(daee_hints) == 1 else None,
                    "priority_score": priority,
                    "status": "queued",
                }

    # 2. Reconcile any extra URLs present in clean_urls.txt
    if CLEAN_URLS_PATH.exists():
        with open(CLEAN_URLS_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                vid = extract_video_id(url)
                if vid and len(vid) == 11 and vid not in by_vid:
                    is_short = "/shorts/" in url.lower()
                    by_vid[vid] = {
                        "video_id": vid,
                        "catalog_order": len(by_vid) + 1,
                        "channel": "SCDawah Arabic",
                        "title_ar": f"مناظرة ميدانية ({vid})",
                        "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                        "is_short": is_short,
                        "daee_hints": [],
                        "primary_daee_hint": None,
                        "priority_score": 65 if is_short else 75,
                        "status": "queued",
                    }

    manifest = list(by_vid.values())
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(target_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    shorts_cnt = sum(1 for m in manifest if m["is_short"])
    full_cnt = len(manifest) - shorts_cnt
    print(f"[+] V2 Manifest built: {len(manifest)} canonical videos ({full_cnt} full-length, {shorts_cnt} #shorts).")
    print(f"[+] Saved to {target_manifest}")
    return manifest


if __name__ == "__main__":
    build_manifest(force_rebuild=True)
