#!/usr/bin/env python3
"""
scripts/reset_for_v2_pipeline.py
Cleanly purges all legacy V1 subtitle-based records and files from yt-spx
(since https://github.com/ibrahim4433/yt-ch preserves the V1 subtitle archive),
rebuilds the 2,640-video canonical V2 manifest from all_channels_videos.csv + clean_urls.txt,
and initializes a fresh V2 SQLite database ready for multi-step audio diarization.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.metadata_linker import build_manifest
from db.database import init_db, get_stats, DATA_DIR


def reset_to_clean_v2() -> None:
    local_data = BASE_DIR / "data"
    local_data.mkdir(parents=True, exist_ok=True)

    legacy_files = [
        local_data / "dawah_knowledge.db",
        local_data / "dawah_knowledge_backup.db",
        local_data / "extracted_qa.jsonl",
        local_data / "extraction_checkpoint.json",
        local_data / "extraction_live.log",
        local_data / "pilot_qa_dataset.json",
        local_data / "video_manifest.json",
    ]

    removed = 0
    for f in legacy_files:
        if f.exists():
            f.unlink()
            removed += 1
            print(f"[-] Removed V1 subtitle artifact: {f.relative_to(BASE_DIR)}")

    # Rebuild canonical 2,640-video V2 manifest
    manifest = build_manifest(force_rebuild=True, output_path=local_data / "video_manifest.json")

    # Initialize fresh V2 SQLite database
    db_file = local_data / "dawah_knowledge.db"
    init_db(db_path=db_file)
    stats = get_stats(db_path=db_file)

    print(f"[+] Clean V2 reset complete ({removed} legacy files purged).")
    print(f"[+] Fresh V2 DB initialized at {db_file}:")
    print(f"    Cataloged Videos : {stats['total_videos']} ({stats['full_debates_count']} full debates, {stats['shorts_count']} #shorts)")
    print(f"    Legacy Q&A Rows  : {stats['total_qa']} (100% clean slate)")


if __name__ == "__main__":
    reset_to_clean_v2()
