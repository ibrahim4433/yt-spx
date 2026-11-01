#!/usr/bin/env python3
"""
scripts/ingest_pilot.py
Ingests video manifest and pilot structured Q&A units into SQLite database.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import json
from db.database import init_db, insert_qa_unit, get_db_connection
DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "video_manifest.json"
PILOT_QA_PATH = DATA_DIR / "pilot_qa_dataset.json"


def ingest():
    init_db()
    conn = get_db_connection()

    # 1. Ingest videos from manifest
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        with conn:
            for v in manifest:
                conn.execute("""
                    INSERT OR REPLACE INTO videos (
                        video_id, channel, title_ar, youtube_url, srt_file, file_size, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    v["video_id"],
                    v.get("channel", "SCDawah Arabic"),
                    v.get("title_ar", ""),
                    v.get("youtube_url", ""),
                    v.get("srt_file", ""),
                    v.get("file_size", 0),
                    v.get("status", "ready")
                ))
        print(f"[+] Ingested {len(manifest)} videos into database.")

    # 2. Ingest pilot Q&A dataset
    if PILOT_QA_PATH.exists():
        with open(PILOT_QA_PATH, "r", encoding="utf-8") as f:
            pilot_items = json.load(f)
        
        for item in pilot_items:
            insert_qa_unit(item)
            
        print(f"[+] Ingested {len(pilot_items)} structured Q&A dialectical units into database.")

    conn.close()


if __name__ == "__main__":
    ingest()
