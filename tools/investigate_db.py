#!/usr/bin/env python3
"""List all qa_units cards with video metadata + run card-level audit summary."""
import sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
conn = sqlite3.connect(str(REPO / "data" / "dawah_knowledge.db"))
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cards = cur.execute("SELECT * FROM qa_units ORDER BY video_id, id").fetchall()
print(f"qa_units total: {len(cards)}\n")

def _rowget(r, k, d=None):
    return r[k] if k in r.keys() else d

videos = {r["video_id"]: r for r in cur.execute("SELECT video_id, title_ar, duration_sec, speakers_count, turns_count FROM videos")}
speakers = {}
for r in cur.execute("SELECT video_id, speaker_id, role, speaking_sec, turn_count, display_name_en FROM video_speakers"):
    speakers.setdefault(r["video_id"], []).append(r)

for c in cards:
    v = videos.get(c["video_id"]) or {}
    spk = speakers.get(c["video_id"], [])
    print(f"== {c['id']}  video={c['video_id']}")
    print(f"   q_en: {(c['question_en'] or '')[:100]}")
    print(f"   video: dur={_rowget(v,'duration_sec')}s speakers_count={_rowget(v,'speakers_count')} turns={_rowget(v,'turns_count')}")
    for s in spk:
        print(f"     spk: {s['speaker_id']} role={s['role']} speaking_sec={s['speaking_sec']} turns={s['turn_count']} name_en={s['display_name_en']}")
    print()

conn.close()
