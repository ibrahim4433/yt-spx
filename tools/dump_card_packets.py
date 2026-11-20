#!/usr/bin/env python3
"""Dump per-card audit packets: qa_units row + video metadata + speaker roster + scenario_turns ground truth.

Read-only w.r.t. data/dawah_knowledge.db. Writes JSON packets to reports/card_packets/.
Used by the manual rating team agents (reports/manual_audit/).
"""
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "dawah_knowledge.db"
OUT = REPO / "reports" / "card_packets"
OUT.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
cur = conn.cursor()


def json_field(row, raw_key, json_key):
    val = row[raw_key]
    if val is None:
        val = row[json_key]
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return [val] if val.strip() else []
    return val if val is not None else []


def card_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    for raw_k, out_k in [
        ("logical_premises_json", "logical_premises"),
        ("logical_premises_ar_json", "logical_premises_ar"),
        ("dialogue_exchange_json", "dialogue_exchange"),
        ("source_turn_ids_json", "source_turn_ids"),
        ("tags_json", "tags"),
        ("analogies_json", "analogies"),
        ("analogies_ar_json", "analogies_ar"),
        ("scriptural_citations_json", "scriptural_citations"),
    ]:
        d[out_k] = json_field(d, raw_k, out_k)
    return d


cards = cur.execute("SELECT * FROM qa_units ORDER BY video_id, id").fetchall()
packets = []
for r in cards:
    card = card_to_dict(r)
    vid = card["video_id"]

    video = cur.execute("SELECT * FROM videos WHERE video_id = ?", (vid,)).fetchone()
    video_d = dict(video) if video else {}

    speakers = [dict(s) for s in cur.execute(
        "SELECT * FROM video_speakers WHERE video_id = ? ORDER BY speaking_sec DESC", (vid,))]

    turns = [dict(t) for t in cur.execute(
        "SELECT * FROM scenario_turns WHERE video_id = ? ORDER BY turn_index ASC, start_ms ASC", (vid,))]
    for t in turns:
        t.pop("words_json", None)

    # Anchor window [ts-60s, ts+180s] around question anchor
    anchor_ms = (card.get("timestamp_seconds") or 0) * 1000
    lo = anchor_ms - 60_000
    hi = anchor_ms + 180_000
    window = [t for t in turns if t["end_ms"] >= lo and t["start_ms"] <= hi]

    packet = {
        "card": card,
        "video_meta": video_d,
        "video_speakers": speakers,
        "transcript_window_around_anchor": window,
        "all_turns_for_video": turns,
    }
    fname = OUT / f"{card['id']}.json"
    fname.write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
    packets.append({"card_id": card["id"], "video_id": vid, "packet": str(fname)})

conn.close()
(REPO / "reports").mkdir(exist_ok=True)
(REPO / "reports" / "card_packets_index.json").write_text(
    json.dumps(packets, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[+] Dumped {len(packets)} card packets to {OUT}")
for p in packets:
    print("  -", p["card_id"], "->", p["packet"])
