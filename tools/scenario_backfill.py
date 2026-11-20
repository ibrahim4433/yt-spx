#!/usr/bin/env python3
"""Backfill scenario JSONs for affected videos from the DB (read-only) into a workspace.

The shared Drive workspace is not mounted in this clone, so data/scenarios/*.json are empty here.
This tool reconstructs scenario payloads from the DB tables (videos, video_speakers, scenario_turns)
for the audited videos, writing them into <ws>/scenarios/. Works both locally (reports/regen_ws)
and on Colab (--ws /content/drive/MyDrive/yt-spx). Read-only w.r.t. the DB.

Usage:
    python tools/scenario_backfill.py [--ws PATH] [--video-ids vid1 vid2 ...]
"""
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from db.database import DB_PATH  # universal: drive DB on Colab, clone DB locally

DEFAULT_WS = REPO / "reports" / "regen_ws"


def _canon_role(raw):
    r = (raw or "").upper()
    if "PEDESTRIAN" in r or "عابرة" in (raw or ""):
        return "PEDESTRIAN"
    if "CO_DAEE" in r or "CO-DAEE" in r:
        return "CO_DAEE"
    if "INTERLOCUTOR" in r or "GUEST" in r or "السائل" in (raw or ""):
        return "INTERLOCUTOR"
    if "DAEE" in r or "الداعية" in (raw or "") or "محاجج" in (raw or ""):
        return "MAIN_DAEE"
    return (raw or "").strip() or "INTERLOCUTOR"


def load_video_ids() -> list:
    try:
        idx = json.loads((REPO / "reports" / "manual_audit_summary.json").read_text(encoding="utf-8"))
        vids = sorted({c.split("_")[0] for c in idx.get("manual_cards", {})})
        return [v for v in vids if v]
    except Exception:
        return []


def backfill(video_ids, ws_root: Path) -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    out = {}
    scen_out = ws_root / "scenarios"
    scen_out.mkdir(parents=True, exist_ok=True)
    for vid in video_ids:
        vrow = cur.execute("SELECT * FROM videos WHERE video_id = ?", (vid,)).fetchone()
        if vrow is None:
            out[vid] = {"status": "MISSING", "reason": "no video row"}
            continue
        speakers = [dict(s) for s in cur.execute(
            "SELECT * FROM video_speakers WHERE video_id = ? ORDER BY speaking_sec DESC", (vid,))]
        turns = [dict(t) for t in cur.execute(
            "SELECT * FROM scenario_turns WHERE video_id = ? ORDER BY turn_index ASC, start_ms ASC", (vid,))]
        if not turns:
            out[vid] = {"status": "NO_TURNS", "reason": "scenario_turns empty"}
            continue
        for t in turns:
            t["words"] = json.loads(t.get("words_json") or "[]")
            t.pop("words_json", None)
            t["speaker_role"] = _canon_role(t.get("speaker_role"))
        scenario = {
            "schema_version": "2.0",
            "video_id": vid,
            "channel": vrow["channel"] or "SCDawah Arabic",
            "title_ar": vrow["title_ar"] or "",
            "youtube_url": vrow["youtube_url"] or f"https://www.youtube.com/watch?v={vid}",
            "duration_sec": float(vrow["duration_sec"] or 0.0),
            "detected_language": vrow["detected_language"] or "en",
            "speakers": [
                {
                    "speaker_id": s["speaker_id"],
                    "role": _canon_role(s["role"]),
                    "display_name_en": s["display_name_en"] or "",
                    "display_name_ar": s["display_name_ar"] or "",
                    "speaking_sec": float(s["speaking_sec"] or 0.0),
                    "turn_count": int(s["turn_count"] or 0),
                    "mean_rms_db": float(s["mean_rms_db"] or -24.0),
                }
                for s in speakers
            ],
            "turns": turns,
            "metrics": {
                "duration_sec": float(vrow["duration_sec"] or 0.0),
                "speakers_count": int(vrow["speakers_count"] or 0),
                "turns_count": int(vrow["turns_count"] or 0),
                "spoken_words_count": int(vrow["spoken_words_count"] or 0),
                "overlap_windows_count": int(vrow["overlap_windows_count"] or 0),
                "pedestrian_turns_count": int(vrow["pedestrian_turns_count"] or 0),
            },
        }
        (scen_out / f"{vid}.json").write_text(json.dumps(scenario, ensure_ascii=False, indent=1), encoding="utf-8")
        out[vid] = {"status": "BACKFILLED", "path": str(scen_out / f"{vid}.json"),
                    "turns": len(turns), "speakers": len(speakers), "duration_sec": scenario["duration_sec"]}
    conn.close()
    return out


def main():
    args = [a for a in sys.argv[1:]]
    video_ids = [a for a in args if not a.startswith("--") and not a.startswith("-")]
    ws_arg = None
    if "--ws" in args:
        i = args.index("--ws")
        ws_arg = Path(args[i + 1]) if i + 1 < len(args) else None
    ws = ws_arg or DEFAULT_WS
    video_ids = video_ids or load_video_ids()
    out = backfill(video_ids, ws)
    n = sum(1 for v in out.values() if v.get("status") == "BACKFILLED")
    print(f"[+] Backfilled {n}/{len(out)} videos into {ws / 'scenarios'} (DB: {DB_PATH})")
    for vid, info in out.items():
        print(f"  - {vid}: {info.get('status')} turns={info.get('turns', '-')} dur={info.get('duration_sec', '-')}")


if __name__ == "__main__":
    main()
