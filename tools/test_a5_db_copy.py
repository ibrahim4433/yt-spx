#!/usr/bin/env python3
"""Test Fix A5 on a COPY of data/dawah_knowledge.db — never touches the real clone.
Uses the exact DB column name extracted from the DB (tools/a5_spellings.json)."""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Exact column name from the DB, extracted by fix_a5_spellings.py
DB_COL = json.loads((REPO / "tools" / "a5_spellings.json").read_text(encoding="utf-8"))[
    "db_interlocator_speaker_id_col_exact"
]

from db.database import get_db_connection, _insert_qa_unit_in_tx  # noqa

tmp = REPO / "reports" / "test_a5_db.sqlite"
shutil.copy(REPO / "data" / "dawah_knowledge.db", str(tmp))

conn = get_db_connection(tmp)
cols = [r[1] for r in conn.execute("PRAGMA table_info(qa_units)").fetchall()]
assert DB_COL in cols, f"expected {DB_COL!r} in qa_units; has={[c for c in cols if 'interloc' in c]}"
row = conn.execute(
    f"SELECT id, video_id, {DB_COL}, daee_speaker_id FROM qa_units WHERE id='jt4BisLcSoI_001'"
).fetchone()
print("BEFORE:", dict(row) if row else None)
unit = {
    "id": "jt4BisLcSoI_001", "video_id": "jt4BisLcSoI",
    DB_COL: "", "daee_speaker_id": "SPK_00",
    "dialogue_exchange": [], "source_turn_ids": [], "tags": [],
}
_insert_qa_unit_in_tx(conn, unit)
row2 = conn.execute(f"SELECT {DB_COL} FROM qa_units WHERE id='jt4BisLcSoI_001'").fetchone()
print("AFTER :", row2[0] if row2 else None)
vspk = conn.execute(
    "SELECT speaker_id, role FROM video_speakers WHERE video_id=? AND role=?",
    ("jt4BisLcSoI", "INTERLOCUTOR"),
).fetchall()
print("GT interlocator speaker rows:", [(r[0], r[1]) for r in vspk])
ok = row2[0] in {r[0] for r in vspk}
print("A5 OK" if ok else "A5 FAIL")
conn.close()
tmp.unlink(missing_ok=True)
sys.exit(0 if ok else 1)
