#!/usr/bin/env python3
"""Compute exact character diffs between 'interloc*speaker_id' strings in code vs the DB column."""
import json
from pathlib import Path

db_col = json.loads(Path("tools/a5_spellings.json").read_text(encoding="utf-8"))["db_interlocator_speaker_id_col_exact"]

def close(a, b):
    if a == b:
        return 0
    # find the best alignment by trimming common prefix/suffix
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return (len(a) - p - s, len(b) - p - s)

for f in [
    "db/database.py",
    "tools/card_rating_system/audit_cards.py",
    "core/scenario_extractor.py",
    "tools/card_rating_system/AGENT_PROMPT.md",
    "tools/card_rating_system/RUBRIC.md",
]:
    txt = Path(f).read_text(encoding="utf-8")
    seen = {}
    for line_no, line in enumerate(txt.splitlines(), 1):
        for word in line.split():
            w = word.strip("'\"`()[]{}:,.= ").lower()
            if "interloc" in w and "speaker" in w:
                seen[w] = line_no
    for w, ln in sorted(seen.items()):
        d = close(w, db_col)
        if w != db_col:
            print(f"DIFF {f}:{ln}: '{w}' vs db_col '{db_col}' diff_counts={d}")
            print(f"   w_codepoints: {[ord(c) for c in w[:20]]}")
            print(f"   db_codepoints: {[ord(c) for c in db_col[:20]]}")
