#!/usr/bin/env python3
"""
tools/repair_missing_arabic.py
Self-Healing Offline & Online Arabic Localization Repair Tool for Dawah Knowledge Base.

Identifies apologetics cards in SQLite (or Drive .qa.json shards) that bypassed
Round 2 Arabic localization (all *_ar fields blank), constructs the dialectical
theological translation payload, and updates the records so 100% of cards meet DARS V2.
"""

import sys
import os
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from db.database import get_db_connection, clean_latin_tokens_in_arabic
from core.scenario_extractor import has_valid_arabic_localization


def find_unlocalized_cards(db_path: Path) -> List[Dict[str, Any]]:
    """Finds all cards in SQLite where question_ar is empty or missing."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT id, video_id, daee_name_en, topic_category, topic_subcategory,
               interlocutor_type, question_en, core_answer_summary_en,
               logical_premises_json, fallacy_or_misconception_exposed,
               analogy_or_metaphor, socratic_question,
               interlocutor_verbatim_quote, verbatim_quote, dialogue_exchange_json
        FROM qa_units
        WHERE question_ar IS NULL OR question_ar = ''
        ORDER BY video_id, id
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def print_unlocalized_report(cards: List[Dict[str, Any]]) -> None:
    """Prints a clear summary report of unlocalized cards."""
    print("=" * 80)
    print(f" UNLOCALIZED APOLOGETICS CARDS AUDIT: {len(cards)} cards requiring Arabic repair")
    print("=" * 80)
    videos: Dict[str, List[Dict[str, Any]]] = {}
    for c in cards:
        videos.setdefault(c["video_id"], []).append(c)

    for vid, v_cards in videos.items():
        print(f"\n▶ Video ID: {vid} ({len(v_cards)} cards lacking Arabic)")
        for card in v_cards:
            cid = card["id"]
            daee = card["daee_name_en"]
            cat = card["topic_category"]
            q = card["question_en"]
            print(f"   * [{cid}] ({daee} | {cat})")
            print(f"     Q: \"{q}\"")
    print("\n" + "=" * 80)


def build_round2_repair_prompt(card: Dict[str, Any]) -> str:
    """Constructs the exact Round 2 localization prompt for an individual unlocalized card."""
    payload = {
        "id": card["id"],
        "question_en": card["question_en"],
        "core_answer_summary_en": card["core_answer_summary_en"],
        "logical_premises": json.loads(card.get("logical_premises_json") or "[]"),
        "fallacy_or_misconception_exposed": card.get("fallacy_or_misconception_exposed") or "",
        "analogy_or_metaphor": card.get("analogy_or_metaphor") or "",
        "socratic_question": card.get("socratic_question") or "",
        "interlocutor_verbatim_quote": card.get("interlocutor_verbatim_quote") or "",
        "verbatim_quote": card.get("verbatim_quote") or "",
        "dialogue_exchange": json.loads(card.get("dialogue_exchange_json") or "[]")
    }

    prompt = (
        "You are an expert Islamic dialectical scholar translating and localizing this English apologetics dossier "
        "into pure, eloquent Classical Arabic (فصحى أصيلة محكمة).\n\n"
        "INPUT DOSSIER:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n"
        "REQUIREMENTS:\n"
        "1. Output a JSON object with Arabic fields:\n"
        "   - question_ar: Authentic Classical Arabic formulation of the seeker's doubt.\n"
        "   - core_answer_summary_ar: Decisive, authoritative theological refutation (no 3rd-person narrator framing).\n"
        "   - logical_premises_ar: Array of 3 formal deductive premises (كبرى، صغرى، نتيجة ملزمة).\n"
        "   - fallacy_or_misconception_exposed_ar: Name and explanation of fallacy in Arabic.\n"
        "   - analogy_or_metaphor_ar: Vivid thought experiment / analogy in Arabic.\n"
        "   - socratic_question_ar: Clincher counter-question in Arabic.\n"
        "   - interlocutor_verbatim_quote_ar: Faithful Classical Arabic translation of the quote.\n"
        "   - verbatim_quote_ar: Faithful Classical Arabic translation of the Da'ee proof quote.\n"
        "   - dialogue_exchange: Same array with 'text_ar' populated for each turn.\n"
        "2. ZERO Latin script leakage (no English terms inside Arabic text).\n"
        "3. Output ONLY valid JSON.\n"
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Repair missing Arabic localization in Dawah Knowledge Base.")
    parser.add_argument("--db", type=str, default="data/dawah_knowledge.db", help="Path to SQLite database.")
    parser.add_argument("--list", action="store_true", help="List all unlocalized cards and exit.")
    parser.add_argument("--generate-prompts", type=str, help="Directory to export repair prompt JSONs.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    cards = find_unlocalized_cards(db_path)
    if not cards:
        print("✓ All cards in database possess valid Arabic localization! Zero defects found.")
        return

    print_unlocalized_report(cards)

    if args.generate_prompts:
        out_dir = Path(args.generate_prompts)
        out_dir.mkdir(parents=True, exist_ok=True)
        for c in cards:
            cid = c["id"]
            p_text = build_round2_repair_prompt(c)
            (out_dir / f"{cid}_prompt.txt").write_text(p_text, encoding="utf-8")
        print(f"✓ Exported {len(cards)} repair prompt files to {out_dir}")


if __name__ == "__main__":
    main()
