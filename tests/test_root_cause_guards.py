#!/usr/bin/env python3
"""
tests/test_root_cause_guards.py
Validates the root-cause fixes across:
  1. Foreign script / Cyrillic / CJK sanitization in text fields
  2. Da'ee attribution protection against generic placeholders ('المحاور', etc.)
  3. Speaker Inversion protection in core/scenario_extractor.py
  4. Dialogue exchange turn-level timestamp and localized speaker label enrichment
"""

import unittest
import sqlite3
import json
from pathlib import Path

from db.database import (
    sanitize_text_field,
    _sanitize_and_validate_qa_unit,
    _insert_qa_unit_in_tx,
    init_db,
    get_db_connection,
)
from core.scenario_extractor import (
    sanitize_arabic_fields,
    anchor_and_enrich_extracted_unit,
)


class TestRootCauseGuards(unittest.TestCase):

    def test_foreign_script_sanitization(self):
        """Ensures Russian Cyrillic and Chinese CJK punctuation are completely removed or mapped."""
        sample_ru = "تناقضات الكتاب المقدس ت дисквалифицируют его как совершенный божественный текст."
        cleaned_ru = sanitize_text_field(sample_ru)
        self.assertNotIn("дисквалифицируют", cleaned_ru)
        self.assertIn("تسقط عنه صفة النص الإلهي الكامل المعصوم", cleaned_ru)

        sample_ped = "هل هذا يعد إباحة لل педوفيليا؟"
        cleaned_ped = sanitize_text_field(sample_ped)
        self.assertNotIn("педوفيليا", cleaned_ped)
        self.assertIn("البيدوفيليا", cleaned_ped)

        sample_cjk = "بالضبط. هذا هو السؤال، صحيح؟。，？，。"
        cleaned_cjk = sanitize_text_field(sample_cjk)
        self.assertNotIn("。", cleaned_cjk)
        self.assertNotIn("，", cleaned_cjk)

    def test_scenario_extractor_sanitizer(self):
        """Ensures core/scenario_extractor.py sanitize_arabic_fields purges foreign scripts."""
        item = {
            "question_ar": "سؤال عن педوفيليا",
            "analogy_or_metaphor_ar": "قياس مع дисквалифицируют его как совершенный божественный текст",
            "socratic_question_ar": "أليس كذلك؟。，？",
            "logical_premises_ar": ["المقدمة الأولى。"]
        }
        cleaned = sanitize_arabic_fields(item)
        self.assertNotIn("педوفيليا", cleaned["question_ar"])
        self.assertNotIn("дисквалифицируют", cleaned["analogy_or_metaphor_ar"])
        self.assertNotIn("。", cleaned["socratic_question_ar"])
        self.assertNotIn("。", cleaned["logical_premises_ar"][0])

    def test_speaker_inversion_protection(self):
        """
        Ensures that if the LLM points answer_turn_id to an INTERLOCUTOR turn,
        anchor_and_enrich_extracted_unit strictly rejects it and assigns MAIN_DAEE.
        """
        scenario = {
            "video_id": "test_vid_001",
            "youtube_url": "https://www.youtube.com/watch?v=test_vid_001",
            "title_ar": "حوار بين هاشم ومحاور مسيحي",
            "speakers": [
                {
                    "speaker_id": "SPK_00",
                    "role": "MAIN_DAEE",
                    "profile_id": "daee_hashim",
                    "display_name_en": "Hashim",
                    "display_name_ar": "هاشم"
                },
                {
                    "speaker_id": "SPK_01",
                    "role": "INTERLOCUTOR",
                    "profile_id": None,
                    "display_name_en": "Interlocutor 1",
                    "display_name_ar": "المحاور 1"
                }
            ],
            "turns": [
                {
                    "turn_id": "test_vid_001_T0001",
                    "turn_index": 1,
                    "speaker_id": "SPK_01",
                    "speaker_role": "INTERLOCUTOR",
                    "speaker_label_en": "Interlocutor 1",
                    "speaker_label_ar": "المحاور 1",
                    "start_ms": 10000,
                    "end_ms": 15000,
                    "start_fmt": "00:10",
                    "end_fmt": "00:15",
                    "text_original": "Is Jesus God in the Bible?"
                },
                {
                    # LLM erroneously picked this interlocutor turn as answer_turn_id!
                    "turn_id": "test_vid_001_T0002",
                    "turn_index": 2,
                    "speaker_id": "SPK_01",
                    "speaker_role": "INTERLOCUTOR",
                    "speaker_label_en": "Interlocutor 1",
                    "speaker_label_ar": "المحاور 1",
                    "start_ms": 16000,
                    "end_ms": 20000,
                    "start_fmt": "00:16",
                    "end_fmt": "00:20",
                    "text_original": "I believe he is God."
                }
            ]
        }

        raw_unit = {
            "question_turn_id": "test_vid_001_T0001",
            "answer_turn_id": "test_vid_001_T0002",  # Bad turn pointing to Interlocutor
            "question_en": "Is Jesus God in the Bible?",
            "question_ar": "هل عيسى هو الله في الكتاب المقدس؟",
            "core_answer_summary_en": "God alone is all-knowing and eternal.",
            "core_answer_summary_ar": "الله وحده هو العليم المستغني عن الخلق.",
        }

        enriched = anchor_and_enrich_extracted_unit(
            raw_unit=raw_unit,
            scenario=scenario,
            episode_index=1,
            window_start_ms=0,
            unit_seq_num=1
        )

        # MUST NOT be assigned 'المحاور 1'! MUST be assigned 'هاشم' / 'daee_hashim'!
        self.assertEqual(enriched["daee_profile_id"], "daee_hashim")
        self.assertEqual(enriched["daee_name_ar"], "هاشم")
        self.assertEqual(enriched["daee_name_en"], "Hashim")
        self.assertEqual(enriched["daee_speaker_id"], "SPK_00")

    def test_db_ingestion_sanitization_and_enrichment(self):
        """
        Tests that when data is inserted into SQLite, _sanitize_and_validate_qa_unit
        automatically sanitizes text and enriches dialogue exchange.
        """
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # create mock tables
        conn.execute("""
            CREATE TABLE videos (
                video_id TEXT PRIMARY KEY,
                primary_daee_hint TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE video_speakers (
                video_id TEXT,
                speaker_id TEXT,
                role TEXT,
                profile_id TEXT,
                display_name_en TEXT,
                display_name_ar TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE speaker_profiles (
                profile_id TEXT PRIMARY KEY,
                name_ar TEXT,
                name_en TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE scenario_turns (
                turn_id TEXT PRIMARY KEY,
                start_ms INTEGER,
                start_fmt TEXT,
                speaker_role TEXT,
                speaker_label_en TEXT,
                speaker_label_ar TEXT
            );
        """)

        conn.execute("INSERT INTO videos VALUES ('vid123', 'daee_uthman');")
        conn.execute("INSERT INTO video_speakers VALUES ('vid123', 'SPK_00', 'MAIN_DAEE', 'daee_uthman', 'Sheikh Uthman Ibn Farooq', 'الشيخ عثمان بن فاروق');")
        conn.execute("INSERT INTO speaker_profiles VALUES ('daee_uthman', 'الشيخ عثمان بن فاروق', 'Sheikh Uthman Ibn Farooq');")
        conn.execute("INSERT INTO scenario_turns VALUES ('T01', 65000, '01:05', 'INTERLOCUTOR', 'Interlocutor', 'المحاور');")
        conn.execute("INSERT INTO scenario_turns VALUES ('T02', 72000, '01:12', 'MAIN_DAEE', 'Sheikh Uthman', 'الشيخ عثمان بن فاروق');")

        raw_unit = {
            "id": "vid123_001",
            "video_id": "vid123",
            "daee_name_ar": "المحاور 1",  # Erroneous generic name
            "daee_profile_id": "",
            "question_ar": "سؤال مع педوفيليا ورموز。",
            "dialogue_exchange": [
                {"turn_id": "T01", "role": "INTERLOCUTOR", "text_ar": "سؤال"},
                {"turn_id": "T02", "role": "MAIN_DAEE", "text_ar": "جواب"}
            ]
        }

        sanitized = _sanitize_and_validate_qa_unit(conn, raw_unit)

        # 1. Daee name resolved
        self.assertEqual(sanitized["daee_name_ar"], "الشيخ عثمان بن فاروق")
        self.assertEqual(sanitized["daee_profile_id"], "daee_uthman")

        # 2. Foreign script cleaned
        self.assertNotIn("педوفيليا", sanitized["question_ar"])
        self.assertIn("البيدوفيليا", sanitized["question_ar"])
        self.assertNotIn("。", sanitized["question_ar"])

        # 3. Dialogue exchange turns enriched
        ex = sanitized["dialogue_exchange"]
        self.assertEqual(ex[0]["timestamp_formatted"], "01:05")
        self.assertEqual(ex[1]["timestamp_formatted"], "01:12")
        self.assertEqual(ex[1]["speaker_label_ar"], "الشيخ عثمان بن فاروق")


if __name__ == "__main__":
    unittest.main()
