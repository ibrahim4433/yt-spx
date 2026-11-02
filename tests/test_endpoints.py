#!/usr/bin/env python3
"""
tests/test_endpoints.py
Automated FastAPI V2 endpoint test suite covering:
  1. Production clean-slate V2 database verification (2,640 cataloged videos, 0 noisy V1 captions)
  2. Full V2 scenario + diarized Q&A ingestion & REST endpoint verification against an isolated temp DB
     (/api/stats, /api/search, /api/speakers, /api/categories, /api/interlocutors,
      /api/videos, /api/videos/{id}/scenario, /api/transcript-search, /api/chat, and / HTML)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
import db.database as db_mod
from web.app import app


class TestV2WebEndpoints(unittest.TestCase):
    def test_01_production_clean_slate_v2_database_and_ui_shell(self):
        """Verifies the reset production DB has 2,640 cataloged videos, 0 V1 records, and serves the V2 Studio UI."""
        client = TestClient(app)

        resp_stats = client.get("/api/stats")
        self.assertEqual(resp_stats.status_code, 200)
        stats = resp_stats.json()
        self.assertEqual(stats["pipeline_version"], "2.0-diarized-scenario")
        self.assertEqual(stats["total_videos"], 2640)
        self.assertGreaterEqual(stats["total_qa"], 0)

        resp_vids = client.get("/api/videos?limit=10")
        self.assertEqual(resp_vids.status_code, 200)
        vdata = resp_vids.json()
        self.assertEqual(vdata["total"], 2640)
        self.assertEqual(len(vdata["videos"]), 10)

        resp_chat = client.get("/api/chat?q=Who+created+God")
        self.assertEqual(resp_chat.status_code, 200)
        self.assertIn(resp_chat.json()["status"], ("success", "no_match"))

        resp_html = client.get("/")
        self.assertEqual(resp_html.status_code, 200)
        html = resp_html.text
        self.assertIn("Dawah Dialectics", html)
        self.assertIn('id="scenario-studio-dialog"', html)
        self.assertIn('closedby="any"', html)
        self.assertIn('id="scenarios-tab"', html)
        self.assertIn('id="speaker-pills"', html)

    def test_02_full_v2_scenario_and_dossier_endpoints_with_isolated_db(self):
        """Ingests a multi-speaker overlapping debate into a temp SQLite DB and verifies every V2 API route."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = Path(tmpdir) / "test_v2.db"
            db_mod.init_db(db_path=temp_db)

            vid = "CsWdrUXYnWg"
            scenario_payload = {
                "schema_version": "2.0",
                "video_id": vid,
                "channel": "SCDawah Arabic",
                "title_ar": "الشيخ عثمان بن فاروق يناظر مبشرين حول التثليث والسببية",
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "duration_sec": 3600.0,
                "detected_language": "en",
                "metrics": {"overlap_windows_count": 2, "overlap_ratio": 0.05},
                "speakers": [
                    {
                        "speaker_id": "speaker_0",
                        "role": "MAIN_DAEE",
                        "interlocutor_index": 0,
                        "profile_id": "uthman_ibn_farooq",
                        "display_name_en": "Sheikh Uthman Ibn Farooq",
                        "display_name_ar": "الشيخ عثمان بن فاروق",
                        "speaking_sec": 2100.0,
                        "turn_count": 2,
                        "mean_rms_db": -17.2,
                    },
                    {
                        "speaker_id": "speaker_1",
                        "role": "INTERLOCUTOR",
                        "interlocutor_index": 1,
                        "profile_id": None,
                        "display_name_en": "Interlocutor 1",
                        "display_name_ar": "المحاور 1",
                        "speaking_sec": 950.0,
                        "turn_count": 1,
                        "mean_rms_db": -18.8,
                    },
                    {
                        "speaker_id": "speaker_2",
                        "role": "PEDESTRIAN",
                        "interlocutor_index": 0,
                        "profile_id": None,
                        "display_name_en": "Pedestrian Interruption",
                        "display_name_ar": "مقاطعة عابرة من المارة",
                        "speaking_sec": 2.0,
                        "turn_count": 1,
                        "mean_rms_db": -34.0,
                    },
                ],
                "turns": [
                    {
                        "turn_id": f"{vid}_T0001",
                        "turn_index": 0,
                        "episode_index": 1,
                        "speaker_id": "speaker_1",
                        "speaker_role": "INTERLOCUTOR",
                        "speaker_label_en": "Interlocutor 1",
                        "speaker_label_ar": "المحاور 1",
                        "start_ms": 60000,
                        "end_ms": 74000,
                        "start_fmt": "01:00",
                        "end_fmt": "01:14",
                        "language": "en",
                        "text_original": "Who created God if everything has a cause?",
                        "text_ar": "",
                        "is_overlap": True,
                        "overlap_group_id": "OV_01",
                        "is_pedestrian_interruption": False,
                        "confidence": 0.95,
                        "words": [],
                    },
                    {
                        "turn_id": f"{vid}_T0002",
                        "turn_index": 1,
                        "episode_index": 1,
                        "speaker_id": "speaker_0",
                        "speaker_role": "MAIN_DAEE",
                        "speaker_label_en": "Sheikh Uthman Ibn Farooq",
                        "speaker_label_ar": "الشيخ عثمان بن فاروق",
                        "start_ms": 61000,
                        "end_ms": 98000,
                        "start_fmt": "01:01",
                        "end_fmt": "01:38",
                        "language": "en",
                        "text_original": "Asking who created the Uncaused Necessary Creator is a category fallacy.",
                        "text_ar": "",
                        "is_overlap": True,
                        "overlap_group_id": "OV_01",
                        "is_pedestrian_interruption": False,
                        "confidence": 0.97,
                        "words": [],
                    },
                    {
                        "turn_id": f"{vid}_T0003",
                        "turn_index": 2,
                        "episode_index": 1,
                        "speaker_id": "speaker_2",
                        "speaker_role": "PEDESTRIAN",
                        "speaker_label_en": "Pedestrian Interruption",
                        "speaker_label_ar": "مقاطعة عابرة من المارة",
                        "start_ms": 99000,
                        "end_ms": 101000,
                        "start_fmt": "01:39",
                        "end_fmt": "01:41",
                        "language": "en",
                        "text_original": "Watch out for the bike!",
                        "text_ar": "",
                        "is_overlap": False,
                        "overlap_group_id": "",
                        "is_pedestrian_interruption": True,
                        "confidence": 0.88,
                        "words": [],
                    },
                ],
            }

            qa_units = [
                {
                    "id": f"{vid}_001",
                    "video_id": vid,
                    "channel": "SCDawah Arabic",
                    "video_title_ar": scenario_payload["title_ar"],
                    "youtube_url": scenario_payload["youtube_url"],
                    "timestamp_url": f"{scenario_payload['youtube_url']}&t=60s",
                    "timestamp_seconds": 60,
                    "timestamp_formatted": "01:00",
                    "answer_timestamp_seconds": 61,
                    "answer_timestamp_formatted": "01:01",
                    "episode_index": 1,
                    "daee_speaker_id": "speaker_0",
                    "daee_profile_id": "uthman_ibn_farooq",
                    "daee_name_en": "Sheikh Uthman Ibn Farooq",
                    "daee_name_ar": "الشيخ عثمان بن فاروق",
                    "interlocutor_speaker_id": "speaker_1",
                    "topic_category": "Theology & God's Existence",
                    "topic_subcategory": "First Cause & Contingency",
                    "interlocutor_type": "Atheist / Materialist",
                    "question_en": "If everything has a cause, who created God?",
                    "question_ar": "إذا كان لكل شيء سبب، فمن خلق الله؟",
                    "core_answer_summary_en": "Only things that begin to exist require a cause; God is the Necessary Uncaused Being.",
                    "core_answer_summary_ar": "كل ما له بداية وجود يحتاج إلى مسبب، أما الله فهو واجب الوجود غير المخلوق.",
                    "logical_premises": [
                        "Whatever begins to exist has an external cause.",
                        "An infinite regress of contingent causes produces nothing.",
                        "Therefore an Uncaused Necessary Creator exists.",
                    ],
                    "logical_premises_ar": [
                        "كل حادث له بداية لا بد له من محدث.",
                        "التسلسل اللانهائي للممكنات مستحيل عقلاً.",
                        "إذن لا بد من خالق واجب الوجود أزلي غير مخلوق.",
                    ],
                    "fallacy_or_misconception_exposed": "Category Fallacy",
                    "fallacy_or_misconception_exposed_ar": "مغالطة الخلط المقولي",
                    "analogy_or_metaphor": "Asking who baked the unbaked baker.",
                    "analogy_or_metaphor_ar": "كسؤال من خبز الخباز.",
                    "socratic_question": "If the creator needed a creator, would we ever get a first effect?",
                    "socratic_question_ar": "لو احتاج الخالق إلى خالق إلى ما لا نهاية، فهل كان سيوجد الكون أصلاً؟",
                    "interlocutor_verbatim_quote": "Who created God if everything has a cause?",
                    "interlocutor_verbatim_quote_ar": "من خلق الله إذا كان لكل شيء سبب؟",
                    "verbatim_quote": "Asking who created the Uncaused Necessary Creator is a category fallacy.",
                    "verbatim_quote_ar": "السؤال عمن خلق واجب الوجود غير المخلوق هو مغالطة مقولية.",
                    "dialogue_exchange": [
                        {
                            "turn_id": f"{vid}_T0001",
                            "role": "INTERLOCUTOR",
                            "speaker_label_en": "Interlocutor 1",
                            "speaker_label_ar": "المحاور 1",
                            "timestamp_seconds": 60,
                            "timestamp_formatted": "01:00",
                            "move_type": "Objection",
                            "text_en": "Who created God?",
                            "text_ar": "من خلق الله؟",
                        },
                        {
                            "turn_id": f"{vid}_T0002",
                            "role": "MAIN_DAEE",
                            "speaker_label_en": "Sheikh Uthman Ibn Farooq",
                            "speaker_label_ar": "الشيخ عثمان بن فاروق",
                            "timestamp_seconds": 61,
                            "timestamp_formatted": "01:01",
                            "move_type": "Refutation",
                            "text_en": "Category fallacy: Necessary Being is uncaused.",
                            "text_ar": "مغالطة مقولية: واجب الوجود غير معلول.",
                        },
                    ],
                    "source_turn_ids": [f"{vid}_T0001", f"{vid}_T0002"],
                    "tags": ["first cause", "contingency"],
                }
            ]

            db_mod.ingest_scenario_payload(scenario_payload, db_path=temp_db)
            db_mod.ingest_extraction_payload(vid, qa_units, db_path=temp_db)

            with patch.object(db_mod, "DB_PATH", temp_db):
                client = TestClient(app)

                # 1. Telemetry stats
                st = client.get("/api/stats").json()
                self.assertEqual(st["total_qa"], 1)
                self.assertEqual(st["stage1_completed"], 1)
                self.assertEqual(st["stage2_completed"], 1)
                self.assertEqual(st["cleaned_hours"], 1.0)
                self.assertEqual(st["overlaps_resolved"], 1)
                self.assertEqual(st["pedestrian_isolated"], 1)

                # 2. Facet breakdowns
                spks = client.get("/api/speakers").json()["speakers"]
                self.assertEqual(len(spks), 1)
                self.assertEqual(spks[0]["speaker_key"], "uthman_ibn_farooq")

                # 3. Search with Da'ee speaker filter & video_id filter
                sr = client.get(f"/api/search?q=cause&speaker=uthman_ibn_farooq&video_id={vid}").json()
                self.assertEqual(sr["total"], 1)
                self.assertEqual(len(sr["results"][0]["dialogue_exchange"]), 2)

                # 4. Scenario Studio endpoint (with & without pedestrian filtering)
                scen_all = client.get(f"/api/videos/{vid}/scenario?include_pedestrians=true").json()
                self.assertEqual(len(scen_all["turns"]), 3)
                scen_clean = client.get(f"/api/videos/{vid}/scenario?include_pedestrians=false").json()
                self.assertEqual(len(scen_clean["turns"]), 2)

                # 5. Direct FTS5 Scenario Turn Search
                ts_hits = client.get("/api/transcript-search?q=category+fallacy&role=MAIN_DAEE").json()
                self.assertEqual(ts_hits["total"], 1)
                self.assertEqual(ts_hits["turns"][0]["turn_id"], f"{vid}_T0002")

                # 6. Arabic Smart Assistant Chat
                chat = client.get("/api/chat?q=%D9%85%D9%86+%D8%AE%D9%84%D9%82+%D8%A7%D9%84%D9%84%D9%87").json()
                self.assertEqual(chat["status"], "success")
                self.assertEqual(chat["response_package"]["daee_name"], "الشيخ عثمان بن فاروق")
                self.assertIn("youtube.com", chat["response_package"]["timestamp_url"])

    def test_03_database_initialization_and_search_without_fts5(self):
        """Verifies init_db and search degrade gracefully to LIKE queries when FTS5 is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = Path(tmpdir) / "no_fts5.db"
            with patch("db.database._supports_fts5", return_value=False):
                # Should not raise sqlite3.OperationalError: no such module: fts5
                db_mod.init_db(temp_db)

                # Ingest a scenario turn and verify LIKE search
                db_mod.ingest_scenario_payload(
                    {
                        "video_id": "test_no_fts",
                        "speakers": [{"speaker_id": "SPK1", "speaker_role": "MAIN_DAEE", "speaking_sec": 10.0}],
                        "turns": [{
                            "turn_id": "T1",
                            "turn_index": 1,
                            "speaker_id": "SPK1",
                            "speaker_role": "MAIN_DAEE",
                            "start_ms": 0,
                            "end_ms": 5000,
                            "text_original": "Is morality objective?",
                            "text_ar": "هل الأخلاق موضوعية؟",
                        }],
                    },
                    db_path=temp_db,
                )
                res = db_mod.search_scenario_turns("morality", db_path=temp_db)
                self.assertEqual(res["total"], 1)
                self.assertEqual(res["turns"][0]["turn_id"], "T1")


if __name__ == "__main__":
    unittest.main()
