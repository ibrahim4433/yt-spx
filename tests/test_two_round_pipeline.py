#!/usr/bin/env python3
"""
tests/test_two_round_pipeline.py
Unit & integration tests for the Two-Round Stage 2 Local GPU extraction pipeline:
  - Round 1 English extraction prompt & Round 2 Arabic theological verification prompt formatting
  - LocalGPUClient & ExtractionClient two-round signatures, Qwen3 default with Qwen2.5 fallback
  - Stage2ScenarioExtractor two-round execution with intermediate .en.json checkpointing
  - Intermediate .en.json resume directly from Round 2 without re-running Round 1
  - Database schema: analogies_json, scriptural_citations_json, dialogue_trajectory_turns_count & FTS5 search
  - Expanded dialogue trajectory (6-12 turns) and scriptural citations enrichment
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.drive_sync import SharedWorkspace
from core.extractor_prompt import (
    ROUND1_ENGLISH_EXTRACTION_PROMPT,
    USER_ROUND1_PROMPT_TEMPLATE,
    ROUND2_ARABIC_VERIFICATION_PROMPT,
    USER_ROUND2_PROMPT_TEMPLATE,
)
from core.colab_runner import LocalGPUClient, ExtractionClient
from core.scenario_extractor import Stage2ScenarioExtractor
from db.database import (
    init_db,
    insert_qa_unit,
    search_qa,
    _hydrate_qa_row,
)


class TestTwoRoundPromptsAndFormatting(unittest.TestCase):
    def test_round1_prompt_formatting_and_constraints(self):
        """Validates Round 1 prompt contains dialectical requirements and formats cleanly."""
        self.assertIn("ROUND 1", ROUND1_ENGLISH_EXTRACTION_PROMPT)
        self.assertIn("150-250", ROUND1_ENGLISH_EXTRACTION_PROMPT)
        self.assertIn("analogies", ROUND1_ENGLISH_EXTRACTION_PROMPT)
        self.assertIn("scriptural_citations", ROUND1_ENGLISH_EXTRACTION_PROMPT)
        self.assertIn("dialogue_exchange", ROUND1_ENGLISH_EXTRACTION_PROMPT)
        self.assertIn("6 to 12 turns", ROUND1_ENGLISH_EXTRACTION_PROMPT)

        formatted_r1 = USER_ROUND1_PROMPT_TEMPLATE.format(
            title_ar="مناظرة الشارع الكبرى",
            youtube_url="https://youtube.com/watch?v=test_r1",
            episode_index=1,
            start_fmt="00:00",
            end_fmt="05:30",
            speakers_summary="MAIN_DAEE vs INTERLOCUTOR",
            target_daee_name="Uthman Ibn Farooq",
            target_daee_id="SPK_01",
            transcript_text="[00:10] (SPK_02): Can you prove God exists?",
        )
        self.assertIn("مناظرة الشارع الكبرى", formatted_r1)
        self.assertIn("Uthman Ibn Farooq", formatted_r1)
        self.assertIn("[00:10] (SPK_02)", formatted_r1)

    def test_round2_prompt_formatting_and_theological_rules(self):
        """Validates Round 2 prompt contains Classical Arabic verification and anti-hallucination rules."""
        self.assertIn("ROUND 2", ROUND2_ARABIC_VERIFICATION_PROMPT)
        self.assertIn("فصحى", ROUND2_ARABIC_VERIFICATION_PROMPT)
        self.assertIn("dialogue_exchange", ROUND2_ARABIC_VERIFICATION_PROMPT)
        self.assertIn("scriptural_citations", ROUND2_ARABIC_VERIFICATION_PROMPT)

        formatted_r2 = USER_ROUND2_PROMPT_TEMPLATE.format(
            title_ar="مناظرة الشارع الكبرى",
            youtube_url="https://youtube.com/watch?v=test_r2",
            episode_index=1,
            start_fmt="00:00",
            end_fmt="05:30",
            target_daee_name="Uthman Ibn Farooq",
            target_daee_id="SPK_01",
            transcript_text="[00:10] (SPK_02): Can you prove God exists?",
            english_units_json='[{"question_en": "Can you prove God exists?"}]',
        )
        self.assertIn("ROUND 1 EXTRACTED ENGLISH DOSSIERS", formatted_r2)
        self.assertIn("Can you prove God exists?", formatted_r2)


class TestClientTwoRoundMethods(unittest.TestCase):
    def test_client_methods_and_defaults(self):
        """Verifies LocalGPUClient and ExtractionClient expose two-round methods."""
        import inspect

        sig_gpu = inspect.signature(LocalGPUClient.__init__)
        self.assertEqual(sig_gpu.parameters["gguf_filename"].default, "Qwen2.5-14B-Instruct-Q5_K_M.gguf")
        self.assertEqual(sig_gpu.parameters["gguf_repo"].default, "bartowski/Qwen2.5-14B-Instruct-GGUF")
        self.assertEqual(sig_gpu.parameters["preferred_gguf_repo"].default, "Qwen/Qwen3-14B-GGUF")
        self.assertEqual(sig_gpu.parameters["preferred_gguf_filename"].default, "Qwen3-14B-Q5_K_M.gguf")

        self.assertTrue(hasattr(LocalGPUClient, "extract_round1_english"))
        self.assertTrue(hasattr(LocalGPUClient, "extract_round2_arabic"))
        self.assertTrue(hasattr(ExtractionClient, "extract_round1_english"))
        self.assertTrue(hasattr(ExtractionClient, "extract_round2_arabic"))

    def test_extraction_client_round_calls(self):
        """Verifies ExtractionClient delegates round extraction to underlying backend."""
        class MockBackend:
            def __init__(self):
                self.calls = []

            def extract(self, prompt, system_prompt=None):
                self.calls.append((prompt, system_prompt))
                return "[]"

            def extract_round1_english(self, prompt):
                return self.extract(prompt, system_prompt=ROUND1_ENGLISH_EXTRACTION_PROMPT)

            def extract_round2_arabic(self, r1_json, prompt):
                return self.extract(prompt, system_prompt=ROUND2_ARABIC_VERIFICATION_PROMPT)

        backend = MockBackend()
        client = ExtractionClient(client=backend)

        res_r1 = client.extract_round1_english("Test R1 Prompt")
        self.assertEqual(res_r1, "[]")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][1], ROUND1_ENGLISH_EXTRACTION_PROMPT)

        res_r2 = client.extract_round2_arabic('[{"q": "test"}]', "Test R2 Prompt")
        self.assertEqual(res_r2, "[]")
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[1][1], ROUND2_ARABIC_VERIFICATION_PROMPT)


class TestTwoRoundPipelineExecutionAndResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "yt-spx"
        self.ws = SharedWorkspace(root_dir=self.root, worker_name="colab-test-worker")
        self.vid = "vid_tworound_001"

        # Create a realistic scenario with cohesive turns and valid word count
        self.scenario = {
            "video_id": self.vid,
            "title_ar": "مناظرة في ركن الخطباء",
            "youtube_url": f"https://youtube.com/watch?v={self.vid}",
            "speakers": [
                {
                    "speaker_id": "SPK_00",
                    "role": "MAIN_DAEE",
                    "display_name_en": "Sheikh Uthman Ibn Farooq",
                    "display_name_ar": "عثمان بن فاروق",
                    "profile_id": "uthman_ibn_farooq",
                },
                {
                    "speaker_id": "SPK_01",
                    "role": "INTERLOCUTOR",
                    "display_name_en": "Atheist Seeker",
                    "display_name_ar": "محاور ملحد",
                },
            ],
            "turns": [
                {
                    "turn_id": f"{self.vid}_T0001",
                    "episode_index": 1,
                    "speaker_id": "SPK_01",
                    "speaker_role": "INTERLOCUTOR",
                    "speaker_label_en": "Atheist Seeker",
                    "speaker_label_ar": "محاور ملحد",
                    "start_ms": 0,
                    "end_ms": 15000,
                    "start_fmt": "00:00",
                    "end_fmt": "00:15",
                    "text_original": "If God is the creator of the universe, who created God then? You cannot just arbitrarily stop at God.",
                    "text_ar": "",
                    "is_pedestrian_interruption": False,
                },
                {
                    "turn_id": f"{self.vid}_T0002",
                    "episode_index": 1,
                    "speaker_id": "SPK_00",
                    "speaker_role": "MAIN_DAEE",
                    "speaker_label_en": "Sheikh Uthman Ibn Farooq",
                    "speaker_label_ar": "عثمان بن فاروق",
                    "start_ms": 15500,
                    "end_ms": 45000,
                    "start_fmt": "00:15",
                    "end_fmt": "00:45",
                    "text_original": "That question commits an infinite regress fallacy. Whatever begins to exist has a cause, but an uncaused First Cause must exist by rational necessity.",
                    "text_ar": "",
                    "is_pedestrian_interruption": False,
                },
                {
                    "turn_id": f"{self.vid}_T0003",
                    "episode_index": 1,
                    "speaker_id": "SPK_01",
                    "speaker_role": "INTERLOCUTOR",
                    "speaker_label_en": "Atheist Seeker",
                    "speaker_label_ar": "محاور ملحد",
                    "start_ms": 45500,
                    "end_ms": 60000,
                    "start_fmt": "00:45",
                    "end_fmt": "01:00",
                    "text_original": "Why cannot the universe itself be the uncreated first cause without needing any transcendent God?",
                    "text_ar": "",
                    "is_pedestrian_interruption": False,
                },
                {
                    "turn_id": f"{self.vid}_T0004",
                    "episode_index": 1,
                    "speaker_id": "SPK_00",
                    "speaker_role": "MAIN_DAEE",
                    "speaker_label_en": "Sheikh Uthman Ibn Farooq",
                    "speaker_label_ar": "عثمان بن فاروق",
                    "start_ms": 60500,
                    "end_ms": 95000,
                    "start_fmt": "01:00",
                    "end_fmt": "01:35",
                    "text_original": "Because the universe has a finite beginning and is composed of changing, contingent matter that requires an external cause.",
                    "text_ar": "",
                    "is_pedestrian_interruption": False,
                },
            ],
        }
        self.ws.atomic_write_json(self.ws.scenarios_dir / f"{self.vid}.json", self.scenario)
        self.ws.advance_micro_step(self.vid, "M5_SCENARIO_BUILT")

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_two_round_extraction_creates_en_checkpoint_and_final(self):
        """Verifies Round 1 writes .en.json and Round 2 merges into final .json."""
        r1_output = [
            {
                "question_turn_id": f"{self.vid}_T0001",
                "answer_turn_id": f"{self.vid}_T0002",
                "topic_category": "Theology & God's Existence",
                "topic_subcategory": "First Cause & Infinite Regress",
                "interlocutor_type": "Atheist / Materialist",
                "question_en": "If God created the universe, who created God?",
                "core_answer_summary_en": "The question commits a category error and infinite regress fallacy. God is by definition the uncreated necessary First Cause.",
                "logical_premises": [
                    "An infinite regress of dependent causes is rationally impossible.",
                    "The chain of dependent causes requires an independent, uncaused First Cause.",
                ],
                "fallacy_or_misconception_exposed": "Category Error / Infinite Regress Fallacy",
                "analogies": [
                    "A row of dominoes cannot fall backwards forever without a first push.",
                    "A soldier cannot fire a gun if each soldier indefinitely waits for prior authorization.",
                ],
                "socratic_question": "If each creator requires a prior creator, can anything ever begin to exist?",
                "scriptural_citations": [
                    {"source": "Quran", "reference": "At-Tur 52:35-36", "significance": "Presents the rational trilemma of cosmic origins."}
                ],
                "interlocutor_verbatim_quote": "If God is the creator of the universe, who created God then? You cannot just arbitrarily stop at God.",
                "verbatim_quote": "Whatever begins to exist has a cause, but an uncaused First Cause must exist by rational necessity.",
                "dialogue_trajectory": [
                    {"turn_id": f"{self.vid}_T0001", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Who created God?"},
                    {"turn_id": f"{self.vid}_T0002", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "That commits an infinite regress fallacy."},
                    {"turn_id": f"{self.vid}_T0003", "role": "INTERLOCUTOR", "move_type": "Counter-Question", "text_en": "Why can't the universe be uncreated?"},
                    {"turn_id": f"{self.vid}_T0004", "role": "MAIN_DAEE", "move_type": "Deduction", "text_en": "Because the universe is contingent and changing."},
                ],
                "tags": ["god", "first cause", "infinite regress"],
            }
        ]

        r2_output = [
            {
                "question_ar": "إذا كان الله هو خالق الكون، فمن خلق الله؟",
                "core_answer_summary_ar": "السؤال يقع في مغالطة التسلسل اللانهائي وخطأ التصنيف؛ فالخالق أزلي واجب الوجود بذاته ولا يقاس بالمخلوقات.",
                "logical_premises_ar": [
                    "التسلسل في الفاعلين والعلل إلى ما لا بداية له ممتنع عقلاً.",
                    "سلسلة الموجودات الممكنة تفتقر حتماً إلى واجب وجود أول غير معلول.",
                ],
                "fallacy_or_misconception_exposed_ar": "مغالطة التسلسل اللانهائي والخلط بين الممكن والواجب",
                "analogies_ar": [
                    "أحجار الدومينو لا يمكن أن تسقط إلى ما لا نهاية دون دفعة أولى.",
                    "الجندي الذي ينتظر إذناً من جندي قبله إلى ما لا نهاية لن يطلق النار أبداً.",
                ],
                "socratic_question_ar": "إذا كان كل خالق يحتاج إلى خالق قبله، فهل كان يمكن لأي مخلوق أن يظهر إلى الوجود؟",
                "scriptural_citations": [
                    {
                        "source": "Quran",
                        "reference": "الطور 52:35-36",
                        "significance": "Presents the rational trilemma of cosmic origins.",
                        "significance_ar": "طرح القسمة العقلية الحاصرة في نشأة الكون: (أَمْ خُلِقُوا مِنْ غَيْرِ شَيْءٍ أَمْ هُمُ الْخَالِقُونَ).",
                    }
                ],
                "interlocutor_verbatim_quote_ar": "إذا كان الله هو خالق الكون، فمن خلق الله؟",
                "verbatim_quote_ar": "السبب الأول غير المخلوق حتمي الوجود عقلاً.",
                "dialogue_exchange": [
                    {"turn_id": f"{self.vid}_T0001", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Who created God?", "text_ar": "من خلق الله؟"},
                    {"turn_id": f"{self.vid}_T0002", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "That commits an infinite regress fallacy.", "text_ar": "هذا اعتراض يستلزم التسلسل الممتنع."},
                    {"turn_id": f"{self.vid}_T0003", "role": "INTERLOCUTOR", "move_type": "Counter-Question", "text_en": "Why can't the universe be uncreated?", "text_ar": "لم لا يكون الكون نفسه أزلياً؟"},
                    {"turn_id": f"{self.vid}_T0004", "role": "MAIN_DAEE", "move_type": "Deduction", "text_en": "Because the universe is contingent and changing.", "text_ar": "لأن الكون متغير حادث وكل متغير محتاج إلى مخصص."},
                ],
            }
        ]

        class TwoRoundMockLLM:
            def __init__(self, ws, vid):
                self.ws = ws
                self.vid = vid
                self.r1_calls = 0
                self.r2_calls = 0
                self.en_checkpoint_existed_during_r2 = False

            def extract_round1_english(self, prompt: str) -> str:
                self.r1_calls += 1
                return json.dumps(r1_output)

            def extract_round2_arabic(self, r1_json_str: str, prompt: str) -> str:
                self.r2_calls += 1
                en_file = self.ws.partial_extractions_dir / f"{self.vid}_win001.en.json"
                if en_file.exists():
                    self.en_checkpoint_existed_during_r2 = True
                return json.dumps(r2_output)

        mock_llm = TwoRoundMockLLM(self.ws, self.vid)
        extractor = Stage2ScenarioExtractor(workspace=self.ws, llm_client=mock_llm)
        units = extractor.extract_video_m6(self.vid, verbose=False)

        # 1. Both rounds called once
        self.assertEqual(mock_llm.r1_calls, 1)
        self.assertEqual(mock_llm.r2_calls, 1)
        self.assertEqual(len(units), 1)

        # 2. Check intermediate .en.json checkpoint file was saved on disk during Round 2 execution
        self.assertTrue(mock_llm.en_checkpoint_existed_during_r2)

        # 3. Check final unit merged Arabic and English fields
        u = units[0]
        self.assertEqual(u["question_en"], "If God created the universe, who created God?")
        self.assertEqual(u["question_ar"], "إذا كان الله هو خالق الكون، فمن خلق الله؟")
        self.assertEqual(len(u["analogies"]), 2)
        self.assertEqual(len(u["analogies_ar"]), 2)
        self.assertEqual(len(u["scriptural_citations"]), 1)
        self.assertEqual(u["dialogue_trajectory_turns_count"], 4)
        self.assertEqual(len(u["dialogue_exchange"]), 4)

    def test_resume_from_intermediate_round1_checkpoint(self):
        """When .en.json checkpoint exists, Round 1 is skipped and execution resumes at Round 2."""
        en_checkpoint = self.ws.partial_extractions_dir / f"{self.vid}_win001.en.json"
        pre_saved_en = [
            {
                "question_turn_id": f"{self.vid}_T0001",
                "answer_turn_id": f"{self.vid}_T0002",
                "topic_category": "Theology & God's Existence",
                "question_en": "Who created the Creator?",
                "core_answer_summary_en": "God is uncreated by definition.",
                "logical_premises": ["Causes cannot regress infinitely."],
                "analogies": ["First train engine pulling carts."],
                "socratic_question": "Does an uncaused cause need a cause?",
            }
        ]
        self.ws.atomic_write_json(en_checkpoint, pre_saved_en)

        class ResumeMockLLM:
            def __init__(self):
                self.r1_calls = 0
                self.r2_calls = 0

            def extract_round1_english(self, prompt: str) -> str:
                self.r1_calls += 1
                return "[]"

            def extract_round2_arabic(self, r1_json_str: str, prompt: str) -> str:
                self.r2_calls += 1
                return json.dumps([
                    {
                        "question_ar": "من خلق الخالق؟",
                        "core_answer_summary_ar": "الله غير مخلوق بذاته.",
                        "logical_premises_ar": ["تسلسل العلل ممتنع."],
                        "analogies_ar": ["محرك القطار الأول."],
                    }
                ])

        mock_llm = ResumeMockLLM()
        extractor = Stage2ScenarioExtractor(workspace=self.ws, llm_client=mock_llm)
        units = extractor.extract_video_m6(self.vid, verbose=False)

        # Round 1 MUST be skipped, Round 2 MUST execute
        self.assertEqual(mock_llm.r1_calls, 0)
        self.assertEqual(mock_llm.r2_calls, 1)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["question_ar"], "من خلق الخالق؟")


class TestDatabaseSchemaAndFTS5(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.tmp.close()
        init_db(self.db_path)

        # Insert parent video
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO videos (video_id, title_ar) VALUES (?, ?)",
                ("vid_db_01", "فيديو تجريبي للعقيدة"),
            )
            conn.commit()

    def tearDown(self):
        try:
            self.db_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_insert_and_fts_search_with_analogies_and_scripture(self):
        """Verifies analogies_json, scriptural_citations_json, and dialogue_trajectory_turns_count."""
        unit = {
            "id": "qa_db_01",
            "qa_id": "qa_db_01",
            "video_id": "vid_db_01",
            "episode_index": 1,
            "unit_seq_num": 1,
            "question_turn_id": "vid_db_01_T0001",
            "answer_turn_id": "vid_db_01_T0002",
            "source_turn_ids": ["vid_db_01_T0001", "vid_db_01_T0002"],
            "timestamp_seconds": 60,
            "timestamp_formatted": "01:00",
            "timestamp_url": "https://youtu.be/vid_db_01?t=60",
            "answer_timestamp_seconds": 75,
            "topic_category": "Theology & God's Existence",
            "topic_subcategory": "First Cause",
            "interlocutor_type": "Atheist / Materialist",
            "daee_profile_id": "uthman_ibn_farooq",
            "daee_name_en": "Uthman Ibn Farooq",
            "daee_name_ar": "عثمان بن فاروق",
            "question_en": "Who created the Creator?",
            "question_ar": "من خلق الخالق الأزلي؟",
            "core_answer_summary_en": "The Creator is uncaused and eternal by rational necessity.",
            "core_answer_summary_ar": "الخالق واجب الوجود أزلي بالضرورة العقلية.",
            "logical_premises": ["Infinite regress is absurd.", "An uncaused cause exists."],
            "logical_premises_ar": ["التسلسل في الفاعلين ممتنع.", "واجب الوجود متحتم."],
            "fallacy_or_misconception_exposed": "Infinite Regress Fallacy",
            "fallacy_or_misconception_exposed_ar": "مغالطة التسلسل اللانهائي",
            "analogy_or_metaphor": "First falling domino.",
            "analogy_or_metaphor_ar": "حجر الدومينو الأول.",
            "analogies": ["First falling domino.", "Soldier awaiting infinite orders."],
            "analogies_ar": ["حجر الدومينو الأول.", "جندي ينتظر إذناً من سلسلة لا تنتهي."],
            "socratic_question": "Can an infinite chain ever reach the present?",
            "socratic_question_ar": "هل يمكن لسلسلة لا نهائية أن تصل إلى الحاضر؟",
            "scriptural_citations": [
                {"source": "Quran", "reference": "At-Tur 52:35", "significance": "Were they created from nothing?"}
            ],
            "dialogue_trajectory_turns_count": 8,
            "dialogue_exchange": [
                {"turn_id": "T1", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Who made God?", "text_ar": "من خلق الله؟"},
                {"turn_id": "T2", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "Uncreated.", "text_ar": "أزلي."},
            ],
            "tags": ["theology", "first cause"],
        }

        insert_qa_unit(unit, db_path=self.db_path)

        # 1. Verify retrieval and hydration
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM qa_units WHERE id = ?", ("qa_db_01",))
            row = cursor.fetchone()
            hydrated = _hydrate_qa_row(row, 1)

        self.assertEqual(len(hydrated["analogies"]), 2)
        self.assertEqual(len(hydrated["analogies_ar"]), 2)
        self.assertEqual(hydrated["analogies_ar"][0], "حجر الدومينو الأول.")
        self.assertEqual(len(hydrated["scriptural_citations"]), 1)
        self.assertEqual(hydrated["scriptural_citations"][0]["reference"], "At-Tur 52:35")
        self.assertEqual(hydrated["dialogue_trajectory_turns_count"], 8)

        # 2. Verify FTS5 search finds keywords in analogies (English + Arabic) and scriptural citations
        res_analogy = search_qa(query="domino", db_path=self.db_path)
        self.assertEqual(len(res_analogy), 1)
        self.assertEqual(res_analogy[0]["id"], "qa_db_01")

        res_ar_analogy = search_qa(query="جندي", db_path=self.db_path)
        self.assertEqual(len(res_ar_analogy), 1)
        self.assertEqual(res_ar_analogy[0]["id"], "qa_db_01")

        res_scripture = search_qa(query="At-Tur", db_path=self.db_path)
        self.assertEqual(len(res_scripture), 1)
        self.assertEqual(res_scripture[0]["id"], "qa_db_01")


class TestTwoRoundZipperMergeAndRobustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "yt-spx"
        self.ws = SharedWorkspace(root_dir=self.root, worker_name="colab-test-worker")
        self.vid = "vid_robust_002"

        self.scenario = {
            "video_id": self.vid,
            "title_ar": "مناظرة جامعة أكسفورد",
            "youtube_url": f"https://youtube.com/watch?v={self.vid}",
            "speakers": [
                {
                    "speaker_id": "SPK_00",
                    "role": "MAIN_DAEE",
                    "display_name_en": "Sheikh Uthman Ibn Farooq",
                    "display_name_ar": "عثمان بن فاروق",
                    "profile_id": "uthman_ibn_farooq",
                },
                {
                    "speaker_id": "SPK_01",
                    "role": "INTERLOCUTOR",
                    "display_name_en": "Skeptic",
                    "display_name_ar": "محاور شكاك",
                },
            ],
            "turns": [
                {
                    "turn_id": f"{self.vid}_T0001",
                    "episode_index": 1,
                    "speaker_id": "SPK_01",
                    "speaker_role": "INTERLOCUTOR",
                    "start_ms": 0,
                    "end_ms": 10000,
                    "start_fmt": "00:00",
                    "end_fmt": "00:10",
                    "text_original": "If God created the universe, who created God then? Explain that to me rationally.",
                },
                {
                    "turn_id": f"{self.vid}_T0002",
                    "episode_index": 1,
                    "speaker_id": "SPK_00",
                    "speaker_role": "MAIN_DAEE",
                    "start_ms": 11000,
                    "end_ms": 35000,
                    "start_fmt": "00:11",
                    "end_fmt": "00:35",
                    "text_original": "Whatever begins to exist requires a cause. God did not begin to exist; an infinite regress of causes is impossible.",
                },
                {
                    "turn_id": f"{self.vid}_T0003",
                    "episode_index": 1,
                    "speaker_id": "SPK_01",
                    "speaker_role": "INTERLOCUTOR",
                    "start_ms": 36000,
                    "end_ms": 50000,
                    "start_fmt": "00:36",
                    "end_fmt": "00:50",
                    "text_original": "Why does gratuitous suffering and evil exist if God is truly all powerful and compassionate?",
                },
                {
                    "turn_id": f"{self.vid}_T0004",
                    "episode_index": 1,
                    "speaker_id": "SPK_00",
                    "speaker_role": "MAIN_DAEE",
                    "start_ms": 51000,
                    "end_ms": 85000,
                    "start_fmt": "00:51",
                    "end_fmt": "01:25",
                    "text_original": "Suffering is an exam by divine wisdom; this earthly life is not paradise, and evil has no objective basis under atheism.",
                },
            ],
        }
        self.ws.atomic_write_json(self.ws.scenarios_dir / f"{self.vid}.json", self.scenario)
        self.ws.advance_micro_step(self.vid, "M5_SCENARIO_BUILT")

    def tearDown(self):
        self.tmp.cleanup()

    def test_out_of_order_and_partial_round2_zipper_merge(self):
        """Verifies zipper merge correctly matches units even when Round 2 reverses order and truncates turns."""
        r1_output = [
            {
                "question_turn_id": f"{self.vid}_T0001",
                "answer_turn_id": f"{self.vid}_T0002",
                "question_en": "If God created the universe, who created God?",
                "core_answer_summary_en": "God is by definition the uncreated necessary First Cause.",
                "logical_premises": ["Infinite regress is impossible.", "Uncaused First Cause exists."],
                "analogies": ["Domino chain requiring first push.", "Soldier permission chain."],
                "scriptural_citations": [
                    {"source": "Quran", "reference": "At-Tur 52:35", "significance": "Were they created from nothing?"}
                ],
                "dialogue_exchange": [
                    {"turn_id": f"{self.vid}_T0001", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Who created God?"},
                    {"turn_id": f"{self.vid}_T0002", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "God is uncreated."},
                    {"turn_id": "extra_turn_3", "role": "INTERLOCUTOR", "move_type": "Question", "text_en": "Why can't the universe be uncreated?"},
                    {"turn_id": "extra_turn_4", "role": "MAIN_DAEE", "move_type": "Proof", "text_en": "Because the universe has a beginning."},
                ],
            },
            {
                "question_turn_id": f"{self.vid}_T0003",
                "answer_turn_id": f"{self.vid}_T0004",
                "question_en": "Why does gratuitous suffering exist?",
                "core_answer_summary_en": "Earthly life is an exam governed by infinite divine wisdom.",
                "logical_premises": ["God is All-Wise.", "Life is a test."],
                "analogies": ["Surgeon cutting a patient for health."],
                "dialogue_exchange": [
                    {"turn_id": f"{self.vid}_T0003", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Why does suffering exist?"},
                    {"turn_id": f"{self.vid}_T0004", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "It is an examination."},
                ],
            },
        ]

        # Round 2 reverses order (Suffering first, First Cause second), provides Arabic dialogue without text_en,
        # truncates dialogue turns to 2 instead of 4, and adds significance_ar to scripture without source/significance
        r2_reversed_output = [
            {
                "question_turn_id": f"{self.vid}_T0003",
                "answer_turn_id": f"{self.vid}_T0004",
                "question_ar": "لماذا يوجد الشر والابتلاء في العالم؟",
                "core_answer_summary_ar": "الحياة دار ابتلاء واختبار محكومة بكمال الحكمة والعدل الإلهي.",
                "logical_premises_ar": ["الله حكيم عليم.", "الدنيا دار اختبار."],
                "analogies_ar": ["مبضع الجراح الذي يؤلم المريض لشفائه."],
                "dialogue_exchange": [
                    {"turn_id": f"{self.vid}_T0003", "text_ar": "لماذا يوجد الشر والمعاناة؟"},
                    {"turn_id": f"{self.vid}_T0004", "text_ar": "الحياة الدنيا امتحان ودار ابتلاء وليست جنة الخلد."},
                ],
            },
            {
                "question_turn_id": f"{self.vid}_T0001",
                "answer_turn_id": f"{self.vid}_T0002",
                "question_ar": "إذا كان الله هو خالق الكون، فمن خلق الله؟",
                "core_answer_summary_ar": "الخالق أزلي واجب الوجود عقلاً ولا يقاس بالمخلوقات الحادثة.",
                "logical_premises_ar": ["التسلسل في الفاعلين ممتنع.", "واجب الوجود متحتم."],
                "analogies_ar": ["أحجار الدومينو وسلسلة الأوامر."],
                "scriptural_citations": [
                    {"reference": "At-Tur 52:35", "significance_ar": "القسمة العقلية الحاصرة في نشأة الكون."}
                ],
                "dialogue_exchange": [
                    {"turn_id": f"{self.vid}_T0001", "text_ar": "من خلق الله إذاً؟"},
                    {"turn_id": f"{self.vid}_T0002", "text_ar": "الله أزلي واجب الوجود."},
                ],
            },
        ]

        class ReversingMockLLM:
            def extract_round1_english(self, p): return json.dumps(r1_output)
            def extract_round2_arabic(self, r1_json, p): return json.dumps(r2_reversed_output)

        extractor = Stage2ScenarioExtractor(workspace=self.ws, llm_client=ReversingMockLLM())
        units = extractor.extract_video_m6(self.vid, verbose=False)

        self.assertEqual(len(units), 2)

        # 1. First unit (Who created God) must match its own Arabic translation, NOT suffering!
        u0 = units[0]
        self.assertEqual(u0["question_en"], "If God created the universe, who created God?")
        self.assertEqual(u0["question_ar"], "إذا كان الله هو خالق الكون، فمن خلق الله؟")
        self.assertIn("واجب الوجود", u0["core_answer_summary_ar"])

        # 2. Dialogue turns: English text must NOT be lost, Arabic text must be added, all 4 turns preserved!
        ex = u0["dialogue_exchange"]
        self.assertEqual(len(ex), 4, "Zipper merge must preserve all 4 dialogue turns even if R2 only translated 2")
        self.assertEqual(ex[0]["text_en"], "Who created God?")
        self.assertEqual(ex[0]["text_ar"], "من خلق الله إذاً؟")
        self.assertEqual(ex[1]["text_en"], "God is uncreated.")
        self.assertEqual(ex[1]["text_ar"], "الله أزلي واجب الوجود.")
        self.assertEqual(ex[2]["text_en"], "Why can't the universe be uncreated?")
        self.assertEqual(ex[3]["text_en"], "Because the universe has a beginning.")

        # 3. Scriptural citations: source & English significance preserved, significance_ar added
        sc = u0["scriptural_citations"]
        self.assertEqual(len(sc), 1)
        self.assertEqual(sc[0]["source"], "Quran")
        self.assertEqual(sc[0]["reference"], "At-Tur 52:35")
        self.assertEqual(sc[0]["significance"], "Were they created from nothing?")
        self.assertEqual(sc[0]["significance_ar"], "القسمة العقلية الحاصرة في نشأة الكون.")

        # 4. Second unit (Suffering) must match its own Arabic translation!
        u1 = units[1]
        self.assertEqual(u1["question_en"], "Why does gratuitous suffering exist?")
        self.assertEqual(u1["question_ar"], "لماذا يوجد الشر والابتلاء في العالم؟")
        self.assertIn("دار ابتلاء", u1["core_answer_summary_ar"])
        self.assertEqual(len(u1["analogies_ar"]), 1)

    def test_dialogue_trajectory_alias_in_round1_and_round2(self):
        """Verifies pipeline correctly handles dialogue_trajectory key alias."""
        r1_alias = [
            {
                "question_turn_id": f"{self.vid}_T0001",
                "answer_turn_id": f"{self.vid}_T0002",
                "question_en": "Who created the Creator?",
                "core_answer_summary_en": "God is uncreated.",
                "dialogue_trajectory": [
                    {"turn_id": f"{self.vid}_T0001", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Who made God?"},
                    {"turn_id": f"{self.vid}_T0002", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "He is eternal."},
                ]
            }
        ]
        r2_alias = [
            {
                "question_turn_id": f"{self.vid}_T0001",
                "answer_turn_id": f"{self.vid}_T0002",
                "question_ar": "من خلق الخالق؟",
                "core_answer_summary_ar": "الله أزلي غير مخلوق.",
                "dialogue_trajectory": [
                    {"turn_id": f"{self.vid}_T0001", "text_ar": "من خلق الله؟"},
                    {"turn_id": f"{self.vid}_T0002", "text_ar": "هو الأزلي الأول."},
                ]
            }
        ]

        class AliasMockLLM:
            def extract_round1_english(self, p): return json.dumps(r1_alias)
            def extract_round2_arabic(self, r1_json, p): return json.dumps(r2_alias)

        extractor = Stage2ScenarioExtractor(workspace=self.ws, llm_client=AliasMockLLM())
        units = extractor.extract_video_m6(self.vid, verbose=False)

        self.assertEqual(len(units), 1)
        u = units[0]
        self.assertEqual(u["dialogue_trajectory_turns_count"], 2)
        self.assertEqual(len(u["dialogue_exchange"]), 2)
        self.assertEqual(u["dialogue_exchange"][0]["text_en"], "Who made God?")
        self.assertEqual(u["dialogue_exchange"][0]["text_ar"], "من خلق الله؟")

    def test_plain_backend_delegation_in_extraction_client(self):
        """Verifies ExtractionClient delegates round extractions to plain LLM backend without round methods."""
        class PlainBackend:
            def __init__(self):
                self.calls = []

            def extract(self, prompt, system_prompt=None):
                self.calls.append((prompt, system_prompt))
                return "[]"

        plain = PlainBackend()
        wrapper = ExtractionClient(client=plain)
        res1 = wrapper.extract_round1_english("Test Prompt R1")
        self.assertEqual(res1, "[]")
        self.assertEqual(len(plain.calls), 1)
        self.assertEqual(plain.calls[0][1], ROUND1_ENGLISH_EXTRACTION_PROMPT)

        res2 = wrapper.extract_round2_arabic('[{"q": "test"}]', "Test Prompt R2")
        self.assertEqual(res2, "[]")
        self.assertEqual(len(plain.calls), 2)
        self.assertEqual(plain.calls[1][1], ROUND2_ARABIC_VERIFICATION_PROMPT)


if __name__ == "__main__":
    unittest.main()

