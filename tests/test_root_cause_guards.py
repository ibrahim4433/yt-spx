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
        anchor_and_enrich_extracted_unit strictly drops the unit (returns None).
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

        # Strictly dropped: must return None rather than force-renaming to Da'ee
        self.assertIsNone(enriched, "anchor_and_enrich_extracted_unit must drop units whose answer_turn_id is an interlocutor")

    def test_db_ingestion_sanitization_and_enrichment(self):
        """
        Tests that when data is inserted into SQLite, _sanitize_and_validate_qa_unit
        automatically sanitizes text and enriches dialogue exchange, while strictly refusing
        to force-rename 'المحاور' to the Da'ee.
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

        # 1. Daee name must NOT be force-renamed from المحاور
        self.assertEqual(sanitized["daee_name_ar"], "المحاور 1")
        self.assertEqual(sanitized["daee_profile_id"], "")

        # When daee_name_ar is a genuine Daee without profile, it resolves properly
        unit_with_daee = dict(raw_unit, daee_name_ar="الشيخ عثمان بن فاروق")
        sanitized_daee = _sanitize_and_validate_qa_unit(conn, unit_with_daee)
        self.assertEqual(sanitized_daee["daee_name_ar"], "الشيخ عثمان بن فاروق")
        self.assertEqual(sanitized_daee["daee_profile_id"], "daee_uthman")

        # 2. Foreign script cleaned
        self.assertNotIn("педوفيليا", sanitized["question_ar"])
        self.assertIn("البيدوفيليا", sanitized["question_ar"])
        self.assertNotIn("。", sanitized["question_ar"])

        # 3. Dialogue exchange turns enriched
        ex = sanitized["dialogue_exchange"]
        self.assertEqual(ex[0]["timestamp_formatted"], "01:05")
        self.assertEqual(ex[1]["timestamp_formatted"], "01:12")
        self.assertEqual(ex[1]["speaker_label_ar"], "الشيخ عثمان بن فاروق")

    def test_category_canonicalization(self):
        """Tests that micro-categories map cleanly to the extensible canonical domains."""
        from db.database import canonicalize_topic_category, get_category_arabic_name
        self.assertEqual(
            canonicalize_topic_category("Textual Preservation & Biblical Manuscripts"),
            "Textual Preservation & Manuscripts"
        )
        self.assertEqual(
            canonicalize_topic_category("Social Inquiries & Feminism"),
            "Shariah & Social Inquiries"
        )
        self.assertEqual(
            get_category_arabic_name("Textual Preservation & Biblical Manuscripts"),
            "حفظ النصوص والمخطوطات"
        )
        self.assertEqual(
            get_category_arabic_name("Social Inquiries & Feminism"),
            "الشريعة والشبهات الاجتماعية"
        )

    def test_invalid_street_interview_questions(self):
        """Tests that casual street interview questions are rejected."""
        from core.scenario_extractor import is_valid_dialectical_question
        self.assertFalse(is_valid_dialectical_question("Where were you shot?", "أين أصبت بالرصاصة؟"))
        self.assertFalse(is_valid_dialectical_question("How old are you?", "كم عمرك؟"))
        self.assertTrue(
            is_valid_dialectical_question(
                "If everything has a creator, who created God?",
                "إذا كان لكل شيء خالق، فمن خلق الله؟"
            )
        )

    def test_turn_pre_merging(self):
        """Tests that consecutive rapid turns by the same speaker are merged into cohesive dialogue."""
        from core.scenario_extractor import merge_consecutive_speaker_turns
        turns = [
            {"turn_id": "T01", "speaker_id": "SPK_00", "episode_index": 1, "start_ms": 1000, "end_ms": 2500, "start_fmt": "00:01", "end_fmt": "00:02", "text_original": "Who is the", "text_ar": "من هو"},
            {"turn_id": "T02", "speaker_id": "SPK_00", "episode_index": 1, "start_ms": 2700, "end_ms": 4000, "start_fmt": "00:02", "end_fmt": "00:04", "text_original": "lamb of God?", "text_ar": "حمل الله؟"},
            {"turn_id": "T03", "speaker_id": "SPK_01", "episode_index": 1, "start_ms": 5000, "end_ms": 8000, "start_fmt": "00:05", "end_fmt": "00:08", "text_original": "Jesus is the sacrifice.", "text_ar": "يسوع هو الذبيحة."}
        ]
        merged = merge_consecutive_speaker_turns(turns, max_gap_ms=2500)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["text_original"], "Who is the lamb of God?")
        self.assertEqual(merged[0]["text_ar"], "من هو حمل الله؟")
        self.assertEqual(merged[0]["end_fmt"], "00:04")
        self.assertEqual(merged[1]["speaker_id"], "SPK_01")

    def test_latin_token_purification(self):
        """Tests that leaked English words in Classical Arabic fields are cleaned."""
        from db.database import sanitize_text_field
        self.assertEqual(sanitize_text_field("العالم والInstinctive للإنسان"), "العالم والفطري للإنسان")
        self.assertEqual(sanitize_text_field("الإسلام يرفض الincest والpedophilia"), "الإسلام يرفض زنا المحارم والبيدوفيليا (استغلال الأطفال)")
        self.assertEqual(sanitize_text_field("الكون يحتاج إلى Cause"), "الكون يحتاج إلى سبب أول")

    def test_programmatic_verbatim_recovery(self):
        """Tests that hallucinated quotes are automatically recovered from real spoken Da'ee turns."""
        from core.scenario_extractor import anchor_and_enrich_extracted_unit
        scenario = {
            "video_id": "vid_hashim_01",
            "youtube_url": "https://www.youtube.com/watch?v=vid_hashim_01",
            "title_ar": "مناظرة الشيخ هاشم",
            "speakers": [
                {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "profile_id": "daee_hashim", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "profile_id": None, "display_name_en": "Interlocutor", "display_name_ar": "المحاور"}
            ],
            "turns": [
                {"turn_id": "T01", "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "start_ms": 1000, "end_ms": 5000, "start_fmt": "00:01", "text_original": "Is Jesus God?"},
                {"turn_id": "T02", "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 6000, "end_ms": 12000, "start_fmt": "00:06", "text_original": "No, Jesus clearly said the Father is greater than I."}
            ]
        }
        # LLM generated a totally fabricated essay for verbatim_quote
        raw_unit = {
            "question_turn_id": "T01",
            "answer_turn_id": "T02",
            "question_en": "Is Jesus God?",
            "question_ar": "هل عيسى هو الله؟",
            "verbatim_quote": "The biblical depiction of Christ as a divine essence is a philosophical contradiction to Monotheism." # Fabricated essay!
        }
        enriched = anchor_and_enrich_extracted_unit(raw_unit, scenario, 1, 0, 1)
        # MUST be recovered from real spoken words in T02!
        self.assertEqual(enriched["verbatim_quote"], "No, Jesus clearly said the Father is greater than I.")

    def test_theological_bias_rejection(self):
        """Ensures that units asserting opponent theological bias or heresy are strictly rejected."""
        from core.scenario_extractor import check_theological_bias as core_bias_check, anchor_and_enrich_extracted_unit
        from db.database import check_theological_bias as db_bias_check

        heretical_units = [
            {"core_answer_summary_en": "The speaker argues that incest is justified under utilitarian philosophy."},
            {"verbatim_quote": "Incest is permissible if there is no reproductive risk."},
            {"core_answer_summary_ar": "الدفاع عن زنا المحارم في حالة عدم الإنجاب."},
            {"core_answer_summary_en": "The Comforter is the Holy Spirit, not Muhammad."},
            {"core_answer_summary_ar": "روح الحق هو الروح القدس وليس نبي الإسلام."},
            {"core_answer_summary_en": "Old Testament laws are void and no longer apply."},
            {"core_answer_summary_ar": "الناموس باطل وملغى بالكامل."},
            {"core_answer_summary_en": "Jesus is the true God incarnate."},
            {"core_answer_summary_ar": "المسيح هو الله الظاهر في الجسد."},
            {"core_answer_summary_en": "Christ died for our sins on the cross."},
            {"core_answer_summary_ar": "يسوع مات تكفيراً عن خطايانا."},
        ]

        for u in heretical_units:
            self.assertIsNotNone(core_bias_check(u), f"Core bias check failed to catch: {u}")
            self.assertIsNotNone(db_bias_check(u), f"DB bias check failed to catch: {u}")

        # Ensure anchor_and_enrich_extracted_unit drops such units entirely (returns None)
        scenario = {
            "video_id": "test_heresy_01",
            "youtube_url": "https://www.youtube.com/watch?v=test_heresy_01",
            "title_ar": "حوار",
            "speakers": [
                {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "profile_id": "daee_01", "display_name_en": "Daee", "display_name_ar": "داعية"},
                {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "profile_id": None, "display_name_en": "Inter", "display_name_ar": "محاور"}
            ],
            "turns": [
                {"turn_id": "T01", "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "start_ms": 1000, "end_ms": 4000, "start_fmt": "00:01", "text_original": "Is incest moral?"},
                {"turn_id": "T02", "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 5000, "end_ms": 10000, "start_fmt": "00:05", "text_original": "Incest is justified under my moral view."}
            ]
        }
        bad_raw_unit = {
            "question_turn_id": "T01",
            "answer_turn_id": "T02",
            "question_en": "Is incest moral?",
            "question_ar": "هل زنا المحارم أخلاقي؟",
            "core_answer_summary_en": "Incest is justified under our worldview.",
            "verbatim_quote": "Incest is justified under my moral view."
        }
        res = anchor_and_enrich_extracted_unit(bad_raw_unit, scenario, 1, 0, 1)
        self.assertIsNone(res, "anchor_and_enrich_extracted_unit should have dropped unit with theological bias")

    def test_quote_quality_and_dangling_preposition(self):
        """Ensures dangling prepositions/conjunctions are trimmed from quotes."""
        from core.scenario_extractor import clean_dangling_quote
        quote_with_and = "We believe that God is the creator of the heavens and"
        cleaned = clean_dangling_quote(quote_with_and)
        self.assertEqual(cleaned, "We believe that God is the creator of the heavens")

        # Test double dangling prepositions and 7-word quote cleaning
        quote_double = "He went to Jerusalem and then spoke to to"
        cleaned_double = clean_dangling_quote(quote_double)
        self.assertEqual(cleaned_double, "He went to Jerusalem and then spoke")

        quote_seven_words = "He went to Jerusalem and then to"
        cleaned_seven = clean_dangling_quote(quote_seven_words)
        self.assertEqual(cleaned_seven, "He went to Jerusalem and then")

    def test_canonical_pillars_textual_preservation(self):
        """Ensures 'Textual Preservation & Manuscripts' is a top-level canonical domain."""
        from core.scenario_extractor import CANONICAL_TOPIC_PILLARS, canonicalize_topic_category as core_canonicalize
        from db.database import BILINGUAL_TOPIC_DOMAINS, canonicalize_topic_category as db_canonicalize

        pillar = "Textual Preservation & Manuscripts"
        self.assertIn(pillar, CANONICAL_TOPIC_PILLARS)
        self.assertIn(pillar, BILINGUAL_TOPIC_DOMAINS)

        self.assertEqual(core_canonicalize("Biblical Manuscripts"), pillar)
        self.assertEqual(core_canonicalize("Dead Sea Scrolls"), pillar)
        self.assertEqual(db_canonicalize("Inerrancy & Preservation"), pillar)
        self.assertEqual(db_canonicalize("Codex Sinaiticus"), pillar)

    def test_colab_runner_q5_k_m_default(self):
        """Ensures LocalGPUClient defaults to Qwen2.5-14B-Instruct-Q5_K_M.gguf for maximum reasoning precision."""
        import inspect
        from core.colab_runner import LocalGPUClient
        sig = inspect.signature(LocalGPUClient.__init__)
        self.assertEqual(sig.parameters["gguf_filename"].default, "Qwen2.5-14B-Instruct-Q5_K_M.gguf")
        self.assertEqual(sig.parameters["gguf_repo"].default, "bartowski/Qwen2.5-14B-Instruct-GGUF")

    def test_expanded_philosophical_vocabulary(self):
        """Ensures new philosophical vocabulary (singular/plural, existence, dependence) is sanitized in Arabic."""
        from db.database import sanitize_text_field
        self.assertEqual(sanitize_text_field("المسيحية تقول بـ two wills و two natures"), "المسيحية تقول بـ إرادتين و طبيعتين")
        self.assertEqual(sanitize_text_field("المخلوق في حالة dependence و existence ممكن"), "المخلوق في حالة افتقار واحتياج و وجود ممكن")
        self.assertEqual(sanitize_text_field("الكون يحتاج إلى First Cause"), "الكون يحتاج إلى سبب أول")

    def test_theological_bias_refutation_not_falsely_rejected(self):
        """Ensures sound Islamic refutations mentioning opponent dogmas are not falsely rejected."""
        from core.scenario_extractor import check_theological_bias as core_bias_check
        from db.database import check_theological_bias as db_bias_check

        sound_refutation_units = [
            {"core_answer_summary_en": "The Da'ee refutes the Christian dogma that Jesus is God by citing Mark 13:32."},
            {"core_answer_summary_ar": "يبطل الإسلام زعم النصارى أن عيسى هو الله بالنصوص الصريحة من الأناجيل."},
            {"core_answer_summary_en": "Islam firmly rejects the notion that Christ died for our sins (Surah 4:157)."},
            {"core_answer_summary_ar": "القرآن يرد شبهة أن المسيح مات فداء عن خطايانا بنفي صلبه وقتله."},
            {"core_answer_summary_en": "The speaker exposes the false claim that biblical laws are void by quoting Matthew 5:17."},
            {"core_answer_summary_ar": "الداعية يدحض دعوى أن الناموس باطل ببيان تصريح المسيح أنه لم يأت لينقض."},
        ]

        for u in sound_refutation_units:
            self.assertIsNone(core_bias_check(u), f"Core bias check falsely rejected sound refutation: {u}")
            self.assertIsNone(db_bias_check(u), f"DB bias check falsely rejected sound refutation: {u}")

    def test_stray_latin_token_regex_purging(self):
        """Ensures unmapped stray Latin tokens in Arabic fields are purged by regex."""
        from db.database import sanitize_text_field
        res = sanitize_text_field("هذا كلام English و randomWord في النص الفصيح.", is_arabic=True)
        self.assertNotIn("English", res)
        self.assertNotIn("randomWord", res)
        self.assertIn("هذا كلام", res)
        self.assertIn("النص الفصيح", res)

        # Tatweel prefix cleaning
        res_prefix = sanitize_text_field("المسألة لـ unknownTerm غير مقبولة.", is_arabic=True)
        self.assertNotIn("unknownTerm", res_prefix)
        self.assertIn("المسألة غير مقبولة", res_prefix)

        # Valid Arabic with tatweel prefix is preserved
        res_valid = sanitize_text_field("المسيحية تقول بـ إرادتين وطبيعتين.", is_arabic=True)
        self.assertEqual(res_valid, "المسيحية تقول بـ إرادتين وطبيعتين.")

    def test_rich_prompt_and_generous_windowing(self):
        """Ensures the extractor prompt is comprehensive with 3 gold-standard examples and windows are generous."""
        import inspect
        from core.extractor_prompt import EXTRACTION_SYSTEM_PROMPT, USER_EXTRACTION_PROMPT_TEMPLATE
        from core.scenario_extractor import build_episode_windows

        # 1. Prompt depth and few-shot coverage
        self.assertIn("EXAMPLE 1 (Comparative Religion & Christianity", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("EXAMPLE 2 (Theology & God's Existence", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("EXAMPLE 3 (Problem of Evil & Suffering", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("CRITICAL RULE 7: SOCRATIC CROSS-EXAMINATIONS", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("CRITICAL RULE 8: RIGOROUS DEDUCTIVE SYLLOGISMS", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("socratic_question_ar", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("logical_premises_ar", EXTRACTION_SYSTEM_PROMPT)

        # Ensure prompt template formats correctly without KeyError
        formatted_prompt = USER_EXTRACTION_PROMPT_TEMPLATE.format(
            title_ar="مناظرة تجريبية",
            youtube_url="https://www.youtube.com/watch?v=sample123",
            episode_index=1,
            start_fmt="00:00",
            end_fmt="05:00",
            speakers_summary="SPK_00=MAIN_DAEE",
            target_daee_name="Sheikh Uthman",
            target_daee_id="SPK_00",
            transcript_text="[T01] [00:00 -> 00:05] [MAIN_DAEE]: Bismillah",
        )
        self.assertIn("مناظرة تجريبية", formatted_prompt)
        self.assertIn("Sheikh Uthman", formatted_prompt)

        # 2. Generous window sizing defaults (2400 words, 4 carry turns)
        sig = inspect.signature(build_episode_windows)
        self.assertEqual(sig.parameters["max_words_per_window"].default, 2400)
        self.assertEqual(sig.parameters["context_carry_turns"].default, 4)

        # An episode with 1500 words must stay whole (1 window)
        dummy_turns = [
            {
                "turn_id": f"turn_{i}",
                "episode_index": 1,
                "speaker_id": "SPK_00" if i % 2 == 0 else "SPK_01",
                "speaker_role": "MAIN_DAEE" if i % 2 == 0 else "INTERLOCUTOR",
                "start_ms": i * 10000,
                "end_ms": (i + 1) * 10000,
                "start_fmt": "00:00",
                "end_fmt": "00:10",
                "text_original": " ".join(f"word_{i}_{j}" for j in range(150)),  # 150 distinct words per turn
            }
            for i in range(10)  # 1500 words total
        ]
        windows_1500 = build_episode_windows(dummy_turns)
        self.assertEqual(len(windows_1500), 1, "1500 words should not be split under generous 2400-word window")

        # An episode with 3000 words must split cleanly and carry forward 4 context turns
        dummy_long = [
            {
                "turn_id": f"turn_{i}",
                "episode_index": 1,
                "speaker_id": "SPK_00" if i % 2 == 0 else "SPK_01",
                "speaker_role": "MAIN_DAEE" if i % 2 == 0 else "INTERLOCUTOR",
                "start_ms": i * 10000,
                "end_ms": (i + 1) * 10000,
                "start_fmt": "00:00",
                "end_fmt": "00:10",
                "text_original": " ".join(f"speech_{i}_{j}" for j in range(150)),
            }
            for i in range(20)  # 3000 words total
        ]
        windows_3000 = build_episode_windows(dummy_long)
        self.assertEqual(len(windows_3000), 2, "3000 words should split into 2 windows")
        self.assertEqual(windows_3000[0]["episode_index"], 1)
        self.assertEqual(windows_3000[1]["episode_index"], 1)

    def test_is_valid_dialectical_question_theological_coverage(self):
        """Ensures concise foundational theological questions are accepted and trivia is rejected."""
        from core.scenario_extractor import is_valid_dialectical_question

        # Foundational theological questions (3 English words, 2-4 Arabic words)
        self.assertTrue(is_valid_dialectical_question("Who created God?", "من خلق الله؟"))
        self.assertTrue(is_valid_dialectical_question("Is Jesus God?", "هل عيسى هو الله؟"))
        self.assertTrue(is_valid_dialectical_question("Can God die?", "هل يمكن للإله أن يموت؟"))
        self.assertTrue(is_valid_dialectical_question("Who made God?", "من صنع الله؟"))
        self.assertTrue(is_valid_dialectical_question("Did Jesus die?", "هل مات يسوع؟"))
        self.assertTrue(is_valid_dialectical_question("Why worship God?", "لماذا نعبد الله؟"))

        # Rejection of trivial street noise and conversational filler
        self.assertFalse(is_valid_dialectical_question("what did you say?", "ماذا قلت؟"))
        self.assertFalse(is_valid_dialectical_question("excuse me?", "عفوا؟"))
        self.assertFalse(is_valid_dialectical_question("let her be?", "دعها وشأنها؟"))
        self.assertFalse(is_valid_dialectical_question("what's your name?", "ما اسمك؟"))
        self.assertFalse(is_valid_dialectical_question("can you hear me?", "هل تسمعني؟"))
        self.assertFalse(is_valid_dialectical_question("why?", "لماذا؟"))

    def test_stray_latin_token_parentheses_cleaning(self):
        """Ensures purging stray Latin tokens in Arabic fields removes residual empty parentheses/brackets."""
        from db.database import sanitize_text_field

        res_paren = sanitize_text_field("إنجيل مرقس (Mark)", is_arabic=True)
        self.assertEqual(res_paren, "إنجيل مرقس")
        self.assertNotIn("()", res_paren)

        res_bracket = sanitize_text_field("الدليل العقلي [First Cause]", is_arabic=True)
        self.assertEqual(res_bracket, "الدليل العقلي [سبب أول]")

        res_unmapped_bracket = sanitize_text_field("مسألة الوجود [RandomLatinWord]", is_arabic=True)
        self.assertEqual(res_unmapped_bracket, "مسألة الوجود")
        self.assertNotIn("[]", res_unmapped_bracket)

    def test_stage2_extractor_default_context_length(self):
        """Ensures Stage2ScenarioExtractor defaults to 16,384 context length matching LocalGPUClient."""
        import inspect
        from core.scenario_extractor import Stage2ScenarioExtractor

        sig = inspect.signature(Stage2ScenarioExtractor.__init__)
        self.assertEqual(sig.parameters["context_length"].default, 16384)

    def test_strip_qwen3_think_blocks_in_json_parsers(self):
        """Ensures closed and unclosed Qwen3 <think> blocks with braces do not corrupt JSON parsing."""
        from core.scenario_extractor import clean_llm_json_array
        from core.colab_runner import clean_llm_json_response

        raw_with_closed_think = (
            "<think>\nLet me check the schema {question_ar, core_answer_summary_ar} and array [0].\n</think>\n"
            '[{"question_turn_id": "T1", "answer_turn_id": "T2", "question_en": "Who created God?", "question_ar": "من خلق الله؟"}]'
        )
        parsed_a = clean_llm_json_array(raw_with_closed_think)
        parsed_b = clean_llm_json_response(raw_with_closed_think)
        self.assertEqual(len(parsed_a), 1)
        self.assertEqual(parsed_a[0]["question_ar"], "من خلق الله؟")
        self.assertEqual(len(parsed_b), 1)
        self.assertEqual(parsed_b[0]["question_ar"], "من خلق الله؟")

        raw_with_unclosed_think = (
            "<think>Translating into Classical Arabic...\n"
            '[{"question_turn_id": "T1", "answer_turn_id": "T2", "question_en": "Is Jesus God?", "question_ar": "هل عيسى هو الله؟"}]'
        )
        parsed_unclosed = clean_llm_json_array(raw_with_unclosed_think)
        self.assertEqual(len(parsed_unclosed), 1)
        self.assertEqual(parsed_unclosed[0]["question_ar"], "هل عيسى هو الله؟")

    def test_substantive_answer_turn_recovery_from_filler_turn(self):
        """Ensures tiny filler turns ('Can' or 'What do you think about that? That deep? Well,') re-anchor to substantive Da'ee turns."""
        from core.scenario_extractor import anchor_and_enrich_extracted_unit

        scenario = {
            "video_id": "rEuKJ26ODgw",
            "youtube_url": "https://www.youtube.com/watch?v=rEuKJ26ODgw",
            "title_ar": "حوار الشيخ عثمان",
            "speakers": [
                {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "profile_id": "daee_uthman", "display_name_en": "Sheikh Uthman Ibn Farooq", "display_name_ar": "الشيخ عثمان بن فاروق"},
                {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "profile_id": None, "display_name_en": "Interlocutor", "display_name_ar": "المحاور"},
            ],
            "turns": [
                {"turn_id": "T0547", "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 2428000, "end_ms": 2429000, "start_fmt": "40:28", "text_original": "Can"},
                {"turn_id": "T0548", "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "start_ms": 2446000, "end_ms": 2452000, "start_fmt": "40:46", "text_original": "You pray. Can you ask and you shall receive."},
                {"turn_id": "T0549", "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 2453000, "end_ms": 2468000, "start_fmt": "40:53", "text_original": "In Islam when we pray to Allah He answers according to His infinite wisdom and our true benefit."},
            ],
        }
        raw_unit = {
            "question_turn_id": "T0548",
            "answer_turn_id": "T0547",
            "question_en": "Why isn't every prayer answered immediately?",
            "question_ar": "لماذا لا يستجاب كل دعاء فوراً؟",
            "core_answer_summary_en": "Prayer in Islam is answered according to divine wisdom.",
            "core_answer_summary_ar": "الدعاء في الإسلام يستجاب وفق الحكمة الإلهية البالغة.",
            "verbatim_quote": "Can",
            "dialogue_exchange": [
                {"turn_id": "T0548", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "You pray. Can you ask and you shall receive."},
                {"turn_id": "T0547", "role": "MAIN_DAEE", "move_type": "Socratic Turn", "text_en": "Can"},
            ],
        }
        enriched = anchor_and_enrich_extracted_unit(raw_unit, scenario, 1, 2400000, 1)
        self.assertIsNotNone(enriched)
        self.assertEqual(enriched["answer_turn_id"], "T0549")
        self.assertEqual(enriched["answer_timestamp_seconds"], 2453)
        self.assertEqual(
            enriched["verbatim_quote"],
            "In Islam when we pray to Allah He answers according to His infinite wisdom and our true benefit.",
        )

    def test_round2_per_unit_arabic_retry_when_batch_r2_truncates(self):
        """Ensures extract_video_m6 retries per-unit Arabic localization if initial Round 2 returns empty []."""
        import tempfile
        import json
        from pathlib import Path
        from core.drive_sync import SharedWorkspace
        from core.scenario_extractor import Stage2ScenarioExtractor

        class TruncatingBatchR2MockClient:
            def __init__(self):
                self.r2_calls = 0

            def extract_round1_english(self, prompt: str) -> str:
                return json.dumps([
                    {
                        "question_turn_id": "V1_T01",
                        "answer_turn_id": "V1_T02",
                        "question_en": "Why do Muslims worship Allah alone instead of created beings?",
                        "topic_category": "Theology & God's Existence",
                        "core_answer_summary_en": "Only the uncreated Necessary Being who originated the universe deserves worship.",
                        "logical_premises": ["P1", "P2", "C"],
                        "dialogue_exchange": [
                            {"turn_id": "V1_T01", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Why worship the Creator alone?"},
                            {"turn_id": "V1_T02", "role": "MAIN_DAEE", "move_type": "Deductive Proof", "text_en": "Because only the uncreated Creator who gave you life deserves worship."},
                        ],
                    }
                ])

            def extract_round2_arabic(self, english_units_json: str, prompt: str) -> str:
                self.r2_calls += 1
                # Simulate initial batch Round 2 truncation / <think> exhaustion returning no valid JSON
                return "<think>Thinking too long without closing JSON..."

            def extract(self, prompt: str, system_prompt: str = None) -> str:
                # Per-unit retry uses COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT via .extract()
                return json.dumps([
                    {
                        "question_turn_id": "V1_T01",
                        "answer_turn_id": "V1_T02",
                        "question_ar": "لماذا يعبد المسلمون الله وحده دون الكائنات المخلوقة؟",
                        "core_answer_summary_ar": "لأن الخالق الأزلي واجب الوجود الذي أوجد الكون من العدم هو وحده المستحق للعبادة بحق.",
                        "logical_premises_ar": ["المقدمة الأولى", "المقدمة الثانية", "النتيجة"],
                        "dialogue_exchange": [
                            {"turn_id": "V1_T01", "role": "INTERLOCUTOR", "move_type": "Objection", "text_ar": "لماذا نعبد الخالق وحده؟"},
                            {"turn_id": "V1_T02", "role": "MAIN_DAEE", "move_type": "Deductive Proof", "text_ar": "لأن الخالق غير المخلوق الذي وهبك الحياة هو وحده المستحق للعبادة."},
                        ],
                    }
                ], ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmp:
            ws = SharedWorkspace(root_dir=Path(tmp), worker_name="test_r2_retry")
            scen = {
                "video_id": "V1",
                "title_ar": "مناظرة التوحيد",
                "youtube_url": "https://www.youtube.com/watch?v=V1",
                "speakers": [
                    {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "profile_id": "daee_uthman", "display_name_en": "Sheikh Uthman", "display_name_ar": "الشيخ عثمان"},
                    {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "profile_id": None, "display_name_en": "Interlocutor", "display_name_ar": "المحاور"},
                ],
                "turns": [
                    {"turn_id": "V1_T01", "episode_index": 1, "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "start_ms": 1000, "end_ms": 6000, "start_fmt": "00:01", "end_fmt": "00:06", "text_original": "Why do Muslims worship the Creator alone instead of worshipping any powerful creature in the world?"},
                    {"turn_id": "V1_T02", "episode_index": 1, "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 7000, "end_ms": 18000, "start_fmt": "00:07", "end_fmt": "00:18", "text_original": "Because only the uncreated Creator who gave you life and sustains the entire universe deserves worship."},
                ],
            }
            ws.atomic_write_json(ws.scenarios_dir / "V1.json", scen)
            ws.advance_micro_step("V1", "M5_SCENARIO_BUILT")

            mock_client = TruncatingBatchR2MockClient()
            extractor = Stage2ScenarioExtractor(workspace=ws, llm_client=mock_client)
            units = extractor.extract_video_m6("V1", verbose=False)
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0]["question_ar"], "لماذا يعبد المسلمون الله وحده دون الكائنات المخلوقة؟")
            self.assertIn("الخالق الأزلي", units[0]["core_answer_summary_ar"])
            self.assertEqual(units[0]["dialogue_exchange"][1]["text_ar"], "لأن الخالق غير المخلوق الذي وهبك الحياة هو وحده المستحق للعبادة.")

    def test_find_valid_drive_gguf_promotes_completed_tmp_file(self):
        """Ensures _find_valid_drive_gguf recovers a completed .tmp GGUF left by DriveFS FUSE."""
        import tempfile
        from pathlib import Path
        from core.colab_runner import _find_valid_drive_gguf

        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp)
            models_dir = drive_root / ".cache" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            tmp_gguf = models_dir / "Qwen3-14B-Q5_K_M.gguf.tmp"
            tmp_gguf.write_bytes(b"x" * 1024)

            found = _find_valid_drive_gguf(drive_root, "Qwen3-14B-Q5_K_M.gguf", min_valid_size=512)
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "Qwen3-14B-Q5_K_M.gguf")
            self.assertFalse(tmp_gguf.exists())


if __name__ == "__main__":
    unittest.main()
