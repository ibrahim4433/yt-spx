#!/usr/bin/env python3
"""
tests/test_audit_root_causes.py
Comprehensive regression tests for the 6 deep root causes identified in the 63-card audit:
  1. Few-shot prompt demonstration leakage prevention (train carriage, first cause, Mark 13:32, etc.)
  2. Round 1 + Round 2 Transactional Barrier (blocking empty Arabic fields)
  3. Apologetic semantic pre-filtering gatekeeper (police encounters, dog petting, outro fundraising)
  4. 1st-person confessional inversion normalization and question syntax enforcement (? and ؟)
  5. Domain-blind machine translation drift glossary repairs (Arabia, synchronicities, Taoism, biblical names)
  6. Speaker role collapse prevention (daee_speaker_id != interlocutor_speaker_id)
"""

import unittest
import tempfile
import sqlite3
from pathlib import Path

from core.scenario_extractor import (
    sanitize_arabic_fields,
    normalize_question_syntax_and_perspective,
    is_substantive_apologetics_window,
    is_valid_dialectical_question,
    has_valid_arabic_localization,
    anchor_and_enrich_extracted_unit,
)
from core.colab_runner import sanitize_extracted_arabic_unit
from db.database import (
    clean_latin_tokens_in_arabic,
    _insert_qa_unit_in_tx,
    init_db,
    get_db_connection,
)


class TestAuditRootCauses(unittest.TestCase):

    def test_01_anti_leakage_gate(self):
        """Ensures cards with prompt demonstration signatures not in transcript are dropped."""
        scen = {
            "video_id": "vid_leak_test",
            "title_ar": "حوار تجريبي",
            "speakers": [
                {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "display_name_en": "Hashim", "display_name_ar": "هاشم"}
            ],
            "turns": [
                {"turn_id": "T1", "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "text_original": "Who created God?", "start_ms": 0, "end_ms": 5000},
                {"turn_id": "T2", "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "text_original": "God is eternal and uncreated.", "start_ms": 5000, "end_ms": 15000}
            ]
        }
        item_leak = {
            "question_en": "Does God have a cause?",
            "core_answer_summary_en": "The rule of science and philosophy is: whatever begins to exist has a cause.",
            "question_turn_id": "T1",
            "answer_turn_id": "T2"
        }
        res = anchor_and_enrich_extracted_unit(item_leak, scen, 1, 0, 1)
        self.assertIsNone(res, "Anti-leakage gate must drop card with ungrounded prompt demonstration text!")

    def test_02_transactional_barrier(self):
        """Ensures cards lacking valid Arabic localization fail the transactional barrier."""
        card_empty_ar = {
            "question_en": "Who created God?",
            "question_ar": "",
            "core_answer_summary_en": "God is eternal.",
            "core_answer_summary_ar": ""
        }
        self.assertFalse(has_valid_arabic_localization(card_empty_ar))

        card_valid_ar = {
            "question_en": "Who created God?",
            "question_ar": "من خلق الله؟",
            "core_answer_summary_en": "God is eternal.",
            "core_answer_summary_ar": "الله أزلي واجب الوجود لا بداية له."
        }
        self.assertTrue(has_valid_arabic_localization(card_valid_ar))

    def test_03_apologetic_semantic_pre_filtering(self):
        """Ensures non-apologetic windows (police encounters, dog petting, outro fundraising) are filtered out."""
        turns_police = [
            {"text_original": "Excuse me officer, what kind of a dog is this? Can I pet the dog?"},
            {"text_original": "He is a German Shepherd, you may pet him. We are patrolling the park today."}
        ]
        is_sub, reason = is_substantive_apologetics_window(turns_police, 1, 1)
        self.assertFalse(is_sub)
        self.assertIn("police", reason.lower())

        turns_outro = [
            {"text_original": "Thank you for watching this full debate. Please donate now to support our dawah at islamnet, link in the description, subscribe to the channel and hit the bell."}
        ]
        is_sub_o, reason_o = is_substantive_apologetics_window(turns_outro, 1, 1)
        self.assertFalse(is_sub_o)
        self.assertIn("promotional", reason_o.lower())

    def test_04_confessional_inversion_normalization(self):
        """Ensures 1st-person confessional questions are converted into objective seeker questions with syntax."""
        q_en = "Why do I oppose Islam if I have not read the Quran"
        q_ar = "لماذا أعارض الإسلام وأنا لم أقرأ القرآن"
        norm_en, norm_ar, valid = normalize_question_syntax_and_perspective(q_en, q_ar)
        self.assertTrue(valid)
        self.assertNotIn("Why do I", norm_en)
        self.assertTrue(norm_en.endswith("?"))
        self.assertNotIn("لماذا أعارض", norm_ar)
        self.assertTrue(norm_ar.endswith("؟"))

    def test_05_domain_blind_translation_glossary_repairs(self):
        """Ensures apologetics domain glossary corrects translation drift across extraction, Colab, and DB."""
        sample_drift = "نشأ في عربة في القرن السادس معتنقاً مذهب الطائفة وكتب الرسول بول في إنجيل ماتيو عن المؤامرات النفسية"

        # scenario_extractor (sanitizes question_ar)
        cleaned_extr = sanitize_arabic_fields({"question_ar": sample_drift})["question_ar"]
        self.assertIn("الجزيرة العربية في القرن السادس", cleaned_extr)
        self.assertIn("المذهب الطاوي", cleaned_extr)
        self.assertIn("بولس الرسول", cleaned_extr)
        self.assertIn("إنجيل متى", cleaned_extr)
        self.assertIn("التزامنات النفسية", cleaned_extr)

        # colab_runner
        cleaned_colab = sanitize_extracted_arabic_unit({"question_ar": sample_drift})["question_ar"]
        self.assertIn("الجزيرة العربية في القرن السادس", cleaned_colab)
        self.assertIn("المذهب الطاوي", cleaned_colab)
        self.assertIn("بولس الرسول", cleaned_colab)

        # db/database
        cleaned_db = clean_latin_tokens_in_arabic(sample_drift)
        self.assertIn("الجزيرة العربية في القرن السادس", cleaned_db)
        self.assertIn("المذهب الطاوي", cleaned_db)
        self.assertIn("بولس الرسول", cleaned_db)

    def test_06_speaker_role_collapse_guard(self):
        """Ensures daee_speaker_id and interlocutor_speaker_id can never be identical."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_collapse.db"
            init_db(db_path)
            conn = get_db_connection(db_path)
            try:
                conn.execute(
                    "INSERT INTO videos (video_id, channel, title_ar, youtube_url) "
                    "VALUES ('test_vid', 'SCDawah Arabic', 'فيديو تجريبي', 'https://youtu.be/test_vid')"
                )
                bad_spk_unit = {
                    "id": "test_collapse_001",
                    "video_id": "test_vid",
                    "daee_speaker_id": "SPK_00",
                    "interlocutor_speaker_id": "SPK_00",  # Collapsed speaker ID!
                    "question_en": "Who is God",
                    "question_ar": "من هو الله",
                    "core_answer_summary_en": "God is the Necessary Creator.",
                    "core_answer_summary_ar": "الله هو الخالق واجب الوجود."
                }
                _insert_qa_unit_in_tx(conn, bad_spk_unit)
                row = conn.execute(
                    "SELECT daee_speaker_id, interlocutor_speaker_id, question_en, question_ar FROM qa_units WHERE id = ?",
                    ("test_collapse_001",)
                ).fetchone()
                self.assertEqual(row[0], "SPK_00")
                self.assertEqual(row[1], "", "Interlocutor ID must be cleared when it matches Da'ee ID!")
                self.assertTrue(row[2].endswith("?"), "English question must end with '?'")
                self.assertTrue(row[3].endswith("؟"), "Arabic question must end with '؟'")
            finally:
                conn.close()


from core.extractor_prompt import ROUND1_ENGLISH_EXTRACTION_PROMPT
from db.database import resolve_turn_speaker_labels, _sanitize_and_validate_qa_unit


class TestDeepAuditRegressions(unittest.TestCase):
    """Regression suite covering exact card defects discovered during the 63-card sample audit."""

    def _base_scenario(self, vid: str = "reg_vid_001"):
        return {
            "video_id": vid,
            "youtube_url": f"https://www.youtube.com/watch?v={vid}",
            "title_ar": "مناظرة تجريبية",
            "speakers": [
                {
                    "speaker_id": "SPK_00",
                    "role": "MAIN_DAEE",
                    "profile_id": "daee_mansur",
                    "display_name_en": "Mansur",
                    "display_name_ar": "منصور",
                },
                {
                    "speaker_id": "SPK_01",
                    "role": "INTERLOCUTOR",
                    "profile_id": None,
                    "display_name_en": "Interlocutor 1",
                    "display_name_ar": "المحاور",
                },
            ],
            "turns": [
                {
                    "turn_id": f"{vid}_T0001",
                    "turn_index": 1,
                    "speaker_id": "SPK_01",
                    "speaker_role": "INTERLOCUTOR",
                    "speaker_label_en": "Mansur",
                    "speaker_label_ar": "منصور",
                    "start_ms": 10000,
                    "end_ms": 20000,
                    "start_fmt": "00:10",
                    "end_fmt": "00:20",
                    "text_original": "Why does God prescribe moral laws for humanity if human societies evolve?",
                },
                {
                    "turn_id": f"{vid}_T0002",
                    "turn_index": 2,
                    "speaker_id": "SPK_00",
                    "speaker_role": "MAIN_DAEE",
                    "speaker_label_en": "Mansur",
                    "speaker_label_ar": "منصور",
                    "start_ms": 21000,
                    "end_ms": 50000,
                    "start_fmt": "00:21",
                    "end_fmt": "00:50",
                    "text_original": "Objective moral duties require a transcendent Lawgiver; human consensus alone is changeable and cannot ground timeless moral obligations.",
                },
            ],
        }

    def test_07_exact_sample_arabic_drift_and_hebrew_repairs(self):
        cases = [
            ("كيف يمكن لِلَّهِ أن يُنظم الصدف والموامرات؟", "التزامنات", "الموامرات"),
            (
                "هل شهادة بول كافية لتأكيد صحة الإنجيل، بعد التركيز على بول قبل ماتيو؟",
                "بولس",
                "ماتيو",
            ),
            ("كيف يمكن لرجل في القرن السادس الميلادي في عربة أن يعرف ذلك؟", "في الجزيرة العربية", "في عربة"),
            ("مقارنة بالديانات الأخرى مثل البوذية والطائفة والكونفوشيوسية؟", "الطاوية", "الطائفة"),
        ]
        for raw_ar, expected_in, forbidden_out in cases:
            extr = sanitize_arabic_fields({"question_ar": raw_ar})["question_ar"]
            colab = sanitize_extracted_arabic_unit({"question_ar": raw_ar})["question_ar"]
            db_res = clean_latin_tokens_in_arabic(raw_ar)
            for out in (extr, colab, db_res):
                self.assertIn(expected_in, out)
                self.assertNotIn(forbidden_out, out)

        heb_item = sanitize_arabic_fields({
            "question_ar": "ما موقف الإسلام من العدل؟",
            "core_answer_summary_ar": "العدل واجب للجميع.",
            "verbatim_quote_ar": "قتل الأبرياء محرم حتى للיהודים وغيرهم.",
            "dialogue_exchange": [{"turn_id": "T1", "role": "MAIN_DAEE", "text_ar": "لا يجوز الاعتداء على יהודים أبرياء."}],
        })
        self.assertIn("لليهود", heb_item["verbatim_quote_ar"])
        self.assertNotRegex(heb_item["verbatim_quote_ar"], r"[\u0590-\u05ff]")
        self.assertNotRegex(heb_item["dialogue_exchange"][0]["text_ar"], r"[\u0590-\u05ff]")

    def test_08_confessional_trailing_clauses_and_declarative_syntax(self):
        n_en1, n_ar1, ok1 = normalize_question_syntax_and_perspective(
            "Why do I hold anti-Islamic views if I haven't read Islamic sources directly, but rather based on surveys and the behavior of some Muslims in society?",
            "لماذا أحمل رؤية معادية للإسلام رغم أنني لم أقرأ المصادر الإسلامية مباشرة، بل استندت إلى استطلاعات الرأي وسلوك بعض المسلمين في المجتمع؟",
        )
        self.assertTrue(ok1)
        self.assertNotRegex(n_en1, r"\?\s+[a-z]")
        self.assertNotIn("if I haven't", n_en1)
        self.assertNotIn("أنني", n_ar1)

        n_en2, n_ar2, ok2 = normalize_question_syntax_and_perspective(
            "Why do I oppose Islam if I haven't read the Quran and only have a superficial understanding of it?",
            "لماذا أعارض الإسلام إذا لم أقرأ القرآن وأملك فهمًا سطحيًا له؟",
        )
        self.assertTrue(ok2)
        self.assertNotRegex(n_en2, r"\?\s+[a-z]")
        self.assertNotIn("وأملك", n_ar2)

        _, _, bad_conf = normalize_question_syntax_and_perspective(
            "Why do I hate Islam and reject all its teachings in my heart?",
            "لماذا أكره الإسلام؟",
        )
        self.assertFalse(bad_conf)

        self.assertFalse(
            is_valid_dialectical_question(
                "You believe that God exists, who's incredibly powerful, incredibly intelligent, and just exists on course."
            )
        )
        self.assertFalse(
            is_valid_dialectical_question(
                "Quantum physics explains it all. Quantum physics explains why... So I have to now assume you understand quantum physics, right?"
            )
        )

    def test_09_prompt_leakage_citations_analogies_and_mark1332(self):
        for leaked_example in ("train engine without wheels", "surgeon operating on an unknowing child", "power station", "Mark 13:32"):
            self.assertNotIn(leaked_example, ROUND1_ENGLISH_EXTRACTION_PROMPT)

        scen = self._base_scenario("leak_vid")
        unit_cit = {
            "question_turn_id": "leak_vid_T0001",
            "answer_turn_id": "leak_vid_T0002",
            "question_en": "Does God have absolute moral authority over humanity?",
            "question_ar": "هل لله السلطة الأخلاقية المطلقة على البشر؟",
            "core_answer_summary_en": "Objective moral duties require a transcendent Lawgiver.",
            "core_answer_summary_ar": "الواجبات الأخلاقية الموضوعية تستلزم مشرعاً متعالياً.",
            "scriptural_citations": [{"source": "Bible", "reference": "Mark 13:32", "significance": "Quoted text"}],
            "verbatim_quote": scen["turns"][1]["text_original"],
            "interlocutor_verbatim_quote": scen["turns"][0]["text_original"],
        }
        self.assertIsNone(anchor_and_enrich_extracted_unit(unit_cit, scen, 1, 0, 1))

        unit_an = dict(unit_cit, scriptural_citations=[], analogies=["A train cannot move on its own without an engine."])
        self.assertIsNone(anchor_and_enrich_extracted_unit(unit_an, scen, 1, 0, 1))

        scen_spoken = self._base_scenario("spoken_mark")
        scen_spoken["turns"][1]["text_original"] += " Even in Mark 13:32 Jesus said only the Father knows the hour."
        unit_spoken = dict(unit_cit, verbatim_quote=scen_spoken["turns"][1]["text_original"])
        unit_spoken["question_turn_id"] = "spoken_mark_T0001"
        unit_spoken["answer_turn_id"] = "spoken_mark_T0002"
        self.assertIsNotNone(anchor_and_enrich_extracted_unit(unit_spoken, scen_spoken, 1, 0, 1))

    def test_10_noise_false_positive_safety_and_greeting_hallucination(self):
        scen = self._base_scenario("fp_vid")
        valid_with_police_analogy = {
            "question_turn_id": "fp_vid_T0001",
            "answer_turn_id": "fp_vid_T0002",
            "question_en": "Why does God prescribe moral laws for humanity if human societies evolve?",
            "question_ar": "لماذا يشرع الله قوانين أخلاقية للبشر مع تطور المجتمعات؟",
            "core_answer_summary_en": "Moral authority comes from the Creator; just as a citizen cannot ignore the law even when no police officer is watching, objective morality does not depend on human consensus.",
            "core_answer_summary_ar": "المرجعية الأخلاقية من الخالق، كمن يلتزم بالقانون حتى دون وجود رقيب.",
            "analogy_or_metaphor": "Like obeying traffic laws even when no police officer is present.",
            "verbatim_quote": scen["turns"][1]["text_original"],
            "interlocutor_verbatim_quote": scen["turns"][0]["text_original"],
        }
        self.assertIsNotNone(anchor_and_enrich_extracted_unit(valid_with_police_analogy, scen, 1, 0, 1))

        # Short remark whose holiday topic does not appear in Da'ee's actual response
        scen_holiday = self._base_scenario("hol_vid")
        scen_holiday["turns"][0]["text_original"] = "That you celebrate Christmas."
        scen_holiday["turns"][1]["text_original"] = "The Hebrew phrase sons of God simply means righteous servants chosen by God."
        hallucinated = {
            "question_turn_id": "hol_vid_T0001",
            "answer_turn_id": "hol_vid_T0002",
            "question_en": "Why do Muslims celebrate Christmas if it is a Christian holiday?",
            "question_ar": "لماذا يحتفل المسلمون بعيد الميلاد؟",
            "core_answer_summary_en": "Celebrating Christmas culturally is separate from theology.",
            "core_answer_summary_ar": "الاحتفال الثقافي يختلف عن العقيدة.",
            "verbatim_quote": scen_holiday["turns"][1]["text_original"],
            "interlocutor_verbatim_quote": "That you celebrate Christmas.",
        }
        self.assertIsNone(anchor_and_enrich_extracted_unit(hallucinated, scen_holiday, 1, 0, 1))

        # Da'ee rhetorical cross-examination rejected as question_en
        self.assertFalse(
            is_valid_dialectical_question(
                "If God is all-loving, why does He not love sinners? Should He love a rapist for their rape or a murderer for their murder?"
            )
        )

        # Last-window fundraising appeal mentioning masjid/Allah (>120 words) still blocked
        outro_turns = [{"text_original": " ".join(["Please support Islamnet and donate today to build the masjid for the sake of Allah."] * 14)}]
        ok_outro, _ = is_substantive_apologetics_window(outro_turns, 2, 3)
        self.assertFalse(ok_outro)

    def test_11_dialogue_speaker_label_integrity_and_hebrew_localization_gate(self):
        scen = self._base_scenario("spk_lbl")
        unit = {
            "question_turn_id": "spk_lbl_T0001",
            "answer_turn_id": "spk_lbl_T0002",
            "question_en": "Why does God prescribe moral laws for humanity?",
            "question_ar": "لماذا يشرع الله قوانين أخلاقية للبشر؟",
            "core_answer_summary_en": "Objective moral duties require a transcendent Lawgiver.",
            "core_answer_summary_ar": "الواجبات الأخلاقية الموضوعية تستلزم مشرعاً متعالياً.",
            "verbatim_quote": scen["turns"][1]["text_original"],
            "interlocutor_verbatim_quote": scen["turns"][0]["text_original"],
            "dialogue_exchange": [
                {"turn_id": "spk_lbl_T0001", "role": "INTERLOCUTOR", "text_en": "Why moral law?", "text_ar": "لم القانون الأخلاقي؟"},
                {"turn_id": "spk_lbl_T0002", "role": "MAIN_DAEE", "text_en": "Transcendent Lawgiver.", "text_ar": "بسبب وجود المشرع المتعالي."},
            ],
        }
        enriched = anchor_and_enrich_extracted_unit(unit, scen, 1, 0, 1)
        self.assertIsNotNone(enriched)
        self.assertEqual(enriched["dialogue_exchange"][0]["speaker_label_en"], "Interlocutor")
        self.assertEqual(enriched["dialogue_exchange"][0]["speaker_label_ar"], "المحاور")
        self.assertEqual(enriched["dialogue_exchange"][1]["speaker_label_en"], "Mansur")

        # Direct DB sanitization path must also never label INTERLOCUTOR with Da'ee name
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE video_speakers (video_id TEXT, role TEXT, profile_id TEXT, display_name_ar TEXT, display_name_en TEXT, speaker_id TEXT)")
        conn.execute("CREATE TABLE videos (video_id TEXT, primary_daee_hint TEXT)")
        conn.execute("CREATE TABLE speaker_profiles (profile_id TEXT, name_ar TEXT, name_en TEXT)")
        conn.execute("CREATE TABLE scenario_turns (turn_id TEXT, start_ms INTEGER, start_fmt TEXT, speaker_role TEXT, speaker_label_en TEXT, speaker_label_ar TEXT)")
        conn.execute("INSERT INTO scenario_turns VALUES ('spk_lbl_T0001', 10000, '00:10', 'INTERLOCUTOR', 'Mansur', 'منصور')")
        sanitized = _sanitize_and_validate_qa_unit(conn, {
            "video_id": "spk_lbl",
            "daee_name_en": "Mansur",
            "daee_name_ar": "منصور",
            "dialogue_exchange": [{"turn_id": "spk_lbl_T0001", "role": "INTERLOCUTOR", "text_en": "Hi", "text_ar": "مرحبًا"}],
        })
        self.assertEqual(sanitized["dialogue_exchange"][0]["speaker_label_en"], "Interlocutor")
        conn.close()

        self.assertFalse(has_valid_arabic_localization({
            "question_ar": "سؤال صحيح بالعربية؟",
            "core_answer_summary_ar": "جواب صحيح بالعربية.",
            "verbatim_quote_ar": "اقتباس فيه كلمة יהודים عبرية",
        }))

    def test_12_daee_counter_ask_perspective_and_rubric_integration(self):
        """Ensures Da'ee defensive counter-asks are inverted/rejected and Round 2 conditional EN repair respects rubric audit."""
        from core.extractor_prompt import ROUND1_ENGLISH_EXTRACTION_PROMPT, ROUND2_ARABIC_VERIFICATION_PROMPT
        from core.scenario_extractor import merge_round1_and_round2_units
        from tools.card_rating_system.audit_cards import audit_single_card, format_pre_round2_rubric_summary

        # 1. Prompt parity checks for Seeker-Intent Polarity & non-forced Round 2 rubric checks
        self.assertIn("SEEKER-INTENT POLARITY TEST", ROUND1_ENGLISH_EXTRACTION_PROMPT)
        self.assertIn("what is the problem with that?", ROUND1_ENGLISH_EXTRACTION_PROMPT)
        self.assertIn("فما المشكل في ذلك؟", ROUND2_ARABIC_VERIFICATION_PROMPT)
        self.assertIn("english_edits_needed", ROUND2_ARABIC_VERIFICATION_PROMPT)

        # 2. Exact Da'ee defensive rejoinder EN + AR normalization to seeker perspective
        norm_en, norm_ar, ok = normalize_question_syntax_and_perspective(
            "If every man in heaven has 72 virgins following him around, what is the problem with that?",
            "إذا كان كل رجل في الجنة يُتبعه 72 فتاة، فما هو المشكل في ذلك؟",
        )
        self.assertTrue(ok)
        self.assertEqual(norm_en, "If every man in heaven has 72 virgins following him around, isn't that a bad thing?")
        self.assertEqual(norm_ar, "إذا كان كل رجل في الجنة يُتبعه 72 فتاة، أليس ذلك أمرًا سيئًا؟")

        # 3. Bare un-rewritable Da'ee defensive filler must be rejected in EN & AR
        _, _, ok_bare_en = normalize_question_syntax_and_perspective("What is the problem with that?", "")
        _, _, ok_bare_ar = normalize_question_syntax_and_perspective("", "فما هو المشكل في ذلك؟")
        self.assertFalse(ok_bare_en)
        self.assertFalse(ok_bare_ar)
        self.assertFalse(is_valid_dialectical_question("What is the problem with that?"))
        self.assertFalse(is_valid_dialectical_question("", "ما المشكل في ذلك؟"))

        # 4. Genuine seeker question mentioning "problem with the Trinity" must stay untouched
        keep_en, _, ok_genuine = normalize_question_syntax_and_perspective(
            "What is the problem with the Trinity in early church councils?",
            "ما الإشكال العقدي في عقيدة التثليث؟",
        )
        self.assertTrue(ok_genuine)
        self.assertEqual(keep_en, "What is the problem with the Trinity in early church councils?")

        # 5. Rubric audit integration + conditional (not forced) Round 2 merge
        scen = self._base_scenario("rubric_vid")
        clean_r1 = {
            "question_turn_id": "rubric_vid_T0001",
            "answer_turn_id": "rubric_vid_T0002",
            "question_en": "Does God prescribe objective moral duties for humanity?",
            "core_answer_summary_en": (
                "Objective moral duties require a transcendent Lawgiver beyond subjective human opinion or evolving cultural conventions, "
                "because binding moral accountability cannot originate from blind unconscious matter or shifting social preference alone. "
                "When secular materialism reduces ethics to biological survival impulses or consensus, it strips moral obligation of any unconditional authority. "
                "Only the All-Wise Creator provides an unchanging ontological foundation grounding justice, human dignity, and universal accountability."
            ),
            "logical_premises": [
                "Premise 1: Binding moral duties cannot arise solely from arbitrary subjective preference.",
                "Premise 2: Objective moral obligations hold universally across cultures.",
                "Conclusion: Therefore an objective transcendent Lawgiver must exist.",
            ],
            "fallacy_or_misconception_exposed": "Moral Relativism: Conflating shifting human conventions with objective moral ontology.",
            "socratic_question": "Without a Creator, why are your moral intuitions more binding than anyone else's?",
            "verbatim_quote": scen["turns"][1]["text_original"],
            "interlocutor_verbatim_quote": scen["turns"][0]["text_original"],
            "dialogue_exchange": [
                {"turn_id": "rubric_vid_T0001", "role": "INTERLOCUTOR", "text_en": scen["turns"][0]["text_original"]},
                {"turn_id": "rubric_vid_T0002", "role": "MAIN_DAEE", "text_en": scen["turns"][1]["text_original"]},
            ],
        }
        pre_eval = audit_single_card(clean_r1, scen["turns"], require_arabic=False)
        self.assertFalse(pre_eval["needs_english_repair"])
        summary_txt = format_pre_round2_rubric_summary([clean_r1], scen["turns"])
        self.assertIn("Status=CLEAN", summary_txt)

        # Clean card with english_edits_needed=false must preserve R1 English untouched even if R2 tries paraphrasing
        merged_clean = merge_round1_and_round2_units(
            [clean_r1],
            [{
                "question_turn_id": "rubric_vid_T0001",
                "rubric_audit": {"score": 95, "english_edits_needed": False, "drop_card": False},
                "question_en": "Attempted unnecessary paraphrase?",
                "question_ar": "هل شرع الله واجبات أخلاقية موضوعية للبشرية؟",
                "core_answer_summary_ar": "الواجبات الأخلاقية الموضوعية تستلزم مشرعًا متعاليًا.",
                "logical_premises_ar": ["المقدمة الأولى", "المقدمة الثانية", "النتيجة"],
                "fallacy_or_misconception_exposed_ar": "النسبية الأخلاقية",
                "analogy_or_metaphor_ar": "",
                "socratic_question_ar": "لولا الخالق فما مصدر الإلزام الأخلاقي؟",
                "verbatim_quote_ar": "الأخلاق الموضوعية تستلزم مشرعًا.",
            }],
        )
        self.assertEqual(merged_clean[0]["question_en"], clean_r1["question_en"])

        # Defective card with english_edits_needed=true applies targeted fix; drop_card=true flags drop
        merged_repaired = merge_round1_and_round2_units(
            [dict(clean_r1, question_en="Why does God order morality, what is the problem with that?")],
            [{
                "question_turn_id": "rubric_vid_T0001",
                "rubric_audit": {"score": 88, "english_edits_needed": True, "issues_fixed": ["DAEE_COUNTER_ASK"], "drop_card": False},
                "question_en": "If God prescribes absolute moral laws, isn't that restrictive?",
                "question_ar": "إذا فرض الله قوانين أخلاقية مطلقة، أليس ذلك تقييدًا للحرية؟",
                "core_answer_summary_ar": "الواجبات الأخلاقية الموضوعية تستلزم مشرعًا متعاليًا.",
            }],
        )
        self.assertEqual(merged_repaired[0]["question_en"], "If God prescribes absolute moral laws, isn't that restrictive?")

        merged_dropped = merge_round1_and_round2_units(
            [clean_r1],
            [{"question_turn_id": "rubric_vid_T0001", "rubric_audit": {"drop_card": True, "hard_fail": "[GATE-NOISE]"}}],
        )
        self.assertTrue(merged_dropped[0].get("_dropped_by_round2_audit"))


if __name__ == "__main__":
    unittest.main()
