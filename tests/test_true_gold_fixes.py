"""
Comprehensive Regression Test Suite for True Gold Root-Cause Fixes.

Covers:
1. Pre-compiled Arabic regex sanitizer speed & broken-punctuation collapse
2. Expanded theological bias guard (Trinitarian divinity, child slaughter, pantheism, LGBTQ distortions vs refutations)
3. Stage 1 complete-linkage speaker clustering & anti-collapse guard (blocking K=1 collapse)
4. Stage 1 M3 anti-collapse diarization recovery & same-voice rejection
5. Stage 1 cameo voiceprint-bonus guard (<15% share -> CO_DAEE) & role canonicalization
6. Stage 1 3-window majority-vote language detection (Arabic intro + English debate -> 'en')
7. Mark 13:32 spaced/punctuation leak detection in audit_cards & anchor_and_enrich
8. Multi-card Round 1/Round 2 merge blocking positional Arabic cross-contamination
9. Pre-Round-2 grounding, 6-gate post-Round-2 drop, and zero-card diagnostic tracking
10. Video-wide semantic deduplication while preserving distinct numbered inquiries
11. Stage 2 reset (qa_count=0, stage2_status='pending', state reset to M5_SCENARIO_BUILT)
12. SmartDialecticBot ubiquitous-token weighting, FTS punctuation stripping, live video qa_count, per-video overlaps, & 13 categories
"""

import json
import sqlite3
import tempfile
import time
import unittest
import wave
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.audio_diarizer import WarmASRAlignmentEngine, _collapse_turn_repetition_loops
from core.drive_sync import SharedWorkspace
from core.extractor_prompt import ROUND2_ARABIC_VERIFICATION_PROMPT
from core.overlap_resolver import (
    WarmNeMoDiarizer,
    _is_collapsed_diarization,
    merge_fragmented_speaker_clusters,
    run_diarization_m3,
)
from core.scenario_extractor import (
    Stage2ScenarioExtractor,
    _ground_dialogue_exchange,
    _is_duplicate_qa,
    _lookup_turn,
    anchor_and_enrich_extracted_unit,
    diagnose_missing_arabic_fields,
    merge_round1_and_round2_units,
)
from core.smart_bot import SmartDialecticBot, _stem_token, normalize_text
from core.speaker_roles import classify_speakers_and_episodes
from db.database import (
    _clean_arabic_unit_str,
    canonicalize_speaker_role,
    check_theological_bias,
    clean_latin_tokens_in_arabic,
    get_categories_breakdown,
    get_stats,
    ingest_extraction_payload,
    init_db,
    list_videos,
    sanitize_fts_query,
)
from tools.card_rating_system.audit_cards import DAEE_CROSS_EXAM_REGEX, audit_single_card
import tools.reset_stage2 as reset_stage2_mod


def _write_silent_wav(path: Path, duration_sec: float, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sr)
    samples = np.zeros(n_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


class TestDatabaseAndSanitizerFixes(unittest.TestCase):
    def test_clean_latin_tokens_in_arabic_speed_and_broken_punctuation(self):
        self.assertEqual(_clean_arabic_unit_str("د. ?. ? , ? ? ,"), "")
        self.assertEqual(_clean_arabic_unit_str("?. , ?"), "")
        self.assertNotIn("?", clean_latin_tokens_in_arabic("نص عربي ?. ? , ? ? ,"))

        sample = "إن الله هو الخالق واجب الوجود وليس كمثله شيء في الوجود كله. " * 25
        t0 = time.perf_counter()
        for _ in range(40):
            res = clean_latin_tokens_in_arabic(sample)
        elapsed = time.perf_counter() - t0
        self.assertIn("واجب الوجود", res)
        self.assertLess(elapsed, 0.5, f"clean_latin_tokens_in_arabic too slow: {elapsed:.3f}s")

    def test_expanded_theological_bias_and_refutation_exceptions(self):
        bad_trinity = {
            "core_answer_summary_en": "Jesus is fully God and part of the Trinity in hypostatic union.",
            "verbatim_quote": "God the Son is co-equal with the Father.",
        }
        self.assertIsNotNone(check_theological_bias(bad_trinity))

        bad_amalekite = {
            "core_answer_summary_en": "Commanding to kill infants and children of the Amalekites was morally justified.",
        }
        self.assertIsNotNone(check_theological_bias(bad_amalekite))

        bad_pantheism = {
            "core_answer_summary_en": "Man reaches union with the divine when the soul dissolves into God.",
        }
        self.assertIsNotNone(check_theological_bias(bad_pantheism))

        bad_nonbinary = {
            "core_answer_summary_en": "The Quran affirms non-binary gender identities and approves same-sex marriage.",
        }
        self.assertIsNotNone(check_theological_bias(bad_nonbinary))

        sound_refutation_1 = {
            "core_answer_summary_en": "Islam categorically refutes the claim that Jesus is fully God or part of a Trinity; prophets only worshipped the One God.",
            "verbatim_quote": "The Bible itself refutes the Trinity because Jesus prayed to the Father.",
        }
        self.assertIsNone(check_theological_bias(sound_refutation_1))

        sound_refutation_2 = {
            "core_answer_summary_en": "Islam strictly prohibits targeting infants and children in warfare and rejects the claim that such slaughter is commanded in revelation.",
        }
        self.assertIsNone(check_theological_bias(sound_refutation_2))

    def test_smart_bot_weighting_fts_punctuation_and_live_metrics(self):
        self.assertEqual(sanitize_fts_query("Who created God؟?!؛"), '"created"* OR "God"*')
        self.assertEqual(sanitize_fts_query("؛ ، . ! ? "), "")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_gold.db"
            init_db(db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                with conn:
                    conn.execute(
                        "INSERT INTO videos (video_id, title_ar, duration_sec, stage1_status, stage2_status, status) "
                        "VALUES (?, ?, ?, 'completed', 'completed', 'completed')",
                        ("V_BOT", "مناظرة وجود الله والصلاة", 600),
                    )
                    conn.execute(
                        """
                        INSERT INTO scenario_turns
                        (turn_id, video_id, episode_index, turn_index, speaker_id, speaker_role,
                         start_ms, end_ms, start_fmt, end_fmt, text_original, is_overlap, overlap_group_id)
                        VALUES
                        ('V_BOT_T01', 'V_BOT', 1, 1, 'SPK_01', 'INTERLOCUTOR', 0, 5000, '00:00', '00:05', 'Question 1', 1, 'OV_001'),
                        ('V_BOT_T02', 'V_BOT', 1, 2, 'SPK_00', 'MAIN_DAEE', 1000, 6000, '00:01', '00:06', 'Overlap stream 2', 1, 'OV_001'),
                        ('V_BOT_T03', 'V_BOT', 1, 3, 'SPK_00', 'MAIN_DAEE', 10000, 20000, '00:10', '00:20', 'Clean turn', 0, NULL)
                        """
                    )

            payload = [
                {
                    "id": "V_BOT_001",
                    "video_id": "V_BOT",
                    "episode_index": 1,
                    "timestamp_seconds": 10,
                    "timestamp_formatted": "00:10",
                    "daee_speaker_id": "SPK_00",
                    "daee_name_en": "Hashim",
                    "daee_name_ar": "هاشم",
                    "topic_category": "Shariah & Jurisprudence",
                    "interlocutor_type": "Skeptic",
                    "question_en": "Why do Muslims pray five times a day?",
                    "question_ar": "لماذا يصلي المسلمون خمس مرات في اليوم؟",
                    "core_answer_summary_en": "Muslims obey God through daily structured prayer for spiritual purification.",
                    "core_answer_summary_ar": "الصلاة صلة يومية بالله وتزكية للنفس وفق الشريعة.",
                    "logical_premises": ["Remembrance of God.", "Daily gratitude."],
                    "logical_premises_ar": ["ذكر الله.", "الشكر اليومي."],
                    "verbatim_quote": "We pray because God our Creator legislated five prayers for our benefit.",
                    "verbatim_quote_ar": "نصلي لأن الله الخالق شرع الصلوات الخمس لصلاحنا.",
                },
                {
                    "id": "V_BOT_002",
                    "video_id": "V_BOT",
                    "episode_index": 1,
                    "timestamp_seconds": 40,
                    "timestamp_formatted": "00:40",
                    "daee_speaker_id": "SPK_00",
                    "daee_name_en": "Hashim",
                    "daee_name_ar": "هاشم",
                    "topic_category": "Theology & God's Existence",
                    "interlocutor_type": "Atheist",
                    "question_en": "Who created God if everything has a cause?",
                    "question_ar": "من خلق الله إذا كان لكل شيء سبب؟",
                    "core_answer_summary_en": "Asking who created an uncreated necessary First Cause commits a category fallacy.",
                    "core_answer_summary_ar": "السؤال عمن خلق الخالق الأزلي واجب الوجود مغالطة مقولية.",
                    "logical_premises": ["Created things begin to exist.", "An uncreated First Cause halts infinite regress."],
                    "logical_premises_ar": ["المخلوق حادث.", "واجب الوجود يوقف التسلسل."],
                    "verbatim_quote": "Only things that begin to exist require a creator; God is eternal and uncreated.",
                    "verbatim_quote_ar": "ما يبدأ بالوجود فقط يحتاج إلى خالق، أما الله فأزلي غير مخلوق.",
                },
            ]
            ingest_extraction_payload("V_BOT", payload, db_path=db_path)

            videos_resp = list_videos(db_path=db_path)
            self.assertEqual(videos_resp["videos"][0]["qa_count"], 2)

            stats = get_stats(db_path=db_path)
            self.assertEqual(stats["overlaps_resolved"], 1)

            cats = {c["topic_category"] for c in get_categories_breakdown(db_path=db_path)}
            self.assertIn("Theology & God's Existence", cats)
            self.assertIn("Shariah & Social Inquiries", cats)

            bot = SmartDialecticBot()
            ans = bot.answer_query("Who created God?", db_path=db_path)
            self.assertEqual(ans["status"], "success")
            self.assertEqual(ans["matched_qa"]["id"], "V_BOT_002")
            self.assertIn("title_ar", ans["response_package"])


class TestStage1DiarizationAndRoleFixes(unittest.TestCase):
    def test_complete_linkage_blocks_bridge_chaining_and_k1_collapse(self):
        # A-B sim=0.85, B-C sim=0.85, but A-C sim=0.60 < 0.80 -> must NOT bridge A+C together
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.85, float(np.sqrt(1 - 0.85**2)), 0.0], dtype=np.float32)
        vec_c = np.array([0.60, 0.80, 0.0], dtype=np.float32)
        # Ensure B-C cosine >= 0.84 by adjusting vec_c in B plane
        # Check dot(vec_b, vec_c) = 0.85*0.60 + 0.5268*0.80 = 0.51 + 0.421 = 0.931 >= 0.80,
        # while dot(vec_a, vec_c) = 0.60 < 0.80!
        segs = [
            {"speaker_id": "SPK_A", "start_ms": 0, "end_ms": 5000},
            {"speaker_id": "SPK_B", "start_ms": 6000, "end_ms": 11000},
            {"speaker_id": "SPK_C", "start_ms": 12000, "end_ms": 17000},
        ]
        merged, cents, mapping = merge_fragmented_speaker_clusters(
            segs,
            {"SPK_A": vec_a, "SPK_B": vec_b, "SPK_C": vec_c},
            threshold=0.80,
        )
        # Complete linkage merges at most one valid pair (B+C at 0.931) and NEVER all 3 into K=1!
        self.assertGreaterEqual(len(cents), 2)
        self.assertNotEqual(mapping["SPK_A"], mapping["SPK_C"])

        # Anti-collapse guard: 2 substantial >=12s speakers at cosine 0.84 (< 0.90) must stay K=2
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.84, float(np.sqrt(1 - 0.84**2))], dtype=np.float32)
        two_segs = [
            {"speaker_id": "SPK_00", "start_ms": 0, "end_ms": 15000},
            {"speaker_id": "SPK_01", "start_ms": 16000, "end_ms": 32000},
        ]
        _, cents2, map2 = merge_fragmented_speaker_clusters(two_segs, {"SPK_00": v1, "SPK_01": v2}, threshold=0.80)
        self.assertEqual(len(cents2), 2)
        self.assertNotEqual(map2.get("SPK_00", "SPK_00"), map2.get("SPK_01", "SPK_01"))

    def test_run_diarization_m3_anti_collapse_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = SharedWorkspace(root_dir=root, worker_name="test_m3")
            wav_p = root / "debate_120s.wav"
            _write_silent_wav(wav_p, duration_sec=120.0)

            collapsed_rttm = (
                "SPEAKER V_M3 1 0.00 60.00 <NA> <NA> SPK_00 <NA> <NA>\n"
                "SPEAKER V_M3 1 60.00 60.00 <NA> <NA> SPK_00 <NA> <NA>\n"
            )
            recovered_rttm = (
                "SPEAKER V_M3 1 0.00 75.00 <NA> <NA> SPK_00 <NA> <NA>\n"
                "SPEAKER V_M3 1 76.00 44.00 <NA> <NA> SPK_01 <NA> <NA>\n"
            )

            emb_0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            emb_1_distinct = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            emb_1_same = np.array([0.97, float(np.sqrt(1 - 0.97**2)), 0.0], dtype=np.float32)

            class MockDiarizer(WarmNeMoDiarizer):
                def __init__(self, split_emb):
                    super().__init__(device="cpu")
                    self.split_emb = split_emb
                    self.calls = 0
                    self.emb_calls = 0

                def run_diarization(self, wav_16k_path, out_rttm_path, is_short=False, forced_num_speakers=None, clustering_overrides=None):
                    self.calls += 1
                    self.emb_calls = 0
                    txt = collapsed_rttm if self.calls == 1 else recovered_rttm
                    out_rttm_path.parent.mkdir(parents=True, exist_ok=True)
                    out_rttm_path.write_text(txt, encoding="utf-8")
                    return txt

                def extract_embeddings_batch(self, slices_16k, batch_size=16):
                    # Pass 1 has 1 speaker (emb_0); Pass 2+ calls once per speaker: SPK_00 -> emb_0, SPK_01 -> split_emb
                    self.emb_calls += 1
                    vec = emb_0 if self.emb_calls == 1 else self.split_emb
                    return [vec for _ in slices_16k]

            # Case A: distinct 2nd speaker (cos_sim=0.0 < 0.85) -> accepted (K=2)
            d_distinct = MockDiarizer(emb_1_distinct)
            segs_ok, cents_ok, _ = run_diarization_m3("V_M3", wav_p, ws, d_distinct)
            self.assertEqual(len(cents_ok), 2)
            self.assertEqual(len({s["speaker_id"] for s in segs_ok}), 2)

            # Case B: same speaker split (cos_sim=0.97 >= 0.85) -> rejected, remains collapsed
            d_same = MockDiarizer(emb_1_same)
            segs_rej, cents_rej, _ = run_diarization_m3("V_M3_B", wav_p, ws, d_same)
            self.assertEqual(len(cents_rej), 1)
            self.assertEqual({s["speaker_id"] for s in segs_rej}, {"SPK_00"})

    def test_cameo_vp_bonus_guard_and_role_canonicalization(self):
        self.assertEqual(canonicalize_speaker_role("MAIN_DAEE (Hashim)"), "MAIN_DAEE")
        self.assertEqual(canonicalize_speaker_role("المُحاجج"), "MAIN_DAEE")
        self.assertEqual(canonicalize_speaker_role("المحاور"), "INTERLOCUTOR")

        class MockRegistry:
            def match_speaker(self, embedding, title_hint_profile_id=None):
                if embedding is not None and float(embedding[0]) > 0.9:
                    return {
                        "profile_id": "daee_hashim",
                        "name_en": "Brother Hashim",
                        "name_ar": "الأخ هاشم",
                        "similarity": 0.95,
                    }
                return None

        # SPK_01 is registered Hashim making a brief <15% cameo (10s vs 110s MAIN_DAEE SPK_00)
        raw_turns = [
            {"speaker_id": "SPK_00", "start_ms": 0, "end_ms": 110000, "text_original": "Long substantive main daee argument."},
            {"speaker_id": "SPK_01", "start_ms": 111000, "end_ms": 121000, "text_original": "Quick greeting from passing daee."},
        ]
        centroids = {
            "SPK_00": np.array([0.0, 1.0], dtype=np.float32),
            "SPK_01": np.array([1.0, 0.0], dtype=np.float32),
        }
        speakers, _ = classify_speakers_and_episodes(
            raw_turns,
            centroids,
            rms_timeline=[-18.0] * 500,
            video_record={"title_ar": "مناظرة", "title": "Debate"},
            registry=MockRegistry(),
        )
        by_id = {s["speaker_id"]: s for s in speakers}
        self.assertEqual(by_id["SPK_00"]["role"], "MAIN_DAEE")
        self.assertEqual(by_id["SPK_01"]["role"], "CO_DAEE")

    def test_three_window_language_vote_prefers_majority_body_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_p = Path(tmp) / "arabic_intro_english_body.wav"
            _write_silent_wav(wav_p, duration_sec=150.0)

            engine = WarmASRAlignmentEngine(device="cpu")
            probe_langs = [("ar", 0.95), ("en", 0.92), ("en", 0.91)]
            probe_idx = [0]

            class FakeWhisper:
                def transcribe(self, waveform, **kwargs):
                    if "language" not in kwargs:
                        lg, pr = probe_langs[probe_idx[0]]
                        probe_idx[0] += 1
                        return [], SimpleNamespace(language=lg, language_probability=pr)
                    return [], SimpleNamespace(language=kwargs["language"], language_probability=1.0)

            import sys
            from contextlib import nullcontext

            engine._whisper = FakeWhisper()
            fake_torch = SimpleNamespace(
                inference_mode=nullcontext,
                cuda=SimpleNamespace(is_available=lambda: False),
            )
            fake_fw = SimpleNamespace(BatchedInferencePipeline=None)
            with patch.dict(sys.modules, {"torch": fake_torch, "faster_whisper": fake_fw}), patch("core.audio_diarizer.ctc_forced_aligner", None, create=True):
                _, detected_lang = engine.transcribe_and_align_wav(
                    wav_p, forced_language=None, use_ctc_aligner=False
                )
            self.assertEqual(detected_lang, "en")


class TestStage2GroundingAuditAndResetFixes(unittest.TestCase):
    def test_mark_13_32_spaced_punctuation_leak_caught(self):
        turns = [
            {"turn_id": "T1", "speaker_role": "INTERLOCUTOR", "text_original": "Is Jesus divine in scripture?"},
            {"turn_id": "T2", "speaker_role": "MAIN_DAEE", "text_original": "The Father alone holds ultimate knowledge of creation."},
        ]
        card = {
            "question_turn_id": "T1",
            "answer_turn_id": "T2",
            "question_en": "Does scripture teach that Jesus has equal knowledge with the Father?",
            "question_ar": "هل يعلم المسيح وقت الساعة عند النصارى؟",
            "core_answer_summary_en": "Jesus himself disclaimed knowing the Hour in Mark 13 .32, refuting omniscience.",
            "core_answer_summary_ar": "نصوص الأناجيل تنفي علم المسيح بالساعة.",
            "fallacy_or_misconception_exposed": "Equating a prophet with the All-Knowing Creator.",
            "fallacy_or_misconception_exposed_ar": "تسوية النبي بالخالق العليم.",
            "socratic_question": "How could God not know the Hour?",
            "socratic_question_ar": "كيف يجهل الإله وقت الساعة؟",
            "verbatim_quote": "The Father alone holds ultimate knowledge of creation.",
            "verbatim_quote_ar": "الرب وحده يملك العلم المطلق بالخلق.",
            "interlocutor_verbatim_quote": "Is Jesus divine in scripture?",
            "interlocutor_verbatim_quote_ar": "هل يسوع إله في الكتاب المقدس؟",
        }
        res = audit_single_card(card, turns, require_arabic=True)
        self.assertEqual(res["hard_fail_triggered"], "[GATE-LEAK]")
        self.assertTrue(any(d["code"] == "LEAK_MARK_13_32" for d in res["defects"]))

    def test_multi_card_positional_merge_blocked_against_contamination(self):
        r1 = [
            {"question_turn_id": "V_T01", "answer_turn_id": "V_T02", "question_en": "Who created God?"},
            {"question_turn_id": "V_T03", "answer_turn_id": "V_T04", "question_en": "Why is the Quran preserved?"},
        ]
        # Round 2 only emits unit 1 (or swapped turn_id V_T99) -> must NOT contaminate unit 2!
        r2 = [
            {
                "question_turn_id": "V_T99",
                "answer_turn_id": "V_T98",
                "question_en": "Something completely different about fasting",
                "question_ar": "ما حكم الصيام في السفر؟",
                "core_answer_summary_ar": "يرخص للمسافر الإفطار ثم القضاء.",
            }
        ]
        merged = merge_round1_and_round2_units(r1, r2)
        self.assertEqual(len(merged), 2)
        self.assertFalse(merged[0].get("question_ar"))
        self.assertFalse(merged[1].get("question_ar"))

    def test_preground_and_six_gate_enforcement_and_zero_card_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = SharedWorkspace(root_dir=Path(tmp), worker_name="test_preground")
            vid = "PG_01"
            scen = {
                "video_id": vid,
                "title_ar": "شبهات العقيدة",
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "duration_sec": 600,
                "metrics": {"speakers_count": 2, "duration_sec": 600},
                "speakers": [
                    {"speaker_id": "SPK_00", "role": "MAIN_DAEE (Hashim)", "profile_id": "daee_hashim", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                    {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "display_name_en": "Seeker", "display_name_ar": "السائل"},
                ],
                "turns": [
                    {
                        "turn_id": f"{vid}_T0001",
                        "episode_index": 1,
                        "speaker_id": "SPK_01",
                        "speaker_role": "INTERLOCUTOR",
                        "start_ms": 1000,
                        "end_ms": 10000,
                        "start_fmt": "00:01",
                        "end_fmt": "00:10",
                        "text_original": "Why did God need to create the universe if He is completely self sufficient?",
                    },
                    {
                        "turn_id": f"{vid}_T0002",
                        "episode_index": 1,
                        "speaker_id": "SPK_00",
                        "speaker_role": "MAIN_DAEE (Hashim)",
                        "start_ms": 11000,
                        "end_ms": 28000,
                        "start_fmt": "00:11",
                        "end_fmt": "00:28",
                        "text_original": "Creating is a manifestation of Allah's generosity and mercy rather than any need on His part.",
                    },
                ],
            }
            ws.atomic_write_json(ws.scenarios_dir / f"{vid}.json", scen)
            ws.advance_micro_step(vid, "M5_SCENARIO_BUILT")

            seen_r2_input = {}

            class GroundingMockLLM:
                def extract_round1_english(self, prompt):
                    # Intentionally provide a summary-like bogus quote and wrong dialogue role
                    return json.dumps([
                        {
                            "question_turn_id": f"{vid}_T0001",
                            "answer_turn_id": f"{vid}_T0002",
                            "question_en": "Why did Allah create the universe if He does not need it?",
                            "topic_category": "Theology & God's Existence",
                            "core_answer_summary_en": "Creation reflects divine wisdom and generosity without divine need.",
                            "logical_premises": ["Allah is self-sufficient.", "Creation shows divine mercy."],
                            "fallacy_or_misconception_exposed": "Confusing generous action with personal need.",
                            "socratic_question": "Does a philanthropist give charity out of poverty or abundance?",
                            "verbatim_quote": "Fabricated essay that never appeared in transcript at all anywhere.",
                            "interlocutor_verbatim_quote": "Fake seeker text.",
                            "dialogue_exchange": [
                                {"turn_id": f"{vid}_T0001", "role": "MAIN_DAEE", "move_type": "Objection", "text_en": "Why create if self-sufficient?"},
                                {"turn_id": f"{vid}_T0002", "role": "INTERLOCUTOR", "move_type": "Proof", "text_en": "Totally invented unrelated speech about quantum gravity and penguins."},
                            ],
                        }
                    ])

                def extract_round2_arabic(self, r1_json_str, prompt):
                    seen_r2_input["units"] = json.loads(r1_json_str)
                    return json.dumps([
                        {
                            "question_turn_id": f"{vid}_T0001",
                            "answer_turn_id": f"{vid}_T0002",
                            "question_ar": "لماذا خلق الله الكون وهو غني عن العالمين؟",
                            "core_answer_summary_ar": "الخلق أثر كرم الله وحكمته ورحمته لا عن حاجة أو افتقار.",
                            "logical_premises_ar": ["الله غني بذاته.", "الخلق مظهر رحمة وحكمة."],
                            "fallacy_or_misconception_exposed_ar": "قياس فعل الغني المطلق على حاجة المخلوق.",
                            "socratic_question_ar": "هل ينفق الكريم من فاقة أم من سعة؟",
                            "verbatim_quote_ar": "الخلق مظهر لجود الله ورحمته لا عن حاجة إليه سبحانه.",
                            "interlocutor_verbatim_quote_ar": "لماذا احتاج الله إلى خلق الكون إن كان غنياً؟",
                            "dialogue_exchange": [
                                {"turn_id": f"{vid}_T0001", "role": "INTERLOCUTOR", "text_ar": "لمَ خلق الكون إن كان غنياً؟"},
                                {"turn_id": f"{vid}_T0002", "role": "MAIN_DAEE", "text_ar": "الخلق مظهر لجود الله ورحمته."},
                            ],
                        }
                    ], ensure_ascii=False)

            units = Stage2ScenarioExtractor(workspace=ws, llm_client=GroundingMockLLM()).extract_video_m6(vid, verbose=False)
            self.assertEqual(len(units), 1)
            r1_seen = seen_r2_input["units"][0]
            self.assertIn("manifestation of Allah's generosity", r1_seen["verbatim_quote"])
            self.assertEqual(r1_seen["dialogue_exchange"][0]["role"], "INTERLOCUTOR")
            self.assertEqual(r1_seen["dialogue_exchange"][1]["role"], "MAIN_DAEE")
            self.assertIn("manifestation of Allah's generosity", r1_seen["dialogue_exchange"][1]["text_en"])

            # Verify dropped GATE-PERSPECTIVE + zero_card_reason
            vid2 = "PG_DROP"
            scen2 = json.loads(json.dumps(scen).replace(vid, vid2))
            ws.atomic_write_json(ws.scenarios_dir / f"{vid2}.json", scen2)
            ws.advance_micro_step(vid2, "M5_SCENARIO_BUILT")

            class CrossExamMockLLM:
                def extract_round1_english(self, prompt):
                    return json.dumps([
                        {
                            "question_turn_id": f"{vid2}_T0001",
                            "answer_turn_id": f"{vid2}_T0002",
                            "question_en": "Since Paul disagrees with James on works versus faith, why trust the Bible?",
                            "core_answer_summary_en": "Allah is All-Wise.",
                            "verbatim_quote": scen2["turns"][1]["text_original"],
                        }
                    ])

                def extract_round2_arabic(self, r1_json_str, prompt):
                    return json.dumps([
                        {
                            "question_turn_id": f"{vid2}_T0001",
                            "answer_turn_id": f"{vid2}_T0002",
                            "question_ar": "بما أن بولس يخالف يعقوب فلماذا يوثق بالكتاب المقدس؟",
                            "core_answer_summary_ar": "الله حكيم عليم.",
                            "verbatim_quote_ar": "الخلق مظهر لجود الله ورحمته.",
                        }
                    ], ensure_ascii=False)

            dropped_units = Stage2ScenarioExtractor(workspace=ws, llm_client=CrossExamMockLLM()).extract_video_m6(vid2, verbose=False)
            self.assertEqual(len(dropped_units), 0)
            st2 = ws.get_video_state(vid2)
            self.assertIn("[GATE-PERSPECTIVE]:1", st2.get("zero_card_reason", ""))

            # Verify [GATE-DIARIZATION] also drops cards when a >180s video collapsed into 1 speaker
            vid3 = "PG_COLLAPSE"
            scen3 = {
                "video_id": vid3,
                "title_ar": "مناظرة انكمشت لمتحدث واحد",
                "youtube_url": f"https://www.youtube.com/watch?v={vid3}",
                "duration_sec": 600,
                "metrics": {"speakers_count": 1, "duration_sec": 600},
                "speakers": [
                    {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                ],
                "turns": [
                    {
                        "turn_id": f"{vid3}_T0001",
                        "episode_index": 1,
                        "speaker_id": "SPK_00",
                        "speaker_role": "MAIN_DAEE",
                        "start_ms": 1000,
                        "end_ms": 10000,
                        "start_fmt": "00:01",
                        "end_fmt": "00:10",
                        "text_original": "Why did God create evil and suffering in this earthly world if He is infinitely merciful and all-powerful?",
                    },
                    {
                        "turn_id": f"{vid3}_T0002",
                        "episode_index": 1,
                        "speaker_id": "SPK_00",
                        "speaker_role": "MAIN_DAEE",
                        "start_ms": 11000,
                        "end_ms": 28000,
                        "start_fmt": "00:11",
                        "end_fmt": "00:28",
                        "text_original": "This earthly life is a temporary moral test established by infinite divine wisdom and perfect justice.",
                    },
                ],
            }
            ws.atomic_write_json(ws.scenarios_dir / f"{vid3}.json", scen3)
            ws.advance_micro_step(vid3, "M5_SCENARIO_BUILT")

            class CollapsedVideoMockLLM:
                def extract_round1_english(self, prompt):
                    return json.dumps([
                        {
                            "question_turn_id": f"{vid3}_T0001",
                            "answer_turn_id": f"{vid3}_T0002",
                            "question_en": "Why does evil exist in the world?",
                            "core_answer_summary_en": "Earthly life is a temporary moral test governed by Allah's infinite wisdom.",
                            "verbatim_quote": scen3["turns"][1]["text_original"],
                        }
                    ])

                def extract_round2_arabic(self, r1_json_str, prompt):
                    return json.dumps([
                        {
                            "question_turn_id": f"{vid3}_T0001",
                            "answer_turn_id": f"{vid3}_T0002",
                            "question_ar": "لماذا يوجد الشر والابتلاء في الدنيا؟",
                            "core_answer_summary_ar": "الحياة الدنيا دار اختبار وابتلاء بحكمة الله وعدله البالغ.",
                            "verbatim_quote_ar": "هذه الحياة الدنيا اختبار أخلاقي مؤقت بحكمة إلهية بالغة.",
                        }
                    ], ensure_ascii=False)

            col_units = Stage2ScenarioExtractor(workspace=ws, llm_client=CollapsedVideoMockLLM()).extract_video_m6(vid3, verbose=False)
            self.assertEqual(len(col_units), 0)
            st3 = ws.get_video_state(vid3)
            self.assertIn("[GATE-DIARIZATION]:1", st3.get("zero_card_reason", ""))

    def test_video_wide_semantic_dedup_preserves_numbered_items(self):
        existing = [
            {
                "question_en": "If God is merciful why do innocent children suffer?",
                "question_ar": "إذا كان الله رحيماً فلماذا يتألم الأطفال الأبرياء؟",
                "timestamp_seconds": 10,
            },
            {
                "question_en": "Why does objector question principle 1 about causality?",
                "question_ar": "لماذا يُطرح الإشكال رقم 1 حول السببية؟",
                "timestamp_seconds": 20,
            },
        ]
        # Far (> 75s) semantic paraphrase duplicate must still be caught
        dup_paraphrase = {
            "question_en": "Why does an all-loving God permit innocent children to suffer?",
            "question_ar": "لم يسمح الإله الرحيم بآلام الأطفال؟",
            "timestamp_seconds": 800,
        }
        self.assertTrue(_is_duplicate_qa(dup_paraphrase, existing))

        # Distinct numbered item must NOT be falsely deduplicated
        numbered_3 = {
            "question_en": "Why does objector question principle 3 about causality?",
            "question_ar": "لماذا يُطرح الإشكال رقم 3 حول السببية؟",
            "timestamp_seconds": 820,
        }
        self.assertFalse(_is_duplicate_qa(numbered_3, existing))

    def test_reset_stage2_extractions_resets_qa_count_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = SharedWorkspace(root_dir=Path(tmp), worker_name="test_reset2")
            db_path = ws.data_dir / "dawah_knowledge.db"
            init_db(db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                with conn:
                    conn.execute(
                        "INSERT INTO videos (video_id, title_ar, qa_count, stage1_status, stage2_status, status) "
                        "VALUES ('V_R1', 'عنوان 1', 5, 'completed', 'completed', 'completed')"
                    )
                    conn.execute(
                        "INSERT INTO videos (video_id, title_ar, qa_count, stage1_status, stage2_status, status) "
                        "VALUES ('V_R2', 'عنوان 2', 2, 'pending', 'completed', 'completed')"
                    )
                    conn.execute(
                        "INSERT INTO qa_units (id, video_id, episode_index, question_en, question_ar, "
                        "core_answer_summary_en, core_answer_summary_ar) VALUES ('V_R1_001', 'V_R1', 1, 'Q?', 'س؟', 'A.', 'ج.')"
                    )

            ws.advance_micro_step("V_R1", "M6_LLM_EXTRACTED")
            st_before = ws.get_video_state("V_R1")
            st_before["error_count"] = 3
            st_before["last_error"] = "stale timeout"
            ws.atomic_write_json(ws.state_path("V_R1"), st_before)
            ws.atomic_write_json(ws.extractions_dir / "V_R1.qa.json", {"units": []})

            with patch.object(reset_stage2_mod, "DB_PATH", db_path):
                reset_stage2_mod.reset_stage2_extractions(ws)

            st_after = ws.get_video_state("V_R1")
            self.assertEqual(st_after["micro_step"], "M5_SCENARIO_BUILT")
            self.assertEqual(st_after["error_count"], 0)
            self.assertIsNone(st_after["last_error"])

            with closing(sqlite3.connect(db_path)) as conn:
                r1 = conn.execute("SELECT qa_count, stage2_status, status FROM videos WHERE video_id='V_R1'").fetchone()
                r2 = conn.execute("SELECT qa_count, stage2_status, status FROM videos WHERE video_id='V_R2'").fetchone()
                cnt = conn.execute("SELECT COUNT(*) FROM qa_units").fetchone()[0]
            self.assertEqual(r1, (0, "pending", "stage1_done"))
            self.assertEqual(r2, (0, "pending", "queued"))
            self.assertEqual(cnt, 0)


class TestV31RootCauseRegressions(unittest.TestCase):
    def test_m3_retry_does_not_poison_rttm_and_catches_90pct_collapse(self):
        # 90.6% top-speaker dominance on >= 300s video must trigger collapse detection
        segs_90pct = [
            {"speaker_id": "SPK_00", "start_ms": 0, "end_ms": 453_000},
            {"speaker_id": "SPK_01", "start_ms": 453_000, "end_ms": 500_000},
        ]
        self.assertTrue(_is_collapsed_diarization(segs_90pct, total_ms=500_000))

        # Edge-only secondary speaker (confined to first 12% intro even with >10% share) triggers recovery
        segs_edge = [
            {"speaker_id": "SPK_01", "start_ms": 0, "end_ms": 55_000},
            {"speaker_id": "SPK_00", "start_ms": 60_000, "end_ms": 500_000},
        ]
        self.assertTrue(_is_collapsed_diarization(segs_edge, total_ms=500_000))

        # Genuine balanced 2-speaker debate must NOT trigger collapse detection
        segs_ok = [
            {"speaker_id": "SPK_00", "start_ms": 0, "end_ms": 260_000},
            {"speaker_id": "SPK_01", "start_ms": 260_000, "end_ms": 500_000},
        ]
        self.assertFalse(_is_collapsed_diarization(segs_ok, total_ms=500_000))

        # Rejected retry must NOT overwrite accepted RTTM on disk or leave .retry.rttm behind
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = SharedWorkspace(root_dir=root, worker_name="test_rttm_atomic")
            wav_p = root / "debate.wav"
            _write_silent_wav(wav_p, duration_sec=120.0)

            collapsed_rttm = "SPEAKER V_DISK 1 0.00 120.00 <NA> <NA> SPK_00 <NA> <NA>\n"
            bad_retry_rttm = (
                "SPEAKER V_DISK 1 0.00 80.00 <NA> <NA> SPK_00 <NA> <NA>\n"
                "SPEAKER V_DISK 1 81.00 39.00 <NA> <NA> SPK_01 <NA> <NA>\n"
            )
            emb_0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            emb_1_same = np.array([0.96, float(np.sqrt(1 - 0.96**2)), 0.0], dtype=np.float32)

            class RetryRejectDiarizer(WarmNeMoDiarizer):
                def __init__(self):
                    super().__init__(device="cpu")
                    self.calls = 0
                    self.emb_calls = 0
                    self.paths_written = []

                def run_diarization(self, wav_16k_path, out_rttm_path, is_short=False, forced_num_speakers=None, clustering_overrides=None):
                    self.calls += 1
                    self.emb_calls = 0
                    self.paths_written.append(out_rttm_path.name)
                    txt = collapsed_rttm if self.calls == 1 else bad_retry_rttm
                    out_rttm_path.parent.mkdir(parents=True, exist_ok=True)
                    out_rttm_path.write_text(txt, encoding="utf-8")
                    return txt

                def extract_embeddings_batch(self, slices_16k, batch_size=16):
                    self.emb_calls += 1
                    vec = emb_0 if self.emb_calls == 1 else emb_1_same
                    return [vec for _ in slices_16k]

            diarizer = RetryRejectDiarizer()
            segs_final, _, _ = run_diarization_m3("V_DISK", wav_p, ws, diarizer)
            self.assertEqual({s["speaker_id"] for s in segs_final}, {"SPK_00"})
            rttm_path = ws.diarization_dir / "V_DISK" / "V_DISK.rttm"
            retry_path = rttm_path.with_suffix(".retry.rttm")
            self.assertFalse(retry_path.exists())
            disk_text = rttm_path.read_text(encoding="utf-8")
            self.assertIn("SPK_00", disk_text)
            self.assertNotIn("SPK_01", disk_text)
            self.assertIn("V_DISK.retry.rttm", diarizer.paths_written)

    def test_stage1_turn_repetition_loop_collapsed(self):
        looped = "Why " + " ".join(["hath"] * 14) + " not the scripture addressed this clearly?"
        collapsed = _collapse_turn_repetition_loops(looped)
        self.assertEqual(collapsed.lower().count("hath"), 1)
        self.assertIn("not the scripture addressed this clearly?", collapsed)

        normal = "This is a completely normal spoken sentence with no repetition loops."
        self.assertEqual(_collapse_turn_repetition_loops(normal), normal)

    def test_arabic_dangling_proclitic_and_expanded_theological_bias(self):
        dirty_ar = "هذا يوافق المنطق لل. ويدل لل الإسلامي كما ورد في إشعياء 18:20 عن الأرحيم ونَزَرَثَ في غاليلية."
        cleaned = clean_latin_tokens_in_arabic(dirty_ar)
        self.assertNotIn("لل.", cleaned)
        self.assertNotIn(" لل ", cleaned)
        self.assertIn("حزقيال 18:20", cleaned)
        self.assertIn("الرحيم", cleaned)
        self.assertIn("الناصرة في الجليل", cleaned)

        self.assertIsNotNone(check_theological_bias({
            "question_en": "Are Father and Son equal?",
            "core_answer_summary_en": "The Son is one of the co-equal persons of the Trinity in divine essence.",
        }))
        self.assertIsNotNone(check_theological_bias({
            "question_en": "Is only Islam acceptable?",
            "core_answer_summary_en": "Surah 3:85 does not mean that all other religions are invalid or rejected.",
        }))
        self.assertIsNotNone(check_theological_bias({
            "question_en": "Does God know the outcome?",
            "core_answer_summary_en": "Even God cannot know the final outcome of human free choices ahead of time.",
        }))
        self.assertIsNotNone(check_theological_bias({
            "question_en": "Is God one?",
            "core_answer_summary_ar": "هذا لا يعني أن الله واحداً لا يولد له أبناء ولا ينفي ذلك.",
        }))

    def test_turn_id_leading_underscore_recovery_and_dialogue_text_grounding(self):
        t1 = {
            "turn_id": "_moXN_uYN1A_T0001",
            "speaker_role": "INTERLOCUTOR",
            "speaker_label_en": "Interlocutor",
            "speaker_label_ar": "المحاور",
            "start_ms": 10_000,
            "start_fmt": "00:10",
            "text_original": "Why does the Quran speak in parables about cosmic origins and truth?",
        }
        t2 = {
            "turn_id": "_moXN_uYN1A_T0002",
            "speaker_role": "MAIN_DAEE",
            "speaker_label_en": "Hashim",
            "speaker_label_ar": "هاشم",
            "start_ms": 25_000,
            "start_fmt": "00:25",
            "text_original": "Rational necessity establishes a transcendent Eternal Creator beyond all created matter.",
        }
        by_id = {"_moXN_uYN1A_T0001": t1, "_moXN_uYN1A_T0002": t2}

        self.assertIs(_lookup_turn("moXN_uYN1A_T0001", by_id, by_id), t1)
        self.assertIs(_lookup_turn("T0002", by_id, by_id), t2)

        raw_ex = [
            {
                "turn_id": "moXN_uYN1A_T0002",
                "role": "MAIN_DAEE",
                "text_en": "Completely wrong shifted sentence talking about unrelated historical manuscripts and councils.",
                "text_ar": "الضرورة العقلية تثبت وجود خالق أزلي متعالٍ.",
            },
            {
                "turn_id": "moXN_uYN1A_T0001",
                "role": "INTERLOCUTOR",
                "text_en": "Why does the Quran speak in parables about cosmic origins and truth?",
                "text_ar": "لماذا يتحدث القرآن بالأمثال عن أصل الكون والحق؟",
            },
        ]
        grounded, matched = _ground_dialogue_exchange(
            raw_ex, by_id, by_id, "Hashim", "هاشم", "SPK_01", 10, repair_english_from_turn=True
        )
        self.assertTrue(matched)
        self.assertEqual([g["turn_id"] for g in grounded], ["_moXN_uYN1A_T0001", "_moXN_uYN1A_T0002"])
        self.assertEqual(grounded[1]["text_en"], t2["text_original"])

    def test_round2_retry_escalates_with_missing_field_diagnostics_and_drop_accounting(self):
        clean_unit_sentinel = {
            "question_ar": "لماذا يفتقر الكون الحادث إلى خالق أزلي واجب الوجود؟",
            "core_answer_summary_ar": "كل ما له بداية وممكن الوجود يفتقر بالضرورة العقلية إلى خالق أزلي واجب الوجود غير مخلوق.",
            "core_answer_summary_en": "Whatever begins to exist requires a Creator by rational necessity.",
            "fallacy_or_misconception_exposed_ar": "مغالطة التسلسل اللانهائي",
            "socratic_question_ar": "هل يمكن للممكن أن يوجد نفسه من العدم؟",
            "verbatim_quote": "An uncaused First Cause must exist.",
            "verbatim_quote_ar": "لا بد من وجود سبب أول غير مخلوق.",
            "analogy_or_metaphor": "No analogy was used in this exchange.",
            "analogy_or_metaphor_ar": "",
        }
        self.assertEqual(diagnose_missing_arabic_fields(clean_unit_sentinel), [])

        with tempfile.TemporaryDirectory() as tmp:
            ws = SharedWorkspace(root_dir=Path(tmp), worker_name="test_r2_escalation")
            scen = {
                "video_id": "V_ESC",
                "title_ar": "اختبار التصعيد العربي",
                "youtube_url": "https://www.youtube.com/watch?v=V_ESC",
                "duration_sec": 300,
                "metrics": {"speakers_count": 2, "duration_sec": 300},
                "speakers": [
                    {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                    {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "display_name_en": "Skeptic", "display_name_ar": "المحاور"},
                ],
                "turns": [
                    {
                        "turn_id": "V_ESC_T0001",
                        "episode_index": 1,
                        "speaker_id": "SPK_01",
                        "speaker_role": "INTERLOCUTOR",
                        "start_ms": 5000,
                        "end_ms": 15000,
                        "start_fmt": "00:05",
                        "end_fmt": "00:15",
                        "text_original": "If God caused the cosmos, why cannot the material universe simply exist infinitely without an eternal Creator?",
                    },
                    {
                        "turn_id": "V_ESC_T0002",
                        "episode_index": 1,
                        "speaker_id": "SPK_00",
                        "speaker_role": "MAIN_DAEE",
                        "start_ms": 16000,
                        "end_ms": 38000,
                        "start_fmt": "00:16",
                        "end_fmt": "00:38",
                        "text_original": "Infinite regress of contingent events is impossible; an eternal Necessary Creator must terminate the chain.",
                    },
                ],
            }
            ws.atomic_write_json(ws.scenarios_dir / "V_ESC.json", scen)
            ws.advance_micro_step("V_ESC", "M5_SCENARIO_BUILT")

            class EscalatingRetryMockLLM:
                def __init__(self, succeed_attempt=3):
                    self.succeed_attempt = succeed_attempt
                    self.r2_prompts = []
                    self.r2_kwargs = []

                def extract_round1_english(self, prompt):
                    return json.dumps([
                        {
                            "question_turn_id": "V_ESC_T0001",
                            "answer_turn_id": "V_ESC_T0002",
                            "question_en": "Why cannot the material universe exist infinitely without a Creator?",
                            "core_answer_summary_en": "An infinite regress of dependent events is rationally impossible, so an eternal Necessary Creator must exist.",
                            "logical_premises": ["Contingent things depend on causes.", "Infinite regress cannot complete.", "A Necessary Creator exists."],
                            "fallacy_or_misconception_exposed": "Infinite regress fallacy.",
                            "socratic_question": "Can an infinite chain of dependent links ever stand without an anchor?",
                            "interlocutor_verbatim_quote": scen["turns"][0]["text_original"],
                            "verbatim_quote": scen["turns"][1]["text_original"],
                            "dialogue_exchange": [
                                {"turn_id": "V_ESC_T0001", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": scen["turns"][0]["text_original"]},
                                {"turn_id": "V_ESC_T0002", "role": "MAIN_DAEE", "move_type": "Deductive Proof", "text_en": scen["turns"][1]["text_original"]},
                            ],
                        }
                    ])

                def extract_round2_arabic(self, r1_json_str, prompt, enable_thinking=False, max_thinking_tokens=0):
                    self.r2_prompts.append(prompt)
                    self.r2_kwargs.append((enable_thinking, max_thinking_tokens))
                    attempt_num = len(self.r2_prompts) - 1  # 0=initial, 1=retry1, 2=retry2, 3=retry3
                    if attempt_num < self.succeed_attempt:
                        return json.dumps([{"question_turn_id": "V_ESC_T0001", "question_ar": "لماذا لا يكون الكون أزلياً؟"}], ensure_ascii=False)
                    return json.dumps([
                        {
                            "question_turn_id": "V_ESC_T0001",
                            "question_ar": "لماذا لا يمكن للكون المادي أن يوجد أزلياً بلا خالق؟",
                            "core_answer_summary_ar": "تسلسل الممكنات والمعتمدات إلى غير نهاية ممتنع عقلاً ولذا وجب وجود خالق أزلي واجب الوجود.",
                            "logical_premises_ar": ["الممكن يفتقر لعلة.", "التسلسل اللانهائي باطل.", "واجب الوجود ثابت عقلاً."],
                            "fallacy_or_misconception_exposed_ar": "مغالطة التسلسل اللانهائي",
                            "socratic_question_ar": "هل تقوم سلسلة لا نهائية من الحلقات المفتقرة بلا أصل مستقل؟",
                            "interlocutor_verbatim_quote_ar": "لماذا لا يوجد الكون المادي أزلياً بدون خالق؟",
                            "verbatim_quote_ar": "التسلسل اللانهائي للممكنات مستحيل ولذا لا بد من خالق واجب الوجود.",
                            "dialogue_exchange": [
                                {"turn_id": "V_ESC_T0001", "role": "INTERLOCUTOR", "text_ar": "لماذا لا يوجد الكون المادي أزلياً بدون خالق؟"},
                                {"turn_id": "V_ESC_T0002", "role": "MAIN_DAEE", "text_ar": "التسلسل اللانهائي للممكنات مستحيل ولذا لا بد من خالق واجب الوجود."},
                            ],
                        }
                    ], ensure_ascii=False)

            mock_ok = EscalatingRetryMockLLM(succeed_attempt=3)
            units_ok = Stage2ScenarioExtractor(workspace=ws, llm_client=mock_ok).extract_video_m6("V_ESC", verbose=False)
            self.assertEqual(len(units_ok), 1)
            self.assertEqual(len(mock_ok.r2_prompts), 4)  # initial + 3 retries
            self.assertIn("RETRY DIAGNOSTIC:", mock_ok.r2_prompts[1])
            self.assertIn("core_answer_summary_ar", mock_ok.r2_prompts[1])
            self.assertNotIn('"core_answer_summary_ar"', mock_ok.r2_prompts[1])
            self.assertEqual(mock_ok.r2_kwargs[3], (True, 384))
            self.assertIn("RUBRIC VERIFICATION", mock_ok.r2_prompts[3].upper())
            self.assertNotIn("/no_think", mock_ok.r2_prompts[3])

            # Full drop accounting check when all retries fail Arabic localization
            ws.atomic_write_json(ws.scenarios_dir / "V_DROP.json", dict(scen, video_id="V_DROP", turns=[
                dict(scen["turns"][0], turn_id="V_DROP_T0001"),
                dict(scen["turns"][1], turn_id="V_DROP_T0002"),
            ]))
            ws.advance_micro_step("V_DROP", "M5_SCENARIO_BUILT")

            class AlwaysFailR2(EscalatingRetryMockLLM):
                def extract_round1_english(self, prompt):
                    return super().extract_round1_english(prompt).replace("V_ESC_T", "V_DROP_T")

            units_fail = Stage2ScenarioExtractor(workspace=ws, llm_client=AlwaysFailR2(succeed_attempt=99)).extract_video_m6("V_DROP", verbose=False)
            self.assertEqual(len(units_fail), 0)
            st_drop = ws.get_video_state("V_DROP")
            self.assertIn("[DROP-AR-RETRY]:1", st_drop.get("zero_card_reason", ""))
            self.assertNotIn("[none]", st_drop.get("zero_card_reason", ""))

    def test_smart_bot_arabic_punctuation_morphological_stemming_and_intent_guard(self):
        norm = normalize_text("من خلق الله؟ وهل الإسلام، حق؛")
        for punct in ("؟", "،", "؛"):
            self.assertNotIn(punct, norm)

        for form in ("خالق", "مخلوق", "مخلوقات", "يخلق", "خلقه"):
            self.assertEqual(_stem_token(form, "ar"), "خلق")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "bot_test.db"
            init_db(db_path)
            with closing(sqlite3.connect(db_path)) as conn:
                with conn:
                    conn.execute("INSERT INTO videos (video_id, title_ar, status) VALUES ('VB1', 'فيديو اختباري', 'completed')")
            payload = [
                {
                    "id": "VB1_001",
                    "video_id": "VB1",
                    "episode_index": 1,
                    "timestamp_seconds": 120,
                    "timestamp_formatted": "02:00",
                    "answer_timestamp_seconds": 135,
                    "answer_timestamp_formatted": "02:15",
                    "topic_category": "Textual Preservation & Manuscripts",
                    "topic_subcategory": "Preservation of Scripture",
                    "question_en": "Has the Bible been preserved without scribal changes across manuscripts?",
                    "question_ar": "هل حُفظ الإنجيل من التحريف عبر المخطوطات القديمة؟",
                    "core_answer_summary_en": "New Testament manuscripts show thousands of variants whereas the Quran has continuous oral and written chains.",
                    "core_answer_summary_ar": "تظهر مخطوطات العهد الجديد آلاف الاختلافات بخلاف القرآن المحفوظ بأسانيد متواترة.",
                    "tags": ["تحريف الإنجيل", "حفظ القرآن", "manuscripts"],
                },
                {
                    "id": "VB1_002",
                    "video_id": "VB1",
                    "episode_index": 1,
                    "timestamp_seconds": 240,
                    "timestamp_formatted": "04:00",
                    "topic_category": "Theology & God's Existence",
                    "topic_subcategory": "Sun Worship",
                    "question_en": "Why do people worship the sun instead of God?",
                    "question_ar": "لماذا يعبد الناس الشمس بدلاً من الله؟",
                    "core_answer_summary_en": "Humans fall into idolatry when they venerate physical celestial objects.",
                    "core_answer_summary_ar": "يقع الناس في الشرك حين يعظمون الأجرام السماوية.",
                    "tags": ["idolatry", "sun worship"],
                },
            ]
            ingest_extraction_payload("VB1", payload, db_path=db_path)

            bot = SmartDialecticBot()
            pres_ans = bot.answer_query("تحريف الإنجيل؟", db_path=db_path)
            self.assertIn(pres_ans["status"], ("success", "partial_match"))
            self.assertEqual(pres_ans["matched_qa"]["id"], "VB1_001")
            self.assertEqual(pres_ans["response_package"]["timestamp_seconds"], 120)
            self.assertEqual(pres_ans["response_package"]["answer_timestamp_seconds"], 135)

            off_topic = bot.answer_query("Who created God?", db_path=db_path)
            self.assertEqual(off_topic["status"], "no_match")

    def test_audit_cards_v31_catches_shifted_turns_topic_drift_functional_collapse_and_tier3_fail(self):
        self.assertIsNotNone(DAEE_CROSS_EXAM_REGEX.search("If the Father is the only true God, how can Christians claim the Trinity?"))
        self.assertIsNone(DAEE_CROSS_EXAM_REGEX.search("Does the Bible affirm that Jesus is a prophet, or does it claim he is divine?"))

        turns = [
            {"turn_id": "V31_T01", "speaker_role": "MAIN_DAEE", "start_ms": 0, "end_ms": 60_000, "text_original": " ".join(["quantum cosmology fine tuning entropy expansion universe mathematical probability"] * 28)},
            {"turn_id": "V31_T02", "speaker_role": "MAIN_DAEE", "start_ms": 61_000, "end_ms": 120_000, "text_original": "The teleological arrangement of constants demonstrates an All-Wise Designer beyond blind chance."},
            {"turn_id": "V31_T03", "speaker_role": "INTERLOCUTOR", "start_ms": 121_000, "end_ms": 125_000, "text_original": "Interesting point about cosmic constants."},
            {"turn_id": "V31_T04", "speaker_role": "MAIN_DAEE", "start_ms": 126_000, "end_ms": 185_000, "text_original": "Every calibrated physical law points away from random chaotic materialism toward purposeful creation."},
        ]
        bad_card = {
            "id": "V31_001",
            "video_id": "V31",
            "daee_speaker_id": "SPK_00",
            "interlocutor_speaker_id": "SPK_01",
            "question_turn_id": "V31_T03",
            "answer_turn_id": "V31_T02",
            "timestamp_seconds": 60,
            "question_en": "Why should anyone trust ancient inheritance decrees regarding property distribution?",
            "question_ar": "لماذا يوثق بقوانين المواريث القديمة المتعلقة بتوزيع الممتلكات؟",
            "core_answer_summary_en": (
                "Classical jurisprudence regulates probate estates, agrarian taxation, testamentary witnesses, "
                "commercial contracts, guild apprenticeships, municipal water canals, and feudal obligations across "
                "medieval provinces using detailed legal registries and arbitral tribunals to settle boundary disputes "
                "and commercial debts fairly for merchants and landowners throughout rural kingdoms. Furthermore, "
                "customary statutory courts inspect boundary charters, heraldic deeds, maritime customs duties, "
                "livestock tithes, apprenticeship indentures, orchard leases, silver coinage standards, and borough "
                "charters whenever heirs contest ancestral estates before regional magistrates under traditional "
                "civil codes governing provincial succession, guild inheritance privileges, and agrarian tenancy rights."
            ),
            "core_answer_summary_ar": "يتناول الفقه تنظيم المواريث والعقود والديون لل.",
            "logical_premises": ["Probate registries record land.", "Guild tribunals arbitrate debts.", "Statutes bind landowners."],
            "logical_premises_ar": ["السجلات توثق الأراضي.", "المحاكم تفصل الديون.", "القوانين تلزم الملاك."],
            "fallacy_or_misconception_exposed": "Anachronism fallacy.",
            "fallacy_or_misconception_exposed_ar": "مغالطة الإسقاط الزمني",
            "socratic_question": "How could oral customs alone settle commercial debt registries without witnesses?",
            "socratic_question_ar": "كيف تحسم الأعراف وحدها الديون التجارية بلا شهود؟",
            "interlocutor_verbatim_quote": "Interesting point about cosmic constants.",
            "interlocutor_verbatim_quote_ar": "نقطة مثيرة حول الثوابت الكونية لل.",
            "verbatim_quote": "The teleological arrangement of constants demonstrates an All-Wise Designer beyond blind chance.",
            "verbatim_quote_ar": "انتظام الثوابت الكونية يدل على مصمم حكيم يتجاوز الصدفة العمياء.",
            "dialogue_exchange": [
                {"turn_id": "V31_T03", "role": "INTERLOCUTOR", "timestamp_seconds": 121, "text_en": "Interesting point about cosmic constants.", "text_ar": "نقطة مثيرة."},
                {"turn_id": "V31_T02", "role": "MAIN_DAEE", "timestamp_seconds": 61, "text_en": "Unrelated fabricated probate commercial registries guilds and feudal obligations text here.", "text_ar": "نص غير مطابق لل."},
            ],
        }
        report = audit_single_card(bad_card, turns, require_arabic=True, video_meta={"duration_sec": 300.0, "speakers_count": 2, "turns_count": 4})
        codes = {d["code"] for d in report["defects"]}
        self.assertIn("FUNCTIONAL_DIARIZATION_COLLAPSE", codes)
        self.assertIn("DIALOGUE_TURN_TEXT_SHIFTED", codes)
        self.assertIn("NON_MONOTONIC_DIALOGUE_TIMESTAMPS", codes)
        self.assertIn("UNGROUNDED_DOSSIER_SUMMARY", codes)
        self.assertIn("CORRUPTED_ARABIC_PREPOSITION", codes)
        self.assertIn("TRUNCATED_ARABIC_SUMMARY", codes)
        self.assertIn("[GATE-DIARIZATION]", report["hard_fails_all"])
        self.assertIn("[GATE-UNGROUNDED]", report["hard_fails_all"])

        # Tier 3 card (50-74 score without hard-fail gates) must be marked FAIL in v3.1
        tier3_card = {
            "id": "T3_001",
            "video_id": "T3",
            "daee_speaker_id": "SPK_00",
            "interlocutor_speaker_id": "SPK_01",
            "question_turn_id": "T3_T01",
            "answer_turn_id": "T3_T02",
            "timestamp_seconds": 10,
            "question_en": "Why does morality require God",  # missing '?' (-4)
            "question_ar": "لماذا تحتاج الأخلاق إلى الله",  # missing '؟' (-2)
            "core_answer_summary_en": "The Da'ee explains that morality is objective.",  # meta narrator (-4) + <60w (-3) = -7
            "core_answer_summary_ar": "الأخلاق الموضوعية تستلزم خالقاً متعالياً.",
            "logical_premises": ["Only one premise"],  # <2 premises (-6)
            "logical_premises_ar": ["مقدمة واحدة"],
            "fallacy_or_misconception_exposed": "",  # (-3)
            "fallacy_or_misconception_exposed_ar": "",
            "socratic_question": "",  # (-3)
            "socratic_question_ar": "",
            "interlocutor_verbatim_quote": "",  # (-2)
            "interlocutor_verbatim_quote_ar": "",
            "verbatim_quote": "",  # (-5)
            "verbatim_quote_ar": "",
            "dialogue_exchange": [],  # (-4)
        }
        t3_turns = [
            {"turn_id": "T3_T01", "speaker_role": "INTERLOCUTOR", "start_ms": 10_000, "end_ms": 15_000, "text_original": "Why does morality require God?"},
            {"turn_id": "T3_T02", "speaker_role": "MAIN_DAEE", "start_ms": 16_000, "end_ms": 30_000, "text_original": "Objective morality presupposes a Lawgiver."},
        ]
        t3_report = audit_single_card(tier3_card, t3_turns, require_arabic=False)
        self.assertIsNone(t3_report["hard_fail_triggered"])
        self.assertGreaterEqual(t3_report["score_pct"], 50)
        self.assertLess(t3_report["score_pct"], 75)
        self.assertEqual(t3_report["status"], "FAIL")


class TestDeepRootCauseAndForensicDebugger(unittest.TestCase):
    """Regression suite for systemic Stage 1/2 root-cause fixes, v3.2 audit upgrades, and forensic debug tracer."""

    def test_mid_clause_flicker_smoothing_preserves_real_backchannels(self):
        from core.overlap_resolver import assign_words_to_speakers_iou, realign_words_preserving_interruptions

        # 1. Mid-clause acoustic flicker: "How | does the Bible | define adultery?" -> smoothed into 1 speaker_0 run
        flicker_words = [
            {"word": "How", "start": 1.000, "end": 1.200},
            {"word": "does", "start": 1.240, "end": 1.400},
            {"word": "the", "start": 1.420, "end": 1.520},
            {"word": "Bible", "start": 1.540, "end": 1.850},
            {"word": "define", "start": 1.910, "end": 2.200},
            {"word": "adultery?", "start": 2.230, "end": 2.650},
        ]
        flicker_rttm = [
            {"speaker_id": "speaker_0", "start_ms": 950, "end_ms": 1220},
            {"speaker_id": "speaker_1", "start_ms": 1230, "end_ms": 1870},
            {"speaker_id": "speaker_0", "start_ms": 1890, "end_ms": 2700},
        ]
        mapped_flicker = assign_words_to_speakers_iou(flicker_words, flicker_rttm)
        smoothed = realign_words_preserving_interruptions(mapped_flicker)
        self.assertEqual({w["speaker_id"] for w in smoothed}, {"speaker_0"})

        # 2. Genuine punctuated backchannel after sentence end -> MUST keep speaker_1 distinct
        backchannel_words = [
            {"word": "God", "start": 1.000, "end": 1.200},
            {"word": "is", "start": 1.220, "end": 1.350},
            {"word": "eternal.", "start": 1.370, "end": 1.700},
            {"word": "Really?", "start": 1.800, "end": 2.100},
            {"word": "Yes,", "start": 2.200, "end": 2.400},
            {"word": "necessarily.", "start": 2.420, "end": 2.900},
        ]
        backchannel_rttm = [
            {"speaker_id": "speaker_0", "start_ms": 950, "end_ms": 1750},
            {"speaker_id": "speaker_1", "start_ms": 1780, "end_ms": 2150},
            {"speaker_id": "speaker_0", "start_ms": 2180, "end_ms": 2950},
        ]
        mapped_bc = assign_words_to_speakers_iou(backchannel_words, backchannel_rttm)
        preserved = realign_words_preserving_interruptions(mapped_bc)
        self.assertEqual([w["speaker_id"] for w in preserved], ["speaker_0", "speaker_0", "speaker_0", "speaker_1", "speaker_0", "speaker_0"])

    def test_m3_multi_strategy_tournament_picks_best_candidate_and_writes_trace(self):
        import numpy as np
        from unittest.mock import patch
        from core.overlap_resolver import run_diarization_m3

        class MockTournamentDiarizer:
            def __init__(self):
                self.calls = []
                self._emb_call = 0

            def run_diarization(
                self,
                wav_path,
                rttm_path,
                is_short=False,
                forced_num_speakers=None,
                clustering_overrides=None,
                msdd_sigmoid_threshold=None,
                clustering_only=False,
            ):
                self.calls.append((forced_num_speakers, msdd_sigmoid_threshold, clustering_only))
                if len(self.calls) == 1:
                    # Initial pass: 100% single-speaker collapse across 240s
                    lines = [f"SPEAKER audio 1 {i*10:.2f} 9.00 <NA> <NA> speaker_0 <NA> <NA>" for i in range(24)]
                elif forced_num_speakers is None:
                    # Strategy 1 (sensitive spectral): weak 8% 2nd speaker
                    lines = [f"SPEAKER audio 1 {i*10:.2f} 9.00 <NA> <NA> {'speaker_1' if i in (5, 15) else 'speaker_0'} <NA> <NA>" for i in range(24)]
                else:
                    # Strategy 2 (bounded 2-speaker): balanced 33% 2nd speaker across the whole video
                    lines = [f"SPEAKER audio 1 {i*10:.2f} 9.00 <NA> <NA> {'speaker_1' if i % 3 == 1 else 'speaker_0'} <NA> <NA>" for i in range(24)]
                text = "\n".join(lines) + "\n"
                Path(rttm_path).parent.mkdir(parents=True, exist_ok=True)
                Path(rttm_path).write_text(text, encoding="utf-8")
                return text

            def extract_embedding(self, samples_slice, sr=16000):
                # Return orthogonal embeddings based on slice mean tag so speaker_0 and speaker_1 stay distinct
                vec = np.zeros(192, dtype=np.float32)
                if float(np.mean(samples_slice)) > 0.25:
                    vec[1] = 1.0
                else:
                    vec[0] = 1.0
                return vec

        # Encode speaker_1 intervals (i % 3 == 1) with amplitude +0.5 so embeddings are genuinely orthogonal!
        sr = 16000
        synth = np.zeros(240 * sr, dtype=np.float32)
        for i in range(24):
            if i % 3 == 1:
                synth[i * 10 * sr : (i * 10 + 9) * sr] = 0.5

        with tempfile.TemporaryDirectory() as tmp:
            ws = SharedWorkspace(Path(tmp))
            vid = "M3_TOURN"
            wav_p = ws.audio_clean_dir / f"{vid}_vocals.wav"
            wav_p.write_bytes(b"RIFF0000WAVEfmt ")
            diar = MockTournamentDiarizer()

            with patch("core.overlap_resolver.read_wav_float32", return_value=(synth, sr)):
                rttm_segs, centroids, overlaps = run_diarization_m3(
                    video_id=vid,
                    leveled_16k_wav=wav_p,
                    workspace=ws,
                    diarizer=diar,
                )
            spk_ids = {s["speaker_id"] for s in rttm_segs}
            self.assertEqual(spk_ids, {"SPK_00", "SPK_01"})
            self.assertGreaterEqual(len(diar.calls), 3)
            trace = ws.read_json(ws.debug_traces_dir / f"{vid}.trace.json", default={})
            m3_tr = trace.get("stage1_m3") or {}
            self.assertTrue(m3_tr.get("collapse_triggered"))
            self.assertTrue(m3_tr.get("recovered"))
            self.assertEqual(m3_tr.get("accepted_strategy"), "bounded 2-speaker recovery")
            self.assertGreaterEqual(len(m3_tr.get("retry_attempts") or []), 2)

    def test_merge_word_cap_raw_subturn_snapping_quote_trimming_and_analogy_grounding(self):
        from core.scenario_extractor import (
            merge_consecutive_speaker_turns,
            _ground_dialogue_exchange,
            _trim_focused_quote,
            _filter_grounded_analogies,
        )

        # 1. Word cap prevents >110w fused blocks when consecutive same-speaker turns are merged
        raw_turns = [
            {
                "turn_id": "CAP_T01",
                "speaker_id": "SPK_00",
                "speaker_role": "MAIN_DAEE",
                "episode_index": 1,
                "start_ms": 0,
                "end_ms": 20000,
                "start_fmt": "00:00",
                "end_fmt": "00:20",
                "text_original": " ".join(f"proofword{i}" for i in range(65)),
            },
            {
                "turn_id": "CAP_T02",
                "speaker_id": "SPK_00",
                "speaker_role": "MAIN_DAEE",
                "episode_index": 1,
                "start_ms": 20500,
                "end_ms": 40000,
                "start_fmt": "00:20",
                "end_fmt": "00:40",
                "text_original": " ".join(f"reasonword{i}" for i in range(65)),
            },
        ]
        merged = merge_consecutive_speaker_turns(raw_turns, max_merged_words=110)
        self.assertEqual(len(merged), 2)

        # 2. Raw sub-turn snapping aligns merged dialogue_exchange turn to exact SQLite turn_id
        sub_raw = [
            {
                "turn_id": "SNP_T01",
                "speaker_id": "SPK_00",
                "speaker_role": "MAIN_DAEE",
                "episode_index": 1,
                "start_ms": 0,
                "end_ms": 5000,
                "start_fmt": "00:00",
                "end_fmt": "00:05",
                "text_original": "No, I did not say that.",
            },
            {
                "turn_id": "SNP_T02",
                "speaker_id": "SPK_00",
                "speaker_role": "MAIN_DAEE",
                "episode_index": 1,
                "start_ms": 5500,
                "end_ms": 15000,
                "start_fmt": "00:05",
                "end_fmt": "00:15",
                "text_original": "God rewards believers in Paradise with eternal peace and real blessings.",
            },
        ]
        cohesive = merge_consecutive_speaker_turns(sub_raw, max_merged_words=110)
        self.assertEqual(len(cohesive), 1)
        ex_out, _ = _ground_dialogue_exchange(
            [{"turn_id": "SNP_T01", "role": "MAIN_DAEE", "text_en": "God rewards believers in Paradise with eternal peace and real blessings."}],
            {c["turn_id"]: c for c in cohesive},
            {r["turn_id"]: r for r in sub_raw},
            "Hashim",
            "هاشم",
            "SPK_01",
            0,
        )
        self.assertEqual(ex_out[0]["turn_id"], "SNP_T02")

        # 3. _trim_focused_quote strips crowd-control logistics and trims overlong speech to <= 65 words
        noisy_speech = (
            "Stop interrupting me! Otherwise, I'll not carry on the conversation. "
            "God is absolute, eternal, and utterly distinct from contingent creation that depends on food, sleep, or decay. "
            + " ".join(["Every created thing depends on an independent creator beyond space and time."] * 10)
        )
        trimmed = _trim_focused_quote(noisy_speech, anchor_text="God eternal distinct contingent creation", max_words=65)
        self.assertNotIn("Stop interrupting", trimmed)
        self.assertLessEqual(len(trimmed.split()), 65)
        self.assertIn("contingent creation", trimmed)

        # 4. _filter_grounded_analogies drops invented textbook analogies absent from transcript, keeps grounded ones
        long_turns = [{"text_original": " ".join(["A finely tuned pocket watch points directly to an intelligent watchmaker behind creation."] * 15)} for _ in range(4)]
        long_tr = " ".join(t["text_original"] for t in long_turns).lower()
        kept_en, kept_ar = _filter_grounded_analogies(
            [
                "Breaking a formal business partnership contract invalidates the agreement.",
                "A finely tuned pocket watch points to a watchmaker.",
            ],
            ["فسخ عقد شراكة تجارية يبطل الاتفاق.", "الساعة المتقنة تدل على صانعها."],
            long_tr,
            long_turns,
        )
        self.assertEqual(kept_en, ["A finely tuned pocket watch points to a watchmaker."])
        self.assertEqual(kept_ar, ["الساعة المتقنة تدل على صانعها."])

    def test_pre_stage2_diarization_health_gate_skips_collapsed_debate_and_generates_debug_report(self):
        from core.debug_tracer import generate_session_debug_report
        from tools.inspect_debug_traces import main as inspect_cli_main

        with tempfile.TemporaryDirectory() as tmp:
            ws = SharedWorkspace(Path(tmp))
            vid = "COLL_SKIP1"
            turns = [
                {
                    "turn_id": f"{vid}_T{i:04d}",
                    "turn_index": i - 1,
                    "episode_index": 1,
                    "speaker_id": "SPK_00",
                    "speaker_role": "MAIN_DAEE",
                    "start_ms": (i - 1) * 25000,
                    "end_ms": i * 25000 - 1000,
                    "start_fmt": f"00:{(i-1)*25:02d}",
                    "end_fmt": f"00:{i*25:02d}",
                    "text_original": "This is a collapsed debate monologue combining both speakers with twenty five words of theological argumentation about scripture and truth today.",
                }
                for i in range(1, 13)
            ]
            ws.atomic_write_json(ws.scenarios_dir / f"{vid}.json", {
                "video_id": vid,
                "title_ar": "مناظرة منهارة التمييز الصوتي",
                "duration_sec": 300.0,
                "metrics": {"duration_sec": 300.0, "speakers_count": 1},
                "speakers": [{"speaker_id": "SPK_00", "role": "MAIN_DAEE", "display_name_en": "Hashim", "display_name_ar": "هاشم"}],
                "turns": turns,
            })
            ws.advance_micro_step(vid, "M5_SCENARIO_BUILT")

            class NeverCalledLLM:
                def __init__(self):
                    self.calls = 0

                def extract_round1_english(self, prompt):
                    self.calls += 1
                    return "[]"

                def extract_round2_arabic(self, r1_json, prompt, enable_thinking=False, max_thinking_tokens=0):
                    self.calls += 1
                    return "[]"

            llm = NeverCalledLLM()
            units = Stage2ScenarioExtractor(workspace=ws, llm_client=llm).extract_video_m6(vid, verbose=False)
            self.assertEqual(units, [])
            self.assertEqual(llm.calls, 0, "Collapsed video must skip LLM extraction with 0 calls")
            trace = ws.read_json(ws.debug_traces_dir / f"{vid}.trace.json", default={})
            self.assertTrue((trace.get("stage2_m6") or {}).get("skipped_pre_stage2_diarization"))
            self.assertIn("SINGLE_SPEAKER_COLLAPSE", (trace.get("stage2_m6") or {}).get("pre_stage2_skip_reason", ""))

            report_path = generate_session_debug_report(ws)
            report_txt = report_path.read_text(encoding="utf-8")
            self.assertIn(vid, report_txt)
            self.assertIn("SKIP_DIAR", report_txt)
            self.assertIn("SINGLE_SPEAKER_COLLAPSE", report_txt)

            # Verify CLI inspector executes cleanly
            rc = inspect_cli_main(["--dir", str(ws.debug_traces_dir), "--failures-only"])
            self.assertEqual(rc, 0)

    def test_implicit_round2_en_repair_anchor_subreasons_and_bilingual_length_mismatch(self):
        from core.scenario_extractor import (
            merge_round1_and_round2_units,
            anchor_and_enrich_extracted_unit,
            diagnose_missing_arabic_fields,
        )

        # 1. allow_implicit_en_repair=True applies R2's rewritten question_en even when english_edits_needed=False
        r1 = [{
            "question_turn_id": "R2F_T01",
            "answer_turn_id": "R2F_T02",
            "question_en": "In Christianity, is adultery the only valid reason for divorce?",
            "core_answer_summary_en": " ".join(["Islamic law protects spouses from harm and abuse through structured divorce rights."] * 5),
        }]
        r2 = [{
            "question_turn_id": "R2F_T01",
            "rubric_audit": {"score": 92, "english_edits_needed": False, "drop_card": False},
            "question_en": "Why does Islam permit divorce for serious harm rather than restricting it solely to adultery?",
            "question_ar": "لماذا يُجيز الإسلام الطلاق لرفع الضرر الجسيم ولا يحصره في الزنا وحده؟",
        }]
        kept_clean = merge_round1_and_round2_units(r1, r2, allow_implicit_en_repair=False)[0]
        self.assertEqual(kept_clean["question_en"], r1[0]["question_en"])
        repaired_flagged = merge_round1_and_round2_units(r1, r2, allow_implicit_en_repair=True)[0]
        self.assertEqual(repaired_flagged["question_en"], r2[0]["question_en"])

        # 2. anchor_and_enrich_extracted_unit populates exact drop sub-reasons
        scenario = {
            "video_id": "SUBR",
            "title_ar": "حوار",
            "speakers": [
                {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "display_name_en": "Interlocutor", "display_name_ar": "المحاور"},
            ],
            "turns": [
                {"turn_id": "SUBR_T01", "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "start_ms": 0, "end_ms": 5000, "start_fmt": "00:00", "end_fmt": "00:05", "text_original": "Does Surah 3:85 reject other religions?"},
                {"turn_id": "SUBR_T02", "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 6000, "end_ms": 20000, "start_fmt": "00:06", "end_fmt": "00:20", "text_original": "Surah 3:85 states whoever seeks a religion other than Islam it will never be accepted."},
            ],
        }
        reasons = []
        res = anchor_and_enrich_extracted_unit(
            raw_unit={
                "question_turn_id": "SUBR_T01",
                "answer_turn_id": "SUBR_T02",
                "question_en": "Does Surah 3:85 mean Islam rejects all other religions?",
                "question_ar": "هل تعني سورة آل عمران 85 أن الإسلام يرفض سائر الأديان؟",
                "core_answer_summary_en": "Surah 3:85 does not imply that non-Muslims are automatically condemned because sincerity matters.",
                "core_answer_summary_ar": "الآية لا تعني رفض الأديان الأخرى.",
                "verbatim_quote": "Surah 3:85 states whoever seeks a religion other than Islam it will never be accepted.",
                "verbatim_quote_ar": "تقرر سورة آل عمران أن من يبتغ غير الإسلام ديناً فلن يقبل منه.",
            },
            scenario=scenario,
            episode_index=1,
            window_start_ms=0,
            unit_seq_num=1,
            drop_reason_out=reasons,
        )
        self.assertIsNone(res)
        self.assertTrue(reasons and reasons[0].startswith("THEOLOGICAL_BIAS:"))

        # 3. diagnose_missing_arabic_fields flags excess bilingual quote length mismatch
        diag = diagnose_missing_arabic_fields({
            "question_en": "Does Surah 3:85 teach salvation exclusivity in Islam?",
            "question_ar": "هل تقرر سورة آل عمران حصرية النجاة في الإسلام؟",
            "core_answer_summary_en": "Islam affirms pure monotheism and exclusive submission to Allah alone.",
            "core_answer_summary_ar": "يقرر الإسلام التوحيد الخالص وإفراد الله بالعبادة والانقياد له وحده.",
            "interlocutor_verbatim_quote": "Why does the Quran say whoever seeks another religion will not be accepted?",
            "interlocutor_verbatim_quote_ar": " ".join(["هذا نص عربي طويل جداً يترجم فقرة أخرى بالكامل لا علاقة لها بالاقتباس الإنجليزي القصير"] * 6),
            "verbatim_quote": "Surah 3:85 clearly establishes submission to God alone.",
            "verbatim_quote_ar": "تقرر سورة آل عمران بوضوح وجوب الاستسلام لله وحده.",
        })
        self.assertIn("length_mismatch_excess:interlocutor_verbatim_quote_ar", diag)

        # 4. End-to-end Stage 2 records R2 <think> excerpt and R2 English repair diff in debug trace
        with tempfile.TemporaryDirectory() as tmp:
            ws = SharedWorkspace(Path(tmp))
            vid = "R2_REPAIR1"
            ws.atomic_write_json(ws.scenarios_dir / f"{vid}.json", {
                "video_id": vid,
                "title_ar": "حوار حول الطلاق",
                "duration_sec": 120.0,
                "metrics": {"duration_sec": 120.0, "speakers_count": 2},
                "speakers": [
                    {"speaker_id": "SPK_00", "role": "MAIN_DAEE", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                    {"speaker_id": "SPK_01", "role": "INTERLOCUTOR", "display_name_en": "Interlocutor", "display_name_ar": "المحاور"},
                ],
                "turns": [
                    {"turn_id": f"{vid}_T0001", "turn_index": 0, "episode_index": 1, "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "start_ms": 0, "end_ms": 15000, "start_fmt": "00:00", "end_fmt": "00:15", "text_original": "In the Gospel divorce is only allowed for sexual immorality and not for other reasons."},
                    {"turn_id": f"{vid}_T0002", "turn_index": 1, "episode_index": 1, "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 16000, "end_ms": 45000, "start_fmt": "00:16", "end_fmt": "00:45", "text_original": "Islam protects spouses by allowing divorce for serious harm, abuse, and abandonment while preserving justice and dignity."},
                    {"turn_id": f"{vid}_T0003", "turn_index": 2, "episode_index": 1, "speaker_id": "SPK_01", "speaker_role": "INTERLOCUTOR", "start_ms": 46000, "end_ms": 65000, "start_fmt": "00:46", "end_fmt": "01:05", "text_original": "Does not permitting divorce too easily weaken the sanctity of marriage vows?"},
                    {"turn_id": f"{vid}_T0004", "turn_index": 3, "episode_index": 1, "speaker_id": "SPK_00", "speaker_role": "MAIN_DAEE", "start_ms": 66000, "end_ms": 95000, "start_fmt": "01:06", "end_fmt": "01:35", "text_original": "Reconciliation is mandated first, yet trapping an abused spouse in permanent harm contradicts divine mercy and justice."},
                ],
            })
            ws.advance_micro_step(vid, "M5_SCENARIO_BUILT")

            class RepairMockLLM:
                def extract_round1_english(self, prompt):
                    return json.dumps([{
                        "question_turn_id": f"{vid}_T0001",
                        "answer_turn_id": f"{vid}_T0002",
                        "topic_category": "Sharia & Islamic Law",
                        "topic_subcategory": "Divorce Law",
                        "interlocutor_type": "Christian / Trinitarian",
                        "question_en": "In Christianity, is the only valid reason for divorce adultery?",
                        "core_answer_summary_en": " ".join(["Islamic jurisprudence balances the sanctity of marriage with protection against serious harm, abuse, and neglect by mandating reconciliation first while granting lawful exit when marriage causes enduring injustice."] * 2),
                        "logical_premises": [
                            "Divine law must uphold both marital stability and human protection from harm.",
                            "Restricting divorce solely to adultery traps abused spouses in permanent injustice.",
                            "Therefore, permitting structured divorce for serious harm reflects supreme divine mercy and justice.",
                        ],
                        "fallacy_or_misconception_exposed": "False dichotomy between strict indissolubility and reckless dissolution.",
                        "socratic_question": "Should an abused spouse remain legally trapped when marriage causes severe ongoing harm?",
                        "interlocutor_verbatim_quote": "Does not permitting divorce too easily weaken the sanctity of marriage vows?",
                        "verbatim_quote": "Islam protects spouses by allowing divorce for serious harm, abuse, and abandonment while preserving justice and dignity.",
                        "dialogue_exchange": [
                            {"turn_id": f"{vid}_T0001", "role": "INTERLOCUTOR", "move_type": "Claim", "text_en": "In the Gospel divorce is only allowed for sexual immorality and not for other reasons."},
                            {"turn_id": f"{vid}_T0002", "role": "MAIN_DAEE", "move_type": "Proof", "text_en": "Islam protects spouses by allowing divorce for serious harm, abuse, and abandonment while preserving justice and dignity."},
                            {"turn_id": f"{vid}_T0003", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Does not permitting divorce too easily weaken the sanctity of marriage vows?"},
                            {"turn_id": f"{vid}_T0004", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "Reconciliation is mandated first, yet trapping an abused spouse in permanent harm contradicts divine mercy and justice."},
                        ],
                    }])

                def extract_round2_arabic(self, r1_json, prompt, enable_thinking=False, max_thinking_tokens=0):
                    # Deliberately leaves english_edits_needed=False while repairing question_en!
                    return "<think>Pre-R2 flagged DAEE_CROSS_EXAM_IN_QUESTION; converting question_en to Islamic-first seeker inquiry.</think>" + json.dumps([{
                        "question_turn_id": f"{vid}_T0001",
                        "rubric_audit": {"score": 95, "english_edits_needed": False, "drop_card": False},
                        "question_en": "Does permitting divorce beyond adultery weaken the sanctity of marriage in Islam?",
                        "question_ar": "هل يُضعف إباحة الطلاق لغير الزنا قدسية الزواج في الإسلام؟",
                        "core_answer_summary_ar": "يوازن التشريع الإسلامي بين قدسية عقد الزواج وحماية الزوجين من الضرر الجسيم والإيذاء والهجر، فيأمر بالإصلاح والتحكيم أولاً، ثم يشرع الطلاق مخرجاً عادلاً عند تعذر العشرة بالمعروف رفعاً للظلم وتحقيقاً للرحمة والعدل الإلهي.",
                        "logical_premises_ar": [
                            "الشريعة الإلهية الكاملة تجمع بين حفظ استقرار الأسرة ودفع الضرر عن الإنسان.",
                            "حصر الطلاق في الزنا وحده يُبقي الزوجة المتضررة رهينة للأذى والظلم الدائم.",
                            "إذن فإباحة الطلاق المنضبط لرفع الضرر تجسد كمال العدل والرحمة الإلهية.",
                        ],
                        "fallacy_or_misconception_exposed_ar": "مغالطة القسمة الثنائية الزائفة بين تحريم الطلاق مطلقاً وبين التفريط في قدسية الزواج.",
                        "socratic_question_ar": "هل يليق بالعدل الإلهي إلزام الزوجة بالبقاء تحت الأذى الدائم دون مخرج شرعي؟",
                        "interlocutor_verbatim_quote_ar": "ألا يُضعف السماح بالطلاق بسهولة قدسية عهد الزواج؟",
                        "verbatim_quote_ar": "يحمي الإسلام الزوجين بإباحة الطلاق عند الضرر الجسيم والإيذاء والهجر مع حفظ العدل والكرامة.",
                        "dialogue_exchange": [
                            {"turn_id": f"{vid}_T0001", "text_ar": "في الإنجيل لا يُباح الطلاق إلا لعلة الفاحشة دون سائر الأسباب."},
                            {"turn_id": f"{vid}_T0002", "text_ar": "يحمي الإسلام الزوجين بإباحة الطلاق عند الضرر الجسيم والإيذاء والهجر مع حفظ العدل والكرامة."},
                            {"turn_id": f"{vid}_T0003", "text_ar": "ألا يُضعف السماح بالطلاق بسهولة قدسية عهد الزواج؟"},
                            {"turn_id": f"{vid}_T0004", "text_ar": "الإصلاح مقدم أولاً، غير أن حبس المتضرر في الأذى الدائم يناقض الرحمة والعدل الإلهي."},
                        ],
                    }], ensure_ascii=False)

            out_units = Stage2ScenarioExtractor(workspace=ws, llm_client=RepairMockLLM()).extract_video_m6(vid, verbose=False)
            self.assertEqual(len(out_units), 1)
            self.assertEqual(out_units[0]["question_en"], "Does permitting divorce beyond adultery weaken the sanctity of marriage in Islam?")
            tr = ws.read_json(ws.debug_traces_dir / f"{vid}.trace.json", default={})
            u_tr = tr["stage2_m6"]["windows"][0]["units"][0]
            self.assertIn("question_en", u_tr["r2_en_repairs"])
            self.assertIn("DAEE_CROSS_EXAM_IN_QUESTION", u_tr["r2_think_excerpt"])
            self.assertEqual(u_tr["outcome"], "KEPT")

    def test_expanded_theological_bias_and_audit_cards_v32_blind_spots(self):
        from db.database import check_theological_bias

        # 1. Expanded check_theological_bias doctrinal rules
        self.assertIsNotNone(check_theological_bias({
            "core_answer_summary_en": "Surah 3:85 does not imply that non-Muslims are automatically condemned.",
        }))
        self.assertIsNotNone(check_theological_bias({
            "core_answer_summary_en": "The concept of 72 virgins is purely symbolic and not a literal reward.",
        }))
        self.assertIsNotNone(check_theological_bias({
            "core_answer_summary_en": "The phrase 72 virgins is explicitly mentioned in the Quran for martyrs.",
        }))
        self.assertIsNotNone(check_theological_bias({
            "core_answer_summary_en": "The claim that a born again Christian cannot sin is a misinterpretation of 1 John 3:9, which means they are not defined by sin.",
        }))

        # 2. Audit v3.2 blind-spot detections
        turns = [
            {
                "turn_id": "V32_T01",
                "speaker_role": "MAIN_DAEE",
                "start_ms": 0,
                "end_ms": 20000,
                "text_original": (
                    "Stop interrupting me! Otherwise I'll not carry on the conversation. "
                    "In Christianity, is the only valid reason for divorce sexual immorality even in severe abuse cases? "
                    + " ".join(["divorce marriage justice mercy protection scripture"] * 20)
                ),
            },
            {
                "turn_id": "V32_T02",
                "speaker_role": "INTERLOCUTOR",
                "start_ms": 21000,
                "end_ms": 40000,
                "text_original": (
                    "The Bible speaks about marriage vows, divorce, abuse, and Christian forgiveness in many chapters "
                    + " ".join(["marriage vows divorce christian forgiveness"] * 12)
                ),
            },
        ]
        card = {
            "id": "V32_001",
            "video_id": "V32",
            "daee_speaker_id": "SPK_00",
            "interlocutor_speaker_id": "SPK_01",
            "question_turn_id": "V32_T02",
            "answer_turn_id": "V32_T01",
            "timestamp_seconds": 10,
            "question_en": "In Christianity, is the only valid reason for divorce adultery?",
            "question_ar": "هل العبارة التي تقولها في دينكم تعني أن الزنا هو سبب الطلاق الوحيد؟",
            "core_answer_summary_en": " ".join(["Islamic jurisprudence regulates divorce and marriage with justice and mercy."] * 10),
            "core_answer_summary_ar": " ".join(["ينظم الفقه الإسلامي أحكام الزواج والطلاق بالعدل والرحمة وحفظ الحقوق."] * 8),
            "logical_premises": ["Premise 1.", "Premise 2.", "Conclusion."],
            "logical_premises_ar": ["مقدمة 1.", "مقدمة 2.", "نتيجة."],
            "fallacy_or_misconception_exposed": "False dichotomy.",
            "fallacy_or_misconception_exposed_ar": "قسمة ثنائية زائفة.",
            "socratic_question": "How does indissolubility protect an abused spouse?",
            "socratic_question_ar": "كيف يحمي تحريم الطلاق الزوجة المتضررة؟",
            # Exact substring from MAIN_DAEE turn (even though topic words also appear in INTERLOCUTOR!)
            "interlocutor_verbatim_quote": "In Christianity, is the only valid reason for divorce sexual immorality even in severe abuse cases?",
            "interlocutor_verbatim_quote_ar": " ".join(["هذا نص عربي طويل جداً لا يطابق طول الاقتباس الإنجليزي الأصلي على الإطلاق"] * 8),
            "verbatim_quote": turns[0]["text_original"],
            "verbatim_quote_ar": " ".join(["توقف عن مقاطعتي وإلا فلن أواصل الحوار فالإسلام يحفظ العدل والرحمة في الزواج والطلاق."] * 8),
            "dialogue_exchange": [
                {"turn_id": "V32_T01", "role": "MAIN_DAEE", "timestamp_seconds": 0, "text_en": " ".join(["justice mercy protection"] * 12), "text_ar": "العدل والرحمة."},
                {"turn_id": "V32_T02", "role": "INTERLOCUTOR", "timestamp_seconds": 21, "text_en": "Okay.", "text_ar": "حسناً."},
                {"turn_id": "V32_T01", "role": "MAIN_DAEE", "timestamp_seconds": 25, "text_en": " ".join(["divorce scripture mercy"] * 12), "text_ar": "الطلاق والرحمة."},
                {"turn_id": "V32_T01", "role": "MAIN_DAEE", "timestamp_seconds": 30, "text_en": " ".join(["marriage justice protection"] * 12), "text_ar": "الزواج والعدل."},
            ],
        }
        rep = audit_single_card(card, turns, require_arabic=True)
        codes = {d["code"] for d in rep["defects"]}
        self.assertIn("[GATE-PERSPECTIVE]", rep["hard_fails_all"])
        self.assertIn("[GATE-ROLE]", rep["hard_fails_all"])
        self.assertIn("DAEE_CROSS_EXAM_IN_QUESTION", codes)
        self.assertIn("QUOTE_ROLE_SWAPPED_INTERLOCUTOR", codes)
        self.assertIn("LOGISTICS_NOISE_QUOTE", codes)
        self.assertIn("OVERLONG_FUSED_QUOTE", codes)
        self.assertIn("DAEE_DOMINATED_DIALOGUE_EXCHANGE", codes)
        self.assertIn("BILINGUAL_QUOTE_LENGTH_MISMATCH", codes)
        self.assertIn("AWKWARD_ARABIC_2ND_PERSON_QUESTION", codes)

        # Also verify 7-word mid-sentence cut without punctuation triggers TRUNCATED_FRAGMENT_QUOTE
        card2 = dict(card)
        card2["interlocutor_verbatim_quote"] = "You said when you're a new born"
        card2["interlocutor_verbatim_quote_ar"] = "قلت حين تكون مولوداً من جديد"
        rep2 = audit_single_card(card2, turns, require_arabic=True)
        self.assertIn("TRUNCATED_FRAGMENT_QUOTE", {d["code"] for d in rep2["defects"]})


if __name__ == "__main__":
    unittest.main()
