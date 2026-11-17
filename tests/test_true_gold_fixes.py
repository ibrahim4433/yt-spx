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

from core.audio_diarizer import WarmASRAlignmentEngine
from core.drive_sync import SharedWorkspace
from core.overlap_resolver import (
    WarmNeMoDiarizer,
    merge_fragmented_speaker_clusters,
    run_diarization_m3,
)
from core.scenario_extractor import (
    Stage2ScenarioExtractor,
    _is_duplicate_qa,
    anchor_and_enrich_extracted_unit,
    merge_round1_and_round2_units,
)
from core.smart_bot import SmartDialecticBot
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
from tools.card_rating_system.audit_cards import audit_single_card
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


if __name__ == "__main__":
    unittest.main()
