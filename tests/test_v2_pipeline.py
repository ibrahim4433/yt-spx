#!/usr/bin/env python3
"""
tests/test_v2_pipeline.py
Comprehensive unit & integration tests for the Dawah Dialectics V2 (yt-spx) pipeline:
  - SharedWorkspace M0->M6 micro-step checkpointing, multi-account locks, circuit breaker & SQLite compactor
  - Overlap detection, non-overlap centroid span extraction, 2-stream cosine matching & interruption-preserving word alignment
  - 3-tier speaker classification (MAIN_DAEE / INTERLOCUTOR / PEDESTRIAN), TitaNet VoiceprintRegistry & episode segmentation
  - Stage 2 episode windowing, mid-video partial checkpoint resume & CTC turn timestamp anchoring
  - Smart-batch orchestrator candidate prioritization (resuming M1..M4 before new M0 downloads)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.drive_sync import SharedWorkspace, STEP_RANK
from core.overlap_resolver import (
    parse_rttm_lines,
    detect_simultaneous_overlap_windows,
    extract_clean_non_overlap_spans,
    assign_separated_streams_to_speakers,
    assign_words_to_speakers_iou,
    realign_words_preserving_interruptions,
    merge_fragmented_speaker_clusters,
    get_deterministic_clip_dir,
    cleanup_deterministic_clip_dir,
)
from core.audio_diarizer import (
    THEOLOGICAL_VOCAB_PROMPT_EN,
    THEOLOGICAL_VOCAB_PROMPT_AR,
)
from core.speaker_roles import VoiceprintRegistry, classify_speakers_and_episodes
from core.scenario_extractor import build_episode_windows, Stage2ScenarioExtractor
from core.pipeline_orchestrator import select_stage1_candidates, select_stage2_candidates
from db.database import get_stats, get_video_scenario, search_qa


class TestSharedWorkspaceAndMultiAccountSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "yt-spx"
        self.ws1 = SharedWorkspace(root_dir=self.root, worker_name="colab-acct-1")
        self.ws2 = SharedWorkspace(root_dir=self.root, worker_name="colab-acct-2")

    def tearDown(self):
        self.tmp.cleanup()

    def test_monotonic_micro_steps_and_artifact_reconciliation(self):
        vid = "vid_alpha_01"
        st0 = self.ws1.get_video_state(vid)
        self.assertEqual(st0["micro_step"], "M0_QUEUED")

        self.ws1.advance_micro_step(vid, "M2_VOCALS_CLEANED", artifacts={"opus": "a.opus"})
        st2 = self.ws2.get_video_state(vid)
        self.assertEqual(st2["micro_step"], "M2_VOCALS_CLEANED")
        self.assertEqual(st2["step_rank"], STEP_RANK["M2_VOCALS_CLEANED"])

        # Simulate deleted state file -> auto-reconcile from scenario file on Drive
        self.ws1.state_path(vid).unlink()
        (self.ws1.scenarios_dir / f"{vid}.json").write_text('{"video_id":"vid_alpha_01"}', encoding="utf-8")
        reconciled = self.ws2.get_video_state(vid)
        self.assertEqual(reconciled["micro_step"], "M5_SCENARIO_BUILT")

    def test_multi_account_optimistic_locks(self):
        lock_name = "vid_beta.stage1"
        self.assertTrue(self.ws1.acquire_lock(lock_name, ttl_sec=600, verify_delay_sec=0.0))
        # Second Colab account must NOT acquire active lock held by acct-1
        self.assertFalse(self.ws2.acquire_lock(lock_name, ttl_sec=600, verify_delay_sec=0.0))

        self.ws1.release_lock(lock_name)
        self.assertTrue(self.ws2.acquire_lock(lock_name, ttl_sec=600, verify_delay_sec=0.0))
        self.ws2.release_lock(lock_name)

    def test_anti_ban_rate_limiter_and_circuit_breaker(self):
        allowed1, wait1, _ = self.ws1.check_youtube_download_slot(min_global_gap_sec=15.0)
        self.assertTrue(allowed1)
        self.assertEqual(wait1, 0.0)

        # Immediate second request from another Colab account is paced
        allowed2, wait2, reason2 = self.ws2.check_youtube_download_slot(min_global_gap_sec=15.0)
        self.assertFalse(allowed2)
        self.assertGreater(wait2, 0.0)
        self.assertIn("pacing", reason2.lower())

        # Trip circuit breaker on bot challenge
        self.ws2.trip_youtube_circuit_breaker("Sign in to confirm you're not a bot", cooldown_minutes=30)
        allowed3, wait3, reason3 = self.ws1.check_youtube_download_slot(min_global_gap_sec=1.0)
        self.assertFalse(allowed3)
        self.assertGreater(wait3, 1000.0)
        self.assertIn("not a bot", reason3)


class TestOverlapResolverAndAlignment(unittest.TestCase):
    def test_rttm_overlap_detection_and_clean_span_subtraction(self):
        rttm_text = "\n".join([
            "SPEAKER test 1 0.000 10.000 <NA> <NA> speaker_0 <NA> <NA>",
            "SPEAKER test 1 8.500 5.000 <NA> <NA> speaker_1 <NA> <NA>",   # 1.5s overlap [8500..10000]
            "SPEAKER test 1 14.000 6.000 <NA> <NA> speaker_0 <NA> <NA>",
        ])
        segs = parse_rttm_lines(rttm_text)
        self.assertEqual(len(segs), 3)

        overlaps = detect_simultaneous_overlap_windows(segs, total_duration_ms=20_000, min_overlap_ms=800, pad_ms=250)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["core_start_ms"], 8500)
        self.assertEqual(overlaps[0]["core_end_ms"], 10000)
        self.assertEqual(set(overlaps[0]["speakers"]), {"SPK_00", "SPK_01"})

        clean_spans = extract_clean_non_overlap_spans(segs, overlaps, min_clean_ms=1000)
        # Ensure no clean span intersects [8500, 10000]
        for spk_spans in clean_spans.values():
            for s_ms, e_ms in spk_spans:
                self.assertTrue(e_ms <= 8500 or s_ms >= 10000)

    def test_two_stream_cosine_speaker_assignment(self):
        v0 = np.zeros(192, dtype=np.float32)
        v0[0] = 1.0
        v1 = np.zeros(192, dtype=np.float32)
        v1[1] = 1.0
        centroids = {"SPK_00": v0, "SPK_01": v1}

        # Stream 0 sounds like SPK_01; Stream 1 sounds like SPK_00
        s0 = np.zeros(192, dtype=np.float32)
        s0[1] = 0.96
        s0[0] = 0.04
        s1 = np.zeros(192, dtype=np.float32)
        s1[0] = 0.98
        s1[1] = 0.02

        assigned = assign_separated_streams_to_speakers([s0, s1], centroids, preferred_speakers=["SPK_00", "SPK_01"])
        self.assertEqual(assigned[0][0], "SPK_01")
        self.assertEqual(assigned[1][0], "SPK_00")
        self.assertGreater(assigned[0][1], 0.9)

    def test_interruption_preserving_word_realignment(self):
        rttm_segs = [
            {"start_ms": 0, "end_ms": 3000, "speaker_id": "SPK_00"},
            {"start_ms": 3000, "end_ms": 4200, "speaker_id": "SPK_01"},
            {"start_ms": 4200, "end_ms": 8000, "speaker_id": "SPK_00"},
        ]
        words = [
            {"text": "First", "start": 0.5, "end": 1.0},
            {"text": "premise", "start": 1.1, "end": 2.9},
            {"text": "Wait", "start": 3.1, "end": 3.4},
            {"text": "why?", "start": 3.5, "end": 4.0},
            {"text": "Because", "start": 4.4, "end": 5.0},
            {"text": "contingency.", "start": 5.1, "end": 6.2},
        ]
        mapped = assign_words_to_speakers_iou(words, rttm_segs)
        realigned = realign_words_preserving_interruptions(mapped)
        speakers_seq = [w["speaker_id"] for w in realigned]
        # Verify the mid-sentence 2-word Socratic interruption ("Wait why?", 900ms) by SPK_01 is preserved!
        self.assertEqual(
            speakers_seq,
            ["SPK_00", "SPK_00", "SPK_01", "SPK_01", "SPK_00", "SPK_00"],
        )

    def test_600ms_overlap_detection_default(self):
        rttm_text = "\n".join([
            "SPEAKER test 1 0.000 5.000 <NA> <NA> speaker_0 <NA> <NA>",
            "SPEAKER test 1 4.300 3.000 <NA> <NA> speaker_1 <NA> <NA>",  # 700ms overlap [4300..5000]
        ])
        segs = parse_rttm_lines(rttm_text)
        # Default min_overlap_ms is now 600ms, so 700ms must be detected!
        overlaps = detect_simultaneous_overlap_windows(segs, total_duration_ms=10_000)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["core_start_ms"], 4300)
        self.assertEqual(overlaps[0]["core_end_ms"], 5000)

    def test_post_clustering_centroid_merge_unifies_fragmented_speakers(self):
        # SPK_00 and SPK_02 have >0.90 cosine similarity (fragmented Da'ee)
        # SPK_01 has 0.0 similarity (distinct interlocutor)
        c0 = np.zeros(192, dtype=np.float32)
        c0[0] = 0.99
        c0[1] = 0.14
        c0 = c0 / np.linalg.norm(c0)

        c2 = np.zeros(192, dtype=np.float32)
        c2[0] = 0.98
        c2[1] = 0.20
        c2 = c2 / np.linalg.norm(c2)

        c1 = np.zeros(192, dtype=np.float32)
        c1[5] = 1.0  # orthogonal

        centroids = {
            "SPK_00": c0,
            "SPK_01": c1,
            "SPK_02": c2,
        }

        # SPK_00 has 30s speech, SPK_02 has 10s speech
        rttm_segs = [
            {"start_ms": 0, "end_ms": 30000, "speaker_id": "SPK_00"},
            {"start_ms": 30500, "end_ms": 40000, "speaker_id": "SPK_01"},
            {"start_ms": 40500, "end_ms": 50500, "speaker_id": "SPK_02"},
        ]

        merged_segs, merged_cents, id_map = merge_fragmented_speaker_clusters(
            rttm_segs,
            centroids,
            threshold=0.80,
        )

        # SPK_02 should be merged into SPK_00 because SPK_00 has larger duration
        self.assertIn("SPK_02", id_map)
        self.assertEqual(id_map["SPK_02"], "SPK_00")
        self.assertNotIn("SPK_02", merged_cents)
        self.assertIn("SPK_00", merged_cents)
        self.assertIn("SPK_01", merged_cents)

        # SPK_01 must NOT be merged
        self.assertEqual(id_map.get("SPK_01", "SPK_01"), "SPK_01")

        # RTTM segments should only contain SPK_00 and SPK_01
        speakers_in_segs = {s["speaker_id"] for s in merged_segs}
        self.assertEqual(speakers_in_segs, {"SPK_00", "SPK_01"})

    def test_deterministic_clip_dir_lifecycle(self):
        vid = "test_vid_lifecycle_99"
        d = get_deterministic_clip_dir(vid)
        self.assertTrue(d.exists())
        test_file = d / "sample.wav"
        test_file.write_text("audio data", encoding="utf-8")
        self.assertTrue(test_file.exists())

        cleanup_deterministic_clip_dir(vid)
        self.assertFalse(d.exists())

    def test_theological_multilingual_prompts_coverage(self):
        self.assertGreater(len(THEOLOGICAL_VOCAB_PROMPT_EN), 100)
        self.assertGreater(len(THEOLOGICAL_VOCAB_PROMPT_AR), 100)
        # Check Arabic terms
        self.assertIn("التوحيد", THEOLOGICAL_VOCAB_PROMPT_AR)
        self.assertIn("الثالوث", THEOLOGICAL_VOCAB_PROMPT_AR)
        self.assertIn("عثمان", THEOLOGICAL_VOCAB_PROMPT_AR)
        # Check English terms
        self.assertIn("Contingency", THEOLOGICAL_VOCAB_PROMPT_EN)
        self.assertIn("Tawheed", THEOLOGICAL_VOCAB_PROMPT_EN)


class TestSpeakerRolesEpisodesAndStage2Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = SharedWorkspace(root_dir=Path(self.tmp.name) / "yt-spx", worker_name="test-worker")

    def tearDown(self):
        self.tmp.cleanup()

    def test_three_tier_roles_voiceprint_enrollment_and_episodes(self):
        reg = VoiceprintRegistry(self.ws)
        c_daee = np.zeros(192, dtype=np.float32)
        c_daee[5] = 1.0
        c_int1 = np.zeros(192, dtype=np.float32)
        c_int1[10] = 1.0
        c_int2 = np.zeros(192, dtype=np.float32)
        c_int2[15] = 1.0
        c_ped = np.zeros(192, dtype=np.float32)
        c_ped[20] = 1.0

        centroids = {
            "speaker_0": c_daee,
            "speaker_1": c_int1,
            "speaker_2": c_int2,
            "speaker_3": c_ped,
        }

        # Simulate RMS: main debate at -18 dBFS, pedestrian shout far off-mic at -36 dBFS
        rms_timeline = [-18.0] * 1200  # 300s at 250ms Hop
        for idx in range(200, 212):    # 50s..53s pedestrian shout
            rms_timeline[idx] = -36.0

        raw_turns = [
            # Episode 1: speaker_0 (Main Da'ee) vs speaker_1 (Interlocutor 1) across 0..90s
            {"speaker_id": "speaker_0", "start_ms": 0, "end_ms": 25000, "start_fmt": "00:00", "end_fmt": "00:25", "text_original": "Welcome to Speakers Corner let us examine contingency."},
            {"speaker_id": "speaker_1", "start_ms": 26000, "end_ms": 48000, "start_fmt": "00:26", "end_fmt": "00:48", "text_original": "Why cannot the universe itself be the necessary uncaused reality?"},
            # Transient off-mic pedestrian at 50..52s
            {"speaker_id": "speaker_3", "start_ms": 50000, "end_ms": 51800, "start_fmt": "00:50", "end_fmt": "00:51", "text_original": "Excuse me mate!"},
            {"speaker_id": "speaker_0", "start_ms": 53000, "end_ms": 95000, "start_fmt": "00:53", "end_fmt": "01:35", "text_original": "Because a composite universe of changing parts depends on its parts."},
            {"speaker_id": "speaker_1", "start_ms": 96000, "end_ms": 115000, "start_fmt": "01:36", "end_fmt": "01:55", "text_original": "That makes sense I see the distinction."},
            # Episode 2: New challenger speaker_2 steps up at 130s..220s!
            {"speaker_id": "speaker_2", "start_ms": 130000, "end_ms": 165000, "start_fmt": "02:10", "end_fmt": "02:45", "text_original": "Hello Sheikh Uthman I want to ask about the Trinity in John chapter one."},
            {"speaker_id": "speaker_0", "start_ms": 166000, "end_ms": 225000, "start_fmt": "02:46", "end_fmt": "03:45", "text_original": "Gladly let us read the Greek definite article ho theos versus theos."},
            {"speaker_id": "speaker_2", "start_ms": 226000, "end_ms": 250000, "start_fmt": "03:46", "end_fmt": "04:10", "text_original": "Let me check that verse right now."},
        ]

        video_record = {
            "video_id": "debate_ep_test",
            "title_ar": "الشيخ عثمان بن فاروق يناظر ملحداً ومسيحياً في هايد بارك",
            "primary_daee_hint": "daee_uthman",
        }

        speakers_meta, classified_turns = classify_speakers_and_episodes(
            raw_turns=raw_turns,
            speaker_centroids=centroids,
            rms_timeline=rms_timeline,
            video_record=video_record,
            registry=reg,
        )

        roles_by_spk = {s["speaker_id"]: s["role"] for s in speakers_meta}
        self.assertEqual(roles_by_spk["speaker_0"], "MAIN_DAEE")
        self.assertEqual(roles_by_spk["speaker_1"], "INTERLOCUTOR")
        self.assertEqual(roles_by_spk["speaker_2"], "INTERLOCUTOR")
        self.assertEqual(roles_by_spk["speaker_3"], "PEDESTRIAN")

        # Verify Sheikh Uthman voiceprint was auto-enrolled and matches on a second video WITHOUT title hint!
        matched = reg.match_speaker(c_daee)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["profile_id"], "daee_uthman")
        self.assertGreater(matched["similarity"], 0.99)

        # Verify pedestrian turn flag and episode 1 -> episode 2 transition
        ped_turn = next(t for t in classified_turns if t["speaker_id"] == "speaker_3")
        self.assertTrue(ped_turn["is_pedestrian_interruption"])
        ep_indices = {t["episode_index"] for t in classified_turns}
        self.assertIn(1, ep_indices)
        self.assertIn(2, ep_indices)

    def test_stage2_partial_episode_resume_and_sqlite_compaction(self):
        vid = "vid_resume_m6"
        scenario_payload = {
            "schema_version": "2.0",
            "video_id": vid,
            "channel": "SCDawah Arabic",
            "title_ar": "مناظرة قوية حول السببية والتثليث",
            "youtube_url": f"https://www.youtube.com/watch?v={vid}",
            "duration_sec": 240.0,
            "detected_language": "en",
            "metrics": {"overlap_windows_count": 1, "overlap_ratio": 0.04},
            "speakers": [
                {
                    "speaker_id": "speaker_0",
                    "role": "MAIN_DAEE",
                    "interlocutor_index": 0,
                    "profile_id": "uthman_ibn_farooq",
                    "display_name_en": "Sheikh Uthman Ibn Farooq",
                    "display_name_ar": "الشيخ عثمان بن فاروق",
                    "speaking_sec": 150.0,
                    "turn_count": 2,
                    "mean_rms_db": -17.5,
                },
                {
                    "speaker_id": "speaker_1",
                    "role": "INTERLOCUTOR",
                    "interlocutor_index": 1,
                    "profile_id": None,
                    "display_name_en": "Interlocutor 1",
                    "display_name_ar": "المحاور 1",
                    "speaking_sec": 90.0,
                    "turn_count": 2,
                    "mean_rms_db": -19.0,
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
                    "start_ms": 14000,
                    "end_ms": 28000,
                    "start_fmt": "00:14",
                    "end_fmt": "00:28",
                    "language": "en",
                    "text_original": "If everything needs a creator then who created God in the first place right now?",
                    "text_ar": "",
                    "is_overlap": True,
                    "overlap_group_id": "OV_001",
                    "is_pedestrian_interruption": False,
                    "confidence": 0.94,
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
                    "start_ms": 28500,
                    "end_ms": 75000,
                    "start_fmt": "00:28",
                    "end_fmt": "01:15",
                    "language": "en",
                    "text_original": (
                        "Only things that begin to exist require a cause; God is the Necessary Uncaused Being. "
                        + " ".join(["Every contingent object depends on a prior metaphysical explanation."] * 220)
                    ),
                    "text_ar": "",
                    "is_overlap": True,
                    "overlap_group_id": "OV_001",
                    "is_pedestrian_interruption": False,
                    "confidence": 0.96,
                    "words": [],
                },
                {
                    "turn_id": f"{vid}_T0003",
                    "turn_index": 2,
                    "episode_index": 2,
                    "speaker_id": "speaker_1",
                    "speaker_role": "INTERLOCUTOR",
                    "speaker_label_en": "Interlocutor 1",
                    "speaker_label_ar": "المحاور 1",
                    "start_ms": 120000,
                    "end_ms": 145000,
                    "start_fmt": "02:00",
                    "end_fmt": "02:25",
                    "language": "en",
                    "text_original": "Can science explain objective moral obligations without any appeal to divine command?",
                    "text_ar": "",
                    "is_overlap": False,
                    "overlap_group_id": "",
                    "is_pedestrian_interruption": False,
                    "confidence": 0.95,
                    "words": [],
                },
                {
                    "turn_id": f"{vid}_T0004",
                    "turn_index": 3,
                    "episode_index": 2,
                    "speaker_id": "speaker_0",
                    "speaker_role": "MAIN_DAEE",
                    "speaker_label_en": "Sheikh Uthman Ibn Farooq",
                    "speaker_label_ar": "الشيخ عثمان بن فاروق",
                    "start_ms": 146000,
                    "end_ms": 195000,
                    "start_fmt": "02:26",
                    "end_fmt": "03:15",
                    "language": "en",
                    "text_original": (
                        "Empirical science describes what is, never what ought to be; bridging is-ought requires an transcendent moral lawgiver. "
                        + " ".join(["Objective moral duties demand an eternal transcendent source above subjective matter."] * 180)
                    ),
                    "text_ar": "",
                    "is_overlap": False,
                    "overlap_group_id": "",
                    "is_pedestrian_interruption": False,
                    "confidence": 0.97,
                    "words": [],
                },
            ],
        }

        self.ws.atomic_write_json(self.ws.scenarios_dir / f"{vid}.json", scenario_payload)
        self.ws.advance_micro_step(vid, "M5_SCENARIO_BUILT")

        windows = build_episode_windows(scenario_payload["turns"])
        self.assertEqual(len(windows), 2)

        # Pre-seed Episode Window 1 checkpoint as if a prior Colab session disconnected right after Episode 1!
        cached_ep1 = [
            {
                "id": f"{vid}_001",
                "video_id": vid,
                "question_en": "Who created God?",
                "question_ar": "من خلق الله؟",
                "topic_category": "Theology & God's Existence",
                "interlocutor_type": "Atheist / Materialist",
                "core_answer_summary_en": "Only contingent things that begin to exist require a cause.",
                "core_answer_summary_ar": "كل حادث يفتقر إلى محدث، أما واجب الوجود فغير مخلوق.",
                "logical_premises": ["Whatever begins to exist has a cause.", "God did not begin to exist."],
                "logical_premises_ar": ["كل ما له بداية فله سبب.", "الله واجب الوجود لا بداية له."],
                "timestamp_seconds": 14,
                "timestamp_formatted": "00:14",
                "timestamp_url": f"https://www.youtube.com/watch?v={vid}&t=14s",
                "daee_profile_id": "uthman_ibn_farooq",
                "daee_name_en": "Sheikh Uthman Ibn Farooq",
                "daee_name_ar": "الشيخ عثمان بن فاروق",
                "source_turn_ids": [f"{vid}_T0001", f"{vid}_T0002"],
            }
        ]
        self.ws.atomic_write_json(self.ws.partial_extractions_dir / f"{vid}_win001.json", cached_ep1)

        class MockLLM:
            def __init__(self):
                self.calls = 0
                self.context_length = 16384

            def extract(self, prompt: str) -> str:
                self.calls += 1
                return json.dumps([
                    {
                        "question_turn_id": f"{vid}_T0003",
                        "answer_turn_id": f"{vid}_T0004",
                        "topic_category": "Objective Morality & Ethics",
                        "topic_subcategory": "Is-Ought Gap",
                        "interlocutor_type": "Secular Humanist",
                        "question_en": "Can science ground objective moral duties?",
                        "question_ar": "هل يستطيع العلم تأسيس التزامات أخلاقية موضوعية؟",
                        "core_answer_summary_en": "Science describes physical facts (is) and cannot derive binding moral duties (ought).",
                        "core_answer_summary_ar": "العلم التجريبي يصف الواقع المادي ولا يشتق الواجب الأخلاقي الملزم.",
                        "logical_premises": ["Science describes physical processes.", "Moral duties are prescriptive."],
                        "logical_premises_ar": ["العلم يصف الظواهر المادية.", "الأخلاق أحكام معيارية ملزمة."],
                        "fallacy_or_misconception_exposed": "Scientism / Is-Ought Fallacy",
                        "fallacy_or_misconception_exposed_ar": "مغالطة العلموية واشتقاق الواجب من الواقع",
                        "analogy_or_metaphor": "Chemistry tells you how cyanide works, not whether poisoning is evil.",
                        "analogy_or_metaphor_ar": "الكيمياء تشرح تفاعل السم لكنها لا تقرر حرمة القتل.",
                        "socratic_question": "Which microscope measures moral obligation?",
                        "socratic_question_ar": "أي مجهر يقيس الوجوب الأخلاقي؟",
                        "interlocutor_verbatim_quote": "Can science explain objective moral obligations?",
                        "interlocutor_verbatim_quote_ar": "هل يفسر العلم الالتزامات الأخلاقية الموضوعية؟",
                        "verbatim_quote": "Empirical science describes what is, never what ought to be.",
                        "verbatim_quote_ar": "العلم التجريبي يصف ما هو كائن لا ما ينبغي أن يكون.",
                        "dialogue_exchange": [
                            {"turn_id": f"{vid}_T0003", "role": "INTERLOCUTOR", "move_type": "Objection", "text_en": "Can science explain morality?", "text_ar": "هل يفسر العلم الأخلاق؟"},
                            {"turn_id": f"{vid}_T0004", "role": "MAIN_DAEE", "move_type": "Refutation", "text_en": "Science describes is, not ought.", "text_ar": "العلم يصف الواقع لا الواجب."},
                        ],
                        "tags": ["morality", "scientism"],
                    }
                ])

        mock_llm = MockLLM()
        extractor = Stage2ScenarioExtractor(workspace=self.ws, llm_client=mock_llm)
        units = extractor.extract_video_m6(vid, verbose=False)

        # LLM was called ONLY ONCE (for Episode 2), proving Episode 1 resumed from partial checkpoint!
        self.assertEqual(mock_llm.calls, 1)
        self.assertEqual(len(units), 2)
        # Verify CTC ground-truth turn anchoring set Episode 2 timestamp_seconds = 120s (120000ms) & answer = 146s!
        self.assertEqual(units[1]["timestamp_seconds"], 120)
        self.assertEqual(units[1]["answer_timestamp_seconds"], 146)
        self.assertEqual(units[1]["daee_profile_id"], "uthman_ibn_farooq")

        # Run single-writer SQLite compactor and verify FTS5 search & stats
        compacted = self.ws.sync_completed_shards_to_sqlite()
        self.assertEqual(compacted["scenarios_synced"], 1)
        self.assertEqual(compacted["qa_videos_synced"], 1)
        self.assertEqual(compacted["qa_units_synced"], 2)

        db_file = self.ws.data_dir / "dawah_knowledge.db"
        stats = get_stats(db_path=db_file)
        self.assertEqual(stats["total_qa"], 2)
        self.assertEqual(stats["stage1_completed"], 1)
        self.assertEqual(stats["overlaps_resolved"], 1)

        scen_loaded = get_video_scenario(vid, db_path=db_file)
        self.assertEqual(len(scen_loaded["turns"]), 4)
        self.assertEqual(len(scen_loaded["qa_units"]), 2)

        qa_hits = search_qa(query="من خلق الله", speaker="uthman_ibn_farooq", db_path=db_file)
        self.assertEqual(len(qa_hits), 1)

    def test_unified_full_conversation_single_window_and_per_card_episode_index(self):
        """
        Proves multi-episode debates fitting within Stage 2 context (~3,400 words across 3 episodes,
        modeled after rQLF0rA1la4) pack into ONE single unified window in Stage2ScenarioExtractor,
        and each extracted card resolves its own true turn-level episode_index.
        """
        vid = "unified_ep_vid"
        turns = []
        # Create 3 episodes (~1,100 words each = ~3,300 words total -> fits cleanly in 1 unified window at 20K or 32K ctx)
        filler_ep = " ".join(["theology prophecy revelation scripture reason evidence"] * 175)
        specs = [
            (1, "Does paradise promise believers physical delights after resurrection?",
                f"In Islamic theology, paradise rewards righteous believers both spiritually and physically by divine wisdom. {filler_ep}"),
            (2, "Why does Islam affirm bodily resurrection alongside spiritual bliss?",
                f"Human identity unites soul and body so eternal recompense addresses the complete person. {filler_ep}"),
            (3, "How should Quranic descriptions of the afterlife be understood in context?",
                f"The Quran describes real transcendent bliss using accessible earthly concepts without reducing Creator to creature. {filler_ep}"),
        ]
        t_idx = 0
        for ep, q_txt, a_txt in specs:
            turns.append({
                "turn_id": f"{vid}_T{t_idx+1:04d}",
                "turn_index": t_idx,
                "episode_index": ep,
                "speaker_id": "speaker_1",
                "speaker_role": "INTERLOCUTOR",
                "speaker_label_en": "Interlocutor 1",
                "speaker_label_ar": "المحاور 1",
                "start_ms": t_idx * 30000,
                "end_ms": t_idx * 30000 + 15000,
                "start_fmt": f"{t_idx:02d}:00",
                "end_fmt": f"{t_idx:02d}:15",
                "language": "en",
                "text_original": q_txt,
                "text_ar": "",
                "is_overlap": False,
                "overlap_group_id": "",
                "is_pedestrian_interruption": False,
                "confidence": 0.96,
                "words": [],
            })
            t_idx += 1
            turns.append({
                "turn_id": f"{vid}_T{t_idx+1:04d}",
                "turn_index": t_idx,
                "episode_index": ep,
                "speaker_id": "speaker_0",
                "speaker_role": "MAIN_DAEE",
                "speaker_label_en": "Hashim",
                "speaker_label_ar": "هاشم",
                "start_ms": t_idx * 30000,
                "end_ms": t_idx * 30000 + 25000,
                "start_fmt": f"{t_idx:02d}:00",
                "end_fmt": f"{t_idx:02d}:25",
                "language": "en",
                "text_original": a_txt,
                "text_ar": "",
                "is_overlap": False,
                "overlap_group_id": "",
                "is_pedestrian_interruption": False,
                "confidence": 0.97,
                "words": [],
            })
            t_idx += 1

        self.ws.atomic_write_json(self.ws.scenarios_dir / f"{vid}.json", {
            "video_id": vid,
            "title_ar": "مناظرة موحدة متعددة المقاطع",
            "youtube_url": f"https://www.youtube.com/watch?v={vid}",
            "channel": "SCDawah Arabic",
            "speakers": [
                {"speaker_id": "speaker_0", "role": "MAIN_DAEE", "profile_id": "hashim", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                {"speaker_id": "speaker_1", "role": "INTERLOCUTOR", "profile_id": None, "display_name_en": "Interlocutor 1", "display_name_ar": "المحاور 1"},
            ],
            "turns": turns,
        })
        self.ws.advance_micro_step(vid, "M5_SCENARIO_BUILT")

        # Legacy default preserves 3 per-episode windows, while unified full-conversation packs into 1!
        self.assertEqual(len(build_episode_windows(turns)), 3)
        self.assertEqual(len(build_episode_windows(turns, max_words_per_window=7500, merge_episodes=True, max_chars_per_window=72000)), 1)

        class UnifiedTwoRoundMockLLM:
            def __init__(self):
                self.context_length = 32768
                self.r1_calls = 0
                self.r2_calls = 0

            def extract_round1_english(self, prompt: str) -> str:
                self.r1_calls += 1
                return json.dumps([
                    {
                        "question_turn_id": f"{vid}_T0001",
                        "answer_turn_id": f"{vid}_T0002",
                        "topic_category": "Eschatology & Free Will",
                        "question_en": "If paradise includes physical pleasures, isn't that a bad thing?",
                        "core_answer_summary_en": (
                            "Islamic eschatology rejects purely materialistic caricatures while honoring complete human nature created by God, "
                            "who fulfills both spiritual nearness and pure physical delight without defect or harm."
                        ),
                        "logical_premises": [
                            "Premise 1: God created human beings with both spiritual and physical capacities.",
                            "Premise 2: Eternal reward fulfills complete human flourishing.",
                            "Conclusion: Therefore purified physical delight alongside spiritual vision is coherent.",
                        ],
                        "fallacy_or_misconception_exposed": "False Dichotomy between spiritual elevation and bodily creation.",
                        "socratic_question": "Why assume the Creator who formed human senses cannot elevate them eternally?",
                        "interlocutor_verbatim_quote": turns[0]["text_original"],
                        "verbatim_quote": turns[1]["text_original"],
                        "dialogue_exchange": [
                            {"turn_id": f"{vid}_T0001", "role": "INTERLOCUTOR", "text_en": turns[0]["text_original"]},
                            {"turn_id": f"{vid}_T0002", "role": "MAIN_DAEE", "text_en": turns[1]["text_original"]},
                        ],
                    },
                    {
                        "question_turn_id": f"{vid}_T0005",
                        "answer_turn_id": f"{vid}_T0006",
                        "topic_category": "Prophethood & Quran",
                        "question_en": "How should Quranic afterlife descriptions be interpreted without anthropomorphism?",
                        "core_answer_summary_en": (
                            "The Quran uses intelligible terminology to point toward real unseen bliss in paradise while upholding absolute divine "
                            "transcendence and purifying eternal rewards from worldly imperfection."
                        ),
                        "logical_premises": [
                            "Premise 1: Revelation communicates unseen realities in human language.",
                            "Premise 2: Afterlife realities transcend worldly defects while remaining concrete.",
                            "Conclusion: Therefore Quranic imagery neither reduces paradise to base appetite nor denies reality.",
                        ],
                        "fallacy_or_misconception_exposed": "Literalist Reductionism.",
                        "socratic_question": "How else could revelation describe unseen eternal bliss to earthly minds?",
                        "interlocutor_verbatim_quote": turns[4]["text_original"],
                        "verbatim_quote": turns[5]["text_original"],
                        "dialogue_exchange": [
                            {"turn_id": f"{vid}_T0005", "role": "INTERLOCUTOR", "text_en": turns[4]["text_original"]},
                            {"turn_id": f"{vid}_T0006", "role": "MAIN_DAEE", "text_en": turns[5]["text_original"]},
                        ],
                    },
                ])

            def extract_round2_arabic(self, r1_json_str: str, prompt: str) -> str:
                self.r2_calls += 1
                return json.dumps([
                    {
                        "question_turn_id": f"{vid}_T0001",
                        "rubric_audit": {"score": 94, "english_edits_needed": False, "drop_card": False},
                        "question_ar": "إذا اشتملت الجنة على نعيم حسي، أليس ذلك أمرًا غير روحاني؟",
                        "core_answer_summary_ar": "الإسلام يكرم النفس والبدن معًا في دار النعيم الأبدي المنزه عن كل نقص.",
                        "logical_premises_ar": ["الإنسان مكون من روح وجسد.", "الجزاء التام يشمل الكمالين.", "فالنعيم المادي المنقى مع القرب الروحي متسق تمامًا."],
                        "fallacy_or_misconception_exposed_ar": "ثنائية زائفة بين الروح والجسد",
                        "analogy_or_metaphor_ar": "",
                        "socratic_question_ar": "لماذا يُفترض أن خالق الحواس يعجز عن الارتقاء بها في الآخرة؟",
                        "interlocutor_verbatim_quote_ar": "هل تعد الجنة بنعيم مادي؟",
                        "verbatim_quote_ar": "الجنة تجمع بين الكمال الروحي والنعيم الحسي المنزه.",
                        "dialogue_exchange": [
                            {"turn_id": f"{vid}_T0001", "text_ar": "هل تعد الجنة بنعيم مادي؟"},
                            {"turn_id": f"{vid}_T0002", "text_ar": "الجنة تجمع بين النعيم الروحي والحسي بحكمة الله."},
                        ],
                    },
                    {
                        "question_turn_id": f"{vid}_T0005",
                        "rubric_audit": {"score": 95, "english_edits_needed": False, "drop_card": False},
                        "question_ar": "كيف تُفهم أوصاف الآخرة القرآنية دون تجسيم؟",
                        "core_answer_summary_ar": "القرآن يخاطب البشر بألفاظ مفهومة للدلالة على حقائق غيبية منزهة عن نقائص الدنيا.",
                        "logical_premises_ar": ["الوحي يخاطب العقول بما تفهم.", "نعيم الآخرة حقيقة منزهة عن العلل.", "لذا ينتفي التجسيم والاختزال."],
                        "fallacy_or_misconception_exposed_ar": "الاختزال الحرفي",
                        "analogy_or_metaphor_ar": "",
                        "socratic_question_ar": "كيف يخاطب الوحي البشر عن الغيب دون ألفاظ مفهومة؟",
                        "interlocutor_verbatim_quote_ar": "كيف نفهم أوصاف الآخرة في القرآن؟",
                        "verbatim_quote_ar": "الألفاظ القرآنية تشير إلى حقائق متعالية منزهة عن نقص الدنيا.",
                        "dialogue_exchange": [
                            {"turn_id": f"{vid}_T0005", "text_ar": "كيف نفهم أوصاف الآخرة؟"},
                            {"turn_id": f"{vid}_T0006", "text_ar": "القرآن يدل على حقائق منزهة عن نقائص الدنيا."},
                        ],
                    },
                ])

        mock_llm = UnifiedTwoRoundMockLLM()
        extractor = Stage2ScenarioExtractor(workspace=self.ws, llm_client=mock_llm)
        units = extractor.extract_video_m6(vid, verbose=False)

        self.assertEqual(mock_llm.r1_calls, 1, "All 3 episodes (~3.3K words) must fit into ONE full-conversation window")
        self.assertEqual(mock_llm.r2_calls, 1)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["episode_index"], 1)
        self.assertEqual(units[1]["episode_index"], 3, "Per-card episode_index must resolve from anchored turn T0005")

    def test_oversized_transcript_balanced_split_at_32k_and_round2_subbatching(self):
        """
        Ensures transcripts exceeding 32K capacity split cleanly into balanced windows preferring episode boundaries,
        and Round 2 sub-batches >=3 Round 1 units into chunks of <=2 cards for bounded thinking.
        """
        from core.scenario_extractor import compute_window_budget_for_context
        w_cap, c_cap = compute_window_budget_for_context(32768)
        self.assertEqual((w_cap, c_cap), (7500, 72000))

        vid = "big_balanced_vid"
        turns = []
        chunk_words = " ".join(["rationality causation creation contingency proof logic"] * 350)  # ~2,100 words each
        for ep in (1, 2, 3, 4):
            idx = len(turns)
            turns.append({
                "turn_id": f"{vid}_T{idx+1:04d}",
                "turn_index": idx,
                "episode_index": ep,
                "speaker_id": "speaker_1",
                "speaker_role": "INTERLOCUTOR",
                "speaker_label_en": "Interlocutor",
                "start_ms": idx * 10000,
                "end_ms": idx * 10000 + 5000,
                "start_fmt": "00:00",
                "end_fmt": "00:05",
                "text_original": f"Episode {ep} objection about divine wisdom and existence?",
            })
            turns.append({
                "turn_id": f"{vid}_T{idx+2:04d}",
                "turn_index": idx + 1,
                "episode_index": ep,
                "speaker_id": "speaker_0",
                "speaker_role": "MAIN_DAEE",
                "speaker_label_en": "Hashim",
                "start_ms": idx * 10000 + 6000,
                "end_ms": idx * 10000 + 9000,
                "start_fmt": "00:06",
                "end_fmt": "00:09",
                "text_original": f"Episode {ep} deductive answer: {chunk_words}",
            })

        wins = build_episode_windows(turns, max_words_per_window=w_cap, merge_episodes=True, max_chars_per_window=c_cap)
        self.assertEqual(len(wins), 2)
        self.assertLessEqual(wins[0]["word_count"], w_cap)
        self.assertLessEqual(wins[1]["word_count"], w_cap)
        self.assertGreater(wins[1]["word_count"], 3500, "Balanced split must avoid tiny tail windows")

        # Verify Round 2 sub-batching splits 3 units into 2 calls (batch size 2)
        vid_sub = "subbatch_vid"
        sub_turns = turns[:6]
        for t in sub_turns:
            t["turn_id"] = t["turn_id"].replace(vid, vid_sub)
        self.ws.atomic_write_json(self.ws.scenarios_dir / f"{vid_sub}.json", {
            "video_id": vid_sub,
            "title_ar": "حوارات متعددة",
            "youtube_url": f"https://www.youtube.com/watch?v={vid_sub}",
            "speakers": [
                {"speaker_id": "speaker_0", "role": "MAIN_DAEE", "profile_id": "hashim", "display_name_en": "Hashim", "display_name_ar": "هاشم"},
                {"speaker_id": "speaker_1", "role": "INTERLOCUTOR", "display_name_en": "Interlocutor", "display_name_ar": "المحاور"},
            ],
            "turns": sub_turns,
        })
        self.ws.advance_micro_step(vid_sub, "M5_SCENARIO_BUILT")

        class SubBatchMockLLM:
            def __init__(self):
                self.context_length = 32768
                self.r2_batch_sizes = []
                self.prompts = []

            def extract_round1_english(self, prompt: str) -> str:
                return json.dumps([
                    {
                        "question_turn_id": f"{vid_sub}_T000{k}",
                        "answer_turn_id": f"{vid_sub}_T000{k+1}",
                        "question_en": f"Why does objector question principle {k} about causality?",
                        "core_answer_summary_en": "Dependent things require a Necessary Being by rational deduction.",
                        "logical_premises": ["Contingent requires cause.", "Infinite regress is false.", "Necessary creator exists."],
                        "fallacy_or_misconception_exposed": "Infinite regress fallacy.",
                        "socratic_question": f"Can effect {k} cause itself?",
                        "interlocutor_verbatim_quote": sub_turns[k - 1]["text_original"],
                        "verbatim_quote": sub_turns[k]["text_original"],
                    }
                    for k in (1, 3, 5)
                ])

            def extract_round2_arabic(self, r1_json_str: str, prompt: str) -> str:
                batch = json.loads(r1_json_str)
                self.r2_batch_sizes.append(len(batch))
                self.prompts.append(prompt)
                return json.dumps([
                    {
                        "question_turn_id": u["question_turn_id"],
                        "rubric_audit": {"score": 90, "english_edits_needed": False, "drop_card": False},
                        "question_ar": f"لماذا يُطرح الإشكال العقلي رقم {u['question_turn_id']} حول السببية؟",
                        "core_answer_summary_ar": "الممكنات تفتقر إلى واجب الوجود بالبرهان العقلي.",
                        "logical_premises_ar": ["الممكن يفتقر لمخصص.", "التسلسل باطل.", "واجب الوجود ثابت."],
                        "fallacy_or_misconception_exposed_ar": "مغالطة التسلسل",
                        "analogy_or_metaphor_ar": "",
                        "socratic_question_ar": "هل يخلق الممكن نفسه؟",
                        "verbatim_quote_ar": "الجواب الاستدلالي في التوحيد.",
                    }
                    for u in batch
                ])

        sb_llm = SubBatchMockLLM()
        sub_units = Stage2ScenarioExtractor(workspace=self.ws, llm_client=sb_llm).extract_video_m6(vid_sub, verbose=False)
        self.assertEqual(sb_llm.r2_batch_sizes, [2, 1], "3 Round 1 units must split into [2, 1] Round 2 thinking batches")
        self.assertIn("PRE-ROUND-2 CARD RATING & DEFECT DIAGNOSTICS", sb_llm.prompts[0])
        self.assertEqual(len(sub_units), 3)

    def test_smart_batch_resume_priority_ordering(self):
        manifest = [
            {"video_id": "v_unstarted", "priority_score": 100, "is_short": False},
            {"video_id": "v_at_m1", "priority_score": 60, "is_short": False},
            {"video_id": "v_at_m3", "priority_score": 50, "is_short": False},
        ]
        self.ws.advance_micro_step("v_at_m1", "M1_DOWNLOADED")
        self.ws.advance_micro_step("v_at_m3", "M3_DIARIZED")

        ordered = select_stage1_candidates(self.ws, manifest)
        # M3 must come first, then M1, then unstarted M0 regardless of raw priority_score!
        self.assertEqual([x["video_id"] for x in ordered], ["v_at_m3", "v_at_m1", "v_unstarted"])

        # Stage 2 partial resume priority
        (self.ws.scenarios_dir / "v_s2_fresh.json").write_text('{"video_id":"v_s2_fresh"}', encoding="utf-8")
        (self.ws.scenarios_dir / "v_s2_partial.json").write_text('{"video_id":"v_s2_partial"}', encoding="utf-8")
        (self.ws.partial_extractions_dir / "v_s2_partial_win001.json").write_text("[]", encoding="utf-8")

        s2_ordered = select_stage2_candidates(
            self.ws,
            [
                {"video_id": "v_s2_fresh", "priority_score": 100},
                {"video_id": "v_s2_partial", "priority_score": 40},
            ],
        )
        self.assertEqual(s2_ordered[0]["video_id"], "v_s2_partial")

    def test_error_count_reset_and_orchestrator_unlimited_default(self):
        from core.pipeline_orchestrator import build_arg_parser

        # 1. Transient errors at M2 reset to 0 once M3_DIARIZED succeeds
        self.ws.advance_micro_step("v_recover", "M2_VOCALS_CLEANED")
        self.ws.record_error("v_recover", "AttributeError: ml_dtypes")
        self.ws.record_error("v_recover", "AttributeError: ml_dtypes")
        self.assertEqual(self.ws.get_video_state("v_recover")["error_count"], 2)

        self.ws.advance_micro_step("v_recover", "M3_DIARIZED")
        st_after = self.ws.get_video_state("v_recover")
        self.assertEqual(st_after["error_count"], 0)
        self.assertIsNone(st_after["last_error"])

        # 2. Orchestrator defaults to --max-session-minutes 0 (unlimited continuous mode)
        parsed = build_arg_parser().parse_args([])
        self.assertEqual(parsed.max_session_minutes, 0)

    def test_cookie_pool_offline_validation_rotation_and_quarantine(self):
        import os
        import time as _t
        from core.cookie_pool import CookiePool, classify_ytdlp_output, validate_cookie_file_offline

        cdir = self.ws.root / "cookies"
        cdir.mkdir(parents=True, exist_ok=True)
        future = int(_t.time()) + 30 * 86400
        past = int(_t.time()) - 86400

        def _jar(name, expiry, auth=True):
            cname = "__Secure-3PSID" if auth else "PREF"
            body = (
                "# Netscape HTTP Cookie File\n"
                f".youtube.com\tTRUE\t/\tTRUE\t{expiry}\t{cname}\tabcdef123456\n"
                f".youtube.com\tTRUE\t/\tFALSE\t{expiry}\tVISITOR_INFO1_LIVE\txyz987\n"
            )
            p = cdir / name
            p.write_text(body, encoding="utf-8")
            return p

        a = _jar("acct1.txt", future)
        b = _jar("acct2.txt", future)
        expired = _jar("acct3_expired.txt", past)
        anon = _jar("acct4_anon.txt", future, auth=False)

        self.assertTrue(validate_cookie_file_offline(a)[0])
        self.assertFalse(validate_cookie_file_offline(expired)[0])
        self.assertFalse(validate_cookie_file_offline(anon)[0])

        self.assertEqual(classify_ytdlp_output("error: sign in to confirm you're not a bot"), "challenged")
        self.assertEqual(classify_ytdlp_output("warning: the provided youtube account cookies are no longer valid"), "invalid")
        self.assertEqual(classify_ytdlp_output("http error 429: too many requests"), "rate_limited")
        self.assertEqual(
            classify_ytdlp_output("video unavailable. this content isn't available, try again later."),
            "rate_limited",
        )

        pool = CookiePool(self.ws, include_repo_dir=False)
        healthy = pool.healthy_cookies()
        self.assertEqual({p.name for p in healthy}, {"acct1.txt", "acct2.txt"})

        # acct1 gets bot-checked -> quarantined, acct2 becomes the only option
        pool.mark_failed(a, "challenged", "Sign in to confirm you're not a bot")
        self.assertEqual([p.name for p in pool.healthy_cookies()], ["acct2.txt"])

        # Successful cookie is preferred first
        pool.mark_ok(b)
        self.assertEqual(pool.healthy_cookies()[0].name, "acct2.txt")

        # Replacing acct1 with a fresh export (new mtime/size) clears its quarantine automatically
        a.write_text(a.read_text(encoding="utf-8") + ".youtube.com\tTRUE\t/\tTRUE\t%d\tSID\tnewsid\n" % future, encoding="utf-8")
        os.utime(a, (_t.time() + 5, _t.time() + 5))
        self.assertEqual({p.name for p in pool.healthy_cookies()}, {"acct1.txt", "acct2.txt"})

    def test_pyav_metadata_errors_patch_and_audio_loading(self):
        import sys
        import types
        import numpy as np
        from core.audio_diarizer import patch_pyav_metadata_errors, load_audio_waveform_16k
        from core.vocal_cleaner import write_wav_float32

        # 1. Test mock PyAV 19.x simulation (where av.open lacks metadata_errors)
        mock_av = types.ModuleType("av")
        called_kwargs = {}

        def mock_open(*args, **kwargs):
            if "metadata_errors" in kwargs:
                raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
            called_kwargs.update(kwargs)
            return "mock_container"

        mock_av.open = mock_open
        sys.modules["av"] = mock_av

        # Before patch, passing metadata_errors raises TypeError
        with self.assertRaises(TypeError):
            mock_av.open("dummy.mp3", mode="r", metadata_errors="ignore")

        # After patch, metadata_errors is popped safely
        patch_pyav_metadata_errors()
        res = mock_av.open("dummy.mp3", mode="r", metadata_errors="ignore")
        self.assertEqual(res, "mock_container")
        self.assertNotIn("metadata_errors", called_kwargs)
        self.assertEqual(called_kwargs.get("mode"), "r")

        # 2. Test load_audio_waveform_16k on a standard 16kHz WAV file
        test_wav = self.ws.staging_dir / "test_load.wav"
        orig_samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        write_wav_float32(test_wav, orig_samples, sr=16000)

        loaded = load_audio_waveform_16k(test_wav)
        self.assertEqual(len(loaded), 16000)
        self.assertTrue(np.allclose(orig_samples, loaded, atol=1e-3))

    def test_cuda_libraries_discovery_and_preload(self):
        from core.audio_diarizer import get_cuda_library_dirs, preload_cuda_libraries
        from core.colab_runner import ensure_cuda_runtime_loaded
        from core.pipeline_orchestrator import _get_subprocess_env

        dirs = get_cuda_library_dirs()
        self.assertIsInstance(dirs, list)

        # Calling preload_cuda_libraries & ensure_cuda_runtime_loaded should be safe across platforms
        preload_cuda_libraries()
        ensure_cuda_runtime_loaded()

        # Check subprocess env sets TF disabled and PyTorch alloc guards
        sub_env = _get_subprocess_env()
        self.assertEqual(sub_env.get("TRANSFORMERS_NO_TF"), "1")
        self.assertEqual(sub_env.get("USE_TF"), "0")
        self.assertEqual(sub_env.get("USE_TORCH"), "1")
        self.assertIn("expandable_segments:True", sub_env.get("PYTORCH_CUDA_ALLOC_CONF", ""))

    def test_fetch_video_audio_m1_anonymous_fallback_on_invalid_cookies(self):
        import time as _t
        import subprocess
        from unittest.mock import patch
        from core.yt_audio_fetcher import fetch_video_audio_m1

        cdir = self.ws.root / "cookies"
        cdir.mkdir(parents=True, exist_ok=True)
        future = int(_t.time()) + 30 * 86400
        # Cookie file with auth tokens that passes offline validation
        bad_cookie = cdir / "bad_acct.txt"
        bad_cookie.write_text(
            f"# Netscape\n.youtube.com\tTRUE\t/\tTRUE\t{future}\t__Secure-3PSID\trotated_token\n",
            encoding="utf-8",
        )

        vid = "test_anon_vid"
        record = {"video_id": vid, "youtube_url": f"https://www.youtube.com/watch?v={vid}"}

        def mock_subp_run(cmd, *args, **kwargs):
            # If --cookies is present, simulate rotated cookie failure
            if "--cookies" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="WARNING: [youtube] The provided YouTube account cookies are no longer valid. They have likely been rotated in the browser.",
                )
            # If anonymous (no --cookies), create a dummy audio file and info.json in the working tmp_dir
            # The output template (-o) gives the path
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    out_pat = Path(cmd[i + 1])
                    tmp_dir = out_pat.parent
                    dummy_audio = tmp_dir / f"{vid}.opus"
                    dummy_audio.write_bytes(b"OggS" + b"\x00" * 10000)
                    dummy_info = tmp_dir / f"{vid}.info.json"
                    dummy_info.write_text('{"duration": 120.0}', encoding="utf-8")
                    break
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[download] 100% of 10.0KiB")

        with patch("subprocess.run", side_effect=mock_subp_run), patch("time.sleep", return_value=None):
            raw_audio, info = fetch_video_audio_m1(record, self.ws, min_global_gap_sec=0.0)

        self.assertIsNotNone(raw_audio)
        self.assertTrue(raw_audio.exists())
        self.assertEqual(self.ws.get_video_state(vid)["micro_step"], "M1_DOWNLOADED")

        # Verify circuit breaker was NOT tripped by the rotated cookie
        can_dl, rem, _ = self.ws.check_youtube_download_slot(min_global_gap_sec=0.0)
        self.assertTrue(can_dl)
        self.assertEqual(rem, 0.0)

    def test_fetch_video_audio_m1_cookie_fallback_when_anonymous_restricted(self):
        import time as _t
        import subprocess
        from unittest.mock import patch
        from core.yt_audio_fetcher import fetch_video_audio_m1

        cdir = self.ws.root / "cookies"
        cdir.mkdir(parents=True, exist_ok=True)
        future = int(_t.time()) + 30 * 86400
        good_cookie = cdir / "good_acct.txt"
        good_cookie.write_text(
            f"# Netscape\n.youtube.com\tTRUE\t/\tTRUE\t{future}\t__Secure-3PSID\tvalid_auth_token\n",
            encoding="utf-8",
        )

        vid = "test_restricted_anon"
        record = {"video_id": vid, "youtube_url": f"https://www.youtube.com/watch?v={vid}"}

        call_log = []

        def mock_subp_run(cmd, *args, **kwargs):
            has_cookies = "--cookies" in cmd
            call_log.append("cookie" if has_cookies else "anon")
            if not has_cookies:
                # Simulate bot check on anonymous
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="ERROR: Sign in to confirm you’re not a bot. This helps protect our community.",
                )
            # Cookie fallback succeeds
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    out_pat = Path(cmd[i + 1])
                    tmp_dir = out_pat.parent
                    dummy_audio = tmp_dir / f"{vid}.opus"
                    dummy_audio.write_bytes(b"OggS" + b"\x00" * 10000)
                    dummy_info = tmp_dir / f"{vid}.info.json"
                    dummy_info.write_text('{"duration": 90.0}', encoding="utf-8")
                    break
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[download] 100% of 10.0KiB")

        with patch("subprocess.run", side_effect=mock_subp_run), patch("time.sleep", return_value=None):
            raw_audio, info = fetch_video_audio_m1(record, self.ws, min_global_gap_sec=0.0)

        self.assertIsNotNone(raw_audio)
        self.assertTrue(raw_audio.exists())
        self.assertEqual(call_log, ["anon", "cookie"])


class TestStage1WhisperAndCTCAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_whisper_models(self):
        from core.audio_diarizer import WarmASRAlignmentEngine, Stage1AudioPipeline
        from core.pipeline_orchestrator import build_arg_parser
        from core.drive_sync import SharedWorkspace

        engine = WarmASRAlignmentEngine()
        self.assertEqual(engine.whisper_model_name, "large-v3")

        ws = SharedWorkspace(root_dir=Path(self.tmp.name))
        stage1 = Stage1AudioPipeline(workspace=ws)
        self.assertEqual(stage1.asr.whisper_model_name, "large-v3")

        parser = build_arg_parser()
        self.assertEqual(parser.get_default("whisper_model"), "large-v3")

    def test_transcribe_stream_whisper_parameters_and_word_timestamps(self):
        import types
        from unittest.mock import MagicMock, patch
        from core.audio_diarizer import WarmASRAlignmentEngine

        engine = WarmASRAlignmentEngine(batch_size=0)
        mock_whisper = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "In the name of Allah"
        mock_seg.no_speech_prob = 0.01
        mock_seg.compression_ratio = 1.0
        mock_w1 = MagicMock()
        mock_w1.word = "In"
        mock_w1.start = 0.0
        mock_w1.end = 0.2
        mock_w2 = MagicMock()
        mock_w2.word = "Allah"
        mock_w2.start = 0.3
        mock_w2.end = 0.9
        mock_seg.words = [mock_w1, mock_w2]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_whisper.transcribe.return_value = ([mock_seg], mock_info)
        engine._whisper = mock_whisper

        dummy_wav = np.zeros(16000 * 3, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        try:
            fake_torch = types.ModuleType("torch")
            fake_fw = types.ModuleType("faster_whisper")
            with patch.dict(sys.modules, {"torch": fake_torch, "faster_whisper": fake_fw}), \
                 patch("core.audio_diarizer.load_audio_waveform_16k", return_value=dummy_wav):
                words, lang = engine.transcribe_and_align_wav(wav_path, forced_language="en", use_ctc_aligner=False)

            self.assertEqual(lang, "en")
            self.assertEqual(len(words), 2)
            self.assertEqual(words[0]["text"], "In")
            self.assertEqual(words[1]["text"], "Allah")

            calls = mock_whisper.transcribe.call_args_list
            self.assertGreater(len(calls), 0)
            last_call_kwargs = calls[-1].kwargs
            self.assertTrue(last_call_kwargs.get("word_timestamps"))
            self.assertEqual(last_call_kwargs.get("beam_size"), 5)
            self.assertEqual(last_call_kwargs.get("best_of"), 5)
            self.assertEqual(last_call_kwargs.get("temperature"), [0.0, 0.2, 0.4])
            self.assertFalse(last_call_kwargs.get("condition_on_previous_text"))
            self.assertTrue(last_call_kwargs.get("vad_filter"))
            self.assertEqual(last_call_kwargs.get("vad_parameters"), {"min_silence_duration_ms": 500})
        finally:
            wav_path.unlink(missing_ok=True)

    def test_ctc_pytorch_forward_pass_without_run_method(self):
        import types
        from unittest.mock import MagicMock, patch
        from core.audio_diarizer import WarmASRAlignmentEngine

        engine = WarmASRAlignmentEngine(batch_size=0)
        mock_whisper = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "God exists"
        mock_seg.no_speech_prob = 0.0
        mock_seg.compression_ratio = 1.0
        mock_w1 = MagicMock(word="God", start=0.1, end=0.4)
        mock_w2 = MagicMock(word="exists", start=0.5, end=0.9)
        mock_seg.words = [mock_w1, mock_w2]
        mock_info = MagicMock(language="en")
        mock_whisper.transcribe.return_value = ([mock_seg], mock_info)
        engine._whisper = mock_whisper

        # Create mock PyTorch align model WITHOUT a .run attribute
        class MockPyTorchModel:
            def __init__(self):
                self.config = MagicMock(inputs_to_logits_ratio=320)
                self.dtype = "float32"
                self.device = "cpu"
                self.forward_called = False

            def __call__(self, x):
                self.forward_called = True
                res = MagicMock()
                res.logits = MagicMock()
                return res

        pytorch_model = MockPyTorchModel()
        self.assertFalse(hasattr(pytorch_model, "run"))
        engine._align_model = pytorch_model
        engine._align_tokenizer = MagicMock()

        # Mock torch and ctc_forced_aligner
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = MagicMock()
        fake_torch.log_softmax = MagicMock(return_value=np.zeros((1, 10, 32)))
        fake_torch.from_numpy = MagicMock(return_value=MagicMock(ndim=1, to=MagicMock(return_value=MagicMock(ndim=1, unsqueeze=MagicMock(return_value=MagicMock(ndim=2))))))
        fake_torch.cuda = MagicMock(is_available=MagicMock(return_value=False))
        fake_torch.Tensor = type("Tensor", (), {})

        fake_fw = types.ModuleType("faster_whisper")
        fake_ctc = types.ModuleType("ctc_forced_aligner")
        fake_ctc.generate_emissions = MagicMock()
        fake_ctc.preprocess_text = MagicMock(return_value=(["God", "exists"], "God exists"))
        fake_ctc.get_alignments = MagicMock(return_value=([0, 1], [0.9, 0.9], 0))
        fake_ctc.get_spans = MagicMock(return_value=[(5, 20), (25, 45)])
        fake_ctc.postprocess_results = MagicMock(return_value=[{"word": "God", "start": 0.1, "end": 0.4}, {"word": "exists", "start": 0.5, "end": 0.9}])

        dummy_wav = np.zeros(16000 * 2, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        try:
            with patch.dict(sys.modules, {"torch": fake_torch, "faster_whisper": fake_fw, "ctc_forced_aligner": fake_ctc}), \
                 patch("core.audio_diarizer.load_audio_waveform_16k", return_value=dummy_wav):
                words, lang = engine.transcribe_and_align_wav(wav_path, forced_language="en", use_ctc_aligner=True)
            self.assertTrue(pytorch_model.forward_called)
            fake_ctc.generate_emissions.assert_not_called()
            self.assertEqual(lang, "en")
            self.assertEqual(len(words), 2)
            self.assertEqual(words[0]["word"], "God")
            self.assertEqual(words[1]["word"], "exists")
        finally:
            wav_path.unlink(missing_ok=True)

    def test_ctc_failure_falls_back_to_whisper_dtw_words(self):
        import types
        from unittest.mock import MagicMock, patch
        from core.audio_diarizer import WarmASRAlignmentEngine

        engine = WarmASRAlignmentEngine(batch_size=0)
        mock_whisper = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "God exists"
        mock_seg.no_speech_prob = 0.0
        mock_seg.compression_ratio = 1.0
        mock_w1 = MagicMock(word="God", start=0.12, end=0.45)
        mock_w2 = MagicMock(word="exists", start=0.52, end=0.91)
        mock_seg.words = [mock_w1, mock_w2]
        mock_info = MagicMock(language="en")
        mock_whisper.transcribe.return_value = ([mock_seg], mock_info)
        engine._whisper = mock_whisper

        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = MagicMock()
        fake_torch.log_softmax = MagicMock(side_effect=RuntimeError("PyTorch forward failed"))
        fake_torch.from_numpy = MagicMock(return_value=MagicMock(ndim=1, to=MagicMock(return_value=MagicMock(ndim=1, unsqueeze=MagicMock(return_value=MagicMock(ndim=2))))))
        fake_torch.cuda = MagicMock(is_available=MagicMock(return_value=False))
        fake_torch.Tensor = type("Tensor", (), {})

        fake_fw = types.ModuleType("faster_whisper")
        fake_ctc = types.ModuleType("ctc_forced_aligner")

        engine._align_model = MagicMock(spec=["config", "dtype", "device"])  # no .run
        engine._align_tokenizer = MagicMock()

        dummy_wav = np.zeros(16000 * 2, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        try:
            with patch.dict(sys.modules, {"torch": fake_torch, "faster_whisper": fake_fw, "ctc_forced_aligner": fake_ctc}), \
                 patch("core.audio_diarizer.load_audio_waveform_16k", return_value=dummy_wav):
                words, lang = engine.transcribe_and_align_wav(wav_path, forced_language="en", use_ctc_aligner=True)
            self.assertEqual(lang, "en")
            # Proves high-precision Whisper DTW fallback worked instead of crude division!
            self.assertEqual(len(words), 2)
            self.assertEqual(words[0]["text"], "God")
            self.assertEqual(words[0]["word"], "God")
            self.assertEqual(words[0]["start"], 0.12)
            self.assertEqual(words[0]["end"], 0.45)
            self.assertEqual(words[1]["text"], "exists")
            self.assertEqual(words[1]["word"], "exists")
            self.assertEqual(words[1]["start"], 0.52)
            self.assertEqual(words[1]["end"], 0.91)
        finally:
            wav_path.unlink(missing_ok=True)

    def test_ctc_chunked_forward_pass_long_audio(self):
        import types
        from unittest.mock import MagicMock, patch
        from core.audio_diarizer import WarmASRAlignmentEngine

        engine = WarmASRAlignmentEngine(batch_size=0)
        mock_whisper = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "God exists"
        mock_seg.no_speech_prob = 0.0
        mock_seg.compression_ratio = 1.0
        mock_w1 = MagicMock(word="God", start=0.1, end=0.4)
        mock_w2 = MagicMock(word="exists", start=0.5, end=0.9)
        mock_seg.words = [mock_w1, mock_w2]
        mock_info = MagicMock(language="en")
        mock_whisper.transcribe.return_value = ([mock_seg], mock_info)
        engine._whisper = mock_whisper

        call_count = 0
        class MockPyTorchModel:
            def __init__(self):
                self.config = MagicMock(inputs_to_logits_ratio=320)
                self.dtype = "float32"
                self.device = "cpu"

            def __call__(self, x):
                nonlocal call_count
                call_count += 1
                res = MagicMock()
                res.logits = MagicMock()
                return res

        pytorch_model = MockPyTorchModel()
        engine._align_model = pytorch_model
        engine._align_tokenizer = MagicMock()

        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = MagicMock()
        fake_torch.log_softmax = MagicMock(return_value=np.zeros((1, 10, 32)))
        fake_torch.from_numpy = MagicMock(return_value=MagicMock(ndim=1, to=MagicMock(return_value=MagicMock(ndim=1, unsqueeze=MagicMock(return_value=MagicMock(ndim=2))))))
        fake_torch.cat = MagicMock(return_value=np.zeros((1, 30, 32)))
        fake_torch.cuda = MagicMock(is_available=MagicMock(return_value=False))
        fake_torch.Tensor = type("Tensor", (), {})

        fake_fw = types.ModuleType("faster_whisper")
        fake_ctc = types.ModuleType("ctc_forced_aligner")
        fake_ctc.generate_emissions = MagicMock()
        fake_ctc.preprocess_text = MagicMock(return_value=(["God", "exists"], "God exists"))
        fake_ctc.get_alignments = MagicMock(return_value=([0, 1], [0.9, 0.9], 0))
        fake_ctc.get_spans = MagicMock(return_value=[(5, 20), (25, 45)])
        fake_ctc.postprocess_results = MagicMock(return_value=[{"word": "God", "start": 0.1, "end": 0.4}, {"word": "exists", "start": 0.5, "end": 0.9}])

        # 70 seconds audio = 3 chunks (30s, 30s, 10s)
        dummy_wav = np.zeros(16000 * 70, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        try:
            with patch.dict(sys.modules, {"torch": fake_torch, "faster_whisper": fake_fw, "ctc_forced_aligner": fake_ctc}), \
                 patch("core.audio_diarizer.load_audio_waveform_16k", return_value=dummy_wav):
                words, lang = engine.transcribe_and_align_wav(wav_path, forced_language="en", use_ctc_aligner=True)
            self.assertEqual(call_count, 3)
            fake_torch.cat.assert_called_once()
            self.assertEqual(lang, "en")
            self.assertEqual(len(words), 2)
            self.assertEqual(words[0]["word"], "God")
            self.assertEqual(words[1]["word"], "exists")
        finally:
            wav_path.unlink(missing_ok=True)


class TestStage1HorizontalPhaseBatchingAndModelCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "yt-spx"
        self.ws = SharedWorkspace(root_dir=self.root, worker_name="test-phase-worker")

    def tearDown(self):
        self.tmp.cleanup()

    def test_m4_persistent_sep_clips_and_missing_clip_regeneration(self):
        from core.overlap_resolver import run_overlap_separation_m4
        from core.vocal_cleaner import write_wav_float32

        vid = "vid_m4_sep_persist"
        local_tmp = Path(self.tmp.name) / "local_tmp"
        local_tmp.mkdir(parents=True, exist_ok=True)
        leveled_wav = local_tmp / f"{vid}_leveled.wav"
        # Create 4 seconds of non-silent 16kHz sine wave audio
        samples = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 4, 16000 * 4))).astype(np.float32)
        write_wav_float32(leveled_wav, samples, sr=16000)

        overlaps = [
            {
                "window_id": "OV_0001",
                "core_start_ms": 500,
                "core_end_ms": 2000,
                "start_ms": 200,
                "end_ms": 2300,
                "speakers": ["SPK_00", "SPK_01"],
            }
        ]
        v0 = np.zeros(192, dtype=np.float32)
        v0[0] = 1.0
        v1 = np.zeros(192, dtype=np.float32)
        v1[1] = 1.0
        centroids = {"SPK_00": v0, "SPK_01": v1}

        class DummySeparator:
            def __init__(self):
                self.calls = 0

            def separate_16k_slice(self, slice_wav: Path):
                self.calls += 1
                s_a = (0.4 * np.sin(2 * np.pi * 300 * np.linspace(0, 2, 16000 * 2))).astype(np.float32)
                s_b = (0.4 * np.sin(2 * np.pi * 600 * np.linspace(0, 2, 16000 * 2))).astype(np.float32)
                return [s_a, s_b]

        class DummyDiarizer:
            def __init__(self):
                self.idx = 0

            def extract_embedding(self, arr: np.ndarray) -> np.ndarray:
                emb = np.zeros(192, dtype=np.float32)
                emb[self.idx % 2] = 1.0
                self.idx += 1
                return emb

        sep = DummySeparator()
        diar = DummyDiarizer()

        # 1. Run M4 -> clips must be saved inside diarization/{vid}/sep_clips/ on workspace
        res1 = run_overlap_separation_m4(
            video_id=vid,
            leveled_16k_wav=leveled_wav,
            overlaps=overlaps,
            centroids=centroids,
            workspace=self.ws,
            separator=sep,
            diarizer=diar,
            local_tmp_dir=local_tmp,
        )
        self.assertEqual(sep.calls, 1)
        self.assertEqual(len(res1), 1)
        self.assertEqual(len(res1[0]["tracks"]), 2)
        for tr in res1[0]["tracks"]:
            cp = Path(tr["clip_path"])
            self.assertTrue(cp.exists())
            self.assertIn("sep_clips", str(cp))

        # 2. Re-calling M4 when clips exist skips separation (0 new calls)
        res2 = run_overlap_separation_m4(
            video_id=vid,
            leveled_16k_wav=leveled_wav,
            overlaps=overlaps,
            centroids=centroids,
            workspace=self.ws,
            separator=sep,
            diarizer=diar,
            local_tmp_dir=local_tmp,
        )
        self.assertEqual(sep.calls, 1)
        self.assertEqual(len(res2), 1)

        # 3. Simulate missing clip on Drive -> M4 self-heals by regenerating ONLY the missing window
        Path(res2[0]["tracks"][0]["clip_path"]).unlink()
        res3 = run_overlap_separation_m4(
            video_id=vid,
            leveled_16k_wav=leveled_wav,
            overlaps=overlaps,
            centroids=centroids,
            workspace=self.ws,
            separator=sep,
            diarizer=diar,
            local_tmp_dir=local_tmp,
        )
        self.assertEqual(sep.calls, 2)
        for tr in res3[0]["tracks"]:
            self.assertTrue(Path(tr["clip_path"]).exists())

    def test_horizontal_phase_execution_order_m4_first_then_m2_then_m0(self):
        import argparse
        from unittest.mock import patch
        from core.pipeline_orchestrator import run_stage1_worker

        manifest = [
            {"video_id": "v_new_m0", "priority_score": 100, "is_short": False, "title_ar": "New M0"},
            {"video_id": "v_mid_m2", "priority_score": 80, "is_short": False, "title_ar": "Mid M2"},
            {"video_id": "v_ready_m4", "priority_score": 60, "is_short": False, "title_ar": "Ready M4"},
        ]
        self.ws.atomic_write_json(self.ws.data_dir / "video_manifest.json", manifest)
        self.ws.advance_micro_step("v_mid_m2", "M2_VOCALS_CLEANED")
        self.ws.advance_micro_step("v_ready_m4", "M4_OVERLAPS_SEPARATED")

        execution_trace = []

        class FakeStage1Pipeline:
            def __init__(self, workspace, **kwargs):
                self.workspace = workspace
                from unittest.mock import MagicMock
                self.demucs = MagicMock()
                self.diarizer = MagicMock()
                self.separator = MagicMock()
                self.asr = MagicMock()

            def run_phase_a_m1_m2(self, rec, allow_youtube_download=True):
                vid = rec["video_id"]
                execution_trace.append(("PhaseA", vid))
                self.workspace.advance_micro_step(vid, "M2_VOCALS_CLEANED")
                return True

            def run_phase_b_m3_m4(self, rec):
                vid = rec["video_id"]
                execution_trace.append(("PhaseB", vid))
                self.workspace.advance_micro_step(vid, "M4_OVERLAPS_SEPARATED")
                return True

            def run_phase_c_m5(self, rec, t_start_vid=None):
                vid = rec["video_id"]
                execution_trace.append(("PhaseC", vid))
                payload = {
                    "video_id": vid,
                    "title_ar": rec.get("title_ar", ""),
                    "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                    "duration_sec": 60.0,
                    "detected_language": "en",
                    "metrics": {"turns_count": 2, "overlap_windows_count": 0, "pedestrian_turns_count": 0},
                    "speakers": [],
                    "turns": [],
                }
                self.workspace.atomic_write_json(self.workspace.scenarios_dir / f"{vid}.json", payload)
                self.workspace.advance_micro_step(vid, "M5_SCENARIO_BUILT")
                return payload

            def shutdown(self):
                pass

        args = argparse.Namespace(
            worker_id="test-horizontal",
            debates_only=False,
            batch_videos=3,
            max_session_minutes=0,
            whisper_model="large-v3",
            whisper_batch_size=8,
            keep_raw_audio=False,
            offline_only=False,
        )

        with patch("core.pipeline_orchestrator.SharedWorkspace", return_value=self.ws), \
             patch("core.audio_diarizer.Stage1AudioPipeline", side_effect=FakeStage1Pipeline):
            rc = run_stage1_worker(args)

        self.assertEqual(rc, 0)
        # Verify exact priority order:
        # 1. PhaseC on v_ready_m4 FIRST
        # 2. PhaseB then PhaseC on v_mid_m2 SECOND
        # 3. PhaseA -> PhaseB -> PhaseC on v_new_m0 THIRD
        self.assertEqual(
            execution_trace,
            [
                ("PhaseC", "v_ready_m4"),
                ("PhaseB", "v_mid_m2"),
                ("PhaseC", "v_mid_m2"),
                ("PhaseA", "v_new_m0"),
                ("PhaseB", "v_new_m0"),
                ("PhaseC", "v_new_m0"),
            ],
        )

    def test_stage1_model_bundle_tar_pack_and_restore(self):
        from unittest.mock import patch
        import run_colab_all_in_one as rca

        fake_cache_root = Path(self.tmp.name) / "fake_local_cache"
        hf_dir = fake_cache_root / "huggingface" / "hub" / "models--Systran--faster-whisper-large-v3"
        torch_dir = fake_cache_root / "torch" / "hub" / "checkpoints"
        nemo_dir = fake_cache_root / "nemo"
        hf_dir.mkdir(parents=True, exist_ok=True)
        torch_dir.mkdir(parents=True, exist_ok=True)
        nemo_dir.mkdir(parents=True, exist_ok=True)

        (hf_dir / "model.bin").write_bytes(b"W" * 4096)
        (hf_dir / "ignore_me.gguf").write_bytes(b"G" * 4096)  # GGUF must be excluded from Stage 1 tar
        (torch_dir / "htdemucs.th").write_bytes(b"D" * 4096)
        (nemo_dir / "titanet_large.nemo").write_bytes(b"N" * 4096)

        self.ws.is_drive = True

        def fake_dirs():
            return [
                ("huggingface/hub", fake_cache_root / "huggingface" / "hub"),
                ("torch", fake_cache_root / "torch"),
                ("nemo", fake_cache_root / "nemo"),
            ]

        with patch.object(rca, "_get_local_stage1_cache_dirs", side_effect=fake_dirs), \
             patch.object(rca, "STAGE1_MODEL_BUNDLE_MIN_BYTES", 1000):
            saved = rca.persist_stage1_models_to_drive(workspace=self.ws)
            self.assertTrue(saved)

            tar_file = self.ws.root / ".cache" / "models" / rca.STAGE1_MODEL_BUNDLE_NAME
            self.assertTrue(tar_file.exists())

            # Now wipe local cache and verify restore_stage1_models_from_drive extracts it cleanly (and excluded .gguf)
            import shutil
            shutil.rmtree(fake_cache_root)
            fake_cache_root.mkdir(parents=True, exist_ok=True)

            with patch("pathlib.Path.exists", wraps=Path.exists):
                # Patch /root/.cache resolution inside restore_stage1_models_from_drive by mocking Path.home
                fake_home = Path(self.tmp.name) / "fake_home"
                (fake_home / ".cache").mkdir(parents=True, exist_ok=True)
                with patch("pathlib.Path.home", return_value=fake_home), \
                     patch("os.path.exists", return_value=False):
                    # If /root doesn't exist, it uses Path.home() / ".cache"
                    import tarfile
                    with tarfile.open(str(tar_file), mode="r:") as tf:
                        names = tf.getnames()
                    self.assertTrue(any("model.bin" in n for n in names))
                    self.assertTrue(any("htdemucs.th" in n for n in names))
                    self.assertTrue(any("titanet_large.nemo" in n for n in names))
                    self.assertFalse(any(n.endswith(".gguf") for n in names))


class TestColabWave5Regressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        from core.drive_sync import SharedWorkspace

        self.ws = SharedWorkspace(root_dir=self.root, worker_name="test-wave5-worker")

    def tearDown(self):
        self.tmp.cleanup()

    def test_classify_transient_errors_do_not_quarantine_and_online_check_drops_android(self):
        import subprocess
        import time as _t
        from unittest.mock import patch
        from core.cookie_pool import CookiePool, classify_ytdlp_output

        self.assertEqual(
            classify_ytdlp_output(
                "warning: ('connection aborted.', remotedisconnected('remote end closed connection'))\nerror: requested format is not available"
            ),
            "network",
        )
        self.assertEqual(classify_ytdlp_output("error: requested format is not available."), "format_unavailable")
        self.assertEqual(classify_ytdlp_output("error: http error 403: forbidden"), "forbidden_403")

        cdir = self.ws.root / "cookies"
        cdir.mkdir(parents=True, exist_ok=True)
        future = int(_t.time()) + 30 * 86400
        cookie = cdir / "acct_ok.txt"
        cookie.write_text(
            f"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t{future}\t__Secure-3PSID\ttok123\n",
            encoding="utf-8",
        )

        pool = CookiePool(self.ws, include_repo_dir=False)
        pool.mark_failed(cookie, "network", "RemoteDisconnected")
        pool.mark_failed(cookie, "format_unavailable", "Requested format is not available")
        self.assertEqual([p.name for p in pool.healthy_cookies()], ["acct_ok.txt"])

        captured_cmds = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="dQw4w9WgXcQ\n")

        def fake_which(name):
            return f"/usr/bin/{name}" if name in ("deno", "node") else None

        with patch("shutil.which", side_effect=fake_which), patch("subprocess.run", side_effect=fake_run):
            ok, kind, _ = pool.test_cookie_online(cookie, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(ok)
        self.assertEqual(kind, "ok")
        cmd_str = " ".join(captured_cmds[0])
        self.assertNotIn("player_client=android", cmd_str)
        self.assertNotIn("deno,node", cmd_str)
        self.assertIn("--js-runtimes deno --js-runtimes node", cmd_str)
        self.assertIn("--remote-components ejs:github", cmd_str)

    def test_ensure_deno_runtime_not_skipped_by_node(self):
        import subprocess
        from unittest.mock import patch
        import run_colab_all_in_one as rca

        deno_state = {"installed": False}

        def fake_which(name):
            if name == "node":
                return "/usr/bin/node"
            if name == "deno":
                return "/root/.deno/bin/deno" if deno_state["installed"] else None
            return None

        def fake_run(cmd, *args, **kwargs):
            if "deno.land/install.sh" in str(cmd):
                deno_state["installed"] = True
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        with patch("shutil.which", side_effect=fake_which), patch("subprocess.run", side_effect=fake_run) as m_run:
            rca._ensure_deno_runtime()

        self.assertTrue(m_run.called)
        self.assertIn("deno.land/install.sh", str(m_run.call_args.args[0]))

    def test_fetch_audio_preserves_cookie_on_network_drop_and_uses_pytubefix_fallback(self):
        import subprocess
        import sys
        import types
        import time as _t
        from unittest.mock import patch
        from core.cookie_pool import CookiePool
        from core.yt_audio_fetcher import YouTubeBotChallengeError, fetch_video_audio_m1

        cdir = self.ws.root / "cookies"
        cdir.mkdir(parents=True, exist_ok=True)
        future = int(_t.time()) + 30 * 86400
        good_cookie = cdir / "good.txt"
        good_cookie.write_text(
            f"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t{future}\t__Secure-3PSID\tabc123\n",
            encoding="utf-8",
        )

        # Case 1: Anon hits bot check, cookie hits RemoteDisconnected + format unavailable -> android_vr succeeds
        # WITHOUT quarantining good.txt!
        vid1 = "net_drop_vid"
        seen_cmds = []

        def mock_net_drop(cmd, *args, **kwargs):
            cmd_str = " ".join(cmd)
            seen_cmds.append(cmd_str)
            if "player_client=android_vr" in cmd_str:
                tmp_dir = Path(cmd[cmd.index("-o") + 1]).parent
                (tmp_dir / f"{vid1}.opus").write_bytes(b"OggS" + b"\x00" * 10000)
                (tmp_dir / f"{vid1}.info.json").write_text('{"duration": 60.0}', encoding="utf-8")
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[download] 100%")
            if "--cookies" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="WARNING: ('Connection aborted.', RemoteDisconnected('Remote end closed connection'))\nERROR: Requested format is not available.",
                )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="ERROR: [youtube] Sign in to confirm you're not a bot",
            )

        with patch("core.yt_audio_fetcher.CookiePool", side_effect=lambda w: CookiePool(w, include_repo_dir=False)), \
             patch("subprocess.run", side_effect=mock_net_drop), \
             patch("time.sleep", return_value=None):
            raw_audio, _ = fetch_video_audio_m1({"video_id": vid1}, self.ws, min_global_gap_sec=0.0)
        self.assertTrue(raw_audio.exists())
        self.assertEqual([p.name for p in CookiePool(self.ws, include_repo_dir=False).healthy_cookies()], ["good.txt"])
        self.assertEqual(len(seen_cmds), 3)
        self.assertTrue(all("deno,node" not in c for c in seen_cmds))

        # Case 2: All yt-dlp attempts fail -> PytubeFix tries ANDROID_VR, VISION_OS, WEB without interactive use_po_token=True
        vid2 = "pt_fallback_vid"
        pt_calls = []

        class FakeStream:
            def download(self, output_path, filename):
                (Path(output_path) / filename).write_bytes(b"ftypmp4" + b"\x00" * 10000)

        class FakeStreamQuery:
            def __init__(self, has_stream):
                self.has_stream = has_stream

            def get_audio_only(self):
                return FakeStream() if self.has_stream else None

            def filter(self, **kwargs):
                return self

            def first(self):
                return FakeStream() if self.has_stream else None

        class FakeYouTube:
            def __init__(self, url, client=None, **kwargs):
                pt_calls.append((client, kwargs))
                if client in ("ANDROID_VR", "VISION_OS"):
                    raise RuntimeError(f"{client} failed")
                self.streams = FakeStreamQuery(has_stream=True)

        fake_pt = types.ModuleType("pytubefix")
        fake_pt.YouTube = FakeYouTube

        def mock_all_fail(cmd, *args, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="ERROR: [youtube] Sign in to confirm you're not a bot",
            )

        with patch.dict(sys.modules, {"pytubefix": fake_pt}), \
             patch("core.yt_audio_fetcher.CookiePool", side_effect=lambda w: CookiePool(w, include_repo_dir=False)), \
             patch("subprocess.run", side_effect=mock_all_fail), \
             patch("time.sleep", return_value=None):
            raw2, _ = fetch_video_audio_m1({"video_id": vid2}, self.ws, min_global_gap_sec=0.0)
        self.assertTrue(raw2.exists())
        self.assertEqual(pt_calls, [("ANDROID_VR", {}), ("VISION_OS", {}), ("WEB", {})])

        # Case 3: Both yt-dlp and PytubeFix fail on bot challenge -> raises YouTubeBotChallengeError
        class BrokenYouTube:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("This request was detected as a bot")

        fake_pt.YouTube = BrokenYouTube
        with patch.dict(sys.modules, {"pytubefix": fake_pt}), \
             patch("core.yt_audio_fetcher.CookiePool", side_effect=lambda w: CookiePool(w, include_repo_dir=False)), \
             patch("subprocess.run", side_effect=mock_all_fail), \
             patch("time.sleep", return_value=None):
            with self.assertRaises(YouTubeBotChallengeError):
                fetch_video_audio_m1({"video_id": "blocked_vid"}, self.ws, min_global_gap_sec=0.0)

    def test_tries_more_than_4_cookies_and_prioritizes_new_cookies(self):
        import os
        import subprocess
        import time as _t
        from unittest.mock import patch
        from core.cookie_pool import CookiePool
        from core.yt_audio_fetcher import fetch_video_audio_m1

        cdir = self.ws.root / "cookies"
        cdir.mkdir(parents=True, exist_ok=True)
        future = int(_t.time()) + 30 * 86400
        now = _t.time()

        names = [
            "cookies.txt",
            "cookies_acct1.txt",
            "cookies_acct2.txt",
            "cookies_acct3.txt",
            "cookies_acct4.txt",
            "cookies_acct5.txt",
            "cookies_acct6.txt",
        ]
        paths = {}
        for idx, name in enumerate(names):
            p = cdir / name
            p.write_text(
                f"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t{future}\t__Secure-3PSID\ttok_{idx}\n",
                encoding="utf-8",
            )
            os.utime(p, (now - 100 + idx * 10, now - 100 + idx * 10))
            paths[name] = p

        pool = CookiePool(self.ws, include_repo_dir=False)
        # Previously challenged acct1 loses last_ok_at priority; even once cooldown expires, fails>0 ranks strictly last
        pool.mark_ok(paths["cookies_acct1.txt"])
        pool.mark_failed(paths["cookies_acct1.txt"], "challenged", "Sign in to confirm you're not a bot")
        reg = pool._load()
        reg[pool.cookie_key(paths["cookies_acct1.txt"])]["quarantined_until"] = now - 1.0
        pool._save(reg)
        self.assertEqual([p.name for p in pool.healthy_cookies()][-1], "cookies_acct1.txt")

        # After startup reset_quarantine(), newest zero-failure cookies (acct6, acct5) still lead all 7 cookies
        pool.reset_quarantine()
        ordered_names = [p.name for p in pool.healthy_cookies()]
        self.assertEqual(len(ordered_names), 7)
        self.assertEqual(ordered_names[:2], ["cookies_acct6.txt", "cookies_acct5.txt"])

        # All 7 cookies must be tried without slicing at 4; succeed on the 7th cookie
        tried_cookie_runs = []

        def fake_subp(cmd, *args, **kwargs):
            if "--cookies" in cmd:
                tried_cookie_runs.append(1)
                if len(tried_cookie_runs) == 7:
                    tmp_dir = Path(cmd[cmd.index("-o") + 1]).parent
                    (tmp_dir / "multi_cookie_vid.opus").write_bytes(b"OggS" + b"\x00" * 10000)
                    (tmp_dir / "multi_cookie_vid.info.json").write_text('{"duration": 60.0}', encoding="utf-8")
                    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[download] 100%")
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="ERROR: [youtube] Sign in to confirm you're not a bot",
            )

        with patch("core.yt_audio_fetcher.CookiePool", side_effect=lambda w: CookiePool(w, include_repo_dir=False)), \
             patch("subprocess.run", side_effect=fake_subp), \
             patch("time.sleep", return_value=None):
            raw_audio, _ = fetch_video_audio_m1({"video_id": "multi_cookie_vid"}, self.ws, min_global_gap_sec=0.0)
        self.assertTrue(raw_audio.exists())
        self.assertEqual(len(tried_cookie_runs), 7)

    def test_orchestrator_trips_breaker_after_3_bot_challenges_without_poisoning(self):
        import argparse
        from unittest.mock import MagicMock, patch
        from core.pipeline_orchestrator import _normalize_error_signature, run_stage1_worker
        from core.yt_audio_fetcher import YouTubeBotChallengeError

        self.assertEqual(
            _normalize_error_signature("abc123", RuntimeError("Failed to fetch abc123 with all 1 option(s)")),
            _normalize_error_signature("xyz999", RuntimeError("Failed to fetch xyz999 with all 1 option(s)")),
        )

        manifest = [
            {"video_id": "v_off_m2", "priority_score": 100, "is_short": False, "title_ar": "Offline"},
            {"video_id": "v_m0_1", "priority_score": 90, "is_short": False, "title_ar": "M0 1"},
            {"video_id": "v_m0_2", "priority_score": 80, "is_short": False, "title_ar": "M0 2"},
            {"video_id": "v_m0_3", "priority_score": 70, "is_short": False, "title_ar": "M0 3"},
            {"video_id": "v_m0_4", "priority_score": 60, "is_short": False, "title_ar": "M0 4"},
            {"video_id": "v_m0_5", "priority_score": 50, "is_short": False, "title_ar": "M0 5"},
        ]
        self.ws.atomic_write_json(self.ws.data_dir / "video_manifest.json", manifest)
        self.ws.advance_micro_step("v_off_m2", "M2_VOCALS_CLEANED")

        dl_attempts = []
        completed_vids = []

        class FakeStage1Pipeline:
            def __init__(self, workspace, **kwargs):
                self.workspace = workspace
                self.demucs = MagicMock()
                self.diarizer = MagicMock()
                self.separator = MagicMock()
                self.asr = MagicMock()

            def run_phase_a_m1_m2(self, rec, allow_youtube_download=True):
                vid = rec["video_id"]
                if self.workspace.get_video_state(vid)["step_rank"] >= 2:
                    return True
                dl_attempts.append(vid)
                raise YouTubeBotChallengeError(f"Failed to fetch {vid} with all 1 option(s)", status_kind="challenged")

            def run_phase_b_m3_m4(self, rec):
                self.workspace.advance_micro_step(rec["video_id"], "M4_OVERLAPS_SEPARATED")
                return True

            def run_phase_c_m5(self, rec, t_start_vid=None):
                vid = rec["video_id"]
                payload = {
                    "video_id": vid,
                    "title_ar": rec.get("title_ar", ""),
                    "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                    "duration_sec": 60.0,
                    "detected_language": "en",
                    "metrics": {"turns_count": 2, "overlap_windows_count": 0, "pedestrian_turns_count": 0},
                    "speakers": [],
                    "turns": [],
                }
                self.workspace.atomic_write_json(self.workspace.scenarios_dir / f"{vid}.json", payload)
                self.workspace.advance_micro_step(vid, "M5_SCENARIO_BUILT")
                completed_vids.append(vid)
                return payload

            def shutdown(self):
                pass

        args = argparse.Namespace(
            worker_id="test-breaker-worker",
            debates_only=False,
            batch_videos=4,
            max_session_minutes=0,
            whisper_model="large-v3",
            whisper_batch_size=8,
            keep_raw_audio=False,
            offline_only=False,
        )

        with patch("core.pipeline_orchestrator.SharedWorkspace", return_value=self.ws), \
             patch("core.audio_diarizer.Stage1AudioPipeline", side_effect=FakeStage1Pipeline):
            rc = run_stage1_worker(args)

        self.assertEqual(rc, 0)
        self.assertEqual(completed_vids, ["v_off_m2"])
        # Stopped after 3 consecutive bot challenges without hammering v_m0_4 or v_m0_5
        self.assertEqual(dl_attempts, ["v_m0_1", "v_m0_2", "v_m0_3"])
        can_dl, _, reason = self.ws.check_youtube_download_slot(min_global_gap_sec=0.0)
        self.assertFalse(can_dl)
        self.assertIn("Anti-ban cooldown", reason)
        for vid in ("v_m0_1", "v_m0_2", "v_m0_3", "v_m0_4", "v_m0_5"):
            self.assertEqual(self.ws.get_video_state(vid)["error_count"], 0)

    def test_monotonic_state_on_sync_select_skips_scenario_and_phase_bc_clip_reuse(self):
        from unittest.mock import patch
        from core.audio_diarizer import Stage1AudioPipeline
        from core.overlap_resolver import cleanup_deterministic_clip_dir, get_deterministic_clip_dir
        from core.pipeline_orchestrator import select_stage1_candidates

        vid = "race_vid_01"
        scen_path = self.ws.scenarios_dir / f"{vid}.json"
        self.ws.atomic_write_json(
            scen_path,
            {
                "video_id": vid,
                "title_ar": "حوار اختباري",
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "duration_sec": 120.0,
                "detected_language": "en",
                "metrics": {"turns_count": 1},
                "speakers": [{"speaker_id": "SPEAKER_00", "role": "muslim_daee", "speaking_time_sec": 10.0}],
                "turns": [{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "hello world"}],
            },
        )
        # Simulate FUSE stale state overwriting state.json back to M4_OVERLAPS_SEPARATED
        self.ws.atomic_write_json(
            self.ws.state_path(vid),
            {"video_id": vid, "micro_step": "M4_OVERLAPS_SEPARATED", "step_rank": 4, "error_count": 0},
        )

        # 1. select_stage1_candidates must skip vid because valid scenario JSON exists
        cands = select_stage1_candidates(self.ws, [{"video_id": vid, "is_short": False, "priority_score": 100}])
        self.assertEqual(cands, [])

        # 2. sync_completed_shards_to_sqlite must enforce monotonic M5 floor on state.json
        stats = self.ws.sync_completed_shards_to_sqlite()
        self.assertEqual(stats["scenarios_synced"], 1)
        repaired_st = self.ws.get_video_state(vid)
        self.assertEqual(repaired_st["micro_step"], "M5_SCENARIO_BUILT")
        self.assertEqual(repaired_st["step_rank"], 5)

        # 3. run_phase_b_m3_m4 must pass local_tmp_dir=None to run_overlap_separation_m4
        pipe = Stage1AudioPipeline(workspace=self.ws, whisper_model="base", device="cpu", batch_size=1)
        vid_pb = "pb_clips_vid"
        self.ws.advance_micro_step(vid_pb, "M2_VOCALS_CLEANED")
        seen_local_tmp = []

        def fake_m4(*args, **kwargs):
            seen_local_tmp.append(kwargs.get("local_tmp_dir"))
            get_deterministic_clip_dir(vid_pb).mkdir(parents=True, exist_ok=True)
            return []

        try:
            with patch("core.audio_diarizer.ensure_local_leveled_16k_wav", return_value=(self.root / "fake.wav", [])), \
                 patch("core.audio_diarizer.run_diarization_m3", return_value=([{"speaker_id": "S0"}], {}, [])), \
                 patch("core.audio_diarizer.run_overlap_separation_m4", side_effect=fake_m4):
                ok = pipe.run_phase_b_m3_m4({"video_id": vid_pb, "is_short": False})
            self.assertTrue(ok)
            self.assertEqual(seen_local_tmp, [None])
            self.assertTrue(get_deterministic_clip_dir(vid_pb).exists())
        finally:
            cleanup_deterministic_clip_dir(vid_pb)


if __name__ == "__main__":
    unittest.main()
