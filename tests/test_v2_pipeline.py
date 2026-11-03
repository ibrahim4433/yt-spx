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
                    "text_original": "Only things that begin to exist require a cause; God is the Necessary Uncaused Being.",
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
                    "text_original": "Empirical science describes what is, never what ought to be; bridging is-ought requires an transcendent moral lawgiver.",
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


if __name__ == "__main__":
    unittest.main()
