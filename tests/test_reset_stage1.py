#!/usr/bin/env python3
"""
tests/test_reset_stage1.py
Unit tests verifying the Surgical Stage 1 Reset Tool:
- Guarantees 100% preservation of downloaded raw and clean audio files.
- Verifies complete purging of legacy scenarios, diarized SRTs, and old extractions.
- Validates proper micro-step rank fallback (M2_VOCALS_CLEANED vs M4_OVERLAPS_SEPARATED vs M1_DOWNLOADED).
- Confirms database and lock cleanup.
"""

import os
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.drive_sync import SharedWorkspace, STEP_RANK
from tools.reset_stage1 import reset_stage1_pipeline
from db.database import init_db


class TestResetStage1Pipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.test_dir.name)
        self.workspace = SharedWorkspace(root_dir=self.root, worker_name="test-worker")

        # Set up a mock database
        self.db_path = self.workspace.data_dir / "dawah_knowledge.db"
        init_db(self.db_path)

        conn = sqlite3.connect(str(self.db_path))
        with conn:
            conn.execute("""
                INSERT INTO videos (video_id, title_ar, stage1_status, stage2_status, turns_count)
                VALUES ('TEST_VID_01', 'فيديو اختباري', 'completed', 'completed', 15)
            """)
            conn.execute("""
                INSERT INTO video_speakers (video_id, speaker_id, role)
                VALUES ('TEST_VID_01', 'SPK_00', 'MAIN_DAEE')
            """)
            conn.execute("""
                INSERT INTO scenario_turns (turn_id, video_id, turn_index, speaker_id, speaker_role, start_ms, end_ms, start_fmt, end_fmt, text_original)
                VALUES ('TEST_VID_01_T0001', 'TEST_VID_01', 1, 'SPK_00', 'MAIN_DAEE', 0, 5000, '00:00', '00:05', 'Hello test')
            """)
            conn.execute("""
                INSERT INTO qa_units (id, video_id, topic_category, daee_name_ar)
                VALUES ('TEST_VID_01_001', 'TEST_VID_01', 'عقيدة', 'هاشم')
            """)
        conn.close()

    def tearDown(self):
        import gc
        gc.collect()
        self.test_dir.cleanup()

    def test_reset_stage1_preserves_audio_and_purges_scenarios(self):
        vid = "TEST_VID_01"

        # 1. Populate mock audio files
        raw_audio = self.workspace.audio_raw_dir / f"{vid}.mp4"
        raw_audio.write_bytes(b"FAKE_RAW_AUDIO_CONTENT" * 3000)  # > 50KB
        raw_info = self.workspace.audio_raw_dir / f"{vid}.info.json"
        raw_info.write_text(json.dumps({"video_id": vid, "duration": 120.0}))

        clean_opus = self.workspace.audio_clean_dir / f"{vid}.opus"
        clean_opus.write_bytes(b"FAKE_CLEAN_OPUS_CONTENT" * 200)  # > 1KB
        rms_json = self.workspace.audio_clean_dir / f"{vid}_rms.json"
        rms_json.write_text(json.dumps({"video_id": vid, "rms_db": [-22.5, -20.1]}))

        # 2. Populate mock scenario & extraction artifacts
        scen_json = self.workspace.scenarios_dir / f"{vid}.json"
        scen_json.write_text(json.dumps({"video_id": vid, "turns": []}))
        scen_srt = self.workspace.scenarios_dir / f"{vid}.diarized.srt"
        scen_srt.write_text("1\n00:00:00,000 --> 00:00:05,000\nHello")

        qa_shard = self.workspace.extractions_dir / f"{vid}.qa.json"
        qa_shard.write_text(json.dumps([{"unit_id": f"{vid}_001"}]))
        partial_win = self.workspace.partial_extractions_dir / f"{vid}_win0.json"
        partial_win.write_text(json.dumps([{"unit_id": f"{vid}_001"}]))

        # 3. Populate mock diarization dir & locks
        diar_dir = self.workspace.diarization_dir / vid
        diar_dir.mkdir(parents=True, exist_ok=True)
        (diar_dir / f"{vid}.rttm").write_text("SPEAKER vid 1 0.0 5.0 <NA> <NA> SPK_00 <NA> <NA>")

        lock_file = self.workspace.locks_dir / f"{vid}.stage1.lock.json"
        lock_file.write_text(json.dumps({"worker": "test"}))

        # 4. Populate video state
        st_file = self.workspace.state_path(vid)
        self.workspace.atomic_write_json(st_file, {
            "video_id": vid,
            "micro_step": "M6_LLM_EXTRACTED",
            "step_rank": STEP_RANK["M6_LLM_EXTRACTED"],
            "error_count": 0,
        })

        # Run reset_stage1_pipeline (default: purge diarization, reset to M2)
        summary = reset_stage1_pipeline(self.workspace, keep_diarization=False)

        # Assertions
        # A) CRITICAL: YouTube audio & clean vocal files are 100% PRESERVED
        self.assertTrue(raw_audio.exists(), "Raw YouTube audio must NOT be deleted!")
        self.assertTrue(raw_info.exists(), "Raw info.json must NOT be deleted!")
        self.assertTrue(clean_opus.exists(), "Clean Opus vocal file must NOT be deleted!")
        self.assertTrue(rms_json.exists(), "RMS loudness json must NOT be deleted!")

        # B) Scenarios and extractions are purged
        self.assertFalse(scen_json.exists(), "Old scenario JSON must be purged!")
        self.assertFalse(scen_srt.exists(), "Old diarized SRT must be purged!")
        self.assertFalse(qa_shard.exists(), "Old QA shard must be purged!")
        self.assertFalse(partial_win.exists(), "Old partial window must be purged!")
        self.assertFalse(diar_dir.exists(), "Diarization directory must be purged when keep_diarization=False!")
        self.assertFalse(lock_file.exists(), "Locks must be cleared!")

        # C) State reset to M2_VOCALS_CLEANED because clean Opus is present
        st_after = self.workspace.get_video_state(vid)
        self.assertEqual(st_after["micro_step"], "M2_VOCALS_CLEANED")
        self.assertEqual(st_after["step_rank"], STEP_RANK["M2_VOCALS_CLEANED"])

        # D) SQLite database purged
        conn = sqlite3.connect(str(self.db_path))
        t_count = conn.execute("SELECT COUNT(*) FROM scenario_turns").fetchone()[0]
        spk_count = conn.execute("SELECT COUNT(*) FROM video_speakers").fetchone()[0]
        qa_count = conn.execute("SELECT COUNT(*) FROM qa_units").fetchone()[0]
        row_vid = conn.execute("SELECT stage1_status, stage2_status, turns_count FROM videos WHERE video_id = ?", (vid,)).fetchone()
        conn.close()

        self.assertEqual(t_count, 0)
        self.assertEqual(spk_count, 0)
        self.assertEqual(qa_count, 0)
        self.assertEqual(row_vid[0], "pending")
        self.assertEqual(row_vid[1], "pending")
        self.assertEqual(row_vid[2], 0)

    def test_reset_stage1_keep_diarization(self):
        vid = "TEST_VID_02"

        # Setup audio and diarization
        clean_opus = self.workspace.audio_clean_dir / f"{vid}.opus"
        clean_opus.write_bytes(b"OPUS" * 300)
        rms_json = self.workspace.audio_clean_dir / f"{vid}_rms.json"
        rms_json.write_text(json.dumps({"video_id": vid, "rms_db": [-20.0]}))

        diar_dir = self.workspace.diarization_dir / vid
        diar_dir.mkdir(parents=True, exist_ok=True)
        rttm_file = diar_dir / f"{vid}.rttm"
        rttm_file.write_text("SPEAKER vid 1 0.0 5.0 <NA> <NA> SPK_00 <NA> <NA>")

        scen_json = self.workspace.scenarios_dir / f"{vid}.json"
        scen_json.write_text(json.dumps({"video_id": vid}))

        st_file = self.workspace.state_path(vid)
        self.workspace.atomic_write_json(st_file, {
            "video_id": vid,
            "micro_step": "M5_SCENARIO_BUILT",
            "step_rank": STEP_RANK["M5_SCENARIO_BUILT"],
        })

        # Run reset with keep_diarization=True
        reset_stage1_pipeline(self.workspace, keep_diarization=True)

        # Diarization is preserved
        self.assertTrue(rttm_file.exists(), "RTTM must be kept when keep_diarization=True")
        self.assertFalse(scen_json.exists(), "Scenario must still be purged")

        # State resets to M4_OVERLAPS_SEPARATED
        st_after = self.workspace.get_video_state(vid)
        self.assertEqual(st_after["micro_step"], "M4_OVERLAPS_SEPARATED")
        self.assertEqual(st_after["step_rank"], STEP_RANK["M4_OVERLAPS_SEPARATED"])

    def test_reset_stage1_only_raw_audio(self):
        vid = "TEST_VID_03"

        # Only raw audio exists (no clean opus)
        raw_audio = self.workspace.audio_raw_dir / f"{vid}.webm"
        raw_audio.write_bytes(b"RAW" * 20000)

        scen_json = self.workspace.scenarios_dir / f"{vid}.json"
        scen_json.write_text(json.dumps({"video_id": vid}))

        st_file = self.workspace.state_path(vid)
        self.workspace.atomic_write_json(st_file, {
            "video_id": vid,
            "micro_step": "M5_SCENARIO_BUILT",
            "step_rank": STEP_RANK["M5_SCENARIO_BUILT"],
        })

        reset_stage1_pipeline(self.workspace, keep_diarization=False)

        # Raw audio preserved
        self.assertTrue(raw_audio.exists(), "Raw audio must be kept!")
        self.assertFalse(scen_json.exists(), "Scenario must be purged!")

        # State resets to M1_DOWNLOADED
        st_after = self.workspace.get_video_state(vid)
        self.assertEqual(st_after["micro_step"], "M1_DOWNLOADED")
        self.assertEqual(st_after["step_rank"], STEP_RANK["M1_DOWNLOADED"])


if __name__ == "__main__":
    unittest.main()
