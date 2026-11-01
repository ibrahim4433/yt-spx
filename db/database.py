#!/usr/bin/env python3
"""
db/database.py
V2 SQLite + Dual-FTS5 Database Engine for Dawah Dialectics (yt-spx).
Stores both:
  1. Complete Multi-Speaker Diarized Scenario Transcripts (video_speakers, scenario_turns, scenario_turns_fts)
  2. Verified Dual-Speaker Dialectical Claim-and-Proof Dossiers (qa_units, qa_fts)
Supports bilingual (Arabic & English) search, Da'ee speaker voiceprint filtering,
and single-writer shard ingestion across shared 5 TB Google Drive Colab accounts.
"""

import os
import json
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_default_data_dir() -> Path:
    env_drive = os.environ.get("YT_SPX_DRIVE_DIR", "").strip()
    if env_drive:
        d = Path(env_drive) / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d
    colab_drive = Path("/content/drive/MyDrive/yt-spx/data")
    if Path("/content/drive/MyDrive").exists():
        colab_drive.mkdir(parents=True, exist_ok=True)
        return colab_drive
    local_data = BASE_DIR / "data"
    local_data.mkdir(parents=True, exist_ok=True)
    return local_data


DATA_DIR = _resolve_default_data_dir()
DB_PATH = DATA_DIR / "dawah_knowledge.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target = Path(db_path) if db_path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def};")


_SQLITE_FTS5_SUPPORTED: Optional[bool] = None


def _supports_fts5(conn: Optional[sqlite3.Connection] = None) -> bool:
    """Returns True if the runtime sqlite3 engine supports FTS5 virtual tables."""
    global _SQLITE_FTS5_SUPPORTED
    if _SQLITE_FTS5_SUPPORTED is not None:
        return _SQLITE_FTS5_SUPPORTED
    close_after = False
    target = conn
    if target is None:
        target = sqlite3.connect(":memory:")
        close_after = True
    try:
        target.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _test_fts5_probe USING fts5(probe);")
        target.execute("DROP TABLE IF EXISTS _test_fts5_probe;")
        _SQLITE_FTS5_SUPPORTED = True
    except sqlite3.OperationalError:
        _SQLITE_FTS5_SUPPORTED = False
    finally:
        if close_after:
            target.close()
    return _SQLITE_FTS5_SUPPORTED


def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_db_connection(db_path)
    with conn:
        # 1. Canonical Videos Inventory + Acoustic Telemetry
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                catalog_order INTEGER DEFAULT 0,
                channel TEXT DEFAULT 'SCDawah Arabic',
                title_ar TEXT DEFAULT '',
                youtube_url TEXT DEFAULT '',
                is_short INTEGER DEFAULT 0,
                priority_score INTEGER DEFAULT 75,
                primary_daee_hint TEXT,
                duration_sec REAL DEFAULT 0.0,
                detected_language TEXT DEFAULT 'en',
                speakers_count INTEGER DEFAULT 0,
                episodes_count INTEGER DEFAULT 0,
                turns_count INTEGER DEFAULT 0,
                spoken_words_count INTEGER DEFAULT 0,
                overlap_windows_count INTEGER DEFAULT 0,
                overlap_ratio REAL DEFAULT 0.0,
                pedestrian_turns_count INTEGER DEFAULT 0,
                qa_count INTEGER DEFAULT 0,
                clean_opus_path TEXT DEFAULT '',
                rttm_path TEXT DEFAULT '',
                scenario_json_path TEXT DEFAULT '',
                stage1_status TEXT DEFAULT 'queued',
                stage2_status TEXT DEFAULT 'queued',
                status TEXT DEFAULT 'queued',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Ensure backwards-compatible upgrades if table already existed
        for col, cdef in [
            ("catalog_order", "INTEGER DEFAULT 0"),
            ("is_short", "INTEGER DEFAULT 0"),
            ("priority_score", "INTEGER DEFAULT 75"),
            ("primary_daee_hint", "TEXT"),
            ("duration_sec", "REAL DEFAULT 0.0"),
            ("detected_language", "TEXT DEFAULT 'en'"),
            ("speakers_count", "INTEGER DEFAULT 0"),
            ("episodes_count", "INTEGER DEFAULT 0"),
            ("turns_count", "INTEGER DEFAULT 0"),
            ("spoken_words_count", "INTEGER DEFAULT 0"),
            ("overlap_windows_count", "INTEGER DEFAULT 0"),
            ("overlap_ratio", "REAL DEFAULT 0.0"),
            ("pedestrian_turns_count", "INTEGER DEFAULT 0"),
            ("qa_count", "INTEGER DEFAULT 0"),
            ("clean_opus_path", "TEXT DEFAULT ''"),
            ("rttm_path", "TEXT DEFAULT ''"),
            ("scenario_json_path", "TEXT DEFAULT ''"),
            ("stage1_status", "TEXT DEFAULT 'queued'"),
            ("stage2_status", "TEXT DEFAULT 'queued'"),
        ]:
            _ensure_column(conn, "videos", col, cdef)

        # 2. Global Speaker Voiceprint Profiles (Recurring Da'ees)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS speaker_profiles (
                profile_id TEXT PRIMARY KEY,
                name_en TEXT NOT NULL,
                name_ar TEXT NOT NULL,
                role_default TEXT DEFAULT 'MAIN_DAEE',
                videos_count INTEGER DEFAULT 0,
                embedding_path TEXT DEFAULT ''
            );
        """)

        # 3. Per-Video Diarized Speakers & Semantic Roles
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_speakers (
                video_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                role TEXT NOT NULL,
                interlocutor_index INTEGER DEFAULT 0,
                profile_id TEXT,
                display_name_en TEXT DEFAULT '',
                display_name_ar TEXT DEFAULT '',
                speaking_sec REAL DEFAULT 0.0,
                turn_count INTEGER DEFAULT 0,
                mean_rms_db REAL DEFAULT -24.0,
                PRIMARY KEY (video_id, speaker_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            );
        """)

        # 4. Chronological Scenario Turns (Full Diarized & Overlap-Resolved Conversation)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scenario_turns (
                turn_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                episode_index INTEGER DEFAULT 1,
                speaker_id TEXT NOT NULL,
                speaker_role TEXT NOT NULL,
                speaker_label_en TEXT DEFAULT '',
                speaker_label_ar TEXT DEFAULT '',
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                start_fmt TEXT NOT NULL,
                end_fmt TEXT NOT NULL,
                language TEXT DEFAULT 'en',
                text_original TEXT NOT NULL,
                text_ar TEXT DEFAULT '',
                is_overlap INTEGER DEFAULT 0,
                overlap_group_id TEXT DEFAULT '',
                is_pedestrian_interruption INTEGER DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                words_json TEXT DEFAULT '[]',
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_video_order ON scenario_turns(video_id, turn_index);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_video_time ON scenario_turns(video_id, start_ms);")

        # 5. FTS5 Full-Text Index on Raw Scenario Turns (if supported by sqlite3 runtime)
        if _supports_fts5(conn):
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS scenario_turns_fts USING fts5(
                    turn_id UNINDEXED,
                    video_id UNINDEXED,
                    speaker_label_en,
                    speaker_label_ar,
                    text_original,
                    text_ar,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
            """)

            conn.execute("DROP TRIGGER IF EXISTS turns_ai;")
            conn.execute("DROP TRIGGER IF EXISTS turns_ad;")
            conn.execute("DROP TRIGGER IF EXISTS turns_au;")

            conn.execute("""
                CREATE TRIGGER turns_ai AFTER INSERT ON scenario_turns BEGIN
                    INSERT INTO scenario_turns_fts(turn_id, video_id, speaker_label_en, speaker_label_ar, text_original, text_ar)
                    VALUES (new.turn_id, new.video_id, new.speaker_label_en, new.speaker_label_ar, new.text_original, new.text_ar);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER turns_ad AFTER DELETE ON scenario_turns BEGIN
                    DELETE FROM scenario_turns_fts WHERE turn_id = old.turn_id;
                END;
            """)
            conn.execute("""
                CREATE TRIGGER turns_au AFTER UPDATE ON scenario_turns BEGIN
                    DELETE FROM scenario_turns_fts WHERE turn_id = old.turn_id;
                    INSERT INTO scenario_turns_fts(turn_id, video_id, speaker_label_en, speaker_label_ar, text_original, text_ar)
                    VALUES (new.turn_id, new.video_id, new.speaker_label_en, new.speaker_label_ar, new.text_original, new.text_ar);
                END;
            """)

        # 6. V2 Dialectical Q&A Dossier Units
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qa_units (
                id TEXT PRIMARY KEY,
                video_id TEXT,
                channel TEXT DEFAULT 'SCDawah Arabic',
                video_title_ar TEXT DEFAULT '',
                youtube_url TEXT DEFAULT '',
                timestamp_url TEXT DEFAULT '',
                timestamp_seconds INTEGER DEFAULT 0,
                timestamp_formatted TEXT DEFAULT '00:00',
                answer_timestamp_seconds INTEGER DEFAULT 0,
                answer_timestamp_formatted TEXT DEFAULT '00:00',
                episode_index INTEGER DEFAULT 1,
                daee_speaker_id TEXT DEFAULT '',
                daee_profile_id TEXT DEFAULT '',
                daee_name_en TEXT DEFAULT '',
                daee_name_ar TEXT DEFAULT '',
                interlocutor_speaker_id TEXT DEFAULT '',
                topic_category TEXT DEFAULT '',
                topic_subcategory TEXT DEFAULT '',
                interlocutor_type TEXT DEFAULT '',
                question_en TEXT DEFAULT '',
                question_ar TEXT DEFAULT '',
                core_answer_summary_en TEXT DEFAULT '',
                core_answer_summary_ar TEXT DEFAULT '',
                logical_premises_json TEXT DEFAULT '[]',
                logical_premises_ar_json TEXT DEFAULT '[]',
                fallacy_or_misconception_exposed TEXT DEFAULT '',
                fallacy_or_misconception_exposed_ar TEXT DEFAULT '',
                analogy_or_metaphor TEXT DEFAULT '',
                analogy_or_metaphor_ar TEXT DEFAULT '',
                socratic_question TEXT DEFAULT '',
                socratic_question_ar TEXT DEFAULT '',
                interlocutor_verbatim_quote TEXT DEFAULT '',
                interlocutor_verbatim_quote_ar TEXT DEFAULT '',
                verbatim_quote TEXT DEFAULT '',
                verbatim_quote_ar TEXT DEFAULT '',
                dialogue_exchange_json TEXT DEFAULT '[]',
                source_turn_ids_json TEXT DEFAULT '[]',
                tags_json TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE SET NULL
            );
        """)

        for col, cdef in [
            ("answer_timestamp_seconds", "INTEGER DEFAULT 0"),
            ("answer_timestamp_formatted", "TEXT DEFAULT '00:00'"),
            ("episode_index", "INTEGER DEFAULT 1"),
            ("daee_speaker_id", "TEXT DEFAULT ''"),
            ("daee_profile_id", "TEXT DEFAULT ''"),
            ("daee_name_en", "TEXT DEFAULT ''"),
            ("daee_name_ar", "TEXT DEFAULT ''"),
            ("interlocutor_speaker_id", "TEXT DEFAULT ''"),
            ("interlocutor_verbatim_quote", "TEXT DEFAULT ''"),
            ("interlocutor_verbatim_quote_ar", "TEXT DEFAULT ''"),
            ("dialogue_exchange_json", "TEXT DEFAULT '[]'"),
            ("source_turn_ids_json", "TEXT DEFAULT '[]'"),
        ]:
            _ensure_column(conn, "qa_units", col, cdef)

        # 7. FTS5 Full-Text Index on Q&A Dossiers (if supported by sqlite3 runtime)
        if _supports_fts5(conn):
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS qa_fts USING fts5(
                    id UNINDEXED,
                    question_en,
                    question_ar,
                    core_answer_summary_en,
                    core_answer_summary_ar,
                    logical_premises,
                    logical_premises_ar,
                    analogy_or_metaphor,
                    analogy_or_metaphor_ar,
                    socratic_question,
                    socratic_question_ar,
                    verbatim_quote,
                    verbatim_quote_ar,
                    tags,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
            """)

            conn.execute("DROP TRIGGER IF EXISTS qa_ai;")
            conn.execute("DROP TRIGGER IF EXISTS qa_ad;")
            conn.execute("DROP TRIGGER IF EXISTS qa_au;")

            conn.execute("""
                CREATE TRIGGER qa_ai AFTER INSERT ON qa_units BEGIN
                    INSERT INTO qa_fts(
                        id, question_en, question_ar, core_answer_summary_en, core_answer_summary_ar,
                        logical_premises, logical_premises_ar, analogy_or_metaphor, analogy_or_metaphor_ar,
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags
                    ) VALUES (
                        new.id, new.question_en, new.question_ar, new.core_answer_summary_en, new.core_answer_summary_ar,
                        new.logical_premises_json, new.logical_premises_ar_json, new.analogy_or_metaphor, new.analogy_or_metaphor_ar,
                        new.socratic_question, new.socratic_question_ar, new.verbatim_quote, new.verbatim_quote_ar, new.tags_json
                    );
                END;
            """)
            conn.execute("""
                CREATE TRIGGER qa_ad AFTER DELETE ON qa_units BEGIN
                    DELETE FROM qa_fts WHERE id = old.id;
                END;
            """)
            conn.execute("""
                CREATE TRIGGER qa_au AFTER UPDATE ON qa_units BEGIN
                    DELETE FROM qa_fts WHERE id = old.id;
                    INSERT INTO qa_fts(
                        id, question_en, question_ar, core_answer_summary_en, core_answer_summary_ar,
                        logical_premises, logical_premises_ar, analogy_or_metaphor, analogy_or_metaphor_ar,
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags
                    ) VALUES (
                        new.id, new.question_en, new.question_ar, new.core_answer_summary_en, new.core_answer_summary_ar,
                        new.logical_premises_json, new.logical_premises_ar_json, new.analogy_or_metaphor, new.analogy_or_metaphor_ar,
                        new.socratic_question, new.socratic_question_ar, new.verbatim_quote, new.verbatim_quote_ar, new.tags_json
                    );
                END;
            """)

            # Synchronize qa_fts if counts diverge
            qa_count = conn.execute("SELECT COUNT(*) FROM qa_units").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM qa_fts").fetchone()[0]
            if qa_count != fts_count:
                conn.execute("DELETE FROM qa_fts;")
                conn.execute("""
                    INSERT INTO qa_fts(
                        id, question_en, question_ar, core_answer_summary_en, core_answer_summary_ar,
                        logical_premises, logical_premises_ar, analogy_or_metaphor, analogy_or_metaphor_ar,
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags
                    )
                    SELECT
                        id, question_en, question_ar, core_answer_summary_en, core_answer_summary_ar,
                        logical_premises_json, logical_premises_ar_json, analogy_or_metaphor, analogy_or_metaphor_ar,
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags_json
                    FROM qa_units;
                """)

        # Seed default speaker profiles
        try:
            from core.metadata_linker import get_default_speaker_profiles
            for prof in get_default_speaker_profiles():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO speaker_profiles (profile_id, name_en, name_ar, role_default)
                    VALUES (?, ?, ?, ?)
                    """,
                    (prof["profile_id"], prof["name_en"], prof["name_ar"], prof["role_default"]),
                )
        except Exception:
            pass

        # Seed canonical videos from manifest if videos table is empty
        if conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0:
            target_dir = (Path(db_path).parent if db_path else DATA_DIR)
            manifest_file = target_dir / "video_manifest.json"
            if not manifest_file.exists():
                manifest_file = BASE_DIR / "data" / "video_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
                    rows = [
                        (
                            v.get("video_id", ""),
                            int(v.get("catalog_order", idx)),
                            v.get("channel", "SCDawah Arabic"),
                            v.get("title_ar", ""),
                            v.get("youtube_url", ""),
                            1 if v.get("is_short") else 0,
                            int(v.get("priority_score", 75)),
                            v.get("primary_daee_hint"),
                            "queued",
                            "queued",
                            "queued",
                        )
                        for idx, v in enumerate(manifest_data, start=1)
                        if v.get("video_id")
                    ]
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO videos (
                            video_id, catalog_order, channel, title_ar, youtube_url,
                            is_short, priority_score, primary_daee_hint,
                            stage1_status, stage2_status, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                except Exception as e:
                    print(f"[-] Note on video manifest seeding: {e}")

    conn.close()


# ----------------------------------------------------------------------
# Idempotent Shard Ingestion (Used by Single-Writer Compactor & Stage 1/2)
# ----------------------------------------------------------------------

def ingest_scenario_payload(scenario: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """Atomically ingests a complete Stage 1 Scenario Transcript JSON into SQLite."""
    vid = scenario["video_id"]
    speakers = scenario.get("speakers", [])
    turns = scenario.get("turns", [])
    metrics = scenario.get("metrics", {})

    spoken_words = sum(len((t.get("text_original") or "").split()) for t in turns)
    ped_turns = sum(1 for t in turns if t.get("is_pedestrian_interruption"))
    episodes_cnt = max([int(t.get("episode_index", 1)) for t in turns], default=1)

    conn = get_db_connection(db_path)
    with conn:
        # Ensure video record exists
        conn.execute(
            """
            INSERT OR IGNORE INTO videos (video_id, channel, title_ar, youtube_url, status)
            VALUES (?, ?, ?, ?, 'stage1_done')
            """,
            (
                vid,
                scenario.get("channel", "SCDawah Arabic"),
                scenario.get("title_ar", ""),
                scenario.get("youtube_url", f"https://www.youtube.com/watch?v={vid}"),
            ),
        )

        conn.execute(
            """
            UPDATE videos SET
                duration_sec = ?,
                detected_language = ?,
                speakers_count = ?,
                episodes_count = ?,
                turns_count = ?,
                spoken_words_count = ?,
                overlap_windows_count = ?,
                overlap_ratio = ?,
                pedestrian_turns_count = ?,
                clean_opus_path = ?,
                rttm_path = ?,
                scenario_json_path = ?,
                stage1_status = 'completed',
                status = CASE WHEN stage2_status = 'completed' THEN 'completed' ELSE 'stage1_done' END,
                updated_at = CURRENT_TIMESTAMP
            WHERE video_id = ?
            """,
            (
                float(scenario.get("duration_sec", metrics.get("duration_sec", 0.0))),
                scenario.get("detected_language", "en"),
                len(speakers),
                episodes_cnt,
                len(turns),
                spoken_words,
                int(metrics.get("overlap_windows_count", 0)),
                float(metrics.get("overlap_ratio", 0.0)),
                ped_turns,
                scenario.get("clean_opus_path", ""),
                scenario.get("rttm_path", ""),
                scenario.get("scenario_json_path", ""),
                vid,
            ),
        )

        conn.execute("DELETE FROM video_speakers WHERE video_id = ?", (vid,))
        for spk in speakers:
            conn.execute(
                """
                INSERT INTO video_speakers (
                    video_id, speaker_id, role, interlocutor_index, profile_id,
                    display_name_en, display_name_ar, speaking_sec, turn_count, mean_rms_db
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vid,
                    spk["speaker_id"],
                    spk.get("role", "INTERLOCUTOR"),
                    int(spk.get("interlocutor_index") or 0),
                    spk.get("profile_id"),
                    spk.get("display_name_en", spk["speaker_id"]),
                    spk.get("display_name_ar", spk["speaker_id"]),
                    float(spk.get("speaking_sec", 0.0)),
                    int(spk.get("turn_count", 0)),
                    float(spk.get("mean_rms_db", -24.0)),
                ),
            )

        conn.execute("DELETE FROM scenario_turns WHERE video_id = ?", (vid,))
        for idx, t in enumerate(turns):
            turn_id = t.get("turn_id") or f"{vid}_T{idx + 1:04d}"
            conn.execute(
                """
                INSERT INTO scenario_turns (
                    turn_id, video_id, turn_index, episode_index, speaker_id, speaker_role,
                    speaker_label_en, speaker_label_ar, start_ms, end_ms, start_fmt, end_fmt,
                    language, text_original, text_ar, is_overlap, overlap_group_id,
                    is_pedestrian_interruption, confidence, words_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    vid,
                    int(t.get("turn_index", idx)),
                    int(t.get("episode_index", 1)),
                    t.get("speaker_id", "SPK_00"),
                    t.get("speaker_role", "INTERLOCUTOR"),
                    t.get("speaker_label_en", ""),
                    t.get("speaker_label_ar", ""),
                    int(t.get("start_ms", 0)),
                    int(t.get("end_ms", 0)),
                    t.get("start_fmt", "00:00"),
                    t.get("end_fmt", "00:00"),
                    t.get("language", "en"),
                    t.get("text_original", ""),
                    t.get("text_ar", ""),
                    1 if t.get("is_overlap") else 0,
                    t.get("overlap_group_id", ""),
                    1 if t.get("is_pedestrian_interruption") else 0,
                    float(t.get("confidence", 1.0)),
                    json.dumps(t.get("words", []), ensure_ascii=False),
                ),
            )

        # Refresh speaker_profiles video counts
        conn.execute("""
            UPDATE speaker_profiles
            SET videos_count = (
                SELECT COUNT(DISTINCT video_id) FROM video_speakers
                WHERE video_speakers.profile_id = speaker_profiles.profile_id
            );
        """)
    conn.close()


def insert_qa_unit(unit: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    conn = get_db_connection(db_path)
    with conn:
        _insert_qa_unit_in_tx(conn, unit)
    conn.close()


def _insert_qa_unit_in_tx(conn: sqlite3.Connection, unit: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO qa_units (
            id, video_id, channel, video_title_ar, youtube_url, timestamp_url,
            timestamp_seconds, timestamp_formatted, answer_timestamp_seconds, answer_timestamp_formatted,
            episode_index, daee_speaker_id, daee_profile_id, daee_name_en, daee_name_ar,
            interlocutor_speaker_id, topic_category, topic_subcategory, interlocutor_type,
            question_en, question_ar, core_answer_summary_en, core_answer_summary_ar,
            logical_premises_json, logical_premises_ar_json,
            fallacy_or_misconception_exposed, fallacy_or_misconception_exposed_ar,
            analogy_or_metaphor, analogy_or_metaphor_ar,
            socratic_question, socratic_question_ar,
            interlocutor_verbatim_quote, interlocutor_verbatim_quote_ar,
            verbatim_quote, verbatim_quote_ar,
            dialogue_exchange_json, source_turn_ids_json, tags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit.get("id"),
            unit.get("video_id"),
            unit.get("channel", "SCDawah Arabic"),
            unit.get("video_title_ar", ""),
            unit.get("youtube_url", ""),
            unit.get("timestamp_url", ""),
            int(unit.get("timestamp_seconds", 0)),
            unit.get("timestamp_formatted", "00:00"),
            int(unit.get("answer_timestamp_seconds", unit.get("timestamp_seconds", 0))),
            unit.get("answer_timestamp_formatted", unit.get("timestamp_formatted", "00:00")),
            int(unit.get("episode_index", 1)),
            unit.get("daee_speaker_id", ""),
            unit.get("daee_profile_id", ""),
            unit.get("daee_name_en", ""),
            unit.get("daee_name_ar", ""),
            unit.get("interlocutor_speaker_id", ""),
            unit.get("topic_category", "Theology & God's Existence"),
            unit.get("topic_subcategory", ""),
            unit.get("interlocutor_type", "Inquiring Seeker"),
            unit.get("question_en", ""),
            unit.get("question_ar", ""),
            unit.get("core_answer_summary_en", ""),
            unit.get("core_answer_summary_ar", ""),
            json.dumps(unit.get("logical_premises", []), ensure_ascii=False),
            json.dumps(unit.get("logical_premises_ar", []), ensure_ascii=False),
            unit.get("fallacy_or_misconception_exposed", ""),
            unit.get("fallacy_or_misconception_exposed_ar", ""),
            unit.get("analogy_or_metaphor", ""),
            unit.get("analogy_or_metaphor_ar", ""),
            unit.get("socratic_question", ""),
            unit.get("socratic_question_ar", ""),
            unit.get("interlocutor_verbatim_quote", ""),
            unit.get("interlocutor_verbatim_quote_ar", ""),
            unit.get("verbatim_quote", ""),
            unit.get("verbatim_quote_ar", ""),
            json.dumps(unit.get("dialogue_exchange", []), ensure_ascii=False),
            json.dumps(unit.get("source_turn_ids", []), ensure_ascii=False),
            json.dumps(unit.get("tags", []), ensure_ascii=False),
        ),
    )


def ingest_extraction_payload(video_id: str, units: List[Dict[str, Any]], db_path: Optional[Path] = None) -> None:
    """Atomically replaces a video's extracted Q&A units from its completed Stage 2 shard."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM qa_units WHERE video_id = ?", (video_id,))
        for u in units:
            _insert_qa_unit_in_tx(conn, u)
        conn.execute(
            """
            UPDATE videos
            SET qa_count = ?, stage2_status = 'completed', status = 'completed', updated_at = CURRENT_TIMESTAMP
            WHERE video_id = ?
            """,
            (len(units), video_id),
        )
    conn.close()


# ----------------------------------------------------------------------
# Bilingual Concept Expansion & Search Engine
# ----------------------------------------------------------------------

BILINGUAL_CONCEPT_SYNONYMS = {
    "حجاب": ["حجاب", "الحجاب", "لباس", "ستر", "المرأة", "نساء", "شريعة", "hijab", "modesty", "women"],
    "مرأه": ["المرأة", "مرأة", "نساء", "زوجات", "حجاب", "ميراث", "قوامة", "women", "female", "wives"],
    "مرأتان": ["المرأة", "مرأة", "نساء", "شهادة", "ميراث", "women"],
    "تعدد": ["تعدد", "زوجات", "المرأة", "زواج", "polygamy", "wives", "marriage"],
    "عبوديه": ["عبودية", "ملك اليمين", "رق", "سبايا", "slavery", "concubines"],
    "رده": ["ردة", "المرتد", "حرية الاعتقاد", "إكراه", "apostasy", "freedom"],
    "شر": ["الشر", "معاناة", "ألم", "ابتلاء", "مصائب", "evil", "suffering", "pain"],
    "الحاد": ["إلحاد", "ملحد", "مادي", "صدفة", "الكون", "سببية", "atheism", "atheist", "cause"],
    "تطور": ["تطور", "داروين", "خلق", "علم", "evolution", "darwin", "science"],
    "ثالوث": ["ثالوث", "تثليث", "أقانيم", "المسيح", "يسوع", "ابن الله", "trinity", "jesus"],
    "صلب": ["صلب", "فداء", "خطيئة", "المسيح", "يسوع", "crucifixion", "atonement"],
    "تحريف": ["تحريف", "مخطوطات", "إنجيل", "عهد", "قرآن", "حفظ", "preservation", "manuscripts", "bible"],
}


def _normalize_arabic_variants(token: str) -> List[str]:
    variants = {token}
    for prefix in ("وال", "فال", "بال", "كال", "لال", "ال"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            variants.add(token[len(prefix):])
    alef_norm = re.sub(r"[أإآ]", "ا", token)
    variants.add(alef_norm)
    for prefix in ("وال", "فال", "بال", "كال", "لال", "ال"):
        if alef_norm.startswith(prefix) and len(alef_norm) - len(prefix) >= 2:
            variants.add(alef_norm[len(prefix):])

    extra = set()
    for v in variants:
        if v.endswith("ة"):
            extra.add(v[:-1] + "ه")
        elif v.endswith("ه"):
            extra.add(v[:-1] + "ة")
    variants.update(extra)

    stem_key = alef_norm
    for prefix in ("وال", "فال", "بال", "كال", "لال", "ال"):
        if stem_key.startswith(prefix) and len(stem_key) - len(prefix) >= 2:
            stem_key = stem_key[len(prefix):]
            break
    stem_key_norm = stem_key[:-1] + "ه" if stem_key.endswith("ة") else stem_key
    if stem_key_norm in BILINGUAL_CONCEPT_SYNONYMS:
        variants.update(BILINGUAL_CONCEPT_SYNONYMS[stem_key_norm])

    return [v for v in variants if len(v) >= 2]


_SEARCH_STOPWORDS = {
    "من", "في", "على", "عن", "إلى", "الى", "ما", "لا", "أو", "او", "هو", "هي",
    "ان", "أن", "إن", "كان", "هل", "مع", "كل", "هذا", "هذه", "ذلك", "التي", "الذي",
    "the", "and", "for", "with", "from", "that", "this", "what", "who", "how", "why",
    "in", "of", "to", "is", "or", "on", "at", "an", "by", "as", "be", "are", "was",
}


def _extract_meaningful_tokens(cleaned_query: str, min_len: int = 2) -> List[str]:
    tokens = [t.strip() for t in cleaned_query.split() if len(t.strip()) >= min_len]
    non_stop = [t for t in tokens if t.lower() not in _SEARCH_STOPWORDS]
    return non_stop if non_stop else tokens


def sanitize_fts_query(query: str) -> str:
    cleaned = re.sub(r'[*":^()\[\]{}~/-]', " ", query)
    raw_tokens = _extract_meaningful_tokens(cleaned, min_len=2)
    if not raw_tokens:
        return ""
    all_variants = []
    for token in raw_tokens:
        for v in _normalize_arabic_variants(token):
            if v not in all_variants:
                all_variants.append(v)
    return " OR ".join(f'"{v}"*' for v in all_variants)


def _build_search_where(
    query: str = "",
    category: Optional[str] = None,
    interlocutor: Optional[str] = None,
    speaker: Optional[str] = None,
    video_id: Optional[str] = None,
):
    sql_parts = []
    params = []

    if query and query.strip():
        cleaned = re.sub(r'[*":^()\[\]{}~/-]', " ", query.strip())
        raw_tokens = _extract_meaningful_tokens(cleaned, min_len=2) or [query.strip()]

        all_variants = []
        for token in raw_tokens:
            for v in _normalize_arabic_variants(token):
                if v not in all_variants:
                    all_variants.append(v)

        sub_clauses = []
        fts_query = sanitize_fts_query(query)
        if fts_query and _supports_fts5():
            sub_clauses.append("qa_units.id IN (SELECT id FROM qa_fts WHERE qa_fts MATCH ?)")
            params.append(fts_query)

        searchable_cols = [
            "question_en", "question_ar",
            "core_answer_summary_en", "core_answer_summary_ar",
            "logical_premises_json", "logical_premises_ar_json",
            "fallacy_or_misconception_exposed", "fallacy_or_misconception_exposed_ar",
            "analogy_or_metaphor", "analogy_or_metaphor_ar",
            "socratic_question", "socratic_question_ar",
            "interlocutor_verbatim_quote", "interlocutor_verbatim_quote_ar",
            "verbatim_quote", "verbatim_quote_ar",
            "daee_name_en", "daee_name_ar",
            "tags_json", "topic_category", "topic_subcategory", "video_title_ar",
        ]
        for variant in all_variants:
            like_val = f"%{variant}%"
            col_ors = " OR ".join([f"{col} LIKE ?" for col in searchable_cols])
            sub_clauses.append(f"({col_ors})")
            params.extend([like_val] * len(searchable_cols))

        sql_parts.append("(" + " OR ".join(sub_clauses) + ")")

    if category and category != "all":
        sql_parts.append("topic_category = ?")
        params.append(category)

    if interlocutor and interlocutor != "all":
        sql_parts.append("interlocutor_type = ?")
        params.append(interlocutor)

    if speaker and speaker != "all":
        sql_parts.append("(daee_profile_id = ? OR daee_name_en = ?)")
        params.extend([speaker, speaker])

    if video_id and video_id != "all":
        sql_parts.append("video_id = ?")
        params.append(video_id)

    where_clause = " AND ".join(sql_parts) if sql_parts else "1=1"
    return where_clause, params


def count_qa(
    query: str = "",
    category: Optional[str] = None,
    interlocutor: Optional[str] = None,
    speaker: Optional[str] = None,
    video_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    conn = get_db_connection(db_path)
    where_clause, params = _build_search_where(query, category, interlocutor, speaker, video_id)
    total = conn.execute(f"SELECT COUNT(*) FROM qa_units WHERE {where_clause}", params).fetchone()[0]
    conn.close()
    return total


def _hydrate_qa_row(row: sqlite3.Row, serial_num: int) -> Dict[str, Any]:
    d = dict(row)
    d["serial_number"] = serial_num
    for json_col, out_key in [
        ("logical_premises_json", "logical_premises"),
        ("logical_premises_ar_json", "logical_premises_ar"),
        ("dialogue_exchange_json", "dialogue_exchange"),
        ("source_turn_ids_json", "source_turn_ids"),
        ("tags_json", "tags"),
    ]:
        try:
            d[out_key] = json.loads(d.get(json_col) or "[]")
        except Exception:
            d[out_key] = []
    return d


def search_qa(
    query: str = "",
    category: Optional[str] = None,
    interlocutor: Optional[str] = None,
    speaker: Optional[str] = None,
    video_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort: str = "default",
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    where_clause, params = _build_search_where(query, category, interlocutor, speaker, video_id)
    order_map = {
        "default": "rowid ASC",
        "timestamp_asc": "timestamp_seconds ASC",
        "timestamp_desc": "timestamp_seconds DESC",
        "category": "topic_category ASC, timestamp_seconds ASC",
    }
    order_clause = order_map.get(sort, "rowid ASC")
    sql = f"SELECT * FROM qa_units WHERE {where_clause} ORDER BY {order_clause} LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [_hydrate_qa_row(r, offset + i + 1) for i, r in enumerate(rows)]


def get_categories_breakdown(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    rows = conn.execute("""
        SELECT topic_category, COUNT(*) as count
        FROM qa_units
        WHERE topic_category IS NOT NULL AND topic_category != ''
        GROUP BY topic_category
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_interlocutor_breakdown(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    rows = conn.execute("""
        SELECT interlocutor_type, COUNT(*) as count
        FROM qa_units
        WHERE interlocutor_type IS NOT NULL AND interlocutor_type != ''
        GROUP BY interlocutor_type
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_speakers_breakdown(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns identified Da'ee speaker breakdown across extracted Q&As and diarized videos."""
    conn = get_db_connection(db_path)
    rows = conn.execute("""
        SELECT
            COALESCE(NULLIF(daee_profile_id, ''), daee_name_en) AS speaker_key,
            MAX(daee_name_en) AS name_en,
            MAX(daee_name_ar) AS name_ar,
            COUNT(*) AS qa_count,
            COUNT(DISTINCT video_id) AS videos_count
        FROM qa_units
        WHERE (daee_profile_id IS NOT NULL AND daee_profile_id != '')
           OR (daee_name_en IS NOT NULL AND daee_name_en != '')
        GROUP BY speaker_key
        ORDER BY qa_count DESC
    """).fetchall()

    results = [dict(r) for r in rows]
    if not results:
        # Fallback to stage-1 diarized video_speakers if stage-2 hasn't populated qa_units yet
        vs_rows = conn.execute("""
            SELECT
                COALESCE(NULLIF(profile_id, ''), display_name_en) AS speaker_key,
                MAX(display_name_en) AS name_en,
                MAX(display_name_ar) AS name_ar,
                0 AS qa_count,
                COUNT(DISTINCT video_id) AS videos_count
            FROM video_speakers
            WHERE role IN ('MAIN_DAEE', 'CO_DAEE')
              AND display_name_en != ''
            GROUP BY speaker_key
            ORDER BY videos_count DESC
        """).fetchall()
        results = [dict(r) for r in vs_rows]

    conn.close()
    return results


def list_videos(
    query: str = "",
    speaker: Optional[str] = None,
    only_diarized: bool = False,
    limit: int = 30,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    clauses = []
    params: List[Any] = []

    if only_diarized:
        clauses.append("v.stage1_status = 'completed'")
    if query and query.strip():
        clauses.append("(v.title_ar LIKE ? OR v.video_id LIKE ?)")
        q_like = f"%{query.strip()}%"
        params.extend([q_like, q_like])
    if speaker and speaker != "all":
        clauses.append("""
            (v.primary_daee_hint = ? OR EXISTS (
                SELECT 1 FROM video_speakers vs
                WHERE vs.video_id = v.video_id AND (vs.profile_id = ? OR vs.display_name_en = ?)
            ))
        """)
        params.extend([speaker, speaker, speaker])

    where_sql = " AND ".join(clauses) if clauses else "1=1"
    total = conn.execute(f"SELECT COUNT(*) FROM videos v WHERE {where_sql}", params).fetchone()[0]

    sql = f"""
        SELECT v.*
        FROM videos v
        WHERE {where_sql}
        ORDER BY
            CASE WHEN v.stage1_status = 'completed' THEN 0 ELSE 1 END ASC,
            v.qa_count DESC,
            v.priority_score DESC,
            v.catalog_order ASC
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(sql, params + [limit, offset]).fetchall()
    conn.close()
    return {"videos": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}


def get_video_scenario(
    video_id: str,
    include_pedestrians: bool = True,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    v_row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if not v_row:
        conn.close()
        return None

    speakers = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM video_speakers WHERE video_id = ? ORDER BY speaking_sec DESC",
            (video_id,),
        ).fetchall()
    ]

    turn_sql = "SELECT * FROM scenario_turns WHERE video_id = ?"
    if not include_pedestrians:
        turn_sql += " AND is_pedestrian_interruption = 0"
    turn_sql += " ORDER BY turn_index ASC, start_ms ASC"

    turns = []
    for r in conn.execute(turn_sql, (video_id,)).fetchall():
        td = dict(r)
        try:
            td["words"] = json.loads(td.get("words_json") or "[]")
        except Exception:
            td["words"] = []
        td.pop("words_json", None)
        turns.append(td)

    qa_rows = conn.execute(
        "SELECT * FROM qa_units WHERE video_id = ? ORDER BY timestamp_seconds ASC",
        (video_id,),
    ).fetchall()
    qas = [_hydrate_qa_row(r, idx + 1) for idx, r in enumerate(qa_rows)]
    conn.close()

    return {
        "video": dict(v_row),
        "speakers": speakers,
        "turns": turns,
        "qa_units": qas,
    }


def search_scenario_turns(
    query: str,
    role: Optional[str] = None,
    limit: int = 40,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not query or not query.strip():
        return {"turns": [], "total": 0}

    conn = get_db_connection(db_path)
    fts_q = sanitize_fts_query(query)
    clauses = []
    params: List[Any] = []

    if fts_q and _supports_fts5(conn):
        clauses.append("st.turn_id IN (SELECT turn_id FROM scenario_turns_fts WHERE scenario_turns_fts MATCH ?)")
        params.append(fts_q)
    else:
        clauses.append("(st.text_original LIKE ? OR st.text_ar LIKE ?)")
        params.extend([f"%{query.strip()}%", f"%{query.strip()}%"])

    if role and role != "all":
        clauses.append("st.speaker_role = ?")
        params.append(role)

    where_sql = " AND ".join(clauses)
    total = conn.execute(f"SELECT COUNT(*) FROM scenario_turns st WHERE {where_sql}", params).fetchone()[0]
    rows = conn.execute(
        f"""
        SELECT st.*, v.title_ar AS video_title_ar, v.youtube_url AS youtube_url
        FROM scenario_turns st
        JOIN videos v ON v.video_id = st.video_id
        WHERE {where_sql}
        ORDER BY st.start_ms ASC
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d.pop("words_json", None)
        sec = int(d.get("start_ms", 0) // 1000)
        yt = d.get("youtube_url") or f"https://www.youtube.com/watch?v={d['video_id']}"
        d["timestamp_url"] = f"{yt}&t={sec}s"
        results.append(d)

    return {"turns": results, "total": total, "limit": limit, "offset": offset}


def get_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    total_qa = conn.execute("SELECT COUNT(*) FROM qa_units").fetchone()[0]
    total_videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    stage1_completed = conn.execute("SELECT COUNT(*) FROM videos WHERE stage1_status = 'completed'").fetchone()[0]
    stage2_completed = conn.execute("SELECT COUNT(*) FROM videos WHERE stage2_status = 'completed'").fetchone()[0]
    shorts_count = conn.execute("SELECT COUNT(*) FROM videos WHERE is_short = 1").fetchone()[0]
    categories_count = conn.execute("SELECT COUNT(DISTINCT topic_category) FROM qa_units WHERE topic_category != ''").fetchone()[0]
    interlocutors_count = conn.execute("SELECT COUNT(DISTINCT interlocutor_type) FROM qa_units WHERE interlocutor_type != ''").fetchone()[0]
    extracted_videos_count = conn.execute("SELECT COUNT(DISTINCT video_id) FROM qa_units WHERE video_id IS NOT NULL AND video_id != ''").fetchone()[0]
    total_turns = conn.execute("SELECT COUNT(*) FROM scenario_turns").fetchone()[0]
    total_spoken_words = conn.execute("SELECT COALESCE(SUM(spoken_words_count), 0) FROM videos").fetchone()[0]
    total_duration_sec = conn.execute("SELECT COALESCE(SUM(duration_sec), 0) FROM videos WHERE stage1_status = 'completed'").fetchone()[0]
    overlaps_resolved = conn.execute("SELECT COUNT(DISTINCT overlap_group_id) FROM scenario_turns WHERE is_overlap = 1 AND overlap_group_id != ''").fetchone()[0]
    pedestrian_isolated = conn.execute("SELECT COUNT(*) FROM scenario_turns WHERE is_pedestrian_interruption = 1").fetchone()[0]
    speakers_identified = conn.execute("SELECT COUNT(DISTINCT COALESCE(NULLIF(profile_id, ''), video_id || ':' || speaker_id)) FROM video_speakers WHERE role != 'PEDESTRIAN'").fetchone()[0]
    conn.close()

    return {
        "pipeline_version": "2.0-diarized-scenario",
        "total_qa": total_qa,
        "total_videos": total_videos,
        "shorts_count": shorts_count,
        "full_debates_count": max(0, total_videos - shorts_count),
        "stage1_completed": stage1_completed,
        "stage2_completed": stage2_completed,
        "extracted_videos_count": extracted_videos_count,
        "categories_count": categories_count,
        "interlocutors_count": interlocutors_count,
        "total_turns": total_turns,
        "total_spoken_words": int(total_spoken_words),
        "cleaned_hours": round(float(total_duration_sec) / 3600.0, 2),
        "overlaps_resolved": overlaps_resolved,
        "pedestrian_isolated": pedestrian_isolated,
        "speakers_identified": speakers_identified,
    }
