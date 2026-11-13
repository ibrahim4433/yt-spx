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
                analogies_json TEXT DEFAULT '[]',
                analogies_ar_json TEXT DEFAULT '[]',
                scriptural_citations_json TEXT DEFAULT '[]',
                dialogue_trajectory_turns_count INTEGER DEFAULT 0,
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
            ("analogies_json", "TEXT DEFAULT '[]'"),
            ("analogies_ar_json", "TEXT DEFAULT '[]'"),
            ("scriptural_citations_json", "TEXT DEFAULT '[]'"),
            ("dialogue_trajectory_turns_count", "INTEGER DEFAULT 0"),
        ]:
            _ensure_column(conn, "qa_units", col, cdef)

        # 7. FTS5 Full-Text Index on Q&A Dossiers (if supported by sqlite3 runtime)
        if _supports_fts5(conn):
            try:
                # Upgrade existing qa_fts table if missing new columns
                cols_info = conn.execute("PRAGMA table_info(qa_fts)").fetchall()
                col_names = [c[1] for c in cols_info]
                if col_names and ("analogies" not in col_names or "scriptural_citations" not in col_names):
                    conn.execute("DROP TABLE IF EXISTS qa_fts;")
            except Exception:
                pass

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
                    analogies,
                    scriptural_citations,
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
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags,
                        analogies, scriptural_citations
                    ) VALUES (
                        new.id, new.question_en, new.question_ar, new.core_answer_summary_en, new.core_answer_summary_ar,
                        new.logical_premises_json, new.logical_premises_ar_json, new.analogy_or_metaphor, new.analogy_or_metaphor_ar,
                        new.socratic_question, new.socratic_question_ar, new.verbatim_quote, new.verbatim_quote_ar, new.tags_json,
                        (new.analogies_json || ' ' || coalesce(new.analogies_ar_json, '')), new.scriptural_citations_json
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
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags,
                        analogies, scriptural_citations
                    ) VALUES (
                        new.id, new.question_en, new.question_ar, new.core_answer_summary_en, new.core_answer_summary_ar,
                        new.logical_premises_json, new.logical_premises_ar_json, new.analogy_or_metaphor, new.analogy_or_metaphor_ar,
                        new.socratic_question, new.socratic_question_ar, new.verbatim_quote, new.verbatim_quote_ar, new.tags_json,
                        (new.analogies_json || ' ' || coalesce(new.analogies_ar_json, '')), new.scriptural_citations_json
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
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags,
                        analogies, scriptural_citations
                    )
                    SELECT
                        id, question_en, question_ar, core_answer_summary_en, core_answer_summary_ar,
                        logical_premises_json, logical_premises_ar_json, analogy_or_metaphor, analogy_or_metaphor_ar,
                        socratic_question, socratic_question_ar, verbatim_quote, verbatim_quote_ar, tags_json,
                        (analogies_json || ' ' || coalesce(analogies_ar_json, '')), scriptural_citations_json
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


_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]")

_PHILOSOPHICAL_VOCABULARY = [
    # Multi-word theological/philosophical phrases
    ("two natures and two wills", "طبيعتين وإرادتين"),
    ("two natures", "طبيعتين"),
    ("two wills", "إرادتين"),
    ("first cause", "سبب أول"),
    ("moral ontology", "الأساس الوجودي للأخلاق"),
    ("objective morality", "الأخلاق الموضوعية"),
    ("hypostatic union", "الاتحاد الأقنومي"),
    ("dead sea scrolls", "مخطوطات البحر الميت"),
    ("old testament", "العهد القديم"),
    ("new testament", "العهد الجديد"),
    
    # Singular and plural theological/philosophical vocabulary
    ("natures", "طبائع"),
    ("nature", "طبيعة"),
    ("wills", "إرادات"),
    ("will", "إرادة"),
    ("two", "اثنين"),
    ("dependence", "افتقار واحتياج"),
    ("dependent", "مفتقر ومحتاج"),
    ("independency", "استغناء"),
    ("independent", "مستغنٍ"),
    ("existed", "وُجد"),
    ("exist", "يوجد"),
    ("existence", "وجود"),
    ("contingency", "إمكان الوجود"),
    ("contingent", "ممكن الوجود"),
    ("necessary", "واجب الوجود"),
    ("causes", "أسباب أولى"),
    ("cause", "سبب أول"),
    ("causality", "السببية"),
    ("essence", "جوهر"),
    ("essences", "جواهر"),
    ("substance", "جوهر"),
    ("substances", "جواهر"),
    ("person", "أقنوم"),
    ("persons", "أقانيم"),
    ("omniscience", "العلم الكلي الشامل"),
    ("omniscient", "العليم بكل شيء"),
    ("omnipotence", "القدرة المطلقة"),
    ("omnipotent", "القدير على كل شيء"),
    ("omnipresence", "الإحاطة بكل مكان"),
    ("omnipresent", "محيط بكل مكان"),
    ("transcendence", "التنزيه والتعالي"),
    ("transcendent", "منزه ومتعالٍ"),
    ("anthropomorphism", "التجسيم والتشبيه"),
    ("monotheism", "التوحيد"),
    ("polytheism", "الشرك وتعدد الآلهة"),
    ("pantheism", "وحدة الوجود"),
    ("divine", "إلهي"),
    ("human", "بشري"),
    ("instinctive", "فطري"),
    ("instinct", "فطرة"),
    ("incest", "زنا المحارم"),
    ("pedophilia", "البيدوفيليا (استغلال الأطفال)"),
    ("condemn", "يُدين"),
    ("reduced", "مختزل"),
    ("nihilists", "العدميون"),
    ("nihilist", "عدمي"),
    ("nihilism", "العدمية"),
    ("theodicy", "العدل الإلهي ومعضلة الشر"),
    ("hermeneutics", "علم التأويل"),
    ("inerrancy", "عصمة النص"),
    ("preservation", "حفظ النص"),
    ("corruption", "التحريف"),
    ("interpolation", "الإقحام والتحريف"),
    ("syllogism", "قياس برهاني"),
    ("premises", "مقدمات"),
    ("premise", "مقدمة"),
    ("axioms", "بديهيات عقلية"),
    ("axiom", "بديهية عقلية"),
    ("morality", "الأخلاق"),
    ("moral", "أخلاقي"),
    ("miracles", "المعجزات"),
    ("machines", "الآلات"),
    ("messenger", "رسول"),
    ("scribes", "النساخ"),
    ("bikinis", "لباس البحر الفاضح"),
]

_SCRIPT_REPLACEMENTS = [
    ("ت дисквалифицируют его как совершенный божественный текст", "تسقط عنه صفة النص الإلهي الكامل المعصوم"),
    ("дисквалифицируют его как совершенный божественный текст", "تسقط عنه صفة النص الإلهي الكامل المعصوم"),
    ("педوفيليا", "البيدوفيليا (استغلال الأطفال)"),
    ("따라서،", "لذلك،"), ("따라서", "لذلك"), ("여전히", "لا تزال"),
    ("المiracles", "المعجزات"), ("فالمachines", "فالآلات"), ("كمessenger", "كرسول"),
    ("للمonotheism", "للتوحيد"), ("السcribes", "النساخ"), ("للسcribes", "للنساخ"),
    ("نihilist", "عدمي"), ("النihilists", "العدميون"), ("بprocesseات", "بعمليات"),
    ("والInstinctive", "والفطري"), ("الInstinctive", "الفطري"), ("Instinctive", "فطري"),
    ("Instinct", "فطرة"), ("Reduced", "مختزلاً"), ("reduced", "مختزل"),
    ("condemn", "يُدين"),
    ("والincest", "وزنا المحارم"),
    ("الincest", "زنا المحارم"),
    ("incest", "زنا المحارم"),
    ("two natures and two wills", "طبيعتين وإرادتين"),
    ("two natures", "طبيعتين"),
    ("two wills", "إرادتين"),
    ("bikinis", "لباس البحر الفاضح"),
    ("والpedophilia", "والبيدوفيليا (استغلال الأطفال)"),
    ("الpedophilia", "البيدوفيليا (استغلال الأطفال)"),
    ("pedophilia", "البيدوفيليا (استغلال الأطفال)"),
    ("Hermeneutics", "علم التأويل"), ("Hagia Sophia", "آيا صوفيا"),
    ("Muawiyah", "معاوية"), ("Hassan", "الحسن"), ("omnipresent", "محيط بكل مكان"),
    ("SUBJX", "الفاعل"), ("rijal", "علم الرجال"),
    # Classical Arabic apologetics glossary repairs (domain-blind machine translation drift)
    ("للיהודים", "لليهود"),
    ("الיהודים", "اليهود"),
    ("יהודים", "يهود"),
    ("في القرن السادس الميلادي في عربة", "في القرن السادس الميلادي في الجزيرة العربية"),
    ("في القرن السابع الميلادي في عربة", "في القرن السابع الميلادي في الجزيرة العربية"),
    ("في القرن السادس في عربة", "في القرن السادس في الجزيرة العربية"),
    ("في القرن السابع في عربة", "في القرن السابع في الجزيرة العربية"),
    ("في عربة في القرن السادس", "في الجزيرة العربية في القرن السادس"),
    ("في عربة في القرن السابع", "في الجزيرة العربية في القرن السابع"),
    ("في عربة القرن السادس", "في الجزيرة العربية في القرن السادس"),
    ("في عربة القرن السابع", "في الجزيرة العربية في القرن السابع"),
    ("عربة في القرن السادس", "الجزيرة العربية في القرن السادس"),
    ("عربة في القرن السابع", "الجزيرة العربية في القرن السابع"),
    ("عربة في القرن", "الجزيرة العربية في القرن"),
    ("الصدف والمؤامرات", "الصدف والتزامنات"),
    ("الصدف والموامرات", "الصدف والتزامنات"),
    ("والموامرات", "والتزامنات"),
    ("الموامرات", "التزامنات"),
    ("المؤامرات النفسية", "التزامنات النفسية"),
    ("المؤامرات غير السببية", "التزامنات غير السببية"),
    ("البوذية والطائفة", "البوذية والطاوية"),
    ("الطائفة والكونفوشيوسية", "الطاوية والكونفوشيوسية"),
    ("والطائفة و", "والطاوية و"),
    ("فلسفة الطائفة", "فلسفة الطاوية"),
    ("ديانة الطائفة", "الديانة الطاوية"),
    ("مذهب الطائفة", "المذهب الطاوي"),
    ("الرسول بول", "بولس الرسول"),
    ("بول الرسول", "بولس الرسول"),
    ("رسائل بول", "رسائل بولس"),
    ("إنجيل ماتيو", "إنجيل متى"),
    ("إنجيل مارك", "إنجيل مرقس"),
    ("إنجيل لوك", "إنجيل لوقا"),
    ("إنجيل جون", "إنجيل يوحنا"),
    ("القديس بيتر", "القديس بطرس"),
    ("سمعان بيتر", "سمعان بطرس"),
    ("الرسول جيمس", "الرسول يعقوب"),
    ("。，？，。", "؟"),
    ("。", "."),
    ("，", "،"),
    ("？", "؟"),
    ("！", "!"),
]


def _is_refutation_context(text: str, match_start: int) -> bool:
    """Checks whether a matched theological statement is quoted as a refutation target or is negated."""
    window = text[max(0, match_start - 80):match_start].lower()
    refutation_markers = [
        "refut", "reject", "deni", "deny", "false", "fallacy", "misconception",
        "claim", "dogma", "notion", "myth", "alleg", "argument that", "oppose",
        "disprov", "counter", "refuting", "rejecting", "denying", "refutation",
        "زعم", "دعوى", "شبهة", "باطل", "بطلان", "رد", "دحض", "إبطال", "نفي", "إنكار",
        "ليس", "لا يمكن", "يستحيل", "كيف يكون", "كيف يعقل", "تناقض", "مغالطة"
    ]
    return any(marker in window for marker in refutation_markers)


def clean_latin_tokens_in_arabic(text: str) -> str:
    """Replaces leaked Latin/English words in Arabic fields using philosophical vocabulary and cleans artifacts."""
    if not text:
        return ""
    val = text

    # 1. Exact script replacements and overrides first
    for old_s, new_s in _SCRIPT_REPLACEMENTS:
        if old_s in val:
            val = val.replace(old_s, new_s)

    # 2. Arabic prefix mapping: (prefix_if_ar_term_has_al, prefix_if_ar_term_lacks_al)
    prefix_map = {
        "وال": ("و", "وال"),
        "فال": ("ف", "فال"),
        "بال": ("ب", "بال"),
        "كال": ("ك", "كال"),
        "ال": ("", "ال"),
        "و": ("و", "و"),
        "ف": ("ف", "ف"),
        "ب": ("ب", "ب"),
        "ك": ("ك", "ك"),
        "ل": ("ل", "ل"),
    }

    for eng_term, ar_term in _PHILOSOPHICAL_VOCABULARY:
        has_al = (
            ar_term.startswith("ال")
            or ar_term.startswith("الأ")
            or ar_term.startswith("الإ")
            or ar_term.startswith("الآ")
        )

        # Handle prefixed Latin terms (e.g., والInstinctive -> والفطري, والcausality -> والسببية)
        for p_ar, (with_al, without_al) in prefix_map.items():
            pattern_pre = r'(?<![\w\u0600-\u06ff])' + re.escape(p_ar) + re.escape(eng_term) + r'\b'
            replacement = (with_al + ar_term) if has_al else (without_al + ar_term)
            val = re.sub(pattern_pre, replacement, val, flags=re.IGNORECASE)

        # Standalone term
        pattern = r'\b' + re.escape(eng_term) + r'\b'
        val = re.sub(pattern, ar_term, val, flags=re.IGNORECASE)

    # 3. Handle known compound Latin-Arabic splices
    val = re.sub(r'بيرمINGTON', 'برمنغهام', val, flags=re.IGNORECASE)
    val = re.sub(r'فutility', 'عبثية وعدم جدوى', val, flags=re.IGNORECASE)

    # 3B. Targeted regex replacements for historical/apologetic terms
    val = re.sub(r'(القرن\s+(?:السادس|السابع)(?:\s+الميلادي)?\s+في\s+)عربة\b', r'\1الجزيرة العربية', val)
    val = re.sub(r'\b(في\s+)?عربة\s+في\s+القرن\s+(السادس|السابع)\b', r'في الجزيرة العربية في القرن \2', val)
    val = re.sub(r'\bفي\s+عربة\s+القرن\s+(السادس|السابع)\b', r'في الجزيرة العربية في القرن \1', val)
    val = re.sub(r'\bعربة\s+في\s+القرن\s+(السادس|السابع)\b', r'الجزيرة العربية في القرن \1', val)
    val = re.sub(r'(الصدف\s+و)ال(?:مؤ|مو)امرات\b', r'\1التزامنات', val)
    val = re.sub(r'\bال(?:مؤ|مو)امرات(\s+(?:النفسية|غير السببية))', r'التزامنات\1', val)
    val = re.sub(r'\b(مذهب|فلسفة|ديانة|الديانة)\s+الطائفة\b', r'\1 الطاوية', val)
    val = re.sub(r'\b(البوذية|الهندوسية|الكونفوشيوسية|الشنتوية)(\s+و\s*)الطائفة\b', r'\1\2الطاوية', val)
    val = re.sub(r'\bالطائفة(\s+و\s*(?:البوذية|الهندوسية|الكونفوشيوسية|الشنتوية))', r'الطاوية\1', val)
    val = re.sub(r'\bالرسول\s+بول\b', 'بولس الرسول', val)
    val = re.sub(r'\bبول\s+الرسول\b', 'بولس الرسول', val)
    val = re.sub(r'\b(شهادة|كلام|رسالة|رسائل|تعاليم|التركيز\s+على|قبل|عن|مع|على|من|باسم|مثل)\s+بول\b', r'\1 بولس', val)
    val = re.sub(r'(?<![\w\u0600-\u06ff])([وفبكل]?)ماتيو\b', r'\1متى', val)
    val = re.sub(r'\bإنجيل\s+ماتيو\b', 'إنجيل متى', val)
    val = re.sub(r'\bإنجيل\s+مارك\b', 'إنجيل مرقس', val)
    val = re.sub(r'\bإنجيل\s+لوك\b', 'إنجيل لوقا', val)
    val = re.sub(r'\bإنجيل\s+جون\b', 'إنجيل يوحنا', val)
    val = re.sub(r'\bالقديس\s+بيتر\b', 'القديس بطرس', val)
    if any(ctx in val for ctx in ("الإنجيل", "العهد الجديد", "متى", "مرقس", "لوقا", "يوحنا", "المسيح", "يسوع", "شاهد")):
        val = re.sub(r'(?<![\w\u0600-\u06ff])([وفبكل]?)بول\b', r'\1بولس', val)

    # 4. Purge any remaining stray Latin/English tokens and Hebrew/CJK/Cyrillic from Arabic fields
    val = re.sub(r'(?:^|\s)(?:[وفبكل]?الـ?|[وفبكل]ـ?)\s*(?=[a-zA-Z])', ' ', val)
    val = re.sub(r'[A-Za-z]+', '', val)
    val = _HEBREW_RE.sub('', val)
    val = re.sub(r'\(\s*\)', '', val)
    val = re.sub(r'\[\s*\]', '', val)
    val = re.sub(r'[ \t]+', ' ', val).strip()
    val = re.sub(r'\s+([،.؟!:;])', r'\1', val)

    return val


def check_theological_bias(unit: Dict[str, Any], a_turn: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Detects whether a unit expresses opponent theological bias / heresy."""
    bias_patterns = [
        # 1. Defending / justifying incest
        (r'\b(defend(ing|s)?|justif(y|ying|ies)|permit(ting|s)?|allow(ing|s)?)\s+(incest)\b', 'Defending incest', False),
        (r'\b(incest\s+(is|was)\s+(justified|permissible|allowed|moral|acceptable|necessary))\b', 'Justifying incest', False),
        (r'(تبرير|إباحة|جواز|الدفاع عن)\s+زنا المحارم', 'تبرير زنا المحارم', False),
        (r'زنا المحارم\s+(مباح|جائز|حلال|مبرر)', 'زنا المحارم مباح', False),

        # 2. Spirit of Truth is Holy Spirit (denying it is Prophet Muhammad)
        (r'\b(spirit of truth|comforter|paraclete)\s+(is|means)\s+(the\s+)?holy spirit\b', 'Claiming Spirit of Truth is Holy Spirit', True),
        (r'\b(spirit of truth|comforter)\s+is\s+not\s+(prophet\s+)?muhammad\b', 'Denying Comforter is Muhammad', False),
        (r'(روح الحق|المعزي|البارقليط)\s+(هو|يعني)\s+الروح القدس', 'روح الحق هو الروح القدس', True),
        (r'(روح الحق|المعزي|البارقليط)\s+ليس\s+(هو\s+)?محمد', 'المعزي ليس محمدا', False),

        # 3. Biblical laws are void / abolished
        (r'\b(biblical|old testament|mosaic)\s+laws?\s+(are|is)\s+(void|abolished|nullified)\b', 'Biblical laws void', True),
        (r'\b(the\s+law\s+is\s+(void|abolished|null and void))\b', 'Law is void', True),
        (r'(الناموس|شريعة العهد القديم|الشريعة الموسوية)\s+(باطل|ملغى|منسوخ بالكلية لإسقاط الأخلاق)', 'الناموس باطل', True),

        # 4. Asserting Christ is God or Trinity as the Da'ee's position
        (r'\b(jesus|christ)\s+is\s+(the\s+)?(true\s+)?god\b', 'Christ is God asserted as truth', True),
        (r'(المسيح|يسوع|عيسى)\s+هو\s+الله', 'المسيح هو الله', True),
        (r'(الثالوث|التثليث)\s+حق', 'الثالوث حق', True),

        # 5. Asserting vicarious crucifixion / atonement as the Da'ee's position
        (r'\b(jesus|christ)\s+died\s+for\s+(our\s+)?sins\b', 'Crucifixion atonement asserted as truth', True),
        (r'(المسيح|يسوع)\s+مات\s+(تكفيراً|فداءً)\s+عن\s+خطايانا', 'المسيح مات فداء عن خطايانا', True),
    ]

    texts_to_check = [
        str(unit.get("core_answer_summary_en") or ""),
        str(unit.get("core_answer_summary_ar") or ""),
        str(unit.get("verbatim_quote") or ""),
        str(unit.get("verbatim_quote_ar") or ""),
    ]

    for pk in ("logical_premises", "logical_premises_ar"):
        val = unit.get(pk)
        if isinstance(val, list):
            texts_to_check.extend(str(x) for x in val)

    if a_turn:
        texts_to_check.append(str(a_turn.get("text_original") or ""))
        texts_to_check.append(str(a_turn.get("text_ar") or ""))

    combined = " ".join(texts_to_check)
    for pattern, reason, check_refutation in bias_patterns:
        m = re.search(pattern, combined, re.IGNORECASE)
        if m:
            if check_refutation and _is_refutation_context(combined, m.start()):
                continue
            return reason
    return None


BILINGUAL_TOPIC_DOMAINS = {
    "Textual Preservation & Manuscripts": {
        "ar": "حفظ النصوص والمخطوطات",
        "variants": ["Textual Preservation", "Manuscript", "Inerrancy", "Biblical Criticism", "Biblical Reliability", "Biblical Manuscripts", "Dead Sea Scrolls", "Codex"]
    },
    "Comparative Religion & Christianity": {
        "ar": "مقارنة الأديان والمسيحية",
        "variants": ["Christianity", "Trinity", "Crucifixion", "Atonement", "Christology", "Gospel"]
    },
    "Theology & God's Existence": {
        "ar": "العقيدة ووجود الله",
        "variants": ["Theology", "God's Existence", "First Cause", "Causality", "Tawheed", "Divine Unity", "Creation", "Contingency"]
    },
    "Prophethood & Quran": {
        "ar": "النبوة وإعجاز القرآن",
        "variants": ["Prophethood", "Quran", "Prophet", "Miracles", "Inimitability", "Seerah"]
    },
    "Problem of Evil & Suffering": {
        "ar": "معضلة الشر والمعاناة",
        "variants": ["Problem of Evil", "Suffering", "Theodicy", "Disasters", "Evil", "Ibtilaa"]
    },
    "Objective Morality & Ethics": {
        "ar": "الأخلاق الموضوعية والقيم",
        "variants": ["Objective Morality", "Moral Ontology", "Ethics & Morality", "Moral Argument", "Nihilism"]
    },
    "Epistemology & Science": {
        "ar": "المعرفة والعلم التجريبي",
        "variants": ["Epistemology", "Science", "Evolution", "Darwin", "Scientism", "Consciousness", "Philosophy of Mind"]
    },
    "Shariah & Social Inquiries": {
        "ar": "الشريعة والشبهات الاجتماعية",
        "variants": ["Shariah", "Social Inquiries", "Women", "Hijab", "Polygamy", "Inheritance", "Apostasy", "Hudood", "Slavery Context"]
    },
    "Comparative Religion & Judaism": {
        "ar": "مقارنة الأديان واليهودية",
        "variants": ["Judaism", "Jewish", "Torah", "Talmud", "Old Testament Prophecies"]
    },
    "Comparative Religion & Eastern Traditions": {
        "ar": "مقارنة الأديان والمعتقدات الشرقية",
        "variants": ["Hinduism", "Buddhism", "Reincarnation", "Karma", "Eastern Traditions"]
    },
    "Hadith & Sunnah Preservation": {
        "ar": "حفظ السنة والحديث النبوي",
        "variants": ["Hadith", "Sunnah", "Isnad", "Bukhari", "Fabrication", "Narrators"]
    },
    "Islamic History & Civilizations": {
        "ar": "التاريخ والحضارة الإسلامية",
        "variants": ["Islamic History", "Caliphate", "Conquests", "Andalusia", "Crusades", "Ottoman"]
    },
    "Eschatology & Free Will": {
        "ar": "القدر واليوم الآخر والجنة والنار",
        "variants": ["Eschatology", "Free Will", "Qadr", "Destiny", "Afterlife", "Hell", "Heaven", "Resurrection"]
    },
}

CANONICAL_TOPIC_PILLARS = {k: v["variants"] for k, v in BILINGUAL_TOPIC_DOMAINS.items()}


def get_category_arabic_name(en_category: str) -> str:
    """Returns authentic Classical Arabic title for any category, existing or new."""
    if not en_category:
        return "العقيدة ووجود الله"
    en_clean = en_category.strip()
    if en_clean in BILINGUAL_TOPIC_DOMAINS:
        return BILINGUAL_TOPIC_DOMAINS[en_clean]["ar"]
    for dom_name, dom_info in BILINGUAL_TOPIC_DOMAINS.items():
        if any(v.lower() in en_clean.lower() for v in dom_info["variants"]):
            return dom_info["ar"]
    # Semantic word-level fallback translation for arbitrary new domains:
    word_map = {
        "Comparative Religion": "مقارنة الأديان",
        "Textual Preservation": "حفظ النصوص",
        "Manuscripts": "المخطوطات",
        "Theology": "العقيدة",
        "God's Existence": "وجود الله",
        "Prophethood": "النبوة",
        "Quran": "القرآن",
        "Hadith": "الحديث النبوي",
        "Sunnah": "السنة النبوية",
        "History": "التاريخ",
        "Civilization": "الحضارة",
        "Philosophy": "الفلسفة",
        "Ethics": "الأخلاق",
        "Morality": "القيم",
        "Science": "العلوم",
        "Epistemology": "نظرية المعرفة",
        "Shariah": "الشريعة",
        "Feminism": "النسوية",
        "Social": "المجتمع",
        "Inquiries": "القضايا والشبهات",
        "Suffering": "المعاناة",
        "Evil": "الشر",
        "Christianity": "المسيحية",
        "Judaism": "اليهودية",
        "Islam": "الإسلام",
        "&": "و"
    }
    ar_name = en_clean
    for en_word, ar_word in word_map.items():
        ar_name = re.sub(r'\b' + re.escape(en_word) + r'\b', ar_word, ar_name, flags=re.IGNORECASE)
    ar_name = ar_name.replace("&", "و").strip()
    return ar_name if re.search(r'[\u0600-\u06ff]', ar_name) else "قضايا فكرية وبحوث مقارنة"


def canonicalize_topic_category(raw_category: str) -> str:
    """Normalizes minor semantic variants without artificially restricting new major domains."""
    if not raw_category:
        return "Theology & God's Existence"
    cat_clean = raw_category.strip()
    if cat_clean in BILINGUAL_TOPIC_DOMAINS:
        return cat_clean
    for pillar, dom_info in BILINGUAL_TOPIC_DOMAINS.items():
        if any(v.lower() in cat_clean.lower() for v in dom_info["variants"]):
            return pillar
    return cat_clean


def sanitize_text_field(text: Optional[str], is_arabic: Optional[bool] = None) -> str:
    """Purges foreign scripts, Cyrillic, and CJK artifacts; purifies leaked Latin words in Arabic fields."""
    if not text or not isinstance(text, str):
        return ""
    val = text
    has_arabic = bool(re.search(r"[\u0600-\u06ff]", val))
    apply_arabic_cleaning = is_arabic if is_arabic is not None else has_arabic

    if apply_arabic_cleaning:
        val = clean_latin_tokens_in_arabic(val)
        val = _HEBREW_RE.sub("", val)
    else:
        for old_s, new_s in [
            ("。，？，。", "؟"),
            ("。", "."),
            ("，", ","),
            ("？", "?"),
            ("！", "!"),
        ]:
            if old_s in val:
                val = val.replace(old_s, new_s)

    val = _CYRILLIC_RE.sub("", val)
    val = _CJK_RE.sub("", val)
    return re.sub(r"[ \t]{2,}", " ", val).strip()


ARABIC_UNIT_SCALAR_KEYS = (
    "question_ar",
    "core_answer_summary_ar",
    "fallacy_or_misconception_exposed_ar",
    "analogy_or_metaphor_ar",
    "socratic_question_ar",
    "verbatim_quote_ar",
    "interlocutor_verbatim_quote_ar",
)


def sanitize_arabic_unit_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical cleaner for all Classical Arabic fields across scalars, lists, citations, and dialogue turns."""
    if not isinstance(item, dict):
        return item

    for k in ARABIC_UNIT_SCALAR_KEYS:
        val = item.get(k)
        if isinstance(val, str) and val:
            item[k] = sanitize_text_field(val, is_arabic=True)

    for list_key in ("logical_premises_ar", "analogies_ar"):
        raw_list = item.get(list_key)
        if isinstance(raw_list, str) and raw_list.strip().startswith("["):
            try:
                raw_list = json.loads(raw_list)
            except Exception:
                pass
        if isinstance(raw_list, list):
            item[list_key] = [
                cleaned
                for p in raw_list
                if isinstance(p, str)
                for cleaned in (sanitize_text_field(p, is_arabic=True),)
                if cleaned
            ]

    citations = item.get("scriptural_citations")
    if isinstance(citations, list):
        for cit in citations:
            if isinstance(cit, dict) and isinstance(cit.get("significance_ar"), str) and cit["significance_ar"]:
                cit["significance_ar"] = sanitize_text_field(cit["significance_ar"], is_arabic=True)

    for ex_key in ("dialogue_exchange", "dialogue_trajectory"):
        exchanges = item.get(ex_key)
        if isinstance(exchanges, list):
            for ex in exchanges:
                if isinstance(ex, dict) and isinstance(ex.get("text_ar"), str) and ex["text_ar"]:
                    ex["text_ar"] = sanitize_text_field(ex["text_ar"], is_arabic=True)

    return item


def _row_field(row_or_dict: Any, key: str) -> Any:
    if row_or_dict is None:
        return None
    if isinstance(row_or_dict, dict):
        return row_or_dict.get(key)
    try:
        return row_or_dict[key]
    except (KeyError, IndexError, TypeError):
        return None


def resolve_turn_speaker_labels(
    role: str,
    turn_obj: Any,
    daee_name_en: str,
    daee_name_ar: str,
) -> tuple[str, str]:
    """Returns role-consistent (speaker_label_en, speaker_label_ar), preventing Da'ee names on interlocutor turns."""
    is_daee = role in ("MAIN_DAEE", "CO_DAEE")
    fallback_en = daee_name_en or "Main Da'ee" if is_daee else "Interlocutor"
    fallback_ar = daee_name_ar or "الداعية" if is_daee else "المحاور"

    if turn_obj is None:
        return fallback_en, fallback_ar

    turn_role = _row_field(turn_obj, "speaker_role")
    if turn_role and (turn_role in ("MAIN_DAEE", "CO_DAEE")) != is_daee:
        return fallback_en, fallback_ar

    lbl_en = str(_row_field(turn_obj, "speaker_label_en") or "").strip()
    lbl_ar = str(_row_field(turn_obj, "speaker_label_ar") or "").strip()

    if not is_daee:
        if daee_name_en and daee_name_en.lower() in lbl_en.lower():
            lbl_en = fallback_en
        if daee_name_ar and daee_name_ar in lbl_ar:
            lbl_ar = fallback_ar

    return (lbl_en or fallback_en), (lbl_ar or fallback_ar)


def _sanitize_and_validate_qa_unit(conn: sqlite3.Connection, unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Root-cause ingestion guard: enforces 100% clean scripts, authentic Da'ee attribution,
    and turn-aligned metadata before committing to SQLite.
    """
    u = dict(unit)

    # 1. Sanitize text fields (respecting English vs Arabic domain boundaries)
    for k in (
        "question_en",
        "core_answer_summary_en",
        "fallacy_or_misconception_exposed",
        "analogy_or_metaphor",
        "socratic_question",
        "interlocutor_verbatim_quote",
        "verbatim_quote",
    ):
        if k in u:
            u[k] = sanitize_text_field(u[k], is_arabic=False)

    # 2. Sanitize English premises and all Classical Arabic fields via shared helper
    if "logical_premises" in u:
        val = u.get("logical_premises")
        if isinstance(val, list):
            u["logical_premises"] = [sanitize_text_field(p, is_arabic=False) for p in val if sanitize_text_field(p, is_arabic=False)]
        elif isinstance(val, str) and val.startswith("["):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    u["logical_premises"] = [sanitize_text_field(p, is_arabic=False) for p in parsed if sanitize_text_field(p, is_arabic=False)]
            except Exception:
                pass

    sanitize_arabic_unit_fields(u)

    # 3. Swap / repair Arabic premises if misplaced in English premises
    p_en = u.get("logical_premises") or []
    p_ar = u.get("logical_premises_ar") or []
    if not p_ar and p_en and any(re.search(r"[\u0600-\u06ff]", str(p)) for p in p_en):
        u["logical_premises_ar"] = list(p_en)

    # 4. Enforce Da'ee attribution integrity (block generic placeholders & guard against heresy)
    vid = u.get("video_id")
    daee_ar = u.get("daee_name_ar") or ""
    daee_prof = u.get("daee_profile_id") or ""

    # Strictly remove blind force-renaming of interlocutors to Da'ee
    if daee_ar not in ("المحاور", "المحاور 1", "المحاور 2", "Interlocutor", "Interlocutor 1", "Interlocutor 2"):
        if vid and not daee_prof:
            spk = conn.execute(
                "SELECT profile_id, display_name_ar, display_name_en, speaker_id FROM video_speakers WHERE video_id = ? AND role = 'MAIN_DAEE'",
                (vid,)
            ).fetchone()
            if spk and spk["display_name_ar"]:
                if not daee_ar or daee_ar == spk["display_name_ar"]:
                    u["daee_profile_id"] = spk["profile_id"] or ""
                    u["daee_name_ar"] = spk["display_name_ar"]
                    u["daee_name_en"] = spk["display_name_en"] or "Main Da'ee"
                    u["daee_speaker_id"] = spk["speaker_id"] or u.get("daee_speaker_id", "")
            else:
                vhint = conn.execute("SELECT primary_daee_hint FROM videos WHERE video_id = ?", (vid,)).fetchone()
                if vhint and vhint[0]:
                    prof_id = vhint[0]
                    prof = conn.execute("SELECT name_ar, name_en FROM speaker_profiles WHERE profile_id = ?", (prof_id,)).fetchone()
                    if prof:
                        if not daee_ar or daee_ar == prof["name_ar"]:
                            u["daee_profile_id"] = prof_id
                            u["daee_name_ar"] = prof["name_ar"]
                            u["daee_name_en"] = prof["name_en"]

    # 5. Enrich dialogue_exchange with timestamps and localized speaker names if missing
    exchanges = u.get("dialogue_exchange") or []
    if isinstance(exchanges, str) and exchanges.startswith("["):
        try:
            exchanges = json.loads(exchanges)
        except Exception:
            exchanges = []
    if isinstance(exchanges, list) and exchanges:
        enriched = []
        for ex in exchanges:
            if not isinstance(ex, dict):
                continue
            ex_copy = dict(ex)
            tid = ex_copy.get("turn_id", "")
            turn = conn.execute(
                "SELECT start_ms, start_fmt, speaker_role, speaker_label_en, speaker_label_ar FROM scenario_turns WHERE turn_id = ?",
                (tid,)
            ).fetchone() if tid else None

            role = ex_copy.get("role") or (turn["speaker_role"] if turn else "MAIN_DAEE")
            ex_copy["role"] = role
            lbl_en, lbl_ar = resolve_turn_speaker_labels(
                role=role,
                turn_obj=turn,
                daee_name_en=str(u.get("daee_name_en") or "Main Da'ee"),
                daee_name_ar=str(u.get("daee_name_ar") or "الداعية"),
            )
            ex_copy["speaker_label_ar"] = lbl_ar
            ex_copy["speaker_label_en"] = lbl_en
            if turn and ("timestamp_formatted" not in ex_copy or ex_copy["timestamp_formatted"] in ("00:00", "")):
                ex_copy["timestamp_formatted"] = turn["start_fmt"]
                ex_copy["timestamp_seconds"] = int(turn["start_ms"] // 1000)
            enriched.append(ex_copy)
        u["dialogue_exchange"] = enriched

    return u


def insert_qa_unit(unit: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    conn = get_db_connection(db_path)
    with conn:
        _insert_qa_unit_in_tx(conn, unit)
    conn.close()


def _insert_qa_unit_in_tx(conn: sqlite3.Connection, unit: Dict[str, Any]) -> None:
    if check_theological_bias(unit):
        return
    unit = _sanitize_and_validate_qa_unit(conn, unit)
    if unit.get("daee_name_ar") in ("المحاور", "المحاور 1", "المحاور 2", "Interlocutor", "Interlocutor 1", "Interlocutor 2"):
        return

    # Prepare analogies_json
    raw_an = unit.get("analogies")
    if isinstance(raw_an, list) and raw_an:
        analogies_list = [str(a).strip() for a in raw_an if str(a).strip()]
    elif unit.get("analogy_or_metaphor"):
        analogies_list = [str(unit["analogy_or_metaphor"]).strip()]
    else:
        analogies_list = []
    analogies_json_val = json.dumps(analogies_list, ensure_ascii=False)

    # Prepare analogies_ar_json
    raw_an_ar = unit.get("analogies_ar")
    if isinstance(raw_an_ar, list) and raw_an_ar:
        analogies_ar_list = [str(a).strip() for a in raw_an_ar if str(a).strip()]
    elif unit.get("analogy_or_metaphor_ar"):
        analogies_ar_list = [str(unit["analogy_or_metaphor_ar"]).strip()]
    else:
        analogies_ar_list = []
    analogies_ar_json_val = json.dumps(analogies_ar_list, ensure_ascii=False)

    # Prepare scriptural_citations_json
    raw_sc = unit.get("scriptural_citations")
    citations_list = raw_sc if isinstance(raw_sc, list) else []
    scriptural_citations_json_val = json.dumps(citations_list, ensure_ascii=False)

    # Prepare dialogue_trajectory_turns_count
    exchange_list = unit.get("dialogue_exchange") or unit.get("dialogue_trajectory") or []
    if not isinstance(exchange_list, list):
        exchange_list = []
    turn_count = int(unit.get("dialogue_trajectory_turns_count") or len(exchange_list))

    # Speaker Integrity Guard: Da'ee and Interlocutor speaker IDs cannot be identical
    daee_spk = str(unit.get("daee_speaker_id") or "").strip()
    inter_spk = str(unit.get("interlocutor_speaker_id") or "").strip()
    if daee_spk and inter_spk and daee_spk == inter_spk:
        inter_spk = ""

    # Interrogative Syntax Normalization
    q_en = str(unit.get("question_en") or "").strip()
    if q_en and not q_en.endswith("?"):
        q_en = re.sub(r'[\s.,:;!?؟]+$', '', q_en) + "?"
    q_ar = str(unit.get("question_ar") or "").strip()
    if q_ar and not q_ar.endswith("؟"):
        q_ar = re.sub(r'[\s.,:;!?؟]+$', '', q_ar) + "؟"

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
            dialogue_exchange_json, source_turn_ids_json, tags_json,
            analogies_json, analogies_ar_json, scriptural_citations_json, dialogue_trajectory_turns_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            daee_spk,
            unit.get("daee_profile_id", ""),
            unit.get("daee_name_en", ""),
            unit.get("daee_name_ar", ""),
            inter_spk,
            unit.get("topic_category", "Theology & God's Existence"),
            unit.get("topic_subcategory", ""),
            unit.get("interlocutor_type", "Inquiring Seeker"),
            q_en,
            q_ar,
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
            analogies_json_val,
            analogies_ar_json_val,
            scriptural_citations_json_val,
            turn_count,
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
    alef_norm = re.sub(r"[أإآ]", "ا", token)
    if alef_norm in ("الله", "لله", "والله", "بالله"):
        return ["الله", "لله"]

    variants = {token}
    for prefix in ("وال", "فال", "بال", "كال", "لال", "ال"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            variants.add(token[len(prefix):])
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

    variants.discard("له")
    variants.discard("لة")
    return [v for v in variants if len(v) >= 2]


_SEARCH_STOPWORDS = {
    "من", "في", "على", "عن", "إلى", "الى", "ما", "لا", "أو", "او", "هو", "هي", "له",
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
            "analogies_json", "analogies_ar_json", "scriptural_citations_json",
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
        variants = CANONICAL_TOPIC_PILLARS.get(category, [])
        if variants:
            clauses = ["topic_category = ?"] + ["topic_category LIKE ?" for _ in variants]
            sql_parts.append(f"({' OR '.join(clauses)})")
            params.append(category)
            params.extend([f"%{v}%" for v in variants])
        else:
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
        ("analogies_json", "analogies"),
        ("analogies_ar_json", "analogies_ar"),
        ("scriptural_citations_json", "scriptural_citations"),
    ]:
        try:
            d[out_key] = json.loads(d.get(json_col) or "[]")
        except Exception:
            d[out_key] = []

    # Fallback for analogies if empty but analogy_or_metaphor exists
    if not d.get("analogies") and d.get("analogy_or_metaphor"):
        d["analogies"] = [d["analogy_or_metaphor"]]
    if not d.get("analogies_ar") and d.get("analogy_or_metaphor_ar"):
        d["analogies_ar"] = [d["analogy_or_metaphor_ar"]]

    # Ensure dialogue_trajectory_turns_count is hydrated as int
    d["dialogue_trajectory_turns_count"] = int(
        d.get("dialogue_trajectory_turns_count") or len(d.get("dialogue_exchange") or [])
    )

    # Dynamic read-time sanitization: guarantees zero Russian/CJK artifacts even if DB is untouched clone
    for text_key in (
        "question_en", "question_ar",
        "core_answer_summary_en", "core_answer_summary_ar",
        "fallacy_or_misconception_exposed", "fallacy_or_misconception_exposed_ar",
        "analogy_or_metaphor", "analogy_or_metaphor_ar",
        "socratic_question", "socratic_question_ar",
        "interlocutor_verbatim_quote", "interlocutor_verbatim_quote_ar",
        "verbatim_quote", "verbatim_quote_ar",
    ):
        if text_key in d and isinstance(d[text_key], str):
            d[text_key] = sanitize_text_field(d[text_key])

    # Dynamic premise repair: if logical_premises_ar is empty but logical_premises has Arabic premises
    if not d.get("logical_premises_ar") and d.get("logical_premises"):
        p_en = d["logical_premises"]
        if any(re.search(r"[\u0600-\u06ff]", str(p)) for p in p_en):
            d["logical_premises_ar"] = list(p_en)

    # Dynamic Da'ee attribution resolution: if generic ('المحاور', etc.)
    if d.get("daee_name_ar") in ("المحاور", "المحاور 1", "المحاور 2", ""):
        title = d.get("video_title_ar") or ""
        if "هاشم" in title:
            d["daee_name_ar"] = "هاشم"
            d["daee_name_en"] = "Hashim"
            d["daee_profile_id"] = "daee_hashim"
        elif "منصور" in title:
            d["daee_name_ar"] = "منصور"
            d["daee_name_en"] = "Mansur"
            d["daee_profile_id"] = "daee_mansur"
        elif "حجاب" in title:
            d["daee_name_ar"] = "محمد حجاب"
            d["daee_name_en"] = "Mohammed Hijab"
            d["daee_profile_id"] = "daee_hijab"
        elif "عثمان" in title or "فاروق" in title:
            d["daee_name_ar"] = "الشيخ عثمان بن فاروق"
            d["daee_name_en"] = "Sheikh Uthman Ibn Farooq"
            d["daee_profile_id"] = "daee_uthman"

    # Dynamic dialogue exchange enrichment: ensure localized names and proper timestamps
    exchanges = d.get("dialogue_exchange") or []
    if isinstance(exchanges, list) and exchanges:
        enriched_ex = []
        for ex in exchanges:
            if not isinstance(ex, dict):
                continue
            ex_copy = dict(ex)
            role = ex_copy.get("role") or ""
            is_daee = "DAEE" in role.upper()
            if not ex_copy.get("speaker_label_ar") or ex_copy.get("speaker_label_ar") in ("MAIN_DAEE", "INTERLOCUTOR", "CO_DAEE"):
                ex_copy["speaker_label_ar"] = (d.get("daee_name_ar") or "الداعية المحاور") if is_daee else "المحاور"
            if not ex_copy.get("speaker_label_en") or ex_copy.get("speaker_label_en") in ("MAIN_DAEE", "INTERLOCUTOR", "CO_DAEE"):
                ex_copy["speaker_label_en"] = (d.get("daee_name_en") or "Main Da'ee") if is_daee else "Interlocutor"
            if not ex_copy.get("timestamp_formatted"):
                ex_copy["timestamp_formatted"] = d.get("timestamp_formatted", "00:00")
            enriched_ex.append(ex_copy)
        d["dialogue_exchange"] = enriched_ex

    # Dynamic category canonicalization
    raw_cat = d.get("topic_category") or ""
    canonical_cat = canonicalize_topic_category(raw_cat)
    if canonical_cat != raw_cat:
        if not d.get("topic_subcategory"):
            d["topic_subcategory"] = raw_cat
        d["topic_category"] = canonical_cat

    # Ensure topic_category_ar is always populated with authentic Classical Arabic
    d["topic_category_ar"] = get_category_arabic_name(d.get("topic_category") or "")

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

    from collections import defaultdict
    counts = defaultdict(int)
    for r in rows:
        c_pillar = canonicalize_topic_category(r["topic_category"])
        counts[c_pillar] += r["count"]

    ordered_pillars = [
        "Comparative Religion & Christianity",
        "Theology & God's Existence",
        "Prophethood & Quran",
        "Shariah & Social Inquiries",
        "Problem of Evil & Suffering",
        "Epistemology & Science",
        "Objective Morality & Ethics"
    ]

    res = []
    for p in ordered_pillars:
        if counts[p] > 0:
            res.append({
                "topic_category": p,
                "topic_category_ar": get_category_arabic_name(p),
                "count": counts[p]
            })
    for p, cnt in counts.items():
        if p not in ordered_pillars and cnt > 0:
            res.append({
                "topic_category": p,
                "topic_category_ar": get_category_arabic_name(p),
                "count": cnt
            })

    return res


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
