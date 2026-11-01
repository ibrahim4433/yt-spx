#!/usr/bin/env python3
"""
core/drive_sync.py
Shared 5 TB Google Drive Workspace, 6-Stage Micro-Step Checkpointing (M1 -> M6),
Anti-Ban Smart Batch Rate-Limiter, and Multi-Account Atomic Lock & Shard Engine.

Designed specifically for running multiple Free-Tier Google Colab T4 accounts
(2-4 hours/day each) safely against a single shared Google Drive folder
(/content/drive/MyDrive/yt-spx) with zero redone work and zero SQLite FUSE corruption.
"""

import os
import json
import time
import uuid
import socket
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

# Micro-step progression order (strictly monotonic per video)
MICRO_STEPS = [
    "M0_QUEUED",
    "M1_DOWNLOADED",          # Raw YouTube audio + info.json saved to Drive staging
    "M2_VOCALS_CLEANED",      # Chunked Demucs vocals + 64kbps Opus + pre-norm RMS saved
    "M3_DIARIZED",            # NeMo MSDD RTTM + TitaNet-Large 192-D centroids + overlap windows saved
    "M4_OVERLAPS_SEPARATED",  # MossFormer2 2-stream overlap clips separated & speaker-matched
    "M5_SCENARIO_BUILT",      # Faster-Whisper + CTC word align + 3-tier roles -> scenarios/{vid}.json
    "M6_LLM_EXTRACTED",       # Qwen 2.5 14B episode-by-episode Q&A extraction -> extractions/{vid}.qa.json
]

STEP_RANK = {step: idx for idx, step in enumerate(MICRO_STEPS)}


def resolve_drive_root() -> Optional[Path]:
    """
    Resolves the shared 'yt-spx' workspace on Google Drive across both the primary
    5 TB owner account and collaborator Colab accounts using 'Add shortcut to My Drive'.
    """
    env_override = os.environ.get("YT_SPX_DRIVE_DIR", "").strip()
    if env_override:
        p = Path(env_override)
        p.mkdir(parents=True, exist_ok=True)
        return p

    candidates = [
        Path("/content/drive/MyDrive/yt-spx"),
        Path("/content/drive/My Drive/yt-spx"),
    ]

    # Check Shared Drives if mounted
    shared_drives_root = Path("/content/drive/Shareddrives")
    if shared_drives_root.exists():
        for sd in shared_drives_root.iterdir():
            if sd.is_dir() and (sd / "yt-spx").exists():
                candidates.insert(0, sd / "yt-spx")

    for cand in candidates:
        if cand.parent.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand

    return None


class SharedWorkspace:
    """
    Manages directory layout, micro-step checkpoints, multi-account locks,
    anti-ban YouTube circuit breakers, and single-writer SQLite shard compaction.
    """

    def __init__(self, root_dir: Optional[Path] = None, worker_name: Optional[str] = None):
        drive_root = root_dir or resolve_drive_root()
        self.is_drive = drive_root is not None
        self.root: Path = drive_root if drive_root is not None else (BASE_DIR / "data")

        # Subdirectories for micro-step artifacts
        self.data_dir = self.root / "data" if self.is_drive else self.root
        self.staging_dir = self.root / "staging"
        self.audio_raw_dir = self.root / "audio_raw"
        self.audio_clean_dir = self.root / "audio_clean"
        self.diarization_dir = self.root / "diarization"
        self.scenarios_dir = self.root / "scenarios"
        self.extractions_dir = self.root / "extractions"
        self.partial_extractions_dir = self.extractions_dir / "partial"
        self.state_dir = self.root / "state"
        self.locks_dir = self.root / "locks"
        self.profiles_dir = self.root / "profiles"
        self.logs_dir = self.root / "logs"

        for d in (
            self.data_dir,
            self.staging_dir,
            self.audio_raw_dir,
            self.audio_clean_dir,
            self.diarization_dir,
            self.scenarios_dir,
            self.extractions_dir,
            self.partial_extractions_dir,
            self.state_dir,
            self.locks_dir,
            self.profiles_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        host = socket.gethostname()[:16]
        short_uid = uuid.uuid4().hex[:6]
        env_worker = os.environ.get("YT_SPX_WORKER_ID", "").strip()
        self.worker_id = worker_name or env_worker or f"colab-{host}-{os.getpid()}-{short_uid}"

    # ------------------------------------------------------------------
    # Atomic JSON Read/Write Helpers (Safe on Google Drive FUSE)
    # ------------------------------------------------------------------
    @staticmethod
    def atomic_write_json(target: Path, payload: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)

    @staticmethod
    def read_json(target: Path, default: Any = None) -> Any:
        if not target.exists():
            return default
        try:
            with open(target, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    # ------------------------------------------------------------------
    # Micro-Step State Tracking (M0 -> M6)
    # ------------------------------------------------------------------
    def state_path(self, video_id: str) -> Path:
        return self.state_dir / f"{video_id}.state.json"

    def get_video_state(self, video_id: str) -> Dict[str, Any]:
        default_state = {
            "video_id": video_id,
            "micro_step": "M0_QUEUED",
            "step_rank": 0,
            "updated_at": 0.0,
            "worker_id": None,
            "artifacts": {},
            "metrics": {},
            "episode_progress": {"completed_episodes": [], "total_episodes": 0},
            "overlap_progress": {"completed_windows": 0, "total_windows": 0},
            "demucs_progress": {"completed_chunks": 0, "total_chunks": 0},
            "db_scenario_synced": False,
            "db_qa_synced": False,
            "error_count": 0,
            "last_error": None,
        }
        loaded = self.read_json(self.state_path(video_id))
        if not isinstance(loaded, dict):
            # Reconcile from existing artifacts on disk if state file was missing
            return self._reconcile_state_from_artifacts(video_id, default_state)
        merged = {**default_state, **loaded}
        merged["step_rank"] = STEP_RANK.get(merged.get("micro_step", "M0_QUEUED"), 0)
        return merged

    def _reconcile_state_from_artifacts(self, video_id: str, base_state: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-recovers micro-step progress purely from existing Drive files if needed."""
        qa_shard = self.extractions_dir / f"{video_id}.qa.json"
        scenario_file = self.scenarios_dir / f"{video_id}.json"
        overlap_file = self.diarization_dir / video_id / "overlap_sep.json"
        rttm_file = self.diarization_dir / video_id / f"{video_id}.rttm"
        clean_opus = self.audio_clean_dir / f"{video_id}.opus"
        raw_candidates = list(self.audio_raw_dir.glob(f"{video_id}.*"))

        if qa_shard.exists():
            base_state["micro_step"] = "M6_LLM_EXTRACTED"
        elif scenario_file.exists():
            base_state["micro_step"] = "M5_SCENARIO_BUILT"
        elif overlap_file.exists():
            base_state["micro_step"] = "M4_OVERLAPS_SEPARATED"
        elif rttm_file.exists():
            base_state["micro_step"] = "M3_DIARIZED"
        elif clean_opus.exists():
            base_state["micro_step"] = "M2_VOCALS_CLEANED"
        elif any(p.suffix not in (".json", ".part", ".ytdl") for p in raw_candidates):
            base_state["micro_step"] = "M1_DOWNLOADED"

        base_state["step_rank"] = STEP_RANK[base_state["micro_step"]]
        return base_state

    def advance_micro_step(
        self,
        video_id: str,
        new_step: str,
        artifacts: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically persists completion of a micro-step to Google Drive."""
        if new_step not in STEP_RANK:
            raise ValueError(f"Unknown micro-step: {new_step}")

        state = self.get_video_state(video_id)
        if STEP_RANK[new_step] > state["step_rank"]:
            state["error_count"] = 0
        if STEP_RANK[new_step] >= state["step_rank"]:
            state["micro_step"] = new_step
            state["step_rank"] = STEP_RANK[new_step]
        state["updated_at"] = time.time()
        state["worker_id"] = self.worker_id
        state["last_error"] = None

        if artifacts:
            state.setdefault("artifacts", {}).update(artifacts)
        if metrics:
            state.setdefault("metrics", {}).update(metrics)
        if extra_updates:
            state.update(extra_updates)

        self.atomic_write_json(self.state_path(video_id), state)
        return state

    def update_substep_progress(self, video_id: str, key: str, progress_payload: Dict[str, Any]) -> None:
        """Saves granular mid-step progress (Demucs chunks, overlap windows, or LLM episodes)."""
        state = self.get_video_state(video_id)
        state[key] = progress_payload
        state["updated_at"] = time.time()
        state["worker_id"] = self.worker_id
        self.atomic_write_json(self.state_path(video_id), state)

    def record_error(self, video_id: str, error_msg: str) -> Dict[str, Any]:
        state = self.get_video_state(video_id)
        state["error_count"] = int(state.get("error_count", 0)) + 1
        state["last_error"] = str(error_msg)[:500]
        state["updated_at"] = time.time()
        state["worker_id"] = self.worker_id
        self.atomic_write_json(self.state_path(video_id), state)
        return state

    # ------------------------------------------------------------------
    # Multi-Account Optimistic Lock with Read-After-Write Verification
    # ------------------------------------------------------------------
    def lock_path(self, name: str) -> Path:
        return self.locks_dir / f"{name}.lock.json"

    def acquire_lock(self, lock_name: str, ttl_sec: int = 2400, verify_delay_sec: float = 0.35) -> bool:
        """
        Acquires a cross-Colab-account lock on Google Drive FUSE using
        optimistic claim + jittered read-back verification.
        """
        lp = self.lock_path(lock_name)
        now = time.time()

        existing = self.read_json(lp)
        if isinstance(existing, dict):
            expires_at = float(existing.get("expires_at", 0.0))
            owner = existing.get("worker_id")
            if expires_at > now and owner != self.worker_id:
                return False

        claim_token = uuid.uuid4().hex
        payload = {
            "lock_name": lock_name,
            "worker_id": self.worker_id,
            "claim_token": claim_token,
            "claimed_at": now,
            "heartbeat_at": now,
            "expires_at": now + ttl_sec,
        }
        try:
            self.atomic_write_json(lp, payload)
        except Exception:
            return False

        if verify_delay_sec > 0:
            time.sleep(verify_delay_sec + random.uniform(0.05, 0.25))

        verified = self.read_json(lp)
        return isinstance(verified, dict) and verified.get("claim_token") == claim_token

    def heartbeat_lock(self, lock_name: str, ttl_sec: int = 2400) -> bool:
        lp = self.lock_path(lock_name)
        existing = self.read_json(lp)
        if not isinstance(existing, dict) or existing.get("worker_id") != self.worker_id:
            return False
        now = time.time()
        existing["heartbeat_at"] = now
        existing["expires_at"] = now + ttl_sec
        self.atomic_write_json(lp, existing)
        return True

    def release_lock(self, lock_name: str) -> None:
        lp = self.lock_path(lock_name)
        existing = self.read_json(lp)
        if isinstance(existing, dict) and existing.get("worker_id") == self.worker_id:
            try:
                lp.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Shared Anti-Ban YouTube Rate Limiter & Circuit Breaker
    # ------------------------------------------------------------------
    def check_youtube_download_slot(self, min_global_gap_sec: float = 12.0) -> Tuple[bool, float, str]:
        """
        Coordinates YouTube downloads across ALL active Colab accounts so multiple
        accounts never burst-download simultaneously or burn shared cookies.
        Returns (allowed, wait_seconds, reason).
        """
        rl_path = self.locks_dir / "_youtube_ratelimit.json"
        now = time.time()
        data = self.read_json(rl_path, default={}) or {}

        cooldown_until = float(data.get("circuit_breaker_until", 0.0))
        reason_str = str(data.get("reason", "")).lower()
        if (
            cooldown_until > now
            and "invalid" in reason_str
            and "bot" not in reason_str
            and "429" not in reason_str
            and "too many" not in reason_str
        ):
            # Auto-clear false-positive cooldown tripped by rotated/invalid cookies (anonymous download still works!)
            data["circuit_breaker_until"] = 0.0
            self.atomic_write_json(rl_path, data)
            cooldown_until = 0.0

        if cooldown_until > now:
            rem = round(cooldown_until - now, 1)
            return False, rem, data.get("reason", "Anti-ban cooldown active")

        last_dl = float(data.get("last_download_at", 0.0))
        elapsed = now - last_dl
        if elapsed < min_global_gap_sec:
            return False, round(min_global_gap_sec - elapsed + random.uniform(1.5, 5.0), 1), "Global download pacing"

        data["last_download_at"] = now
        data["last_worker"] = self.worker_id
        self.atomic_write_json(rl_path, data)
        return True, 0.0, "ok"

    def trip_youtube_circuit_breaker(self, reason: str, cooldown_minutes: int = 45) -> None:
        """
        Trips a shared cooldown across Drive if YouTube returns a 403/429 or
        'Sign in to confirm you're not a bot' challenge, protecting cookies.txt
        while allowing GPU workers to keep running M2->M6 offline!
        """
        rl_path = self.locks_dir / "_youtube_ratelimit.json"
        now = time.time()
        data = self.read_json(rl_path, default={}) or {}
        data.update({
            "circuit_breaker_until": now + cooldown_minutes * 60,
            "tripped_at": now,
            "tripped_by": self.worker_id,
            "reason": reason[:300],
        })
        self.atomic_write_json(rl_path, data)

    # ------------------------------------------------------------------
    # Single-Writer SQLite Compactor (Merges Per-Video Shards Safely)
    # ------------------------------------------------------------------
    def sync_completed_shards_to_sqlite(self, force_reingest: bool = False) -> Dict[str, int]:
        """
        Acquires the exclusive '_sqlite_compactor' lock and merges any unsynced
        scenarios/{vid}.json and extractions/{vid}.qa.json shards into SQLite.
        Safe across multiple Colab accounts because only one compactor writes SQLite at a time.
        """
        stats = {"scenarios_synced": 0, "qa_videos_synced": 0, "qa_units_synced": 0}
        if not self.acquire_lock("_sqlite_compactor", ttl_sec=600, verify_delay_sec=0.25 if self.is_drive else 0.0):
            return stats

        try:
            from db.database import init_db, ingest_scenario_payload, ingest_extraction_payload
            init_db(db_path=self.data_dir / "dawah_knowledge.db")

            # 1. Sync Scenario Transcripts (M5+)
            for scen_path in sorted(self.scenarios_dir.glob("*.json")):
                vid = scen_path.stem
                st = self.get_video_state(vid)
                if st.get("db_scenario_synced") and not force_reingest:
                    continue
                payload = self.read_json(scen_path)
                if isinstance(payload, dict) and payload.get("video_id"):
                    ingest_scenario_payload(payload, db_path=self.data_dir / "dawah_knowledge.db")
                    st["db_scenario_synced"] = True
                    self.atomic_write_json(self.state_path(vid), st)
                    stats["scenarios_synced"] += 1

            # 2. Sync Extracted Q&A Units (M6)
            for qa_path in sorted(self.extractions_dir.glob("*.qa.json")):
                vid = qa_path.name.replace(".qa.json", "")
                st = self.get_video_state(vid)
                if st.get("db_qa_synced") and not force_reingest:
                    continue
                payload = self.read_json(qa_path)
                units = payload.get("units", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
                ingest_extraction_payload(vid, units, db_path=self.data_dir / "dawah_knowledge.db")
                st["db_qa_synced"] = True
                self.atomic_write_json(self.state_path(vid), st)
                stats["qa_videos_synced"] += 1
                stats["qa_units_synced"] += len(units)

            return stats
        finally:
            self.release_lock("_sqlite_compactor")
