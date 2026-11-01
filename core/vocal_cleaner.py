#!/usr/bin/env python3
"""
core/vocal_cleaner.py
Micro-Step M2: Street-Safe Human Vocal Stem Isolation, Pre-Norm Mic Proximity Metering,
Conversational Loudness Leveling, and 64 kbps Opus Archival on 5 TB Google Drive.

Why this beats standard single-speaker denoisers on Speakers' Corner:
  1. Single-speaker denoisers suppress quiet off-axis voices as 'babble noise', erasing interlocutors!
  2. htdemucs (--two-stems=vocals) isolates ALL human speech (close-mic Da'ee + off-mic interlocutor)
     while stripping Hyde Park wind roar, traffic rumble, sirens, and nasheeds.
  3. Chunk-Checkpointed Execution (15-min chunks with 2s linear crossfades):
     Every 15-min chunk is saved to Drive staging immediately so even a 2.5-hour debate
     disconnected mid-pass on Free Colab resumes from the exact unfinished chunk!
  4. Measures true pre-normalization RMS dBFS (100ms frames) BEFORE loudness leveling so
     downstream role classification knows who wears the primary wireless microphone.
"""

import os
import math
import json
import wave
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from core.drive_sync import SharedWorkspace, STEP_RANK


CHUNK_SEC = 900.0       # 15-minute Demucs chunks (prevents Colab 12.7 GB RAM OOM)
CROSSFADE_SEC = 2.0     # Smooth 2-second linear crossfade between vocal chunks
SAMPLE_RATE_16K = 16000


def decode_audio_to_16k_wav(src_path: Path, dst_wav: Path) -> None:
    """Decodes any media file to 16kHz mono 16-bit PCM WAV using FFmpeg."""
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE_16K), "-c:a", "pcm_s16le",
        str(dst_wav),
    ]
    subprocess.run(cmd, check=True, timeout=600)


def read_wav_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Reads a 16-bit mono PCM WAV into a float32 [-1.0, 1.0] numpy array."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def write_wav_float32(wav_path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE_16K) -> None:
    """Writes a float32 [-1.0, 1.0] mono array to 16-bit PCM WAV."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(clipped.tobytes())


def compute_frame_rms_db(samples: np.ndarray, sr: int = SAMPLE_RATE_16K, hop_sec: float = 0.25) -> List[float]:
    """
    Computes raw pre-normalization RMS loudness (in dBFS) every 250ms.
    Used by speaker_roles.py to distinguish close wireless-mic speakers from
    1.5m interlocutors and 5m passing pedestrians.
    """
    hop = max(1, int(sr * hop_sec))
    rms_db: List[float] = []
    for start in range(0, len(samples), hop):
        frame = samples[start:start + hop]
        if len(frame) == 0:
            continue
        mean_sq = float(np.mean(np.square(frame, dtype=np.float64)))
        db = 10.0 * math.log10(max(mean_sq, 1e-10))
        rms_db.append(round(db, 2))
    return rms_db


def lookup_span_mean_rms_db(rms_timeline: List[float], start_sec: float, end_sec: float, hop_sec: float = 0.25) -> float:
    """Returns mean raw RMS dBFS across [start_sec, end_sec]."""
    if not rms_timeline:
        return -26.0
    i0 = max(0, int(start_sec / hop_sec))
    i1 = min(len(rms_timeline), max(i0 + 1, int(math.ceil(end_sec / hop_sec))))
    seg = rms_timeline[i0:i1]
    return round(sum(seg) / len(seg), 2) if seg else -26.0


class WarmDemucsCleaner:
    """
    Keeps Demucs htdemucs loaded in GPU VRAM (or runs CLI fallback) across a video batch,
    with 15-minute chunk checkpointing to Google Drive staging.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._sr = 44100

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from demucs.pretrained import get_model
        self._model = get_model("htdemucs")
        self._model.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            self._model.to("cuda")
        self._sr = int(getattr(self._model, "samplerate", 44100))

    def isolate_vocals_44k_to_16k(self, chunk_in_wav: Path, chunk_out_16k_wav: Path) -> None:
        """Runs htdemucs --two-stems=vocals on a single <=15-min chunk without RAM spikes."""
        import torch
        from demucs.apply import apply_model

        self._ensure_loaded()
        tmp_44k = chunk_in_wav.with_name(f"{chunk_in_wav.stem}_44k.wav")
        tmp_voc_44k = chunk_in_wav.with_name(f"{chunk_in_wav.stem}_voc44k.wav")
        try:
            # Convert 16k mono slice -> 44.1k stereo float input expected by htdemucs
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(chunk_in_wav),
                    "-ac", "2", "-ar", str(self._sr), "-c:a", "pcm_s16le",
                    str(tmp_44k),
                ],
                check=True,
                timeout=180,
            )

            with wave.open(str(tmp_44k), "rb") as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            stereo = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, 2).T / 32768.0
            wav_tensor = torch.from_numpy(stereo).unsqueeze(0)  # (1, 2, T)

            ref = wav_tensor.mean(0)
            mean = ref.mean()
            std = ref.std() + 1e-8
            wav_norm = (wav_tensor - mean) / std

            target_dev = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            with torch.inference_mode():
                sources = apply_model(
                    self._model,
                    wav_norm.to(target_dev),
                    segment=7.0,
                    overlap=0.25,
                    shifts=0,
                    split=True,
                    device=target_dev,
                )
            # Locate 'vocals' stem index in htdemucs.sources
            vocal_idx = list(self._model.sources).index("vocals")
            vocals = (sources[0, vocal_idx].cpu() * std + mean).mean(dim=0).numpy()
            del sources, wav_tensor, wav_norm
            if target_dev == "cuda":
                torch.cuda.empty_cache()

            write_wav_float32(tmp_voc_44k, vocals, sr=self._sr)
            decode_audio_to_16k_wav(tmp_voc_44k, chunk_out_16k_wav)
        finally:
            tmp_44k.unlink(missing_ok=True)
            tmp_voc_44k.unlink(missing_ok=True)

    def unload(self):
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def stitch_crossfaded_chunks(chunk_files: List[Path], crossfade_sec: float = CROSSFADE_SEC) -> np.ndarray:
    """Stitches 16kHz vocal chunks smoothly using linear crossfades across overlaps."""
    if not chunk_files:
        return np.zeros(0, dtype=np.float32)
    merged, sr = read_wav_float32(chunk_files[0])
    xfade_samples = int(sr * crossfade_sec)

    for cf in chunk_files[1:]:
        nxt, _ = read_wav_float32(cf)
        if xfade_samples > 0 and len(merged) >= xfade_samples and len(nxt) >= xfade_samples:
            fade_out = np.linspace(1.0, 0.0, xfade_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)
            blended = merged[-xfade_samples:] * fade_out + nxt[:xfade_samples] * fade_in
            merged = np.concatenate([merged[:-xfade_samples], blended, nxt[xfade_samples:]])
        else:
            merged = np.concatenate([merged, nxt])
    return merged


def ensure_local_leveled_16k_wav(
    video_id: str,
    workspace: SharedWorkspace,
    local_tmp_dir: Path,
) -> Tuple[Path, List[float]]:
    """
    Fast-resume helper for M3/M4/M5: decodes the already-archived 64kbps clean Opus
    from Google Drive into local /tmp 16kHz WAV in ~1.5 seconds without re-running Demucs!
    """
    clean_opus = workspace.audio_clean_dir / f"{video_id}.opus"
    rms_file = workspace.audio_clean_dir / f"{video_id}_rms.json"
    if not clean_opus.exists():
        raise FileNotFoundError(f"Clean Opus archive missing for {video_id}: {clean_opus}")

    local_wav = local_tmp_dir / f"{video_id}_leveled_16k.wav"
    if not local_wav.exists():
        decode_audio_to_16k_wav(clean_opus, local_wav)

    rms_data = workspace.read_json(rms_file, default={}) or {}
    rms_timeline = rms_data.get("rms_db", [])
    return local_wav, rms_timeline


def clean_vocals_m2(
    video_id: str,
    raw_audio_path: Optional[Path],
    workspace: SharedWorkspace,
    cleaner: WarmDemucsCleaner,
    local_tmp_dir: Path,
    keep_raw_audio_on_drive: bool = False,
    enable_stemming: bool = True,
) -> Tuple[Path, List[float]]:
    """
    Executes Micro-Step M2 (Chunk-Checkpointed Vocal Isolation + Loudness Leveling + 64kbps Opus Archive).
    Returns (local_leveled_16k_wav_path, raw_rms_db_timeline).
    """
    clean_opus = workspace.audio_clean_dir / f"{video_id}.opus"
    rms_json = workspace.audio_clean_dir / f"{video_id}_rms.json"
    state = workspace.get_video_state(video_id)

    # Fast-skip if M2 already completed on a previous Colab session!
    if state["step_rank"] >= STEP_RANK["M2_VOCALS_CLEANED"] and clean_opus.exists() and rms_json.exists():
        return ensure_local_leveled_16k_wav(video_id, workspace, local_tmp_dir)

    if raw_audio_path is None or not raw_audio_path.exists():
        raise FileNotFoundError(f"Raw audio required for M2 on {video_id}, but got {raw_audio_path}")

    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_16k_wav = local_tmp_dir / f"{video_id}_raw16k.wav"
    decode_audio_to_16k_wav(raw_audio_path, raw_16k_wav)
    raw_samples, sr = read_wav_float32(raw_16k_wav)
    total_sec = len(raw_samples) / float(sr)

    if enable_stemming and total_sec > 1.0:
        vid_stage_dir = workspace.staging_dir / video_id
        vid_stage_dir.mkdir(parents=True, exist_ok=True)

        step_samples = int((CHUNK_SEC - CROSSFADE_SEC) * sr)
        win_samples = int(CHUNK_SEC * sr)
        starts = list(range(0, max(1, len(raw_samples)), step_samples))
        if len(starts) > 1 and (len(raw_samples) - starts[-1]) < int(15 * sr):
            starts.pop()

        chunk_paths: List[Path] = []
        for idx, s0 in enumerate(starts):
            s1 = len(raw_samples) if idx == len(starts) - 1 else min(len(raw_samples), s0 + win_samples)
            staged_chunk = vid_stage_dir / f"demucs_chunk_{idx:03d}_of_{len(starts):03d}.wav"
            if not staged_chunk.exists() or staged_chunk.stat().st_size < 1024:
                chunk_in = local_tmp_dir / f"in_chunk_{idx:03d}.wav"
                chunk_out = local_tmp_dir / f"out_chunk_{idx:03d}.wav"
                write_wav_float32(chunk_in, raw_samples[s0:s1], sr=sr)
                cleaner.isolate_vocals_44k_to_16k(chunk_in, chunk_out)
                tmp_stage = vid_stage_dir / f".demucs_chunk_{idx:03d}.tmp.wav"
                shutil.copy2(chunk_out, tmp_stage)
                os.replace(tmp_stage, staged_chunk)
                chunk_in.unlink(missing_ok=True)
                chunk_out.unlink(missing_ok=True)

                workspace.update_substep_progress(
                    video_id,
                    "demucs_progress",
                    {"completed_chunks": idx + 1, "total_chunks": len(starts)},
                )
            chunk_paths.append(staged_chunk)

        vocal_raw_samples = stitch_crossfaded_chunks(chunk_paths, crossfade_sec=CROSSFADE_SEC)
    else:
        vocal_raw_samples = raw_samples

    # 1. Measure raw pre-normalization mic proximity energy BEFORE dynaudnorm!
    rms_timeline = compute_frame_rms_db(vocal_raw_samples, sr=SAMPLE_RATE_16K, hop_sec=0.25)
    workspace.atomic_write_json(
        rms_json,
        {"video_id": video_id, "hop_sec": 0.25, "duration_sec": round(total_sec, 2), "rms_db": rms_timeline},
    )

    vocal_unleveled_wav = local_tmp_dir / f"{video_id}_vocals_unleveled.wav"
    write_wav_float32(vocal_unleveled_wav, vocal_raw_samples, sr=SAMPLE_RATE_16K)
    del raw_samples, vocal_raw_samples

    # 2. Apply High-Pass 80Hz wind/rumble cut + Conversational Dynamic Loudness Leveling
    leveled_wav = local_tmp_dir / f"{video_id}_leveled_16k.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(vocal_unleveled_wav),
            "-af", "highpass=f=80,dynaudnorm=f=150:g=13:p=0.92:m=8.0",
            "-ac", "1", "-ar", str(SAMPLE_RATE_16K), "-c:a", "pcm_s16le",
            str(leveled_wav),
        ],
        check=True,
        timeout=600,
    )
    vocal_unleveled_wav.unlink(missing_ok=True)
    raw_16k_wav.unlink(missing_ok=True)

    # 3. Archive crisp 64 kbps 24kHz VoIP Opus permanently to the 5 TB Google Drive
    tmp_opus = workspace.audio_clean_dir / f".{video_id}.tmp.opus"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(leveled_wav),
            "-ac", "1", "-ar", "24000",
            "-c:a", "libopus", "-b:a", "64k", "-vbr", "on", "-application", "voip",
            str(tmp_opus),
        ],
        check=True,
        timeout=600,
    )
    os.replace(tmp_opus, clean_opus)

    # 4. Clean up Drive staging chunks (and raw download unless user wants both raw + clean)
    shutil.rmtree(workspace.staging_dir / video_id, ignore_errors=True)
    if not keep_raw_audio_on_drive and raw_audio_path and raw_audio_path.exists():
        try:
            raw_audio_path.unlink(missing_ok=True)
        except OSError:
            pass

    workspace.advance_micro_step(
        video_id,
        "M2_VOCALS_CLEANED",
        artifacts={
            "clean_opus": str(clean_opus),
            "rms_json": str(rms_json),
        },
        metrics={
            "duration_sec": round(total_sec, 2),
            "clean_opus_mb": round(clean_opus.stat().st_size / (1024 * 1024), 2),
        },
    )

    return leveled_wav, rms_timeline
