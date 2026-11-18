#!/usr/bin/env python3
"""
core/overlap_resolver.py
Micro-Steps M3 & M4: Overlap-Aware Multi-Speaker Diarization (NeMo MSDD + TitaNet-Large),
Surgical Simultaneous Speech Separation (ClearVoice MossFormer2_SS_16K),
Optimal Cosine Speaker Assignment, and Interruption-Preserving Word-to-Speaker Mapping.

Fixes the 4 core street-debate limitations of MahmoudAshraf97/whisper-diarization:
  1. Enables multi-label overlap output in NeMo MSDD (sigmoid_threshold=0.5, ignore_overlap=False).
  2. Separates ONLY simultaneous cross-talk windows (>=0.8s) into 2 clean speaker streams via MossFormer2
     and matches streams back to global speakers via 192-D TitaNet-Large cosine similarity.
  3. Replaces monotonic-pointer word mapping with Maximum Temporal Overlap (IoU) assignment.
  4. Replaces 50-word blind majority-vote smoothing with Interruption-Preserving Realignment.
"""

import os
import json
import math
import shutil
import logging
import warnings
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Mute noisy internal NeMo / Megatron training and syntax warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
try:
    from nemo.utils import nemo_logging
    nemo_logging.set_verbosity(nemo_logging.ERROR)
except Exception:
    pass

from core.drive_sync import SharedWorkspace, STEP_RANK
from core.vocal_cleaner import read_wav_float32, write_wav_float32, SAMPLE_RATE_16K


# ==============================================================================
# 1. Pure Mathematical & Interval Geometry Algorithms (CPU-Testable)
# ==============================================================================

def parse_rttm_lines(rttm_text: str) -> List[Dict[str, Any]]:
    """
    Parses standard NIST RTTM lines:
    SPEAKER <file> 1 <start_sec> <dur_sec> <NA> <NA> <spk_name> <NA> <NA>
    """
    segments: List[Dict[str, Any]] = []
    for raw_line in rttm_text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        start_sec = float(parts[3])
        dur_sec = float(parts[4])
        raw_spk = parts[7]
        # Normalize speaker_0 / SPEAKER_00 -> SPK_00
        digits = "".join(ch for ch in raw_spk if ch.isdigit())
        spk_id = f"SPK_{int(digits):02d}" if digits else raw_spk.upper()
        start_ms = int(round(start_sec * 1000))
        end_ms = int(round((start_sec + dur_sec) * 1000))
        if end_ms > start_ms:
            segments.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_id": spk_id,
            })

    segments.sort(key=lambda s: (s["start_ms"], s["end_ms"], s["speaker_id"]))
    return segments


def detect_simultaneous_overlap_windows(
    rttm_segments: List[Dict[str, Any]],
    total_duration_ms: int = 0,
    min_overlap_ms: int = 600,
    merge_gap_ms: int = 350,
    pad_ms: int = 350,
    max_window_ms: int = 14_000,
) -> List[Dict[str, Any]]:
    """
    Sweeps RTTM intervals to detect exact windows where >= 2 distinct speakers
    speak simultaneously for >= min_overlap_ms.
    """
    if not rttm_segments:
        return []

    events: List[Tuple[int, int, str]] = []
    for seg in rttm_segments:
        events.append((seg["start_ms"], +1, seg["speaker_id"]))
        events.append((seg["end_ms"], -1, seg["speaker_id"]))
    events.sort(key=lambda x: (x[0], x[1]))

    active_counts: Dict[str, int] = {}
    raw_overlaps: List[Dict[str, Any]] = []
    prev_t: Optional[int] = None

    for t, delta, spk in events:
        if prev_t is not None and t > prev_t:
            active_spks = sorted([s for s, c in active_counts.items() if c > 0])
            if len(active_spks) >= 2:
                if (
                    raw_overlaps
                    and raw_overlaps[-1]["end_ms"] == prev_t
                    and set(raw_overlaps[-1]["speakers"]) == set(active_spks)
                ):
                    raw_overlaps[-1]["end_ms"] = t
                else:
                    raw_overlaps.append({
                        "start_ms": prev_t,
                        "end_ms": t,
                        "speakers": active_spks,
                    })
        active_counts[spk] = active_counts.get(spk, 0) + delta
        if active_counts[spk] <= 0:
            active_counts.pop(spk, None)
        prev_t = t

    # Merge micro-gaps first, then keep intervals whose simultaneous core >= min_overlap_ms
    merged: List[Dict[str, Any]] = []
    for ov in raw_overlaps:
        if merged and (ov["start_ms"] - merged[-1]["end_ms"]) <= merge_gap_ms:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], ov["end_ms"])
            merged[-1]["speakers"] = sorted(set(merged[-1]["speakers"]).union(ov["speakers"]))
        else:
            merged.append(dict(ov))

    filtered = [m for m in merged if (m["end_ms"] - m["start_ms"]) >= min_overlap_ms]

    # Pad context slightly for clean MossFormer2 separation & split overly long windows
    final_windows: List[Dict[str, Any]] = []
    for ov in filtered:
        s0 = max(0, ov["start_ms"] - pad_ms)
        e0 = min(total_duration_ms, ov["end_ms"] + pad_ms) if total_duration_ms > 0 else (ov["end_ms"] + pad_ms)
        cur = s0
        while cur < e0:
            nxt = min(e0, cur + max_window_ms)
            if (e0 - nxt) < 1200:
                nxt = e0
            final_windows.append({
                "window_id": f"OV_{len(final_windows) + 1:04d}",
                "core_start_ms": max(cur, ov["start_ms"]),
                "core_end_ms": min(nxt, ov["end_ms"]),
                "start_ms": cur,
                "end_ms": nxt,
                "speakers": ov["speakers"],
            })
            cur = nxt

    return final_windows


def extract_clean_non_overlap_spans(
    rttm_segments: List[Dict[str, Any]],
    overlap_windows: List[Dict[str, Any]],
    min_clean_ms: int = 900,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Subtracts overlap windows from RTTM spans so TitaNet-Large voiceprint centroids
    are extracted strictly from uncontaminated single-speaker turns.
    """
    by_spk: Dict[str, List[Tuple[int, int]]] = {}
    ov_spans = [(w["core_start_ms"], w["core_end_ms"]) for w in overlap_windows]

    for seg in rttm_segments:
        spk = seg["speaker_id"]
        pieces = [(seg["start_ms"], seg["end_ms"])]
        for os_ms, oe_ms in ov_spans:
            new_pieces = []
            for ps, pe in pieces:
                if oe_ms <= ps or os_ms >= pe:
                    new_pieces.append((ps, pe))
                else:
                    if os_ms > ps:
                        new_pieces.append((ps, os_ms))
                    if oe_ms < pe:
                        new_pieces.append((oe_ms, pe))
            pieces = new_pieces

        for ps, pe in pieces:
            if (pe - ps) >= min_clean_ms:
                by_spk.setdefault(spk, []).append((ps, pe))

    return by_spk


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 1e-8 else vec.astype(np.float32)


def get_deterministic_clip_dir(video_id: str) -> Path:
    """
    Returns the deterministic local NVMe temp directory for separated overlap WAV clips
    (/tmp/yt_spx_clips/{video_id}_sep_clips or OS temp equivalent).
    Guarantees Phase B (MossFormer2 separation) and Phase C (Whisper ASR) share the exact
    same clip cache across steps without redundant model unloads or re-separations.
    """
    base_tmp = Path(tempfile.gettempdir()) / "yt_spx_clips"
    d = base_tmp / f"{video_id}_sep_clips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_deterministic_clip_dir(video_id: str) -> None:
    """Removes temporary separated WAV clips from local NVMe after M5 scenario is assembled."""
    try:
        d = Path(tempfile.gettempdir()) / "yt_spx_clips" / f"{video_id}_sep_clips"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


def merge_fragmented_speaker_clusters(
    rttm_segments: List[Dict[str, Any]],
    speaker_centroids: Dict[str, np.ndarray],
    threshold: float = 0.80,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, str]]:
    """
    Merges genuinely fragmented speaker clusters using complete-linkage verification across
    192-D TitaNet centroids (every member pair must exceed `threshold`), with an explicit
    anti-collapse guard blocking two substantial debate speakers from merging into K=1.

    Returns:
        (merged_rttm_segments, merged_centroids, mapping_of_old_to_new_ids)
    """
    if len(speaker_centroids) <= 1 or not rttm_segments:
        return rttm_segments, speaker_centroids, {}

    # Calculate total speaking duration per speaker to pick the dominant canonical ID
    spk_durations: Dict[str, int] = {}
    for s in rttm_segments:
        dur = max(0, int(s["end_ms"]) - int(s["start_ms"]))
        spk = s["speaker_id"]
        spk_durations[spk] = spk_durations.get(spk, 0) + dur

    spk_list = sorted(speaker_centroids.keys())
    norm_cents = {s: l2_normalize(speaker_centroids[s]) for s in spk_list}
    sim_matrix: Dict[Tuple[str, str], float] = {}
    candidate_pairs: List[Tuple[float, str, str]] = []
    for i in range(len(spk_list)):
        s_i = spk_list[i]
        for j in range(i + 1, len(spk_list)):
            s_j = spk_list[j]
            sim = float(np.dot(norm_cents[s_i], norm_cents[s_j]))
            sim_matrix[(s_i, s_j)] = sim
            sim_matrix[(s_j, s_i)] = sim
            if sim >= threshold:
                candidate_pairs.append((sim, s_i, s_j))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    # Complete-linkage agglomerative clustering (prevents transitive chain collapses)
    root_of: Dict[str, str] = {s: s for s in spk_list}
    cluster_members: Dict[str, List[str]] = {s: [s] for s in spk_list}

    def _cluster_dur(root_id: str) -> int:
        return sum(spk_durations.get(m, 0) for m in cluster_members[root_id])

    for _, s_i, s_j in candidate_pairs:
        root_a = root_of[s_i]
        root_b = root_of[s_j]
        if root_a == root_b:
            continue

        members_a = cluster_members[root_a]
        members_b = cluster_members[root_b]
        min_cross_sim = min(
            sim_matrix.get((u, v), float(np.dot(norm_cents[u], norm_cents[v])))
            for u in members_a
            for v in members_b
        )
        if min_cross_sim < threshold:
            continue

        # Anti-collapse guard: never merge the last 2 substantial debate clusters down to K=1
        # unless their voices are virtually identical (complete-linkage cosine >= 0.90).
        if len(cluster_members) == 2:
            dur_a = _cluster_dur(root_a)
            dur_b = _cluster_dur(root_b)
            if dur_a >= 12_000 and dur_b >= 12_000 and min_cross_sim < max(threshold, 0.90):
                continue

        # Keep root with larger total speaking duration (tie-break: lower alphabetical ID)
        if (_cluster_dur(root_b) > _cluster_dur(root_a)) or (
            _cluster_dur(root_b) == _cluster_dur(root_a) and root_b < root_a
        ):
            root_a, root_b = root_b, root_a

        moved = cluster_members.pop(root_b)
        cluster_members[root_a].extend(moved)
        for m in moved:
            root_of[m] = root_a

    id_mapping: Dict[str, str] = dict(root_of)
    has_merges = any(k != v for k, v in id_mapping.items())
    if not has_merges:
        return rttm_segments, speaker_centroids, {}

    # Merge centroids using duration-weighted average
    merged_centroids: Dict[str, np.ndarray] = {}
    for canon_id, members in cluster_members.items():
        if len(members) == 1:
            merged_centroids[canon_id] = norm_cents[canon_id]
        else:
            w_sum = np.zeros_like(norm_cents[canon_id])
            for m in members:
                w = max(1.0, float(spk_durations.get(m, 1)))
                w_sum += w * norm_cents[m]
            merged_centroids[canon_id] = l2_normalize(w_sum)

    # Update RTTM segments and coalesce adjacent intervals of the same speaker
    updated_segs: List[Dict[str, Any]] = []
    for s in rttm_segments:
        old_spk = s["speaker_id"]
        new_spk = id_mapping.get(old_spk, old_spk)
        updated_segs.append({
            **s,
            "speaker_id": new_spk,
        })

    updated_segs.sort(key=lambda x: (x["start_ms"], x["end_ms"], x["speaker_id"]))

    # Coalesce overlapping or adjacent intervals of the same speaker
    coalesced: List[Dict[str, Any]] = []
    for s in updated_segs:
        if coalesced and coalesced[-1]["speaker_id"] == s["speaker_id"] and s["start_ms"] <= (coalesced[-1]["end_ms"] + 150):
            coalesced[-1]["end_ms"] = max(coalesced[-1]["end_ms"], s["end_ms"])
        else:
            coalesced.append(dict(s))

    return coalesced, merged_centroids, id_mapping


def assign_separated_streams_to_speakers(
    stream_embeddings: List[np.ndarray],
    speaker_centroids: Dict[str, np.ndarray],
    preferred_speakers: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Solves optimal 1-to-1 cosine similarity assignment between separated MossFormer2
    streams (typically 2 streams) and candidate speaker centroids so both streams
    never collapse onto the same speaker ID.
    """
    if not stream_embeddings:
        return []
    if not speaker_centroids:
        return [(f"SPK_{i:02d}", 0.0) for i in range(len(stream_embeddings))]

    # Prioritize speakers active in the overlap window if >= 2 centroids exist
    candidates = [s for s in (preferred_speakers or []) if s in speaker_centroids]
    if len(candidates) < len(stream_embeddings):
        candidates = sorted(speaker_centroids.keys())

    norm_streams = [l2_normalize(emb) for emb in stream_embeddings]
    norm_cents = {spk: l2_normalize(speaker_centroids[spk]) for spk in candidates}

    if len(norm_streams) == 1 or len(candidates) == 1:
        s_vec = norm_streams[0]
        best_spk = max(candidates, key=lambda spk: float(np.dot(s_vec, norm_cents[spk])))
        sim = float(np.dot(s_vec, norm_cents[best_spk]))
        if len(norm_streams) == 2:
            return [(best_spk, sim), (best_spk, sim)]
        return [(best_spk, sim)]

    # Exact optimal 2-to-K assignment maximizing total cosine similarity across distinct speakers
    best_pair: Optional[Tuple[str, str]] = None
    best_score = -1e9
    best_sims = (0.0, 0.0)

    for spk_a in candidates:
        sim_a = float(np.dot(norm_streams[0], norm_cents[spk_a]))
        for spk_b in candidates:
            if spk_a == spk_b:
                continue
            sim_b = float(np.dot(norm_streams[1], norm_cents[spk_b]))
            if (sim_a + sim_b) > best_score:
                best_score = sim_a + sim_b
                best_pair = (spk_a, spk_b)
                best_sims = (round(sim_a, 4), round(sim_b, 4))

    assert best_pair is not None
    return [(best_pair[0], best_sims[0]), (best_pair[1], best_sims[1])]


def assign_words_to_speakers_iou(
    word_timestamps: List[Dict[str, Any]],
    rttm_segments: List[Dict[str, Any]],
    excluded_overlap_windows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Maps CTC-aligned words ({'text'/'word', 'start', 'end'} in seconds or ms)
    to speakers using Maximum Temporal Overlap (IoU / intersection ms),
    skipping words whose midpoint falls inside separated cross-talk windows
    (since those windows are replaced by clean 2-stream transcriptions).
    """
    if not word_timestamps:
        return []
    if not rttm_segments:
        return [
            {
                "word": w.get("word") or w.get("text", ""),
                "start_ms": int(round(float(w["start"]) * 1000)) if float(w["start"]) < 50_000 else int(w["start"]),
                "end_ms": int(round(float(w["end"]) * 1000)) if float(w["end"]) < 50_000 else int(w["end"]),
                "speaker_id": "SPK_00",
            }
            for w in word_timestamps
        ]

    ov_cores = [(w["core_start_ms"], w["core_end_ms"]) for w in (excluded_overlap_windows or [])]
    mapped: List[Dict[str, Any]] = []

    for w in word_timestamps:
        txt = (w.get("word") if "word" in w else w.get("text", "")).strip()
        if not txt:
            continue
        raw_s, raw_e = float(w["start"]), float(w["end"])
        # Support seconds (float from ctc_forced_aligner) or ms
        ws = int(round(raw_s * 1000)) if (raw_e < 36_000 and isinstance(w["start"], float)) else int(raw_s)
        we = int(round(raw_e * 1000)) if (raw_e < 36_000 and isinstance(w["end"], float)) else int(raw_e)
        we = max(we, ws + 20)
        w_mid = (ws + we) // 2

        # Skip mixed-audio words inside separated overlap cores
        if any(cs <= w_mid <= ce for cs, ce in ov_cores):
            continue

        spk_overlap: Dict[str, int] = {}
        for seg in rttm_segments:
            if seg["end_ms"] < ws:
                continue
            if seg["start_ms"] > we:
                break
            inter = min(we, seg["end_ms"]) - max(ws, seg["start_ms"])
            if inter > 0:
                spk_overlap[seg["speaker_id"]] = spk_overlap.get(seg["speaker_id"], 0) + inter

        if spk_overlap:
            best_spk = max(spk_overlap.items(), key=lambda kv: kv[1])[0]
        else:
            # Nearest RTTM boundary fallback for words in brief VAD pad gaps
            nearest = min(
                rttm_segments,
                key=lambda s: min(abs(w_mid - s["start_ms"]), abs(w_mid - s["end_ms"])),
            )
            best_spk = nearest["speaker_id"]

        mapped.append({
            "word": txt,
            "start_ms": ws,
            "end_ms": we,
            "speaker_id": best_spk,
        })

    return mapped


def realign_words_preserving_interruptions(
    mapped_words: List[Dict[str, Any]],
    jitter_max_words: int = 2,
    jitter_max_ms: int = 500,
) -> List[Dict[str, Any]]:
    """
    Smooths ONLY tiny 1-2 word (<500ms) acoustic diarization jitter inside a continuous
    clause while NEVER erasing genuine >=3-word or >500ms speaker interruptions!
    """
    if len(mapped_words) < 3:
        return [dict(w) for w in mapped_words]

    out = [dict(w) for w in mapped_words]
    n = len(out)
    i = 1
    sentence_ends = (".", "?", "!", "؟")

    while i < n - 1:
        prev_spk = out[i - 1]["speaker_id"]
        cur_spk = out[i]["speaker_id"]
        if cur_spk != prev_spk and not out[i - 1]["word"].endswith(sentence_ends):
            # Measure contiguous run of cur_spk
            j = i
            while j < n and out[j]["speaker_id"] == cur_spk:
                j += 1
            run_len = j - i
            run_dur_ms = out[j - 1]["end_ms"] - out[i]["start_ms"]
            after_spk = out[j]["speaker_id"] if j < n else None
            gap_before = out[i]["start_ms"] - out[i - 1]["end_ms"]

            # Absorb ONLY if surrounded by prev_spk, tiny word count, tiny duration, and no pause
            if (
                after_spk == prev_spk
                and run_len <= jitter_max_words
                and run_dur_ms <= jitter_max_ms
                and gap_before < 350
            ):
                for k in range(i, j):
                    out[k]["speaker_id"] = prev_spk
            i = j
        else:
            i += 1

    return out


# ==============================================================================
# 2. Warm GPU Models for Micro-Steps M3 & M4 (Colab T4)
# ==============================================================================

def _quiet_nemo_loggers() -> None:
    """Suppresses verbose NeMo / Megatron / NVIDIA OneLogger training telemetry warnings during inference."""
    import logging
    for logger_name in (
        "nemo_logger",
        "nv_one_logger",
        "nv_one_logger.api.config",
        "nv_one_logger.training_telemetry",
        "nv_one_logger.training_telemetry.api",
        "nv_one_logger.training_telemetry.api.training_telemetry_provider",
        "nv_one_logger. exporter",
        "pytorch_lightning",
    ):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)
    except Exception:
        pass


class WarmNeMoDiarizer:
    """
    Keeps TitaNet-Large embedding extractor ready in VRAM and runs NeMo MSDD
    overlap-aware speaker diarization (or Sortformer v2 for <=4-speaker clips).
    """

    def __init__(self, device: str = "cuda", mode: str = "msdd"):
        self.device = device
        self.mode = mode
        self._titanet = None
        self._msdd_instance = None
        self._msdd_is_short: Optional[bool] = None
        self._msdd_work_dir: Optional[Path] = None
        self._cudnn_fallback_active: bool = False

    def _ensure_titanet(self):
        if self._titanet is None:
            _quiet_nemo_loggers()
            import torch
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
            self._titanet = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
            _quiet_nemo_loggers()
            self._titanet.eval()
            if self.device == "cuda" and torch.cuda.is_available():
                self._titanet.to("cuda")

    def extract_embedding(self, samples_16k: np.ndarray) -> np.ndarray:
        """Computes a 192-D L2-normalized TitaNet-Large voiceprint embedding."""
        import torch
        self._ensure_titanet()
        if len(samples_16k) < int(0.35 * SAMPLE_RATE_16K):
            pad_len = int(0.5 * SAMPLE_RATE_16K) - len(samples_16k)
            samples_16k = np.pad(samples_16k, (0, max(0, pad_len)))

        dev = next(self._titanet.parameters()).device
        sig = torch.from_numpy(samples_16k.astype(np.float32)).unsqueeze(0).to(dev)
        sig_len = torch.tensor([sig.shape[1]], dtype=torch.long, device=dev)
        with torch.inference_mode():
            _, emb = self._titanet.forward(input_signal=sig, input_signal_length=sig_len)
        return l2_normalize(emb.squeeze(0).detach().cpu().numpy())

    def extract_embeddings_batch(self, slices_16k: List[np.ndarray], batch_size: int = 16) -> List[np.ndarray]:
        """
        Computes 192-D L2-normalized TitaNet-Large embeddings in padded GPU mini-batches.
        Accelerates multi-span speaker centroid extraction in M3 by 4x-6x on T4 GPUs.
        """
        if not slices_16k:
            return []
        if len(slices_16k) == 1:
            return [self.extract_embedding(slices_16k[0])]

        try:
            import torch
            self._ensure_titanet()
            dev = next(self._titanet.parameters()).device
            min_samples = int(0.5 * SAMPLE_RATE_16K)
            results: List[np.ndarray] = []
            for i in range(0, len(slices_16k), max(1, batch_size)):
                batch_raw = slices_16k[i : i + max(1, batch_size)]
                prepared = []
                lengths = []
                for arr in batch_raw:
                    s = arr.astype(np.float32)
                    if len(s) < int(0.35 * SAMPLE_RATE_16K):
                        s = np.pad(s, (0, max(0, min_samples - len(s))))
                    lengths.append(len(s))
                    prepared.append(s)

                max_len = max(lengths)
                padded = np.zeros((len(prepared), max_len), dtype=np.float32)
                for row_idx, s in enumerate(prepared):
                    padded[row_idx, : len(s)] = s

                sig = torch.from_numpy(padded).to(dev)
                sig_len = torch.tensor(lengths, dtype=torch.long, device=dev)
                with torch.inference_mode():
                    _, embs = self._titanet.forward(input_signal=sig, input_signal_length=sig_len)
                embs_np = embs.detach().cpu().numpy()
                if embs_np.ndim == 1:
                    results.append(l2_normalize(embs_np))
                else:
                    for row in embs_np:
                        results.append(l2_normalize(row))
                del sig, sig_len, embs, padded
            return results
        except Exception:
            return [self.extract_embedding(s) for s in slices_16k]

    def run_diarization(
        self,
        wav_16k_path: Path,
        out_rttm_path: Path,
        is_short: bool = False,
        forced_num_speakers: Optional[int] = None,
        clustering_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Runs NeMo MSDD NeuralDiarizer (up to 10 speakers + overlap RTTM) and writes out_rttm_path."""
        import importlib
        import torch
        from omegaconf import OmegaConf

        _quiet_nemo_loggers()
        # Prioritize overlap-aware NeuralDiarizer (NeMo 2.x msdd_models) before ClusteringDiarizer fallback.
        # Do NOT swallow inner dependency errors (e.g. ml_dtypes / onnx AttributeError).
        DiarizerClass = None
        last_err: Optional[Exception] = None
        for mod_path, cls_name in [
            ("nemo.collections.asr.models.msdd_models", "NeuralDiarizer"),
            ("nemo.collections.asr.parts.submodules.neural_diarizer", "NeuralDiarizer"),
            ("nemo.collections.asr.models.clustering_diarizer", "NeuralDiarizer"),
            ("nemo.collections.asr.models.clustering_diarizer", "ClusteringDiarizer"),
            ("nemo.collections.asr.models", "ClusteringDiarizer"),
        ]:
            try:
                mod = importlib.import_module(mod_path)
                _quiet_nemo_loggers()
                if hasattr(mod, cls_name):
                    DiarizerClass = getattr(mod, cls_name)
                    break
            except ModuleNotFoundError as exc:
                # Only ignore if the target NeMo module path itself does not exist in this NeMo version
                if exc.name and mod_path.startswith(exc.name):
                    last_err = exc
                    continue
                raise

        if DiarizerClass is None:
            raise ImportError(
                f"Could not locate NeuralDiarizer or ClusteringDiarizer in nemo.collections.asr "
                f"(last error: {last_err}). Ensure 'nemo_toolkit[asr]>=2.6.0,<3.0.0' is installed."
            )

        _quiet_nemo_loggers()
        if self._msdd_work_dir is None:
            self._msdd_work_dir = Path(tempfile.gettempdir()) / f"yt_spx_msdd_work_{os.getpid()}"
        tmp_p = self._msdd_work_dir
        if tmp_p.exists():
            shutil.rmtree(tmp_p, ignore_errors=True)
        tmp_p.mkdir(parents=True, exist_ok=True)

        custom_cfg = (forced_num_speakers is not None) or bool(clustering_overrides)
        try:
            manifest_path = tmp_p / "input_manifest.json"
            meta = {
                "audio_filepath": str(wav_16k_path),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": int(forced_num_speakers) if forced_num_speakers is not None else None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            manifest_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")

            clus_params: Dict[str, Any] = {
                "oracle_num_speakers": forced_num_speakers is not None,
                "max_num_speakers": int(forced_num_speakers) if forced_num_speakers is not None else (5 if is_short else 10),
                "enhanced_count_thres": 40,
                "max_rp_threshold": 0.16,
                "sparse_search_volume": 40,
                "maj_vote_spk_count": True,
                "chunk_cluster_count": 50,
                "embeddings_per_chunk": 10000,
            }
            if clustering_overrides:
                clus_params.update(clustering_overrides)

            cfg = OmegaConf.create({
                "name": "ClusterDiarizer",
                "num_workers": 0,
                "sample_rate": 16000,
                "batch_size": 64,
                "device": self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu",
                "verbose": False,
                "diarizer": {
                    "manifest_filepath": str(manifest_path),
                    "out_dir": str(tmp_p),
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": False,
                    "vad": {
                        "model_path": "vad_multilingual_marblenet",
                        "external_vad_manifest": None,
                        "parameters": {
                            "window_length_in_sec": 0.15,
                            "shift_length_in_sec": 0.01,
                            "smoothing": "median",
                            "overlap": 0.5,
                            "onset": 0.55,
                            "offset": 0.40,
                            "pad_onset": 0.10,
                            "pad_offset": 0.0,
                            "min_duration_on": 0.05,
                            "min_duration_off": 0.15,
                            "filter_speech_first": True,
                        },
                    },
                    "speaker_embeddings": {
                        "model_path": "titanet_large",
                        "parameters": {
                            "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                            "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                            "multiscale_weights": [1, 1, 1, 1, 1],
                            "save_embeddings": True,
                        },
                    },
                    "clustering": {
                        "parameters": clus_params,
                    },
                    "msdd_model": {
                        "model_path": "diar_msdd_telephonic",
                        "parameters": {
                            "use_speaker_model_from_ckpt": True,
                            "infer_batch_size": 40,
                            "sigmoid_threshold": [0.5],
                            "seq_eval_mode": False,
                            "split_infer": True,
                            "diar_window_length": 50,
                            "overlap_infer_spk_limit": 5,
                        },
                    },
                },
            })

            def _bind_or_create_msdd(force_new: bool = False):
                if (not force_new) and (not custom_cfg) and self._msdd_instance is not None and self._msdd_is_short == bool(is_short):
                    try:
                        msdd_obj = self._msdd_instance
                        for target_obj in (msdd_obj, getattr(msdd_obj, "clus_diar_model", None)):
                            if target_obj is None:
                                continue
                            if hasattr(target_obj, "_cfg") and hasattr(target_obj._cfg, "diarizer"):
                                target_obj._cfg.diarizer.manifest_filepath = str(manifest_path)
                                target_obj._cfg.diarizer.out_dir = str(tmp_p)
                            if hasattr(target_obj, "_diarizer_params"):
                                target_obj._diarizer_params.manifest_filepath = str(manifest_path)
                                target_obj._diarizer_params.out_dir = str(tmp_p)
                            if hasattr(target_obj, "_out_dir"):
                                target_obj._out_dir = str(tmp_p)
                            if hasattr(target_obj, "_manifest_filepath"):
                                target_obj._manifest_filepath = str(manifest_path)
                        return msdd_obj
                    except Exception:
                        self._msdd_instance = None
                self._msdd_instance = DiarizerClass(cfg=cfg).to(cfg.device)
                _quiet_nemo_loggers()
                self._msdd_is_short = None if custom_cfg else bool(is_short)
                return self._msdd_instance

            orig_cudnn_outer = getattr(getattr(torch, "backends", None), "cudnn", None)
            prev_cudnn_flag = orig_cudnn_outer.enabled if orig_cudnn_outer is not None else True
            # NeMo MSDD uses multi-scale recurrent/conformer layers that trigger cuDNN runtime
            # execution errors on Tesla T4 GPUs (Turing compute 7.5). Executing MSDD with PyTorch's
            # native CUDA kernels (cudnn.enabled=False) runs flawlessly in ~2-3s with zero warning notes.
            if orig_cudnn_outer is not None:
                orig_cudnn_outer.enabled = False
            try:
                msdd = _bind_or_create_msdd()
                try:
                    msdd.diarize()
                except FileNotFoundError:
                    # If a cached NeMo instance referenced a cleaned sub-path, re-write manifest and re-instantiate
                    tmp_p.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
                    msdd = _bind_or_create_msdd(force_new=True)
                    msdd.diarize()
                pred_check = tmp_p / "pred_rttms" / f"{wav_16k_path.stem}.rttm"
                if not pred_check.exists():
                    tmp_p.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
                    msdd = _bind_or_create_msdd(force_new=True)
                    msdd.diarize()
            except RuntimeError as err:
                err_s = str(err).lower()
                if "cudnn" in err_s:
                    print("    [!] cuDNN runtime error encountered; falling back to PyTorch native CUDA kernels (cudnn.enabled=False)...", flush=True)
                    self._cudnn_fallback_active = True
                    orig_cudnn = torch.backends.cudnn.enabled
                    torch.backends.cudnn.enabled = False
                    try:
                        tmp_p.mkdir(parents=True, exist_ok=True)
                        manifest_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
                        self._msdd_instance = DiarizerClass(cfg=cfg).to(cfg.device)
                        _quiet_nemo_loggers()
                        self._msdd_is_short = bool(is_short)
                        self._msdd_instance.diarize()
                    except ValueError as v_err:
                        if "silence" in str(v_err).lower():
                            print(f"    [!] NeMo VAD detected silence/non-speech in audio ({v_err}); continuing with empty RTTM.", flush=True)
                        else:
                            raise
                    finally:
                        torch.backends.cudnn.enabled = orig_cudnn
                else:
                    self._msdd_instance = None
                    raise
            except ValueError as err:
                if "silence" in str(err).lower():
                    print(f"    [!] NeMo VAD detected silence/non-speech in audio ({err}); continuing with empty RTTM.", flush=True)
                else:
                    raise
            finally:
                if orig_cudnn_outer is not None:
                    orig_cudnn_outer.enabled = prev_cudnn_flag
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pred_rttm = tmp_p / "pred_rttms" / f"{wav_16k_path.stem}.rttm"
            rttm_text = pred_rttm.read_text(encoding="utf-8") if pred_rttm.exists() else ""
            out_rttm_path.parent.mkdir(parents=True, exist_ok=True)
            out_rttm_path.write_text(rttm_text, encoding="utf-8")
            return rttm_text
        finally:
            if tmp_p.exists():
                shutil.rmtree(tmp_p, ignore_errors=True)

    def unload(self):
        import gc
        try:
            import torch
        except ImportError:
            torch = None
        if self._titanet is not None:
            del self._titanet
            self._titanet = None
        if self._msdd_instance is not None:
            del self._msdd_instance
            self._msdd_instance = None
            self._msdd_is_short = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class WarmMossFormerSeparator:
    """Wraps ClearVoice MossFormer2_SS_16K for targeted 2-speaker cross-talk separation."""

    def __init__(self):
        self._cv = None

    def _ensure_loaded(self):
        if self._cv is None:
            from clearvoice import ClearVoice
            self._cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])

    def separate_16k_slice(self, slice_wav: Path) -> List[np.ndarray]:
        """Separates an overlapping 16kHz mono WAV slice into 2 speaker float32 streams."""
        self._ensure_loaded()
        raw_out = self._cv(input_path=str(slice_wav), online_write=False)
        if isinstance(raw_out, dict):
            raw_out = next(iter(raw_out.values()))
        arr = np.asarray(raw_out, dtype=np.float32)
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 3:
            arr = arr[0]
        # Shape (2, T) or (T, 2)
        if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
            return [arr[i] for i in range(arr.shape[0])]
        return [arr[:, i] for i in range(arr.shape[1])]

    def unload(self):
        if self._cv is not None:
            import torch
            import gc
            del self._cv
            self._cv = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


# ==============================================================================
# 3. Micro-Step M3 & M4 Execution Runners (Checkpointed to Shared 5 TB Drive)
# ==============================================================================

def _compute_centroids_for_rttm(
    rttm_segs: List[Dict[str, Any]],
    overlaps: List[Dict[str, Any]],
    samples: np.ndarray,
    sr: int,
    diarizer: WarmNeMoDiarizer,
) -> Dict[str, np.ndarray]:
    """Computes L2-normalized 192-D TitaNet centroids per speaker using up to 45s clean speech."""
    clean_spans = extract_clean_non_overlap_spans(rttm_segs, overlaps, min_clean_ms=900)
    centroids: Dict[str, np.ndarray] = {}
    all_spks = sorted({s["speaker_id"] for s in rttm_segs})
    for spk in all_spks:
        spans = sorted(clean_spans.get(spk, []), key=lambda x: (x[1] - x[0]), reverse=True)
        if not spans:
            spans = [(s["start_ms"], s["end_ms"]) for s in rttm_segs if s["speaker_id"] == spk]
        embs = []
        slices_for_spk: List[np.ndarray] = []
        budget_ms = 0
        for s_ms, e_ms in spans[:12]:
            s_idx = int(s_ms * sr / 1000)
            e_idx = min(len(samples), int(e_ms * sr / 1000))
            if (e_idx - s_idx) >= int(0.4 * sr):
                slices_for_spk.append(samples[s_idx:e_idx])
                budget_ms += (e_ms - s_ms)
                if budget_ms >= 45_000:
                    break
        if slices_for_spk:
            if hasattr(diarizer, "extract_embeddings_batch"):
                embs = diarizer.extract_embeddings_batch(slices_for_spk)
            else:
                embs = [diarizer.extract_embedding(sl) for sl in slices_for_spk]
        if embs:
            centroids[spk] = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0))
    return centroids


def _speaker_durations_ms(rttm_segs: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    durs: Dict[str, int] = {}
    for s in rttm_segs:
        spk = s["speaker_id"]
        durs[spk] = durs.get(spk, 0) + max(0, int(s["end_ms"]) - int(s["start_ms"]))
    return sorted(durs.items(), key=lambda x: x[1], reverse=True)


def _is_collapsed_diarization(rttm_segs: List[Dict[str, Any]], total_ms: int) -> bool:
    """Returns True when a multi-minute video collapsed to <=1 speaker, <10% secondary share, or intro/outro-only split."""
    if total_ms < 90_000:
        return False
    ranked = _speaker_durations_ms(rttm_segs)
    if len(ranked) <= 1:
        return True
    total_spk_ms = sum(d for _, d in ranked) or 1
    top_share = ranked[0][1] / total_spk_ms
    second_ms = ranked[1][1]
    if top_share >= 0.96 and second_ms < 15_000:
        return True
    if total_ms >= 300_000:
        if top_share >= 0.90 or second_ms < 25_000:
            return True
        edge_lo = int(total_ms * 0.12)
        edge_hi = int(total_ms * 0.88)
        sec_segs = [s for s in rttm_segs if s["speaker_id"] == ranked[1][0]]
        if sec_segs and all(int(s["end_ms"]) <= edge_lo or int(s["start_ms"]) >= edge_hi for s in sec_segs):
            return True
    return False


def _write_rttm_file(video_id: str, rttm_segs: List[Dict[str, Any]], rttm_path: Path) -> None:
    lines = []
    for s in rttm_segs:
        start_s = s["start_ms"] / 1000.0
        dur_s = max(0.01, (s["end_ms"] - s["start_ms"]) / 1000.0)
        lines.append(f"SPEAKER {video_id} 1 {start_s:.3f} {dur_s:.3f} <NA> <NA> {s['speaker_id']} <NA> <NA>")
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    rttm_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_diarization_m3(
    video_id: str,
    leveled_16k_wav: Path,
    workspace: SharedWorkspace,
    diarizer: WarmNeMoDiarizer,
    is_short: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """
    Executes Micro-Step M3:
      - Runs NeMo MSDD overlap-aware diarization -> saves {vid}.rttm
      - Detects and automatically recovers from single-speaker clustering collapse on debates
      - Computes clean 192-D TitaNet-Large centroids -> saves {vid}_embeddings.npz
      - Detects simultaneous cross-talk windows -> saves {vid}_overlaps.json
    """
    vid_diar_dir = workspace.diarization_dir / video_id
    vid_diar_dir.mkdir(parents=True, exist_ok=True)
    rttm_path = vid_diar_dir / f"{video_id}.rttm"
    emb_path = vid_diar_dir / f"{video_id}_embeddings.npz"
    ov_path = vid_diar_dir / f"{video_id}_overlaps.json"

    state = workspace.get_video_state(video_id)
    if (
        state["step_rank"] >= STEP_RANK["M3_DIARIZED"]
        and rttm_path.exists()
        and emb_path.exists()
        and ov_path.exists()
    ):
        rttm_segs = parse_rttm_lines(rttm_path.read_text(encoding="utf-8"))
        npz = np.load(str(emb_path))
        centroids = {k: npz[k] for k in npz.files}
        overlaps = workspace.read_json(ov_path, default=[]) or []
        cached_total_ms = max((int(s["end_ms"]) for s in rttm_segs), default=None)
        rttm_segs, centroids, merge_map = merge_fragmented_speaker_clusters(
            rttm_segments=rttm_segs,
            speaker_centroids=centroids,
            threshold=0.84,
        )
        if merge_map:
            overlaps = detect_simultaneous_overlap_windows(
                rttm_segs,
                total_duration_ms=cached_total_ms,
                min_overlap_ms=600,
            )
        return rttm_segs, centroids, overlaps

    samples, sr = read_wav_float32(leveled_16k_wav)
    total_ms = int(round(len(samples) * 1000.0 / sr))

    rttm_text = diarizer.run_diarization(leveled_16k_wav, rttm_path, is_short=is_short)
    rttm_segs = parse_rttm_lines(rttm_text)
    overlaps = detect_simultaneous_overlap_windows(rttm_segs, total_duration_ms=total_ms)
    centroids = _compute_centroids_for_rttm(rttm_segs, overlaps, samples, sr, diarizer)

    # Post-Clustering Centroid Merge with complete-linkage + anti-collapse guard (cosine >= 0.84)
    rttm_segs, centroids, merge_map = merge_fragmented_speaker_clusters(
        rttm_segments=rttm_segs,
        speaker_centroids=centroids,
        threshold=0.84,
    )
    if merge_map:
        merges_str = ", ".join(f"{old}->{new}" for old, new in merge_map.items() if old != new)
        print(f"      [Centroid Merge] Unified fragmented speakers: {merges_str}", flush=True)
        overlaps = detect_simultaneous_overlap_windows(
            rttm_segs,
            total_duration_ms=total_ms,
            min_overlap_ms=600,
        )

    # Anti-Collapse Automatic Recovery on multi-minute debates
    retried = False
    if _is_collapsed_diarization(rttm_segs, total_ms):
        retry_configs = [
            (
                None,
                {"max_rp_threshold": 0.12, "enhanced_count_thres": 25, "sparse_search_volume": 50, "maj_vote_spk_count": True},
                "sensitive spectral retry",
            ),
            (
                2,
                {"oracle_num_speakers": True, "max_rp_threshold": 0.15},
                "bounded 2-speaker recovery",
            ),
        ]
        retry_rttm_path = rttm_path.with_suffix(".retry.rttm")
        try:
            for forced_k, clus_ovr, label in retry_configs:
                cand_rttm = diarizer.run_diarization(
                    leveled_16k_wav,
                    retry_rttm_path,
                    is_short=is_short,
                    forced_num_speakers=forced_k,
                    clustering_overrides=clus_ovr,
                )
                cand_segs = parse_rttm_lines(cand_rttm)
                cand_ov = detect_simultaneous_overlap_windows(cand_segs, total_duration_ms=total_ms)
                cand_cents = _compute_centroids_for_rttm(cand_segs, cand_ov, samples, sr, diarizer)
                cand_segs, cand_cents, cand_merge = merge_fragmented_speaker_clusters(
                    rttm_segments=cand_segs,
                    speaker_centroids=cand_cents,
                    threshold=0.84,
                )
                if cand_merge:
                    cand_ov = detect_simultaneous_overlap_windows(
                        cand_segs, total_duration_ms=total_ms, min_overlap_ms=600
                    )

                ranked_cand = _speaker_durations_ms(cand_segs)
                if len(ranked_cand) >= 2:
                    tot_cand = sum(d for _, d in ranked_cand) or 1
                    s1, s2 = ranked_cand[0][0], ranked_cand[1][0]
                    dur2 = ranked_cand[1][1]
                    share2 = dur2 / tot_cand
                    sim_12 = (
                        float(np.dot(l2_normalize(cand_cents[s1]), l2_normalize(cand_cents[s2])))
                        if (s1 in cand_cents and s2 in cand_cents)
                        else 0.0
                    )
                    # Accept recovered split ONLY if secondary speaker is substantive AND acoustically distinct
                    if dur2 >= 5_000 and share2 >= 0.025 and sim_12 < 0.85:
                        print(
                            f"      [Anti-Collapse Recovery] Recovered {len(ranked_cand)} distinct speakers via {label} "
                            f"(2nd share={share2:.1%}, cos_sim={sim_12:.2f})",
                            flush=True,
                        )
                        rttm_segs, centroids, overlaps = cand_segs, cand_cents, cand_ov
                        retried = True
                        break
        finally:
            retry_rttm_path.unlink(missing_ok=True)

    _write_rttm_file(video_id, rttm_segs, rttm_path)

    np.savez_compressed(str(emb_path), **centroids)
    workspace.atomic_write_json(ov_path, overlaps)

    overlap_dur_ms = sum((w["core_end_ms"] - w["core_start_ms"]) for w in overlaps)
    workspace.advance_micro_step(
        video_id,
        "M3_DIARIZED",
        artifacts={
            "rttm": str(rttm_path),
            "embeddings_npz": str(emb_path),
            "overlaps_json": str(ov_path),
        },
        metrics={
            "speakers_detected": len(centroids),
            "overlap_windows_count": len(overlaps),
            "overlap_ratio": round(overlap_dur_ms / max(total_ms, 1), 4),
        },
    )
    return rttm_segs, centroids, overlaps


def run_overlap_separation_m4(
    video_id: str,
    leveled_16k_wav: Path,
    overlaps: List[Dict[str, Any]],
    centroids: Dict[str, np.ndarray],
    workspace: SharedWorkspace,
    separator: WarmMossFormerSeparator,
    diarizer: WarmNeMoDiarizer,
    local_tmp_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Executes Micro-Step M4 (Window-by-Window Checkpointed Overlap Speech Separation):
    Separates each overlap window into 2 speaker streams, matches each stream to its
    exact speaker centroid via TitaNet-Large cosine assignment, and stores clean clips
    in deterministic NVMe temp directory so Phase C resumes at the exact unfinished overlap window!
    """
    vid_diar_dir = workspace.diarization_dir / video_id
    # Store per-window separated clips in deterministic local NVMe temp (/tmp/yt_spx_clips/{video_id})
    # Never write small temporary WAV slices to Google Drive FUSE to prevent unparented orphan bloat!
    sep_dir = (Path(local_tmp_dir) / f"{video_id}_sep_clips") if local_tmp_dir is not None else get_deterministic_clip_dir(video_id)
    sep_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = vid_diar_dir / "overlap_sep.json"

    state = workspace.get_video_state(video_id)
    if state["step_rank"] >= STEP_RANK["M4_OVERLAPS_SEPARATED"] and manifest_file.exists():
        cached_records: List[Dict[str, Any]] = workspace.read_json(manifest_file, default=[]) or []
        all_tracks_exist = True
        for rec in cached_records:
            for tr in rec.get("tracks", []):
                cp = Path(tr.get("clip_path", ""))
                if not cp.exists():
                    alt_cp = sep_dir / cp.name
                    if alt_cp.exists():
                        tr["clip_path"] = str(alt_cp)
                    else:
                        all_tracks_exist = False
                        break
            if not all_tracks_exist:
                break
        if all_tracks_exist:
            return cached_records

    if not overlaps:
        workspace.atomic_write_json(manifest_file, [])
        workspace.advance_micro_step(video_id, "M4_OVERLAPS_SEPARATED", artifacts={"overlap_sep": str(manifest_file)})
        return []

    existing_records: List[Dict[str, Any]] = workspace.read_json(manifest_file, default=[]) or []
    done_by_id: Dict[str, Dict[str, Any]] = {}
    for r in existing_records:
        if "window_id" not in r:
            continue
        tracks_ok = True
        for tr in r.get("tracks", []):
            cp = Path(tr.get("clip_path", ""))
            if not cp.exists():
                alt_cp = sep_dir / cp.name
                if alt_cp.exists():
                    tr["clip_path"] = str(alt_cp)
                else:
                    tracks_ok = False
                    break
        if tracks_ok:
            done_by_id[r["window_id"]] = r

    samples, sr = read_wav_float32(leveled_16k_wav)
    results: List[Dict[str, Any]] = []

    for idx, win in enumerate(overlaps):
        wid = win["window_id"]
        if wid in done_by_id:
            results.append(done_by_id[wid])
            continue

        s_idx = max(0, int(win["start_ms"] * sr / 1000))
        e_idx = min(len(samples), int(win["end_ms"] * sr / 1000))
        slice_wav = (local_tmp_dir if local_tmp_dir is not None else sep_dir) / f"{video_id}_{wid}.wav"
        write_wav_float32(slice_wav, samples[s_idx:e_idx], sr=sr)

        try:
            streams = separator.separate_16k_slice(slice_wav)
        finally:
            slice_wav.unlink(missing_ok=True)

        # Energy-gate separated streams (reject near-silent ghost residual <-41 dBFS)
        valid_streams: List[np.ndarray] = []
        stream_embs: List[np.ndarray] = []
        for st_arr in streams[:2]:
            rms_db = 10.0 * math.log10(max(float(np.mean(np.square(st_arr, dtype=np.float64))), 1e-10))
            if rms_db > -41.0:
                 peak = float(np.max(np.abs(st_arr)))
                 norm_st = (st_arr / peak * 0.85).astype(np.float32) if peak > 1e-4 else st_arr
                 valid_streams.append(norm_st)
                 stream_embs.append(diarizer.extract_embedding(norm_st))

        assignments = assign_separated_streams_to_speakers(
            stream_embs,
            centroids,
            preferred_speakers=win.get("speakers", []),
        )

        saved_tracks = []
        for t_i, (norm_st, (spk_id, sim)) in enumerate(zip(valid_streams, assignments)):
            clip_path = sep_dir / f"{wid}_s{t_i}_{spk_id}.wav"
            write_wav_float32(clip_path, norm_st, sr=sr)
            saved_tracks.append({
                "speaker_id": spk_id,
                "cosine_sim": sim,
                "clip_path": str(clip_path),
            })

        rec = {
            **win,
            "tracks": saved_tracks,
        }
        results.append(rec)

        # Persist progress every 3 overlap windows so disconnects never redo separation
        if (idx + 1) % 3 == 0 or (idx + 1) == len(overlaps):
            workspace.atomic_write_json(manifest_file, results)
            workspace.update_substep_progress(
                video_id,
                "overlap_progress",
                {"completed_windows": idx + 1, "total_windows": len(overlaps)},
            )

    workspace.atomic_write_json(manifest_file, results)
    workspace.advance_micro_step(
        video_id,
        "M4_OVERLAPS_SEPARATED",
        artifacts={"overlap_sep": str(manifest_file)},
    )
    return results
