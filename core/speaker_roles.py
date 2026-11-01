#!/usr/bin/env python3
"""
core/speaker_roles.py
3-Tier Street-Debate Speaker Role Classifier, Cross-Video Da'ee Voiceprint Registry,
and Automatic Interlocutor-Episode Segmenter.

Assigns every diarized speaker cluster (SPK_00..SPK_N) to one of:
  - MAIN_DAEE (or CO_DAEE): Primary wireless-mic host / Muslim apologist
  - INTERLOCUTOR (1..K): Active debate partner engaging in multi-turn back-and-forth
  - PEDESTRIAN: Passing street crowd interjection / brief background shout
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from core.drive_sync import SharedWorkspace
from core.metadata_linker import get_default_speaker_profiles
from core.vocal_cleaner import lookup_span_mean_rms_db
from core.overlap_resolver import l2_normalize


class VoiceprintRegistry:
    """
    Manages 192-D TitaNet-Large voiceprint centroids for recurring Speakers' Corner Da'ees
    on the shared 5 TB Google Drive (profiles/speaker_profiles.json + .npz).
    Auto-enrolls and refines centroids using high-confidence single-Da'ee titled videos!
    """

    def __init__(self, workspace: SharedWorkspace):
        self.workspace = workspace
        self.meta_path = workspace.profiles_dir / "speaker_profiles.json"
        self.npz_path = workspace.profiles_dir / "speaker_profiles.npz"
        self.profiles: Dict[str, Dict[str, Any]] = self._load_metadata()
        self.centroids: Dict[str, np.ndarray] = self._load_centroids()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        defaults = {p["profile_id"]: {**p, "enrolled_samples": 0} for p in get_default_speaker_profiles()}
        saved = self.workspace.read_json(self.meta_path, default={})
        if isinstance(saved, dict):
            for k, v in saved.items():
                defaults[k] = {**defaults.get(k, {}), **v}
        return defaults

    def _load_centroids(self) -> Dict[str, np.ndarray]:
        if not self.npz_path.exists():
            return {}
        try:
            npz = np.load(str(self.npz_path))
            return {k: l2_normalize(npz[k]) for k in npz.files}
        except Exception:
            return {}

    def match_speaker(
        self,
        embedding: Optional[np.ndarray],
        title_hint_profile_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Matches a speaker's 192-D TitaNet centroid against enrolled Da'ee voiceprints.
        Uses stricter threshold (0.72) globally, or (0.62) when corroborated by Arabic title hint.
        """
        if embedding is None or not self.centroids:
            return None
        emb = l2_normalize(embedding)
        best_pid = None
        best_sim = -1.0
        for pid, cent in self.centroids.items():
            sim = float(np.dot(emb, cent))
            if sim > best_sim:
                best_sim = sim
                best_pid = pid

        if best_pid is None:
            return None

        threshold = 0.62 if (title_hint_profile_id and best_pid == title_hint_profile_id) else 0.72
        if best_sim >= threshold:
            prof = self.profiles.get(best_pid, {"profile_id": best_pid, "name_en": best_pid, "name_ar": best_pid})
            return {
                "profile_id": best_pid,
                "name_en": prof["name_en"],
                "name_ar": prof["name_ar"],
                "similarity": round(best_sim, 4),
            }
        return None

    def maybe_enroll_or_update(
        self,
        profile_id: str,
        embedding: Optional[np.ndarray],
        speech_share: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Safely updates the shared Da'ee voiceprint centroid when a video has an unambiguous
        single-Da'ee title hint and a clearly dominant wireless-mic MAIN_DAEE (>=32% speech share).
        """
        if not profile_id or embedding is None or speech_share < 0.32:
            return None
        if profile_id not in self.profiles:
            return None

        emb = l2_normalize(embedding)
        if self.workspace.acquire_lock("_voiceprint_registry", ttl_sec=120, verify_delay_sec=0.2 if self.workspace.is_drive else 0.0):
            try:
                self.profiles = self._load_metadata()
                self.centroids = self._load_centroids()
                n_prev = int(self.profiles[profile_id].get("enrolled_samples", 0))
                if profile_id in self.centroids and n_prev > 0:
                    # Guard against mislabeled cameo video: require >=0.52 similarity to existing centroid
                    if float(np.dot(emb, self.centroids[profile_id])) >= 0.52:
                        alpha = 1.0 / min(n_prev + 1, 20)
                        self.centroids[profile_id] = l2_normalize((1.0 - alpha) * self.centroids[profile_id] + alpha * emb)
                        self.profiles[profile_id]["enrolled_samples"] = n_prev + 1
                else:
                    self.centroids[profile_id] = emb
                    self.profiles[profile_id]["enrolled_samples"] = 1

                np.savez_compressed(str(self.npz_path), **self.centroids)
                self.workspace.atomic_write_json(self.meta_path, self.profiles)
            finally:
                self.workspace.release_lock("_voiceprint_registry")

        prof = self.profiles[profile_id]
        return {
            "profile_id": profile_id,
            "name_en": prof["name_en"],
            "name_ar": prof["name_ar"],
            "similarity": 1.0,
        }


def classify_speakers_and_episodes(
    raw_turns: List[Dict[str, Any]],
    speaker_centroids: Dict[str, np.ndarray],
    rms_timeline: List[float],
    video_record: Dict[str, Any],
    registry: Optional[VoiceprintRegistry] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Classifies speakers into MAIN_DAEE / CO_DAEE / INTERLOCUTOR_1..K / PEDESTRIAN
    and annotates every chronological turn with speaker labels, pedestrian flags,
    and automatic Interlocutor Episode indices (1..E).
    """
    if not raw_turns:
        return [], []

    # 1. Aggregate per-speaker acoustic & dialogic metrics
    spk_stats: Dict[str, Dict[str, Any]] = {}
    for idx, t in enumerate(raw_turns):
        spk = t["speaker_id"]
        dur_sec = max(0.05, (t["end_ms"] - t["start_ms"]) / 1000.0)
        rms_db = lookup_span_mean_rms_db(rms_timeline, t["start_ms"] / 1000.0, t["end_ms"] / 1000.0)
        st = spk_stats.setdefault(spk, {
            "speaker_id": spk,
            "speaking_sec": 0.0,
            "turn_count": 0,
            "weighted_rms_sum": 0.0,
            "first_ms": t["start_ms"],
            "alternations": 0,
        })
        st["speaking_sec"] += dur_sec
        st["turn_count"] += 1
        st["weighted_rms_sum"] += rms_db * dur_sec
        if idx > 0 and raw_turns[idx - 1]["speaker_id"] != spk:
            st["alternations"] += 1

    total_speech_sec = max(0.1, sum(s["speaking_sec"] for s in spk_stats.values()))
    for st in spk_stats.values():
        st["mean_rms_db"] = round(st["weighted_rms_sum"] / max(st["speaking_sec"], 0.05), 2)
        st["speech_share"] = st["speaking_sec"] / total_speech_sec

    primary_hint = video_record.get("primary_daee_hint")
    daee_hints = video_record.get("daee_hints") or []

    # 2. Match voiceprints & select MAIN_DAEE
    voice_matches: Dict[str, Dict[str, Any]] = {}
    if registry is not None:
        for spk in spk_stats:
            m = registry.match_speaker(speaker_centroids.get(spk), title_hint_profile_id=primary_hint)
            if m:
                voice_matches[spk] = m

    # Score candidate MAIN_DAEE: voiceprint match + speech share + wireless-mic loudness + turn count
    max_rms = max(s["mean_rms_db"] for s in spk_stats.values())

    def _daee_score(s: Dict[str, Any]) -> float:
        spk = s["speaker_id"]
        rms_bonus = max(0.0, 1.0 - (max_rms - s["mean_rms_db"]) / 15.0) * 0.25
        vp_bonus = 0.45 if spk in voice_matches else 0.0
        return s["speech_share"] * 0.65 + rms_bonus + vp_bonus

    main_daee_spk = max(spk_stats.values(), key=_daee_score)["speaker_id"]
    main_stats = spk_stats[main_daee_spk]

    # Bootstrap/refine voiceprint if unambiguous single Da'ee in title
    main_match = voice_matches.get(main_daee_spk)
    if main_match is None and primary_hint:
        if registry is not None and speaker_centroids.get(main_daee_spk) is not None:
            main_match = registry.maybe_enroll_or_update(
                primary_hint,
                speaker_centroids.get(main_daee_spk),
                main_stats["speech_share"],
            )
        if main_match is None and len(daee_hints) == 1:
            main_match = {
                "profile_id": daee_hints[0]["profile_id"],
                "name_en": daee_hints[0]["name_en"],
                "name_ar": daee_hints[0]["name_ar"],
                "similarity": 0.0,
            }

    # Count direct turn alternations specifically with MAIN_DAEE
    daee_exchanges: Dict[str, int] = {spk: 0 for spk in spk_stats}
    for i in range(1, len(raw_turns)):
        a, b = raw_turns[i - 1]["speaker_id"], raw_turns[i]["speaker_id"]
        if a == main_daee_spk and b != main_daee_spk:
            daee_exchanges[b] += 1
        elif b == main_daee_spk and a != main_daee_spk:
            daee_exchanges[a] += 1

    # 3. Classify remaining speakers into CO_DAEE, INTERLOCUTOR_1..K, or PEDESTRIAN
    other_spks = sorted(
        [s for s in spk_stats.values() if s["speaker_id"] != main_daee_spk],
        key=lambda x: x["first_ms"],
    )

    speaker_records: Dict[str, Dict[str, Any]] = {}
    main_name_en = main_match["name_en"] if main_match else "Main Da'ee"
    main_name_ar = main_match["name_ar"] if main_match else "الداعية المحاور"

    speaker_records[main_daee_spk] = {
        "speaker_id": main_daee_spk,
        "role": "MAIN_DAEE",
        "interlocutor_index": 0,
        "profile_id": main_match["profile_id"] if main_match else None,
        "display_name_en": main_name_en,
        "display_name_ar": main_name_ar,
        "speaking_sec": round(main_stats["speaking_sec"], 2),
        "turn_count": main_stats["turn_count"],
        "mean_rms_db": main_stats["mean_rms_db"],
    }

    interlocutor_counter = 0
    for st in other_spks:
        spk = st["speaker_id"]
        vp = voice_matches.get(spk)
        rms_drop = main_stats["mean_rms_db"] - st["mean_rms_db"]
        exch = daee_exchanges.get(spk, 0)

        if vp is not None:
            speaker_records[spk] = {
                "speaker_id": spk,
                "role": "CO_DAEE",
                "interlocutor_index": 0,
                "profile_id": vp["profile_id"],
                "display_name_en": vp["name_en"],
                "display_name_ar": vp["name_ar"],
                "speaking_sec": round(st["speaking_sec"], 2),
                "turn_count": st["turn_count"],
                "mean_rms_db": st["mean_rms_db"],
            }
            continue

        # Pedestrian Interruption Criteria:
        # Very brief total speech (<2.5% share & <12s) AND minimal dialogic alternation (<=2 turns with Da'ee)
        # OR isolated 1-turn far-field shout (>8.5 dB quieter than primary mic & <6s)
        is_pedestrian = (
            len(spk_stats) > 2
            and (
                (st["speech_share"] < 0.025 and st["speaking_sec"] < 12.0 and exch <= 2)
                or (st["turn_count"] == 1 and st["speaking_sec"] < 6.0 and rms_drop > 8.5)
            )
        )

        if is_pedestrian:
            speaker_records[spk] = {
                "speaker_id": spk,
                "role": "PEDESTRIAN",
                "interlocutor_index": 0,
                "profile_id": None,
                "display_name_en": "Pedestrian Interruption",
                "display_name_ar": "مداخلة عابرة من الجمهور",
                "speaking_sec": round(st["speaking_sec"], 2),
                "turn_count": st["turn_count"],
                "mean_rms_db": st["mean_rms_db"],
            }
        else:
            interlocutor_counter += 1
            label_en = "Interlocutor" if len(other_spks) == 1 else f"Interlocutor {interlocutor_counter}"
            label_ar = "المحاور" if len(other_spks) == 1 else f"المحاور {interlocutor_counter}"
            speaker_records[spk] = {
                "speaker_id": spk,
                "role": "INTERLOCUTOR",
                "interlocutor_index": interlocutor_counter,
                "profile_id": None,
                "display_name_en": label_en,
                "display_name_ar": label_ar,
                "speaking_sec": round(st["speaking_sec"], 2),
                "turn_count": st["turn_count"],
                "mean_rms_db": st["mean_rms_db"],
            }

    # 4. Segment Chronological Turns into Interlocutor Episodes (1..E)
    current_episode = 1
    active_interlocutor: Optional[str] = None
    enriched_turns: List[Dict[str, Any]] = []

    for idx, t in enumerate(raw_turns):
        spk = t["speaker_id"]
        srec = speaker_records[spk]
        role = srec["role"]

        if role == "INTERLOCUTOR":
            if active_interlocutor is None:
                active_interlocutor = spk
            elif spk != active_interlocutor:
                # Confirm genuine interlocutor handoff (look ahead: new speaker speaks >= 2 times in next 8 turns)
                lookahead = [rt["speaker_id"] for rt in raw_turns[idx:idx + 8]]
                if lookahead.count(spk) >= 2 and lookahead.count(active_interlocutor) == 0:
                    current_episode += 1
                    active_interlocutor = spk

        enriched_turns.append({
            **t,
            "turn_index": idx,
            "episode_index": current_episode,
            "speaker_role": role,
            "speaker_label_en": srec["display_name_en"],
            "speaker_label_ar": srec["display_name_ar"],
            "is_pedestrian_interruption": (role == "PEDESTRIAN"),
        })

    ordered_speakers = sorted(
        speaker_records.values(),
        key=lambda x: (0 if x["role"] == "MAIN_DAEE" else (1 if x["role"] == "CO_DAEE" else (2 if x["role"] == "INTERLOCUTOR" else 3)), -x["speaking_sec"]),
    )
    return ordered_speakers, enriched_turns
