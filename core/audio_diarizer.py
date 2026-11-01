#!/usr/bin/env python3
"""
core/audio_diarizer.py
Micro-Step M5 (and End-to-End Stage 1 Runner M1 -> M5):
Combines Warm Faster-Whisper (large-v3-turbo float16), CTC Word Forced Aligner
(mms-300m-1130-forced-aligner), Multilingual Punctuation Restoration, Overlap Stream
Transcription, Interval-IoU Speaker Mapping, and 3-Tier Semantic Role Classification
to produce complete, accurate Scenario Transcripts on Shared 5 TB Google Drive.
"""

import os
import re
import gc
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from core.drive_sync import SharedWorkspace, STEP_RANK
from core.yt_audio_fetcher import (
    fetch_video_audio_m1,
    AntiBanCircuitBreakerError,
    PermanentVideoUnavailableError,
)
from core.vocal_cleaner import (
    WarmDemucsCleaner,
    clean_vocals_m2,
    ensure_local_leveled_16k_wav,
    read_wav_float32,
)
from core.overlap_resolver import (
    WarmNeMoDiarizer,
    WarmMossFormerSeparator,
    run_diarization_m3,
    run_overlap_separation_m4,
    assign_words_to_speakers_iou,
    realign_words_preserving_interruptions,
)
from core.speaker_roles import VoiceprintRegistry, classify_speakers_and_episodes


THEOLOGICAL_VOCAB_PROMPT_EN = (
    "Speakers' Corner Dawah philosophical and theological street debate. "
    "Allah, Quran, Surah, Ayah, Prophet Muhammad, Tawheed, Shirk, Sunnah, Hadith, "
    "Trinity, Jesus, Messiah, Christ, Bible, Torah, Injeel, Gospel, John, Isaiah, Paul, "
    "Contingency, Necessary Being, Objective Morality, Scientism, Sheikh Uthman, Hamza, Mohammed Hijab, Mansur."
)

HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by the amara.org community",
    "amara.org",
    "اشترك في القناة",
    "شكرا للمشاهدة",
}

LANG_ISO3 = {
    "en": "eng",
    "ar": "ara",
    "ur": "urd",
    "fr": "fre",
    "de": "ger",
    "es": "spa",
    "tr": "tur",
    "id": "ind",
    "ru": "rus",
}

PUNCT_SUPPORTED_LANGS = {"en", "fr", "de", "es", "it", "nl", "pt", "bg", "pl", "cs", "sk", "sl"}


def ms_to_fmt(ms: int, always_hours: bool = False) -> str:
    ms = max(0, int(ms))
    total_sec = ms // 1000
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0 or always_hours:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def ms_to_srt_ts(ms: int) -> str:
    ms = max(0, int(ms))
    hrs = ms // 3_600_000
    rem = ms % 3_600_000
    mins = rem // 60_000
    rem %= 60_000
    secs = rem // 1_000
    millis = rem % 1_000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def is_hallucinated_segment(text: str, no_speech_prob: float = 0.0, compression_ratio: float = 1.0) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if no_speech_prob > 0.72 and len(t.split()) <= 6:
        return True
    if compression_ratio > 2.65:
        return True
    t_low = t.lower().strip(" .,!?؟")
    if t_low in HALLUCINATION_PHRASES:
        return True
    # Detect pathological single-word/phrase loops ("yeah yeah yeah yeah yeah")
    words = t_low.split()
    if len(words) >= 8 and len(set(words)) <= 2:
        return True
    return False


def build_turns_from_aligned_words(
    realigned_words: List[Dict[str, Any]],
    language: str = "en",
    max_turn_words: int = 85,
    pause_split_ms: int = 1800,
) -> List[Dict[str, Any]]:
    """
    Groups speaker-attributed CTC-aligned words into coherent conversational turns.
    Splits whenever the speaker changes OR at natural sentence boundaries after a long pause/length.
    """
    if not realigned_words:
        return []

    turns: List[Dict[str, Any]] = []
    cur_words: List[Dict[str, Any]] = [realigned_words[0]]
    sent_ends = (".", "?", "!", "؟")

    for w in realigned_words[1:]:
        prev = cur_words[-1]
        speaker_changed = w["speaker_id"] != prev["speaker_id"]
        gap_ms = w["start_ms"] - prev["end_ms"]
        ended_sentence = prev["word"].rstrip().endswith(sent_ends)

        should_split = (
            speaker_changed
            or (ended_sentence and gap_ms >= pause_split_ms and len(cur_words) >= 12)
            or (ended_sentence and len(cur_words) >= max_turn_words)
            or len(cur_words) >= (max_turn_words + 30)
        )

        if should_split:
            turns.append(_pack_word_turn(cur_words, language=language))
            cur_words = [w]
        else:
            cur_words.append(w)

    if cur_words:
        turns.append(_pack_word_turn(cur_words, language=language))

    return turns


def _pack_word_turn(words: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
    text = " ".join(w["word"].strip() for w in words if w.get("word")).strip()
    start_ms = int(words[0]["start_ms"])
    end_ms = int(words[-1]["end_ms"])
    return {
        "speaker_id": words[0]["speaker_id"],
        "start_ms": start_ms,
        "end_ms": max(end_ms, start_ms + 80),
        "start_fmt": ms_to_fmt(start_ms),
        "end_fmt": ms_to_fmt(end_ms),
        "language": language,
        "text_original": text,
        "text_ar": text if language == "ar" else "",
        "is_overlap": False,
        "overlap_group_id": "",
        "confidence": 0.94,
        "words": [
            {"w": w["word"], "s": int(w["start_ms"]), "e": int(w["end_ms"])}
            for w in words
        ],
    }


def merge_clean_and_overlap_turns(
    clean_turns: List[Dict[str, Any]],
    overlap_turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Interleaves non-overlapping turns and 2-stream separated simultaneous cross-talk turns
    in exact chronological order, coalescing adjacent non-overlap fragments of the same speaker.
    """
    combined = list(clean_turns) + list(overlap_turns)
    combined.sort(key=lambda t: (t["start_ms"], 0 if not t.get("is_overlap") else 1, t["end_ms"]))

    coalesced: List[Dict[str, Any]] = []
    for t in combined:
        if not (t.get("text_original") or "").strip():
            continue
        if (
            coalesced
            and not coalesced[-1].get("is_overlap")
            and not t.get("is_overlap")
            and coalesced[-1]["speaker_id"] == t["speaker_id"]
            and (t["start_ms"] - coalesced[-1]["end_ms"]) <= 900
            and len((coalesced[-1]["text_original"] + " " + t["text_original"]).split()) <= 95
        ):
            prev = coalesced[-1]
            prev["end_ms"] = max(prev["end_ms"], t["end_ms"])
            prev["end_fmt"] = ms_to_fmt(prev["end_ms"])
            prev["text_original"] = f"{prev['text_original']} {t['text_original']}".strip()
            if prev.get("language") == "ar":
                prev["text_ar"] = prev["text_original"]
            prev["words"] = prev.get("words", []) + t.get("words", [])
        else:
            coalesced.append(dict(t))

    return coalesced


def format_diarized_srt(turns: List[Dict[str, Any]]) -> str:
    """Renders speaker-labeled, overlap-annotated SRT subtitle text."""
    blocks = []
    for idx, t in enumerate(turns, start=1):
        role_tag = ""
        if t.get("is_pedestrian_interruption"):
            role_tag = " [Pedestrian]"
        elif t.get("is_overlap"):
            role_tag = " [Overlap]"
        spk_label = t.get("speaker_label_en") or t.get("speaker_id", "Speaker")
        clean_txt = (t.get("text_original") or "").strip().replace("-->", "->")
        blocks.append(
            f"{idx}\n"
            f"{ms_to_srt_ts(t['start_ms'])} --> {ms_to_srt_ts(t['end_ms'])}\n"
            f"[{spk_label}{role_tag}]: {clean_txt}\n"
        )
    return "\n".join(blocks)


class WarmASRAlignmentEngine:
    """
    Keeps Faster-Whisper (large-v3-turbo float16), CTC Forced Aligner (mms-300m-1130),
    and Multilingual Punctuation warm in GPU memory across a batch of videos (~4.2 GB VRAM).
    """

    def __init__(
        self,
        whisper_model_name: str = "large-v3-turbo",
        device: str = "cuda",
        batch_size: int = 12,
    ):
        self.whisper_model_name = whisper_model_name
        self.device = device
        self.batch_size = batch_size
        self._whisper = None
        self._whisper_batched = None
        self._align_model = None
        self._align_tokenizer = None
        self._punct_model = None

    def _ensure_whisper(self):
        if self._whisper is None:
            import faster_whisper
            compute_type = "float16" if self.device == "cuda" else "int8"
            self._whisper = faster_whisper.WhisperModel(
                self.whisper_model_name,
                device=self.device,
                compute_type=compute_type,
            )
            self._whisper_batched = faster_whisper.BatchedInferencePipeline(model=self._whisper)

    def _ensure_aligner(self):
        if self._align_model is None:
            import torch
            from ctc_forced_aligner import load_alignment_model
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._align_model, self._align_tokenizer = load_alignment_model(self.device, dtype=dtype)

    def _ensure_punct(self):
        if self._punct_model is None:
            from deepmultilingualpunctuation import PunctuationModel
            self._punct_model = PunctuationModel(model="kredor/punctuate-all")

    def transcribe_and_align_wav(
        self,
        wav_path: Path,
        forced_language: Optional[str] = None,
        use_ctc_aligner: bool = True,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Transcribes a 16kHz WAV file with Faster-Whisper + Hallucination Shield,
        then refines word-level [start, end] timestamps via CTC Forced Aligner.
        """
        import torch
        import faster_whisper

        self._ensure_whisper()
        audio_waveform = faster_whisper.decode_audio(str(wav_path))
        if len(audio_waveform) < 3200:
            return [], forced_language or "en"

        # Detect language on first 45s if not forced
        lang = forced_language
        if not lang:
            _, info_probe = self._whisper.transcribe(audio_waveform[: min(len(audio_waveform), 16000 * 45)], vad_filter=True)
            lang = info_probe.language or "en"

        init_prompt = THEOLOGICAL_VOCAB_PROMPT_EN if lang == "en" else None

        if self.batch_size > 0 and len(audio_waveform) > 16000 * 25:
            segments_iter, info = self._whisper_batched.transcribe(
                audio_waveform,
                language=lang,
                initial_prompt=init_prompt,
                batch_size=self.batch_size,
                word_timestamps=not use_ctc_aligner,
            )
        else:
            segments_iter, info = self._whisper.transcribe(
                audio_waveform,
                language=lang,
                initial_prompt=init_prompt,
                vad_filter=True,
                word_timestamps=not use_ctc_aligner,
            )

        clean_segs = []
        fallback_words: List[Dict[str, Any]] = []
        for seg in segments_iter:
            no_sp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            comp = float(getattr(seg, "compression_ratio", 1.0) or 1.0)
            if is_hallucinated_segment(seg.text, no_sp, comp):
                continue
            clean_segs.append(seg.text.strip())
            if getattr(seg, "words", None):
                for w in seg.words:
                    if w.word and w.word.strip():
                        fallback_words.append({
                            "text": w.word.strip(),
                            "start": float(w.start),
                            "end": float(w.end),
                        })
            else:
                # Approximate proportional word slices if CTC aligner is skipped on tiny overlap clip
                ws_tokens = seg.text.strip().split()
                if ws_tokens:
                    dur = max(0.05, float(seg.end - seg.start)) / len(ws_tokens)
                    for i_tok, tok in enumerate(ws_tokens):
                        fallback_words.append({
                            "text": tok,
                            "start": float(seg.start) + i_tok * dur,
                            "end": float(seg.start) + (i_tok + 1) * dur,
                        })

        full_text = " ".join(clean_segs).strip()
        if not full_text:
            return [], lang

        # Run CTC Forced Aligner for exact phonetic word boundaries
        iso3 = LANG_ISO3.get(lang)
        if use_ctc_aligner and iso3:
            try:
                from ctc_forced_aligner import (
                    generate_emissions,
                    preprocess_text,
                    get_alignments,
                    get_spans,
                    postprocess_results,
                )
                self._ensure_aligner()
                wav_tensor = torch.from_numpy(audio_waveform).to(self._align_model.dtype).to(self._align_model.device)
                emissions, stride = generate_emissions(
                    self._align_model,
                    wav_tensor,
                    batch_size=max(4, self.batch_size // 2),
                )
                del wav_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                tokens_starred, text_starred = preprocess_text(
                    full_text,
                    romanize=True,
                    language=iso3,
                )
                segments, scores, blank_token = get_alignments(
                    emissions,
                    tokens_starred,
                    self._align_tokenizer,
                )
                spans = get_spans(tokens_starred, segments, blank_token)
                word_timestamps = postprocess_results(text_starred, spans, stride, scores)
                return word_timestamps, lang
            except Exception as e:
                print(f"   [!] CTC alignment fallback note ({lang}): {e}")

        return fallback_words, lang

    def restore_punctuation_in_place(self, mapped_words: List[Dict[str, Any]], language: str) -> None:
        if not mapped_words or language not in PUNCT_SUPPORTED_LANGS:
            return
        try:
            self._ensure_punct()
            words_only = [w["word"] for w in mapped_words]
            labeled = self._punct_model.predict(words_only, chunk_size=230)
            ending_puncts = ".?!"
            model_puncts = ".,;:!?"
            is_acronym = lambda s: bool(re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", s))

            for w_dict, (_, punct_char, *_) in zip(mapped_words, labeled):
                w_txt = w_dict["word"]
                if w_txt and punct_char in ending_puncts and (w_txt[-1] not in model_puncts or is_acronym(w_txt)):
                    w_txt = (w_txt + punct_char).rstrip(".") + (punct_char if punct_char != "." else ".")
                    w_dict["word"] = w_txt
        except Exception as e:
            print(f"   [!] Punctuation restoration skipped: {e}")

    def unload(self):
        import torch
        self._whisper = None
        self._whisper_batched = None
        self._align_model = None
        self._align_tokenizer = None
        self._punct_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Stage1AudioPipeline:
    """
    Stateful Stage 1 Worker (M1 -> M5) designed for 2-4 hr/day Free Colab sessions.
    Processes small anti-ban batches, saving every micro-step to the shared 5 TB Drive.
    """

    def __init__(
        self,
        workspace: SharedWorkspace,
        whisper_model: str = "large-v3-turbo",
        device: str = "cuda",
        batch_size: int = 12,
        keep_raw_audio: bool = False,
    ):
        self.workspace = workspace
        self.device = device
        self.keep_raw_audio = keep_raw_audio

        self.demucs = WarmDemucsCleaner(device=device)
        self.diarizer = WarmNeMoDiarizer(device=device)
        self.separator = WarmMossFormerSeparator()
        self.asr = WarmASRAlignmentEngine(whisper_model_name=whisper_model, device=device, batch_size=batch_size)
        self.registry = VoiceprintRegistry(workspace=workspace)

    def process_single_video(self, video_record: Dict[str, Any], allow_youtube_download: bool = True) -> Optional[Dict[str, Any]]:
        """
        Runs all unfinished micro-steps from M1 up to M5_SCENARIO_BUILT for a single video.
        Resumes seamlessly from whichever micro-step was last saved on Google Drive!
        """
        vid = video_record["video_id"]
        scen_path = self.workspace.scenarios_dir / f"{vid}.json"
        srt_path = self.workspace.scenarios_dir / f"{vid}.diarized.srt"

        state = self.workspace.get_video_state(vid)
        if state["step_rank"] >= STEP_RANK["M5_SCENARIO_BUILT"] and scen_path.exists():
            return self.workspace.read_json(scen_path)

        # If download is disallowed (e.g. circuit breaker tripped), only process videos already at >= M1!
        if not allow_youtube_download and state["step_rank"] < STEP_RANK["M1_DOWNLOADED"]:
            return None

        local_tmp = Path(tempfile.mkdtemp(prefix=f"yt_spx_s1_{vid}_"))
        try:
            # --- MICRO-STEP M1: Temp Download -> Drive Staging ---
            raw_audio_path, info_meta = fetch_video_audio_m1(video_record, self.workspace)
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # --- MICRO-STEP M2: Chunked Vocal Isolation + 64kbps Opus Archive ---
            leveled_wav, rms_timeline = clean_vocals_m2(
                video_id=vid,
                raw_audio_path=raw_audio_path,
                workspace=self.workspace,
                cleaner=self.demucs,
                local_tmp_dir=local_tmp,
                keep_raw_audio_on_drive=self.keep_raw_audio,
            )
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # Free Demucs VRAM if tight before NeMo + MossFormer2 + Whisper
            self.demucs.unload()

            # --- MICRO-STEP M3: Overlap-Aware NeMo MSDD Diarization + TitaNet Centroids ---
            rttm_segs, centroids, overlaps = run_diarization_m3(
                video_id=vid,
                leveled_16k_wav=leveled_wav,
                workspace=self.workspace,
                diarizer=self.diarizer,
                is_short=bool(video_record.get("is_short")),
            )
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # --- MICRO-STEP M4: Surgical MossFormer2 Overlap Separation + Cosine Matching ---
            sep_manifest = run_overlap_separation_m4(
                video_id=vid,
                leveled_16k_wav=leveled_wav,
                overlaps=overlaps,
                centroids=centroids,
                workspace=self.workspace,
                separator=self.separator,
                diarizer=self.diarizer,
                local_tmp_dir=local_tmp,
            )
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # --- MICRO-STEP M5: Whisper + CTC Forced Alignment + Scenario Assembly ---
            word_ts, detected_lang = self.asr.transcribe_and_align_wav(
                leveled_wav,
                forced_language=None,
                use_ctc_aligner=True,
            )

            mapped_words = assign_words_to_speakers_iou(
                word_ts,
                rttm_segs,
                excluded_overlap_windows=overlaps,
            )
            self.asr.restore_punctuation_in_place(mapped_words, language=detected_lang)
            realigned_words = realign_words_preserving_interruptions(mapped_words)
            clean_turns = build_turns_from_aligned_words(realigned_words, language=detected_lang)

            # Transcribe each separated 2-stream overlap clip independently
            overlap_turns: List[Dict[str, Any]] = []
            for ov_item in sep_manifest:
                wid = ov_item["window_id"]
                base_start_ms = int(ov_item["start_ms"])
                for track in ov_item.get("tracks", []):
                    clip_p = Path(track["clip_path"])
                    if not clip_p.exists():
                        continue
                    ov_words, _ = self.asr.transcribe_and_align_wav(
                        clip_p,
                        forced_language=detected_lang,
                        use_ctc_aligner=False,
                    )
                    if not ov_words:
                        continue
                    txt = " ".join(w.get("text") or w.get("word", "") for w in ov_words).strip()
                    if not txt:
                        continue
                    rel_s = int(round(float(ov_words[0]["start"]) * 1000))
                    rel_e = int(round(float(ov_words[-1]["end"]) * 1000))
                    abs_s = base_start_ms + rel_s
                    abs_e = max(abs_s + 120, base_start_ms + rel_e)
                    overlap_turns.append({
                        "speaker_id": track["speaker_id"],
                        "start_ms": abs_s,
                        "end_ms": abs_e,
                        "start_fmt": ms_to_fmt(abs_s),
                        "end_fmt": ms_to_fmt(abs_e),
                        "language": detected_lang,
                        "text_original": txt,
                        "text_ar": txt if detected_lang == "ar" else "",
                        "is_overlap": True,
                        "overlap_group_id": wid,
                        "confidence": round(max(0.55, float(track.get("cosine_sim", 0.8))), 3),
                        "words": [
                            {
                                "w": w.get("text") or w.get("word", ""),
                                "s": base_start_ms + int(round(float(w["start"]) * 1000)),
                                "e": base_start_ms + int(round(float(w["end"]) * 1000)),
                            }
                            for w in ov_words
                        ],
                    })

            merged_raw_turns = merge_clean_and_overlap_turns(clean_turns, overlap_turns)

            # Classify MAIN_DAEE / INTERLOCUTOR_1..K / PEDESTRIAN & segment Episodes
            self.registry = VoiceprintRegistry(self.workspace)
            speakers_meta, final_turns = classify_speakers_and_episodes(
                raw_turns=merged_raw_turns,
                speaker_centroids=centroids,
                rms_timeline=rms_timeline,
                video_record=video_record,
                registry=self.registry,
            )

            for idx, ft in enumerate(final_turns):
                ft["turn_id"] = f"{vid}_T{idx + 1:04d}"

            samples_check, sr_check = read_wav_float32(leveled_wav)
            dur_sec = round(len(samples_check) / float(sr_check), 2)
            del samples_check

            overlap_dur_ms = sum((w["core_end_ms"] - w["core_start_ms"]) for w in overlaps)
            scenario_payload = {
                "schema_version": "2.0",
                "video_id": vid,
                "channel": video_record.get("channel", "SCDawah Arabic"),
                "title_ar": video_record.get("title_ar", ""),
                "youtube_url": video_record.get("youtube_url", f"https://www.youtube.com/watch?v={vid}"),
                "is_short": bool(video_record.get("is_short")),
                "duration_sec": dur_sec,
                "detected_language": detected_lang,
                "clean_opus_path": f"audio_clean/{vid}.opus",
                "rttm_path": f"diarization/{vid}/{vid}.rttm",
                "scenario_json_path": f"scenarios/{vid}.json",
                "metrics": {
                    "duration_sec": dur_sec,
                    "speakers_count": len(speakers_meta),
                    "turns_count": len(final_turns),
                    "overlap_windows_count": len(overlaps),
                    "overlap_ratio": round(overlap_dur_ms / max(dur_sec * 1000.0, 1.0), 4),
                    "pedestrian_turns_count": sum(1 for t in final_turns if t.get("is_pedestrian_interruption")),
                },
                "speakers": speakers_meta,
                "turns": final_turns,
            }

            self.workspace.atomic_write_json(scen_path, scenario_payload)
            srt_path.write_text(format_diarized_srt(final_turns), encoding="utf-8-sig")

            # Remove temporary per-window WAV clips now that scenario JSON is finalized
            shutil.rmtree(self.workspace.diarization_dir / vid / "sep_clips", ignore_errors=True)

            self.workspace.advance_micro_step(
                vid,
                "M5_SCENARIO_BUILT",
                artifacts={
                    "scenario_json": str(scen_path),
                    "diarized_srt": str(srt_path),
                },
                metrics=scenario_payload["metrics"],
            )
            return scenario_payload

        finally:
            shutil.rmtree(local_tmp, ignore_errors=True)

    def shutdown(self):
        self.demucs.unload()
        self.diarizer.unload()
        self.separator.unload()
        self.asr.unload()
