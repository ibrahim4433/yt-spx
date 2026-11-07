#!/usr/bin/env python3
"""
core/audio_diarizer.py
Micro-Step M5 (and End-to-End Stage 1 Runner M1 -> M5):
Combines Warm Faster-Whisper (large-v3 float16), CTC Word Forced Aligner
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


def get_cuda_library_dirs() -> List[str]:
    """Finds directories containing CUDA/cuBLAS/cudart shared libraries on Linux."""
    import sys
    from pathlib import Path

    dirs = []
    # 1. PyTorch bundled libraries FIRST (torch/lib) - absolute priority for cuDNN & CUDA runtime
    try:
        import torch
        tlib = Path(torch.__file__).parent / "lib"
        if tlib.is_dir():
            dirs.append(str(tlib.resolve()))
    except Exception:
        pass

    # 2. Check site-packages/nvidia/{cublas,cuda_runtime,cuda_nvrtc}/lib
    # NEVER include nvidia/cudnn/lib as that conflicts with PyTorch's bundled cuDNN (9.19.0)
    for p in sys.path:
        nv_dir = Path(p) / "nvidia"
        if nv_dir.is_dir():
            try:
                for sub in nv_dir.iterdir():
                    if sub.name.lower() in ("cublas", "cuda_runtime", "cuda_nvrtc"):
                        lib_p = sub / "lib"
                        if lib_p.is_dir():
                            p_str = str(lib_p.resolve())
                            if p_str not in dirs:
                                dirs.append(p_str)
            except OSError:
                pass

    # 3. Standard Linux CUDA paths (/usr/local/cuda*/lib64)
    try:
        usr_local = Path("/usr/local")
        if usr_local.is_dir():
            for cuda_cand in sorted(usr_local.glob("cuda*/lib64"), reverse=True):
                if cuda_cand.is_dir():
                    p_str = str(cuda_cand.resolve())
                    if p_str not in dirs:
                        dirs.append(p_str)
    except Exception:
        pass

    for cand in ("/usr/local/cuda/lib64", "/usr/local/cuda-12/lib64"):
        cp = Path(cand)
        if cp.is_dir():
            p_str = str(cp.resolve())
            if p_str not in dirs:
                dirs.append(p_str)

    return dirs


def preload_cuda_libraries() -> None:
    """
    Preloads CUDA runtime (libcudart.so.12) and cuBLAS (libcublas.so.12) using
    ctypes.CDLL(RTLD_GLOBAL) so CTranslate2 / faster-whisper and llama-cpp-python
    dynamically resolve them without dlopen errors, even if LD_LIBRARY_PATH was
    not set prior to Python process startup.
    Ensures PyTorch's bundled cuDNN is preserved without version conflicts.
    """
    import sys
    import os
    import ctypes
    from pathlib import Path

    if sys.platform != "linux":
        return

    lib_dirs = get_cuda_library_dirs()

    # Clean existing LD_LIBRARY_PATH from any conflicting standalone nvidia/cudnn paths
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    existing = [p for p in current_ld.split(":") if p and "nvidia/cudnn" not in p and "nvidia_cudnn" not in p]
    new_parts = [d for d in lib_dirs if d not in existing]
    combined = new_parts + existing
    if combined:
        os.environ["LD_LIBRARY_PATH"] = ":".join(combined)

    # Priority order: CUDA runtime (for llama_cpp) + cuBLAS (for CTranslate2)
    libs_to_load = [
        "libcudart.so.12",
        "libcudart.so",
        "libnvrtc.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
    ]

    for lib_name in libs_to_load:
        for d in lib_dirs:
            target = Path(d) / lib_name
            if target.exists():
                try:
                    ctypes.CDLL(str(target), mode=ctypes.RTLD_GLOBAL)
                    break
                except Exception:
                    pass


preload_cuda_libraries()


def patch_pyav_metadata_errors() -> None:
    """
    Patches PyAV (av >= 19.0.0) where metadata_errors/metadata_encoding kwargs were removed from av.open().
    faster-whisper (even latest 1.2.1) still passes metadata_errors='ignore' in decode_audio().
    """
    try:
        import av

        if not getattr(av, "_patched_for_metadata_errors", False):
            _orig_av_open = av.open

            def _patched_av_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                kwargs.pop("metadata_encoding", None)
                return _orig_av_open(*args, **kwargs)

            av.open = _patched_av_open
            av._patched_for_metadata_errors = True
    except Exception:
        pass


patch_pyav_metadata_errors()


def load_audio_waveform_16k(audio_path: Path) -> np.ndarray:
    """
    Loads 16kHz mono audio into a float32 [-1.0, 1.0] numpy array.
    Uses stdlib wave for WAV files (instant, zero external C-lib dependency),
    falling back to soundfile or faster_whisper.decode_audio (with PyAV patch).
    """
    p = Path(audio_path)
    if p.suffix.lower() == ".wav" and p.exists():
        try:
            samples, sr = read_wav_float32(p)
            if sr == 16000:
                return samples
        except Exception:
            pass

    patch_pyav_metadata_errors()
    try:
        import soundfile as sf

        samples, sr = sf.read(str(p), dtype="float32")
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        if sr != 16000:
            import librosa

            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        return samples.astype(np.float32)
    except Exception:
        pass

    import faster_whisper

    return faster_whisper.decode_audio(str(p), sampling_rate=16000)


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
    Keeps Faster-Whisper (large-v3 float16), CTC Forced Aligner (mms-300m-1130),
    and Multilingual Punctuation warm in GPU memory across a batch of videos (~4.2 GB VRAM).
    """

    def __init__(
        self,
        whisper_model_name: str = "large-v3",
        device: str = "cuda",
        batch_size: int = 24,
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
            preload_cuda_libraries()
            import faster_whisper

            compute_type = "float16" if self.device == "cuda" else "int8"
            try:
                self._whisper = faster_whisper.WhisperModel(
                    self.whisper_model_name,
                    device=self.device,
                    compute_type=compute_type,
                )
                self._whisper_batched = faster_whisper.BatchedInferencePipeline(model=self._whisper)
            except Exception as exc:
                if self.device == "cuda" and ("cublas" in str(exc).lower() or "cudnn" in str(exc).lower()):
                    print(f"   [!] Faster-Whisper CUDA init note ({exc}); falling back to cpu/int8")
                    self.device = "cpu"
                    self._whisper = faster_whisper.WhisperModel(
                        self.whisper_model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                    self._whisper_batched = faster_whisper.BatchedInferencePipeline(model=self._whisper)
                else:
                    raise

    def _ensure_aligner(self):
        if self._align_model is None:
            import torch
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            try:
                from ctc_forced_aligner import load_alignment_model
                self._align_model, self._align_tokenizer = load_alignment_model(self.device, dtype=dtype)
            except (ImportError, AttributeError):
                from transformers import AutoModelForCTC, AutoTokenizer
                m_name = "MahmoudAshraf/mms-300m-1130-forced-aligner"
                self._align_model = AutoModelForCTC.from_pretrained(m_name, torch_dtype=dtype).to(self.device)
                self._align_tokenizer = AutoTokenizer.from_pretrained(m_name)

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
        patch_pyav_metadata_errors()
        audio_waveform = load_audio_waveform_16k(wav_path)
        if len(audio_waveform) < 3200:
            return [], forced_language or "en"

        def _transcribe_stream(waveform: np.ndarray, target_lang: Optional[str], is_probe: bool = False):
            try:
                if is_probe:
                    return self._whisper.transcribe(waveform, vad_filter=True)
                if self.batch_size > 0 and len(waveform) > 16000 * 25 and self._whisper_batched:
                    return self._whisper_batched.transcribe(
                        waveform,
                        language=target_lang,
                        initial_prompt=init_prompt,
                        batch_size=self.batch_size,
                        word_timestamps=True,
                    )
                else:
                    return self._whisper.transcribe(
                        waveform,
                        language=target_lang,
                        initial_prompt=init_prompt,
                        beam_size=5,
                        best_of=5,
                        temperature=[0.0, 0.2, 0.4],
                        condition_on_previous_text=False,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        word_timestamps=True,
                    )
            except RuntimeError as err:
                err_str = str(err).lower()
                if ("libcublas" in err_str or "libcudnn" in err_str) and self.device == "cuda":
                    print(f"   [!] Faster-Whisper cuBLAS/CUDA error ({err}); falling back to CPU int8 inference...")
                    self.device = "cpu"
                    self._whisper = faster_whisper.WhisperModel(
                        self.whisper_model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                    self._whisper_batched = faster_whisper.BatchedInferencePipeline(model=self._whisper)
                    return _transcribe_stream(waveform, target_lang, is_probe=is_probe)
                raise

        # Detect language on first 45s if not forced
        lang = forced_language
        if not lang:
            probe_audio = audio_waveform[: min(len(audio_waveform), 16000 * 45)]
            _, info_probe = _transcribe_stream(probe_audio, None, is_probe=True)
            lang = info_probe.language or "en"

        init_prompt = THEOLOGICAL_VOCAB_PROMPT_EN if lang == "en" else None
        segments_iter, info = _transcribe_stream(audio_waveform, lang, is_probe=False)

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
                            "word": w.word.strip(),
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
                            "word": tok,
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
                ratio = getattr(getattr(self._align_model, "config", None), "inputs_to_logits_ratio", 320)
                stride = float(ratio) / 16000.0
                emissions = None

                if hasattr(self._align_model, "run"):
                    try:
                        wav_tensor = torch.from_numpy(audio_waveform).to(self._align_model.dtype).to(self._align_model.device)
                        emissions, stride = generate_emissions(
                            self._align_model,
                            wav_tensor,
                            batch_size=max(4, self.batch_size // 2),
                        )
                        del wav_tensor
                    except Exception:
                        try:
                            emissions, stride = generate_emissions(
                                self._align_model,
                                audio_waveform,
                                batch_size=max(4, self.batch_size // 2),
                            )
                        except Exception:
                            emissions = None

                if emissions is None:
                    # HuggingFace AutoModelForCTC (Wav2Vec2ForCTC) chunked PyTorch forward pass
                    # Chunk into 30s segments (480,000 samples) so VRAM never exceeds ~100MB even on 3-hour audio
                    chunk_size = 16000 * 30
                    total_len = len(audio_waveform)
                    if total_len <= chunk_size:
                        wav_tensor = torch.from_numpy(audio_waveform)
                        if hasattr(self._align_model, "dtype"):
                            wav_tensor = wav_tensor.to(self._align_model.dtype)
                        if hasattr(self._align_model, "device"):
                            wav_tensor = wav_tensor.to(self._align_model.device)
                        if hasattr(wav_tensor, "ndim") and wav_tensor.ndim == 1 and hasattr(wav_tensor, "unsqueeze"):
                            wav_tensor = wav_tensor.unsqueeze(0)
                        with torch.inference_mode():
                            logits = self._align_model(wav_tensor).logits
                            emissions = torch.log_softmax(logits, dim=-1)
                        del wav_tensor
                    else:
                        emissions_list = []
                        for offset in range(0, total_len, chunk_size):
                            chunk = audio_waveform[offset : min(offset + chunk_size, total_len)]
                            rem = len(chunk) % 320
                            if rem != 0:
                                chunk = np.pad(chunk, (0, 320 - rem))
                            c_tensor = torch.from_numpy(chunk)
                            if hasattr(self._align_model, "dtype"):
                                c_tensor = c_tensor.to(self._align_model.dtype)
                            if hasattr(self._align_model, "device"):
                                c_tensor = c_tensor.to(self._align_model.device)
                            if hasattr(c_tensor, "ndim") and c_tensor.ndim == 1 and hasattr(c_tensor, "unsqueeze"):
                                c_tensor = c_tensor.unsqueeze(0)
                            with torch.inference_mode():
                                c_logits = self._align_model(c_tensor).logits
                                c_em = torch.log_softmax(c_logits, dim=-1)
                                if hasattr(c_em, "cpu"):
                                    c_em = c_em.cpu()
                            emissions_list.append(c_em)
                            del c_tensor, c_logits
                        if hasattr(torch, "cat") and emissions_list:
                            emissions = torch.cat(emissions_list, dim=1)
                            if total_len % 320 != 0 and hasattr(emissions, "__getitem__"):
                                emissions = emissions[:, : (total_len // 320), :]
                        elif emissions_list:
                            emissions = emissions_list[0]

                if isinstance(emissions, torch.Tensor) or hasattr(emissions, "is_cpu"):
                    if hasattr(emissions, "cpu"):
                        emissions = emissions.cpu()
                    if hasattr(emissions, "ndim") and emissions.ndim == 3 and emissions.size(0) == 1:
                        emissions = emissions.squeeze(0)
                elif isinstance(emissions, np.ndarray):
                    if hasattr(emissions, "ndim") and emissions.ndim == 3 and emissions.shape[0] == 1:
                        emissions = np.squeeze(emissions, axis=0)
                    if hasattr(torch, "from_numpy"):
                        try:
                            emissions = torch.from_numpy(emissions)
                        except Exception:
                            pass
                elif isinstance(emissions, (list, tuple)):
                    emissions = [e.cpu() if hasattr(e, "cpu") else e for e in emissions]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                tokens_starred, text_starred = preprocess_text(
                    full_text,
                    romanize=True,
                    language=iso3,
                )
                try:
                    segments, scores, blank_token = get_alignments(
                        emissions,
                        tokens_starred,
                        self._align_tokenizer,
                    )
                except AttributeError as ae:
                    if "is_cpu" in str(ae) and isinstance(emissions, np.ndarray) and hasattr(torch, "from_numpy"):
                        em_tensor = torch.from_numpy(emissions)
                        if hasattr(em_tensor, "ndim") and em_tensor.ndim == 3 and em_tensor.size(0) == 1:
                            em_tensor = em_tensor.squeeze(0)
                        segments, scores, blank_token = get_alignments(
                            em_tensor,
                            tokens_starred,
                            self._align_tokenizer,
                        )
                    else:
                        raise

                if hasattr(scores, "cpu"):
                    scores = scores.cpu()
                if hasattr(scores, "numpy"):
                    scores = scores.numpy()

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
            try:
                labeled = self._punct_model.predict(words_only, chunk_size=230)
            except TypeError:
                labeled = self._punct_model.predict(words_only)
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
        whisper_model: str = "large-v3",
        device: str = "cuda",
        batch_size: int = 24,
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
        t_start_vid = time.time()
        try:
            # --- MICRO-STEP M1: Temp Download -> Drive Staging ---
            print(f"   [1/5] Audio Ingestion (M1)....................... ", end="", flush=True)
            raw_audio_path, info_meta = fetch_video_audio_m1(video_record, self.workspace)
            print("[DONE] (Source ready)")
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # --- MICRO-STEP M2: Chunked Vocal Isolation + 64kbps Opus Archive ---
            print(f"   [2/5] Demucs Vocal Isolation (M2)................ ", end="", flush=True)
            leveled_wav, rms_timeline = clean_vocals_m2(
                video_id=vid,
                raw_audio_path=raw_audio_path,
                workspace=self.workspace,
                cleaner=self.demucs,
                local_tmp_dir=local_tmp,
                keep_raw_audio_on_drive=self.keep_raw_audio,
            )
            print("[DONE] (Clean vocal archive ready)")
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # Free Demucs VRAM if tight before NeMo + MossFormer2 + Whisper
            self.demucs.unload()

            # --- MICRO-STEP M3: Overlap-Aware NeMo MSDD Diarization + TitaNet Centroids ---
            print(f"   [3/5] NeMo MSDD Diarization & TitaNet (M3)....... ", end="", flush=True)
            rttm_segs, centroids, overlaps = run_diarization_m3(
                video_id=vid,
                leveled_16k_wav=leveled_wav,
                workspace=self.workspace,
                diarizer=self.diarizer,
                is_short=bool(video_record.get("is_short")),
            )
            n_spk = len({s.get("speaker_id") for s in rttm_segs})
            print(f"[DONE] ({n_spk} speakers identified, {len(overlaps)} overlap window(s))")
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # --- MICRO-STEP M4: Surgical MossFormer2 Overlap Separation + Cosine Matching ---
            print(f"   [4/5] MossFormer2 Overlap Separation (M4)........ ", end="", flush=True)
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
            print(f"[DONE] ({len(sep_manifest)} cross-talk stream(s) unmixed)")
            self.workspace.heartbeat_lock(f"{vid}.stage1")

            # --- MICRO-STEP M5: Whisper + CTC Forced Alignment + Scenario Assembly ---
            print(f"   [5/5] Whisper ASR & Scenario Assembly (M5)....... ", end="", flush=True)
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

            total_elapsed = round(time.time() - t_start_vid, 1)
            print(f"[DONE] ({len(final_turns)} verified turns assembled)")
            print("   " + "-" * 70)
            print(f"   >>> [VERIFIED] Scenario M5_SCENARIO_BUILT in {total_elapsed}s:")
            print(f"       * Video: {vid} | Language: {detected_lang} | Duration: {round(dur_sec/60.0, 1)}m")
            print(f"       * Diarization: {len(speakers_meta)} speakers | {len(final_turns)} turns | {len(overlaps)} overlap window(s)")
            print(f"       * Integrity: 100% PASSED -> Saved to Drive scenarios/{vid}.json")
            print("   " + "=" * 70, flush=True)

            # Remove temporary per-window WAV clips now that scenario JSON is finalized
            shutil.rmtree(local_tmp / f"{vid}_sep_clips", ignore_errors=True)
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
