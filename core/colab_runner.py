#!/usr/bin/env python3
"""
core/colab_runner.py
Local GPU LLM client (Qwen 2.5 14B Instruct Q5_K_M GGUF + 8-bit KV Cache / 4-bit NF4 fallback)
and JSON stream-recovery helpers for Dawah Dialectics V2 (yt-spx).

Configured for Google Colab Free T4 GPUs (16 GB VRAM) and the shared 5 TB Google Drive
workspace (/content/drive/MyDrive/yt-spx).
"""

import os
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Prevent PyTorch CUDA memory fragmentation on Nvidia GPUs (T4 / L4 / A100)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Force Transformers to never import TensorFlow or Flax (avoids Protobuf version mismatch in Colab)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

try:
    from core.drive_sync import resolve_drive_root
    from core.extractor_prompt import (
        EXTRACTION_SYSTEM_PROMPT,
        ROUND1_ENGLISH_EXTRACTION_PROMPT,
        ROUND2_ARABIC_VERIFICATION_PROMPT,
        COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT,
    )
except (ImportError, ValueError):
    from .drive_sync import resolve_drive_root
    from .extractor_prompt import (
        EXTRACTION_SYSTEM_PROMPT,
        ROUND1_ENGLISH_EXTRACTION_PROMPT,
        ROUND2_ARABIC_VERIFICATION_PROMPT,
        COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT,
    )

try:
    from db.database import sanitize_text_field
except (ImportError, ValueError):
    try:
        from ..db.database import sanitize_text_field
    except Exception:
        sanitize_text_field = None

BASE_DIR = Path(__file__).resolve().parent.parent

# Persistent Google Drive support for yt-spx V2 workspace (never touches legacy yt-ch archive)
_detected_drive = resolve_drive_root()
DRIVE_BASE = _detected_drive if _detected_drive is not None else Path("/content/drive/MyDrive/yt-spx")
if _detected_drive is not None:
    DATA_DIR = DRIVE_BASE / "data"
    DOWNLOADS_DIR = DRIVE_BASE / "audio_raw"
else:
    DATA_DIR = BASE_DIR / "data"
    DOWNLOADS_DIR = BASE_DIR / "data" / "audio_raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LIVE_LOG_FILE = (DRIVE_BASE / "logs" / "extraction_live.log") if _detected_drive else (DATA_DIR / "extraction_live.log")


def log_extracted_unit_live(unit: Dict[str, Any]) -> None:
    """Appends newly extracted V2 Q&A unit to a live human-readable log file on Drive."""
    try:
        LIVE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        ts_now = time.strftime("%Y-%m-%d %H:%M:%S")
        cat = unit.get("topic_category", "General")
        daee = unit.get("daee_name_en") or "Main Da'ee"
        q_text = unit.get("question_ar") or unit.get("question_en") or "Untitled Question"
        ans_text = unit.get("core_answer_summary_ar") or unit.get("core_answer_summary_en") or ""
        with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"[{ts_now}] [{unit.get('timestamp_formatted', '00:00')}] [{cat}] [{daee}]\n"
                f"Q: {q_text}\n"
                f"A: {ans_text[:160]}...\n"
                f"URL: {unit.get('timestamp_url', '')}\n"
                f"{'-' * 75}\n"
            )
    except Exception:
        pass


def _is_dialectical_payload_item(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    keys = (
        "question_en", "question_ar", "question_turn_id", "answer_turn_id",
        "core_answer_summary_en", "core_answer_summary_ar", "question",
        "dialogue_exchange", "dialogue_trajectory",
    )
    return any(k in x for k in keys)


def _normalize_extracted_item(x: Dict[str, Any]) -> Dict[str, Any]:
    if "question" in x:
        q_val = str(x["question"] or "")
        if re.search(r"[\u0600-\u06ff]", q_val):
            x.setdefault("question_ar", q_val)
        else:
            x.setdefault("question_en", q_val)
    if "dialogue_trajectory" in x and not x.get("dialogue_exchange"):
        x["dialogue_exchange"] = x["dialogue_trajectory"]
    return x


def strip_llm_thinking_blocks(raw_text: str) -> str:
    """Removes Qwen3 <think>...</think> reasoning blocks (closed or truncated) before JSON parsing."""
    if not raw_text:
        return ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    if "<think>" in cleaned.lower():
        # Unclosed <think> block: if a JSON array or object starts after <think>, slice from there; else drop <think>...
        idx_think = cleaned.lower().find("<think>")
        after_think = cleaned[idx_think + 7 :]
        json_start = re.search(r"\[\s*\{|\{\s*\"", after_think)
        if json_start:
            cleaned = (cleaned[:idx_think] + after_think[json_start.start() :]).strip()
        else:
            cleaned = cleaned[:idx_think].strip()
    return cleaned


def clean_llm_json_response(raw_text: str) -> List[Dict[str, Any]]:
    """Strips <think> blocks and markdown code fences, parsing JSON safely including partial/truncated streams."""
    if not raw_text:
        return []
    text = strip_llm_thinking_blocks(raw_text)
    if not text:
        return []
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [_normalize_extracted_item(x) for x in data if _is_dialectical_payload_item(x)]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [_normalize_extracted_item(x) for x in v if _is_dialectical_payload_item(x)]
            if _is_dialectical_payload_item(data):
                return [_normalize_extracted_item(data)]
    except Exception:
        pass

    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [_normalize_extracted_item(x) for x in parsed if _is_dialectical_payload_item(x)]
        except Exception:
            pass

    # Balanced-brace stream fallback if max_tokens truncated outer array
    results: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, char in enumerate(text):
        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    obj_str = text[start : i + 1]
                    try:
                        obj = json.loads(obj_str)
                        if _is_dialectical_payload_item(obj):
                            results.append(_normalize_extracted_item(obj))
                    except Exception:
                        pass
                    start = -1
        escape = False

    return results


def sanitize_extracted_arabic_unit(item: Dict[str, Any]) -> Dict[str, Any]:
    """Removes stray CJK/Korean tokens or common Arabic-Latin splices from generated fields."""
    replacements = [
        ("따라서،", "لذلك،"),
        ("따라서", "لذلك"),
        ("여전히", "لا تزال"),
        ("المiracles", "المعجزات"),
        ("فالمachines", "فالآلات"),
        ("كمessenger", "كرسول"),
        ("للمonotheism", "للتوحيد"),
        ("السcribes", "النساخ"),
        ("للسcribes", "للنساخ"),
        ("نihilist", "عدمي"),
        ("النihilists", "العدميون"),
        ("بprocesseات", "بعمليات"),
        ("مutually exclusive", "متنافيين"),
        ("ال hypocritical", "النفاق وازدواجية المعايير"),
        ("بهomo جنسياً", "بميول مثلية"),
        ("التطهيرracial", "التطهير العرقي"),
        ("ت дисквалифицируют его как совершенный божественный текст", "تسقط عنه صفة النص الإلهي الكامل المعصوم"),
        ("дисквалифицируют его как совершенный божественный текст", "تسقط عنه صفة النص الإلهي الكامل المعصوم"),
        ("педوفيليا", "البيدوفيليا (استغلال الأطفال)"),
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
        ("。，？，。", "؟"),
        ("。", "."),
        ("，", "،"),
        ("？", "؟"),
        ("！", "!"),
    ]
    ar_keys = [
        "question_ar",
        "core_answer_summary_ar",
        "fallacy_or_misconception_exposed_ar",
        "analogy_or_metaphor_ar",
        "socratic_question_ar",
        "verbatim_quote_ar",
        "interlocutor_verbatim_quote_ar",
    ]
    for k in ar_keys:
        val = item.get(k)
        if isinstance(val, str) and val:
            if sanitize_text_field is not None:
                item[k] = sanitize_text_field(val, is_arabic=True)
            else:
                for old_s, new_s in replacements:
                    if old_s in val:
                        val = val.replace(old_s, new_s)
                val = re.sub(r"[\u0400-\u04ff\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]+", "", val)
                item[k] = val.strip()

    premises = item.get("logical_premises_ar")
    if isinstance(premises, list):
        clean_premises = []
        for p in premises:
            if isinstance(p, str):
                if sanitize_text_field is not None:
                    cp = sanitize_text_field(p, is_arabic=True)
                else:
                    cp = p
                    for old_s, new_s in replacements:
                        if old_s in cp:
                            cp = cp.replace(old_s, new_s)
                    cp = re.sub(r"[\u0400-\u04ff\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]+", "", cp).strip()
                if cp:
                    clean_premises.append(cp)
        item["logical_premises_ar"] = clean_premises
    return item


def ensure_cuda_runtime_loaded() -> None:
    """Preloads libcudart.so.12 and sets LD_LIBRARY_PATH so llama_cpp loads cleanly on Linux."""
    try:
        from core.audio_diarizer import preload_cuda_libraries
        preload_cuda_libraries()
    except Exception:
        pass

    import sys
    if sys.platform != "linux":
        return

    import ctypes
    search_paths = []
    try:
        import torch
        tlib = Path(torch.__file__).parent / "lib"
        if tlib.is_dir():
            search_paths.append(tlib)
    except Exception:
        pass

    for p in sys.path:
        nv_rt = Path(p) / "nvidia" / "cuda_runtime" / "lib"
        if nv_rt.is_dir():
            search_paths.append(nv_rt)

    try:
        for cdir in sorted(Path("/usr/local").glob("cuda*/lib64"), reverse=True):
            if cdir.is_dir():
                search_paths.append(cdir)
    except Exception:
        pass

    for d in search_paths:
        for cand_name in ("libcudart.so.12", "libcudart.so"):
            target = d / cand_name
            if target.exists():
                try:
                    ctypes.CDLL(str(target), mode=ctypes.RTLD_GLOBAL)
                    curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
                    if str(d) not in curr_ld.split(":"):
                        os.environ["LD_LIBRARY_PATH"] = f"{d}:{curr_ld}" if curr_ld else str(d)
                    return
                except Exception:
                    pass


def _copy_large_file_chunked(src_path: Path, dst_path: Path, label: str = "Copying") -> None:
    """
    Streams large GGUF model files (10GB+) in 64MB chunks with live progress and explicit os.fsync().
    On Google Drive FUSE (/content/drive), writes directly to dst_path instead of renaming a 10GB .tmp
    file (which causes DriveFS background sync to drop or orphan the upload).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    is_drive_target = "/content/drive" in str(dst_path).replace("\\", "/")
    write_target = dst_path if is_drive_target else dst_path.with_suffix(dst_path.suffix + ".tmp")
    total_size = src_path.stat().st_size
    copied = 0
    t0 = time.time()
    last_log = t0
    chunk_size = 64 * 1024 * 1024  # 64 MB

    try:
        with open(src_path, "rb") as fsrc, open(write_target, "wb") as fdst:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                copied += len(buf)
                now = time.time()
                if now - last_log >= 15.0 or copied >= total_size:
                    last_log = now
                    pct = round((copied / max(1, total_size)) * 100, 1)
                    mb_per_sec = round((copied / (1024 * 1024)) / max(0.1, now - t0), 1)
                    print(
                        f"   [{label}] {pct}% ({round(copied / (1024**3), 2)} GB / "
                        f"{round(total_size / (1024**3), 2)} GB at {mb_per_sec} MB/s)...",
                        flush=True,
                    )
            fdst.flush()
            try:
                os.fsync(fdst.fileno())
            except Exception:
                pass
        if write_target != dst_path:
            write_target.replace(dst_path)
    except Exception:
        try:
            if write_target.exists() and write_target.stat().st_size < total_size:
                write_target.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _find_valid_drive_gguf(drive_root: Optional[Path], target_file: str, min_valid_size: int) -> Optional[Path]:
    """
    Searches candidate Google Drive model directories (.cache/models and models),
    auto-recovering any completed .tmp file left behind by DriveFS FUSE renames
    and cleaning up truncated (< min_valid_size) partial files.
    """
    if not drive_root:
        return None
    candidate_dirs = [
        drive_root / ".cache" / "models",
        drive_root / "models",
    ]
    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        exact_path = cdir / target_file
        tmp_path = cdir / f"{target_file}.tmp"
        # Recover completed .tmp file if present
        if tmp_path.exists():
            try:
                if tmp_path.stat().st_size >= min_valid_size:
                    if not exact_path.exists() or exact_path.stat().st_size < min_valid_size:
                        tmp_path.replace(exact_path)
                else:
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        if exact_path.exists():
            try:
                sz = exact_path.stat().st_size
                if sz >= min_valid_size:
                    return exact_path
                else:
                    # Remove broken partial file so it doesn't waste Drive quota
                    exact_path.unlink(missing_ok=True)
            except Exception:
                pass
    return None


def _can_persist_gguf_to_drive(drive_root: Optional[Path], file_size_bytes: int) -> bool:
    """
    Checks if the current Google Drive mount has enough free quota to persist a ~10 GB GGUF file.
    Note: On Google Drive, when a secondary 15 GB collaborator account (Profile 2) creates a new file
    inside a shared folder owned by the 5 TB account, Google Drive charges the file against the
    collaborator's 15 GB quota and silently drops background uploads if free quota is insufficient.
    """
    if not drive_root:
        return False
    try:
        import shutil
        usage = shutil.disk_usage(str(drive_root))
        required_free = file_size_bytes + 1_500_000_000  # 1.5 GB safety headroom
        if usage.free < required_free:
            free_gb = round(usage.free / (1024**3), 2)
            req_gb = round(file_size_bytes / (1024**3), 2)
            print(
                f"[i] Google Drive account has {free_gb} GB free (needs {req_gb} GB + headroom). "
                "Skipping Drive upload on this collaborator account (run Stage 2 once on the 5 TB owner account to cache on Drive).",
                flush=True,
            )
            return False
    except Exception:
        pass
    return True


def _link_or_copy_to_local_ssd(src_path: Path, local_model_file: Path) -> None:
    """Creates a zero-cost hardlink or symlink at /content/models/{target_file} so same-VM reruns take 0s."""
    try:
        local_model_file.parent.mkdir(parents=True, exist_ok=True)
        if local_model_file.exists():
            return
        try:
            os.link(str(src_path), str(local_model_file))
        except Exception:
            os.symlink(str(src_path), str(local_model_file))
    except Exception:
        pass


def _resolve_and_cache_gguf_model(target_repo: str, target_file: str, hf_download_fn: Any) -> str:
    """
    Resolves GGUF model path with 3-tier caching so Colab never has to re-download from HuggingFace:
      Tier 1: Fast local NVMe SSD (/content/models/{target_file} or ~/.cache/huggingface/hub) -> 0s wait
      Tier 2: Persistent Google Drive (/content/drive/MyDrive/yt-spx/.cache/models/{target_file}) -> ~90s copy to local SSD
      Tier 3: First-time Hugging Face download -> links to /content/models & backs up to Google Drive!
    """
    from core.drive_sync import resolve_drive_root

    min_valid_size = 4_000_000_000  # >4 GB sanity check for 14B Q5 GGUF
    drive_root = resolve_drive_root()
    drive_cache_dir = (drive_root / ".cache" / "models") if drive_root else None
    drive_model_file = (drive_cache_dir / target_file) if drive_cache_dir else None
    existing_drive_file = _find_valid_drive_gguf(drive_root, target_file, min_valid_size)

    local_dir = Path("/content/models") if Path("/content").exists() else (Path.home() / ".cache" / "yt-spx" / "models")
    local_model_file = local_dir / target_file

    # Tier 1A: Local NVMe SSD direct path
    if local_model_file.exists() and local_model_file.stat().st_size >= min_valid_size:
        print(f"[+] Found GGUF model on fast local SSD: {local_model_file} (0s wait)")
        if drive_model_file and existing_drive_file is None and _can_persist_gguf_to_drive(drive_root, local_model_file.stat().st_size):
            try:
                print(f"[+] Backing up local model to Google Drive cache ({drive_model_file})...", flush=True)
                _copy_large_file_chunked(local_model_file, drive_model_file, label="Drive Backup")
            except Exception as e:
                print(f"[-] Drive backup note: {e}")
        return str(local_model_file)

    # Tier 1B: Existing Hugging Face local cache in current VM (via local_files_only or hub scan)
    hf_existing: Optional[Path] = None
    try:
        cached_local = hf_download_fn(repo_id=target_repo, filename=target_file, local_files_only=True)
        if cached_local and Path(cached_local).exists() and Path(cached_local).stat().st_size >= min_valid_size:
            hf_existing = Path(cached_local).resolve()
    except Exception:
        pass

    if hf_existing is None:
        try:
            hf_matches = [
                p for p in (Path.home() / ".cache" / "huggingface" / "hub").glob(f"**/{target_file}")
                if p.exists() and p.stat().st_size >= min_valid_size
            ]
            if hf_matches:
                hf_existing = hf_matches[0].resolve()
        except Exception:
            pass

    if hf_existing is not None:
        print(f"[+] Found GGUF model in local HuggingFace cache: {hf_existing} (0s wait)")
        _link_or_copy_to_local_ssd(hf_existing, local_model_file)
        if drive_model_file and existing_drive_file is None and _can_persist_gguf_to_drive(drive_root, hf_existing.stat().st_size):
            try:
                print(f"[+] Persisting model to Google Drive ({drive_model_file}) so future sessions skip HuggingFace...", flush=True)
                _copy_large_file_chunked(hf_existing, drive_model_file, label="Persist to Drive")
            except Exception as e:
                print(f"[-] Drive backup note: {e}")
        return str(hf_existing)

    # Tier 2: Persistent Google Drive cache (~90s high-speed internal copy vs internet download)
    if existing_drive_file is not None:
        print(f"[+] Found persistent GGUF model on Google Drive: {existing_drive_file}")
        print(f"[+] Streaming from Drive to fast local NVMe SSD (~1-2 min, no HuggingFace download needed)...", flush=True)
        try:
            _copy_large_file_chunked(existing_drive_file, local_model_file, label="Drive -> Local SSD")
            return str(local_model_file)
        except Exception as copy_err:
            print(f"[-] Local SSD copy note ({copy_err}); loading directly from Drive path...")
            return str(existing_drive_file)

    # Tier 3: First-time download from Hugging Face
    print(f"[+] First-time download from Hugging Face: {target_repo}/{target_file} (~10.4 GB)...", flush=True)
    downloaded_path = Path(hf_download_fn(repo_id=target_repo, filename=target_file)).resolve()
    _link_or_copy_to_local_ssd(downloaded_path, local_model_file)

    if drive_model_file and downloaded_path.exists() and downloaded_path.stat().st_size >= min_valid_size:
        if _can_persist_gguf_to_drive(drive_root, downloaded_path.stat().st_size):
            try:
                print(f"[+] Saving model permanently to Google Drive ({drive_model_file}) for fast 90s future startups...", flush=True)
                _copy_large_file_chunked(downloaded_path, drive_model_file, label="Save to Drive")
                print(
                    "[OK] Model written to Google Drive mount! "
                    "(Note: DriveFS finishes cloud upload in the background while Stage 2 runs; avoid disconnecting immediately.)"
                )
            except Exception as persist_err:
                print(f"[-] Drive cache warning ({persist_err}); continuing with local model.")

    return str(downloaded_path)


class LocalGPUClient:
    """
    Runs Qwen 3 14B / Qwen 2.5 14B Instruct directly on the Google Colab GPU (T4 15GB / L4 / A100).
    Uses GGUF Q5_K_M (5-bit weights + 8-bit KV Cache + 14,336/16,384 Context) via llama-cpp-python
    (~10.4 GiB VRAM on isolated Stage 2 worker, safely within 15 GiB Colab T4 VRAM), with fallback to Transformers 4-bit NF4.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        load_in_4bit: bool = True,
        use_gguf_q5: bool = True,
        gguf_repo: str = "bartowski/Qwen2.5-14B-Instruct-GGUF",
        gguf_filename: str = "Qwen2.5-14B-Instruct-Q5_K_M.gguf",
        context_length: int = 20480,
        max_new_tokens: int = 6144,
        preferred_gguf_repo: str = "Qwen/Qwen3-14B-GGUF",
        preferred_gguf_filename: str = "Qwen3-14B-Q5_K_M.gguf",
    ):
        import torch

        self.backend = "transformers"
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens

        print(f"[+] Initializing Local GPU Model: {model_name} (Target: 14B Q5_K_M / {context_length} ctx)")
        print(f"[+] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[+] GPU Device: {torch.cuda.get_device_name(0)}")

        # 1. Primary Engine: GGUF Q5_K_M (5-bit) + 8-bit KV Cache via llama-cpp-python
        if use_gguf_q5 and torch.cuda.is_available():
            try:
                ensure_cuda_runtime_loaded()
                try:
                    from llama_cpp import Llama, llama_supports_gpu_offload
                except ImportError:
                    print("[+] Auto-installing pre-built CUDA wheel for llama-cpp-python...")
                    import subprocess, sys
                    for cu_ver in ["cu124", "cu122", "cu121"]:
                        res = subprocess.run(
                            [
                                sys.executable, "-m", "pip", "install", "-q",
                                "--prefer-binary", "llama-cpp-python",
                                f"--extra-index-url=https://abetlen.github.io/llama-cpp-python/whl/{cu_ver}",
                            ],
                            check=False,
                        )
                        if res.returncode == 0:
                            ensure_cuda_runtime_loaded()
                            break
                    from llama_cpp import Llama, llama_supports_gpu_offload

                if llama_supports_gpu_offload():
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError:
                        import subprocess, sys
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
                        from huggingface_hub import hf_hub_download

                    # Primary: Qwen/Qwen3-14B-GGUF with automatic fallback to bartowski/Qwen2.5-14B-Instruct-GGUF
                    is_custom_repo = gguf_repo != "bartowski/Qwen2.5-14B-Instruct-GGUF"
                    target_repo = gguf_repo if is_custom_repo else (preferred_gguf_repo or gguf_repo)
                    target_file = gguf_filename if is_custom_repo else (preferred_gguf_filename or gguf_filename)
                    model_path = None
                    try:
                        print(f"[+] Resolving primary GGUF: {target_repo}/{target_file} (~10.4 GB)...")
                        model_path = _resolve_and_cache_gguf_model(target_repo, target_file, hf_hub_download)
                    except Exception as primary_dl_err:
                        print(
                            f"[-] Primary GGUF ({target_repo}/{target_file}) unavailable: {primary_dl_err}. "
                            f"Falling back to {gguf_repo}/{gguf_filename}..."
                        )
                        model_path = _resolve_and_cache_gguf_model(gguf_repo, gguf_filename, hf_hub_download)

                    # GGML_TYPE_Q8_0 = 8 for 8-bit KV Cache (~1.9 GB at 20,480 context, total ~12.8-13.2 GB VRAM on T4)
                    try:
                        self.llm = Llama(
                            model_path=model_path,
                            n_gpu_layers=-1,
                            n_ctx=context_length,
                            n_batch=2048,
                            type_k=8,
                            type_v=8,
                            flash_attn=True,
                            verbose=False,
                        )
                    except Exception as fa_err:
                        print(f"[-] GGUF flash_attn/20k init note ({fa_err}); retrying with step-down attention/batch...")
                        fallback_ctx = min(context_length, 16384)
                        self.context_length = fallback_ctx
                        self.llm = Llama(
                            model_path=model_path,
                            n_gpu_layers=-1,
                            n_ctx=fallback_ctx,
                            n_batch=1024,
                            type_k=8,
                            type_v=8,
                            flash_attn=False,
                            verbose=False,
                        )
                    self.backend = "gguf"
                    print(f"[+] GGUF Q5_K_M + 8-bit KV Cache ({self.context_length} ctx, n_batch=2048) loaded into VRAM!")
                    return
                else:
                    print("[-] llama-cpp-python lacks CUDA offload; falling back to Transformers 4-bit NF4.")
            except Exception as e:
                print(f"[-] GGUF Q5_K_M init note ({e}); using Transformers 14B 4-bit GPU backend.")

        # 2. Fallback Engine: HuggingFace Transformers 4-bit NF4 + SDPA
        # Colab preinstalls TensorFlow which conflicts with Protobuf 5.x; ensure Transformers never references it
        os.environ["USE_TF"] = "0"
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_config = None
        if load_in_4bit and torch.cuda.is_available():
            try:
                import bitsandbytes
            except ImportError:
                print("[+] Auto-installing bitsandbytes for 4-bit GPU quantization...")
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate"], check=False)

            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as bnb_err:
                print(f"[-] bitsandbytes setup note ({bnb_err}); attempting 4-bit fallback...")

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "quantization_config": quant_config if torch.cuda.is_available() else None,
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "sdpa"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print(f"[+] {model_name} loaded into GPU VRAM successfully!")

    def extract(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        active_sys = system_prompt or EXTRACTION_SYSTEM_PROMPT
        is_round2_arabic = (
            "ROUND 2" in active_sys
            or "CLASSICAL ARABIC LOCALIZATION" in active_sys
            or "ROUND 1 EXTRACTED ENGLISH DOSSIERS" in (prompt or "")
        )
        user_content = prompt or ""
        if "/no_think" not in user_content:
            user_content = f"{user_content.rstrip()}\n/no_think"

        fallback_system = (
            COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT
            if is_round2_arabic
            else (
                "You are an expert Islamic dialectical argument extractor. "
                "Analyze the transcript and extract high-yield Q&A dialectical units as a valid JSON array matching the schema. /no_think"
            )
        )

        messages = [
            {"role": "system", "content": active_sys},
            {"role": "user", "content": user_content},
        ]

        if self.backend == "gguf":
            # Exact or conservative token estimation of formatted prompt
            try:
                formatted = ""
                for m in messages:
                    formatted += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                formatted += "<|im_start|>assistant\n"
                prompt_tokens = len(self.llm.tokenize(formatted.encode("utf-8", errors="replace")))
            except Exception:
                # Conservative fallback: 1 word ≈ 2.2 tokens in Arabic/English
                total_words = sum(len(m.get("content", "").split()) for m in messages)
                prompt_tokens = int(total_words * 2.2)

            safety_margin = 64
            avail = self.context_length - prompt_tokens - safety_margin

            # If available generation headroom is too tight (< 1200 tokens), switch to compact system prompt
            active_messages = messages
            if avail < 1200:
                active_messages = [
                    {"role": "system", "content": fallback_system},
                    {"role": "user", "content": user_content},
                ]
                avail = max(1024, self.context_length - int(len(user_content.split()) * 2.2) - safety_margin)

            # Ensure effective_max_tokens respects the remaining context without trimming critical metadata
            effective_max_tokens = max(512, min(self.max_new_tokens, avail))

            try:
                res = self.llm.create_chat_completion(
                    messages=active_messages,
                    max_tokens=effective_max_tokens,
                    temperature=0.12,
                    top_p=0.9,
                )
                return res["choices"][0]["message"]["content"] or "[]"
            except ValueError as ve:
                if "exceed context window" in str(ve).lower():
                    # Emergency context recovery: compact system prompt and generate
                    compact_messages = [
                        {"role": "system", "content": fallback_system},
                        {"role": "user", "content": user_content},
                    ]
                    try:
                        res = self.llm.create_chat_completion(
                            messages=compact_messages,
                            max_tokens=min(self.max_new_tokens, 2400),
                            temperature=0.12,
                            top_p=0.9,
                        )
                        return res["choices"][0]["message"]["content"] or "[]"
                    except Exception:
                        return "[]"
                raise

        # Transformers fallback path
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        max_gen_tokens = min(self.max_new_tokens, 3200)
        max_prompt_len = min(16384, max(1024, self.context_length - max_gen_tokens - 64))
        inputs = self.tokenizer(
            [text_input],
            return_tensors="pt",
            max_length=max_prompt_len,
            truncation=True,
        ).to(self.model.device)

        eos_ids = [self.tokenizer.eos_token_id]
        for special_token in ["<|im_end|>", "<|endoftext|>"]:
            tok_id = self.tokenizer.convert_tokens_to_ids(special_token)
            if tok_id is not None and tok_id not in eos_ids:
                eos_ids.append(tok_id)

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen_tokens,
                    temperature=0.12,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=eos_ids,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        except torch.OutOfMemoryError:
            print("[-] Transformers GPU OOM encountered; clearing cache and retrying with compact context...")
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fallback_messages = [
                {"role": "system", "content": fallback_system},
                {"role": "user", "content": user_content},
            ]
            fallback_text = self.tokenizer.apply_chat_template(
                fallback_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            compact_inputs = self.tokenizer(
                [fallback_text],
                return_tensors="pt",
                max_length=4096,
                truncation=True,
            ).to(self.model.device)
            with torch.inference_mode():
                gen_ids = self.model.generate(
                    **compact_inputs,
                    max_new_tokens=1600,
                    temperature=0.12,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=eos_ids,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(compact_inputs.input_ids, gen_ids)
            ]
            return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        finally:
            if "inputs" in locals():
                del inputs
            if "compact_inputs" in locals():
                del compact_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_round1_english(self, prompt: str) -> str:
        """Executes Stage 2 Round 1: Pure English dialectical argument and syllogism extraction."""
        return self.extract(prompt, system_prompt=ROUND1_ENGLISH_EXTRACTION_PROMPT)

    def extract_round2_arabic(self, english_units_json: str, transcript_prompt: str) -> str:
        """Executes Stage 2 Round 2: Bilingual theological verification and Classical Arabic localization."""
        if "ROUND 1 EXTRACTED ENGLISH DOSSIERS" in transcript_prompt:
            user_prompt = transcript_prompt
        else:
            user_prompt = (
                f"ORIGINAL TRANSCRIPT CONTEXT:\n{transcript_prompt}\n\n"
                f"ROUND 1 EXTRACTED ENGLISH DOSSIERS:\n{english_units_json}\n\n"
                "Perform Theological Verification and Classical Arabic Localization on the English dossiers. "
                "Translate all fields into pure Classical Arabic (فصحى رصينة), enrich every dialogue exchange turn with text_ar, purge all foreign tokens, and output strictly the completed bilingual JSON array: [ { ... } ]"
            )
        return self.extract(user_prompt, system_prompt=ROUND2_ARABIC_VERIFICATION_PROMPT)


class ExtractionClient:
    """Universal wrapper supporting Local Colab GPU or optional cloud fallback."""

    def __init__(
        self,
        provider: str = "local_gpu",
        model_name: str = "Qwen/Qwen3-14B",
        context_length: int = 20480,
        client: Any = None,
    ):
        self.provider = provider.lower()
        if client is not None:
            self.client = client
        elif self.provider == "local_gpu":
            self.client = LocalGPUClient(model_name=model_name, load_in_4bit=True, context_length=context_length)
        else:
            raise ValueError(f"Unsupported provider '{provider}'. V2 runs 100% free on local Colab T4 GPUs.")

    def extract(self, prompt: str) -> str:
        return self.client.extract(prompt)

    def extract_round1_english(self, prompt: str) -> str:
        if hasattr(self.client, "extract_round1_english"):
            return self.client.extract_round1_english(prompt)
        try:
            return self.client.extract(prompt, system_prompt=ROUND1_ENGLISH_EXTRACTION_PROMPT)
        except TypeError:
            return self.client.extract(prompt)

    def extract_round2_arabic(self, english_units_json: str, transcript_prompt: str) -> str:
        if hasattr(self.client, "extract_round2_arabic"):
            return self.client.extract_round2_arabic(english_units_json, transcript_prompt)
        user_prompt = transcript_prompt
        if "ROUND 1 EXTRACTED ENGLISH DOSSIERS" not in user_prompt:
            user_prompt = (
                f"ORIGINAL TRANSCRIPT CONTEXT:\n{transcript_prompt}\n\n"
                f"ROUND 1 EXTRACTED ENGLISH DOSSIERS:\n{english_units_json}\n\n"
                "Perform Theological Verification and Classical Arabic Localization on the English dossiers. "
                "Translate all fields into pure Classical Arabic (فصحى رصينة), enrich every dialogue exchange turn with text_ar, purge all foreign tokens, and output strictly the completed bilingual JSON array: [ { ... } ]"
            )
        try:
            return self.client.extract(user_prompt, system_prompt=ROUND2_ARABIC_VERIFICATION_PROMPT)
        except TypeError:
            return self.client.extract(user_prompt)

