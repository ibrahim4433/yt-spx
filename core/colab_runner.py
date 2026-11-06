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
    from core.extractor_prompt import EXTRACTION_SYSTEM_PROMPT
except (ImportError, ValueError):
    from .drive_sync import resolve_drive_root
    from .extractor_prompt import EXTRACTION_SYSTEM_PROMPT

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


def clean_llm_json_response(raw_text: str) -> List[Dict[str, Any]]:
    """Strips markdown code fences and parses JSON safely, including partial/truncated streams."""
    if not raw_text:
        return []
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and ("question_en" in x or "question_ar" in x)]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict) and ("question_en" in x or "question_ar" in x)]
            if "question_en" in data or "question_ar" in data:
                return [data]
    except Exception:
        pass

    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict) and ("question_en" in x or "question_ar" in x)]
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
                        if isinstance(obj, dict) and ("question_en" in obj or "question_ar" in obj):
                            results.append(obj)
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
        ("condemn", "يُدين"), ("Cause", "سبب أول"), ("causes", "أسباب أولى"),
        ("والincest", "وزنا المحارم"),
        ("الincest", "زنا المحارم"),
        ("incest", "زنا المحارم"), ("two wills", "إرادتين"), ("wills", "إرادات"),
        ("natures", "طبائع"), ("divine", "إلهي"), ("human", "بشري"),
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
                for old_s, new_s in replacements:
                    if old_s in p:
                        p = p.replace(old_s, new_s)
                p = re.sub(r"[\u0400-\u04ff\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]+", "", p).strip()
                clean_premises.append(p)
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


class LocalGPUClient:
    """
    Runs Qwen 2.5 14B Instruct directly on the Google Colab GPU (T4 15GB / L4 / A100).
    Uses GGUF Q5_K_M (5-bit weights + 8-bit KV Cache + 12,288 Context) via llama-cpp-python
    (~13.2 GiB VRAM on isolated Stage 2 worker), with fallback to Transformers 4-bit NF4.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        load_in_4bit: bool = True,
        use_gguf_q5: bool = True,
        gguf_repo: str = "bartowski/Qwen2.5-14B-Instruct-GGUF",
        gguf_filename: str = "Qwen2.5-14B-Instruct-Q5_K_M.gguf",
        context_length: int = 16384,
        max_new_tokens: int = 2400,
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
                    print(f"[+] Downloading/Loading 5-bit GGUF: {gguf_repo}/{gguf_filename} (~10.5 GB)...")
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError:
                        import subprocess, sys
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
                        from huggingface_hub import hf_hub_download

                    model_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_filename)
                    # GGML_TYPE_Q8_0 = 8 for 8-bit KV Cache (~1.2 GB at 12,288 context)
                    try:
                        self.llm = Llama(
                            model_path=model_path,
                            n_gpu_layers=-1,
                            n_ctx=context_length,
                            n_batch=1024,
                            type_k=8,
                            type_v=8,
                            flash_attn=True,
                            verbose=False,
                        )
                    except Exception as fa_err:
                        print(f"[-] GGUF flash_attn note ({fa_err}); retrying with standard attention...")
                        self.llm = Llama(
                            model_path=model_path,
                            n_gpu_layers=-1,
                            n_ctx=context_length,
                            n_batch=1024,
                            type_k=8,
                            type_v=8,
                            flash_attn=False,
                            verbose=False,
                        )
                    self.backend = "gguf"
                    print(f"[+] Qwen 2.5 14B Q5_K_M + 8-bit KV Cache ({context_length} ctx) loaded into VRAM!")
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
        messages = [
            {"role": "system", "content": system_prompt or EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
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

            # Ensure effective_max_tokens respects the remaining context without trimming critical metadata
            effective_max_tokens = max(256, min(self.max_new_tokens, avail))

            try:
                res = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=0.12,
                    top_p=0.9,
                )
                return res["choices"][0]["message"]["content"] or "[]"
            except ValueError as ve:
                if "exceed context window" in str(ve).lower():
                    # Emergency context recovery: cut max_tokens to 512 and retry
                    res = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=512,
                        temperature=0.12,
                        top_p=0.9,
                    )
                    return res["choices"][0]["message"]["content"] or "[]"
                raise

        # Transformers fallback path
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        max_gen_tokens = min(self.max_new_tokens, 1200)
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
            compact_inputs = self.tokenizer(
                [text_input],
                return_tensors="pt",
                max_length=2500,
                truncation=True,
            ).to(self.model.device)
            with torch.inference_mode():
                gen_ids = self.model.generate(
                    **compact_inputs,
                    max_new_tokens=800,
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ExtractionClient:
    """Universal wrapper supporting Local Colab GPU or optional cloud fallback."""

    def __init__(
        self,
        provider: str = "local_gpu",
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        context_length: int = 14336,
    ):
        self.provider = provider.lower()
        if self.provider == "local_gpu":
            self.client = LocalGPUClient(model_name=model_name, load_in_4bit=True, context_length=context_length)
        else:
            raise ValueError(f"Unsupported provider '{provider}'. V2 runs 100% free on local Colab T4 GPUs.")

    def extract(self, prompt: str) -> str:
        return self.client.extract(prompt)
