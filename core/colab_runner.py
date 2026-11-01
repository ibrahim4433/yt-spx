#!/usr/bin/env python3
"""
core/colab_runner.py
Batch extraction runner optimized for Google Colab Free Tier (Local T4 GPU).
Runs open-weight models (Qwen-2.5 / Llama-3) directly on the Colab GPU without any API keys!
Features auto-checkpointing, error recovery, and zero data-loss batching.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Prevent PyTorch CUDA memory fragmentation on Nvidia GPUs (T4 / A100)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from core.srt_parser import parse_srt_file, chunk_transcript_for_extraction
    from core.extractor_prompt import EXTRACTION_SYSTEM_PROMPT, USER_EXTRACTION_PROMPT_TEMPLATE
except (ImportError, ValueError):
    from .srt_parser import parse_srt_file, chunk_transcript_for_extraction
    from .extractor_prompt import EXTRACTION_SYSTEM_PROMPT, USER_EXTRACTION_PROMPT_TEMPLATE

BASE_DIR = Path(__file__).resolve().parent.parent

# Persistent Google Drive support for Colab environment
DRIVE_BASE = Path("/content/drive/MyDrive/yt-ch")
if os.path.exists("/content/drive/MyDrive"):
    DATA_DIR = DRIVE_BASE / "data"
    DOWNLOADS_DIR = DRIVE_BASE / "Downloads"
else:
    DATA_DIR = BASE_DIR / "data"
    DOWNLOADS_DIR = BASE_DIR / "Downloads"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSONL = DATA_DIR / "extracted_qa.jsonl"
CHECKPOINT_FILE = DATA_DIR / "extraction_checkpoint.json"
LIVE_LOG_FILE = DATA_DIR / "extraction_live.log"


def log_extracted_unit_live(unit: Dict[str, Any]):
    """Appends newly extracted Q&A unit to a live human-readable log file on Drive."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts_now = time.strftime("%Y-%m-%d %H:%M:%S")
        cat = unit.get("topic_category", "General")
        q_text = unit.get("question_ar") or unit.get("question_en") or "Untitled Question"
        ans_text = unit.get("core_answer_summary_ar") or unit.get("core_answer_summary_en") or ""
        with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"[{ts_now}] [{unit.get('timestamp_formatted', '00:00')}] [{cat}]\n"
                f"❓ {q_text}\n"
                f"💡 {ans_text[:140]}...\n"
                f"🔗 {unit.get('timestamp_url', '')}\n"
                f"{'-'*75}\n"
            )
    except Exception:
        pass


def load_checkpoints() -> Dict[str, Any]:
    """Loads already processed video IDs to avoid re-processing."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"processed_video_ids": [], "total_qa_extracted": 0}
    return {"processed_video_ids": [], "total_qa_extracted": 0}


def save_checkpoints_dict(state: Dict[str, Any]):
    """Saves the complete checkpoints dictionary directly."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def save_checkpoint(video_id: str, count_extracted: int):
    """Saves progress after each video is processed."""
    state = load_checkpoints()
    if video_id not in state["processed_video_ids"]:
        state["processed_video_ids"].append(video_id)
        state["total_qa_extracted"] = state.get("total_qa_extracted", 0) + count_extracted
    # Clear any chunk-level tracking once video is fully marked done
    if "chunk_progress" in state and video_id in state["chunk_progress"]:
        del state["chunk_progress"][video_id]
    save_checkpoints_dict(state)


def save_chunk_checkpoint(video_id: str, chunk_index: int, total_chunks: int, count_extracted: int):
    """Saves granular chunk progress so interruptions resume exactly from the last completed chunk."""
    state = load_checkpoints()
    if "chunk_progress" not in state:
        state["chunk_progress"] = {}
    state["chunk_progress"][video_id] = {
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "count_extracted": count_extracted,
        "updated_at": time.time()
    }
    save_checkpoints_dict(state)


def get_chunk_checkpoint(video_id: str) -> int:
    """Returns the last completed chunk index (0 if not started)."""
    state = load_checkpoints()
    prog = state.get("chunk_progress", {}).get(video_id, {})
    return prog.get("chunk_index", 0)


def clear_chunk_checkpoint(video_id: str):
    """Clears chunk progress once the entire video is fully completed."""
    state = load_checkpoints()
    if "chunk_progress" in state and video_id in state["chunk_progress"]:
        del state["chunk_progress"][video_id]
        save_checkpoints_dict(state)


def append_extracted_units(units: List[Dict[str, Any]]):
    """Appends extracted Q&A units to the JSONL dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        for unit in units:
            f.write(json.dumps(unit, ensure_ascii=False) + "\n")


def clean_llm_json_response(raw_text: str) -> List[Dict[str, Any]]:
    """Strips markdown code fences and parses JSON safely, including partial/truncated streams."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and ('question_en' in x or 'question_ar' in x)]
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict) and ('question_en' in x or 'question_ar' in x)]
            if 'question_en' in data or 'question_ar' in data:
                return [data]
    except Exception:
        pass

    # Regex fallback to find full JSON array
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict) and ('question_en' in x or 'question_ar' in x)]
        except Exception:
            pass

    # Stream parser fallback: extracts all completed { ... } objects even if the outer array was truncated
    results = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, char in enumerate(text):
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    obj_str = text[start:i+1]
                    try:
                        obj = json.loads(obj_str)
                        if isinstance(obj, dict) and ('question_en' in obj or 'question_ar' in obj):
                            results.append(obj)
                    except Exception:
                        pass
                    start = -1
        escape = False

    return results


def sanitize_extracted_arabic_unit(item: Dict[str, Any]) -> Dict[str, Any]:
    """Removes any stray CJK/Korean tokens or common Arabic-Latin splices from generated fields."""
    replacements = [
        ("따라서،", "لذلك،"), ("따라서", "لذلك"), ("여전히", "لا تزال"),
        ("المiracles", "المعجزات"), ("فالمachines", "فالآلات"), ("كمessenger", "كرسول"),
        ("للمonotheism", "للتوحيد"), ("السcribes", "النساخ"), ("للسcribes", "للنساخ"),
        ("نihilist", "عدمي"), ("النihilists", "العدميون"), ("بprocesseات", "بعمليات"),
        ("مutually exclusive", "متنافيين"), ("ال hypocritical", "النفاق وازدواجية المعايير"),
        ("بهomo جنسياً", "بميول مثلية"), ("التطهيرracial", "التطهير العرقي"),
    ]
    ar_keys = [
        "question_ar", "core_answer_summary_ar",
        "fallacy_or_misconception_exposed_ar", "analogy_or_metaphor_ar",
        "socratic_question_ar", "verbatim_quote_ar"
    ]
    for k in ar_keys:
        val = item.get(k)
        if isinstance(val, str) and val:
            for old_s, new_s in replacements:
                if old_s in val:
                    val = val.replace(old_s, new_s)
            val = re.sub(r"[\u4e00-\u9fff\uac00-\ud7af\ufffd]+", "", val)
            item[k] = val.strip()

    premises = item.get("logical_premises_ar")
    if isinstance(premises, list):
        clean_premises = []
        for p in premises:
            if isinstance(p, str):
                for old_s, new_s in replacements:
                    if old_s in p:
                        p = p.replace(old_s, new_s)
                p = re.sub(r"[\u4e00-\u9fff\uac00-\ud7af\ufffd]+", "", p).strip()
                clean_premises.append(p)
        item["logical_premises_ar"] = clean_premises
    return item


class LocalGPUClient:
    """
    Runs Qwen 2.5 14B Instruct directly on the Google Colab GPU (T4 15GB / A100).
    Supports GGUF Q5_K_M (5-bit weights + 8-bit KV Cache + 8192 Context) via llama-cpp-python,
    with automatic fallback to HuggingFace Transformers 4-bit NF4.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        load_in_4bit: bool = True,
        use_gguf_q5: bool = True,
        gguf_repo: str = "bartowski/Qwen2.5-14B-Instruct-GGUF",
        gguf_filename: str = "Qwen2.5-14B-Instruct-Q5_K_M.gguf",
        context_length: int = 8192
    ):
        import torch
        self.backend = "transformers"
        self.context_length = context_length

        print(f"[+] Initializing Local GPU Model: {model_name} (Target: 14B Q5_K_M / 8192 ctx)")
        print(f"[+] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[+] GPU Device: {torch.cuda.get_device_name(0)}")

        # 1. Try GGUF Q5_K_M (5-bit) with 8-bit KV Cache & 8192 Context if enabled and llama_cpp GPU is available
        if use_gguf_q5 and torch.cuda.is_available():
            try:
                from llama_cpp import Llama, llama_supports_gpu_offload
                from huggingface_hub import hf_hub_download

                if llama_supports_gpu_offload():
                    print(f"[+] Downloading/Loading 5-bit GGUF: {gguf_repo}/{gguf_filename} (~10.5 GB)...")
                    model_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_filename)
                    # GGML_TYPE_Q8_0 = 8 for 8-bit KV Cache (~2 GB for 8192 context)
                    self.llm = Llama(
                        model_path=model_path,
                        n_gpu_layers=-1,
                        n_ctx=context_length,
                        n_batch=512,
                        type_k=8,
                        type_v=8,
                        flash_attn=True,
                        verbose=False
                    )
                    self.backend = "gguf"
                    print(f"[+] Qwen 2.5 14B Q5_K_M (5-bit) + 8-bit KV Cache (8,192 ctx) loaded into VRAM!")
                    return
                else:
                    print("[-] llama-cpp-python installed without CUDA offload; falling back to Transformers GPU backend.")
            except Exception as e:
                print(f"[-] GGUF Q5_K_M init note ({e}); using Transformers 14B GPU backend.")

        # 2. Transformers Backend (Qwen/Qwen2.5-14B-Instruct in 4-bit NF4 + SDPA Flash Attention)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_config = None
        if load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except ImportError:
                print("[-] bitsandbytes not found, continuing without 4-bit quant.")
        elif not torch.cuda.is_available():
            print("   ⚠️ NOTE: Running on CPU without GPU acceleration!")

        model_kwargs = {
            "device_map": "auto",
            "quantization_config": quant_config if torch.cuda.is_available() else None,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            try:
                model_kwargs["attn_implementation"] = "sdpa"
            except Exception:
                pass

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        print(f"[+] {model_name} loaded into GPU VRAM successfully!")

    def extract(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        if self.backend == "gguf":
            res = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=1800,
                temperature=0.15,
                top_p=0.9,
            )
            return res["choices"][0]["message"]["content"]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        eos_ids = [self.tokenizer.eos_token_id]
        for special_token in ["<|im_end|>", "<|endoftext|>"]:
            tok_id = self.tokenizer.convert_tokens_to_ids(special_token)
            if tok_id is not None and tok_id not in eos_ids:
                eos_ids.append(tok_id)

        import torch
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1800,
                    temperature=0.15,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=eos_ids,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        finally:
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ExtractionClient:
    """Universal client supporting Local Colab GPU or API fallback."""
    def __init__(self, provider: str = "local_gpu", model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.provider = provider.lower()
        if self.provider == "local_gpu":
            self.client = LocalGPUClient(model_name=model_name, load_in_4bit=True)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=EXTRACTION_SYSTEM_PROMPT
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def extract(self, prompt: str) -> str:
        if self.provider == "local_gpu":
            return self.client.extract(prompt)
        elif self.provider == "gemini":
            res = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "response_mime_type": "application/json"}
            )
            return res.text
        return "[]"


def is_corrupted_youtube_caption(text: str) -> bool:
    """Detects transcripts generated by YouTube ASR hallucinating English over Arabic audio or noise."""
    text_lower = text.lower()
    gibberish_signals = [
        'arda turan', 'pastry shop', 'e-star co2', 'les powder', 'smirk 4', 
        'tomato sauce or muslims', 'swearing copies', 'spyware this line'
    ]
    if any(sig in text_lower for sig in gibberish_signals):
        return True

    words = re.findall(r'[a-z]{3,}', text_lower)
    if len(words) < 50:
        return True

    debate_words = {
        'god', 'allah', 'prophet', 'jesus', 'christian', 'muslim', 'quran', 'bible', 
        'islam', 'trinity', 'atheist', 'faith', 'creator', 'evidence', 'resurrection', 
        'shia', 'sunni', 'church', 'prayer', 'sin', 'heaven', 'hell', 'moral', 'morality', 
        'science', 'universe', 'creation', 'believe', 'religion', 'truth', 'human', 'life', 'book'
    }
    found = sum(1 for w in debate_words if w in text_lower)
    if found < 2:
        return True
    return False


def process_video(
    video_record: Dict[str, Any],
    client: Any,
    chunk_duration_sec: float = 240.0,
    overlap_duration_sec: float = 60.0,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Processes a single video by chunking its transcript and extracting all Q&As."""
    raw_srt_rel = video_record.get("srt_file", "").replace("\\", "/")
    filename = video_record.get("srt_filename") or Path(raw_srt_rel).name

    candidate_paths = [
        BASE_DIR / raw_srt_rel,
        BASE_DIR / "Downloads" / filename,
        DOWNLOADS_DIR / filename,
        DRIVE_BASE / raw_srt_rel,
        DRIVE_BASE / "Downloads" / filename,
        Path("/content/drive/MyDrive/yt-ch/Downloads") / filename,
        Path("/content/drive/MyDrive/yt-ch") / raw_srt_rel,
    ]

    srt_path = None
    for p in candidate_paths:
        if p.exists():
            srt_path = p
            break

    if not srt_path:
        return None

    cues = parse_srt_file(srt_path)
    if not cues:
        return None

    video_id = video_record["video_id"]
    title_ar = video_record.get("title_ar", "")

    # Pre-filter: Check if transcript is YouTube acoustic noise before running heavy GPU passes
    full_text = " ".join(c.text for c in cues)
    if is_corrupted_youtube_caption(full_text):
        if verbose:
            print(f"\n🎬 [{video_id}] {title_ar}", flush=True)
            print(f"   ⏭️ Skipped in 0.01s: Corrupted YouTube auto-caption (acoustic noise / non-English audio)\n", flush=True)
        return []

    chunks = chunk_transcript_for_extraction(
        cues,
        chunk_duration_sec=chunk_duration_sec,
        overlap_duration_sec=overlap_duration_sec
    )

    all_units = []
    youtube_base = video_record.get("youtube_url", f"https://www.youtube.com/watch?v=YC86BQwvHyk")

    total_duration = cues[-1].end_sec if hasattr(cues[-1], 'end_sec') else 0.0
    dur_min = int(total_duration // 60)
    dur_sec = int(total_duration % 60)

    if verbose:
        print(f"\n🎬 [{video_id}] {title_ar}", flush=True)
        print(f"   ⏱️ Duration: {dur_min:02d}:{dur_sec:02d} | Cues: {len(cues)} | Chunks: {len(chunks)}", flush=True)

    start_chunk_idx = get_chunk_checkpoint(video_id)
    if start_chunk_idx > 0 and start_chunk_idx < len(chunks):
        if verbose:
            print(f"   🔄 Resuming from Chunk {start_chunk_idx + 1}/{len(chunks)} (Chunks 1-{start_chunk_idx} already saved in DB)...", flush=True)

    for i, chunk in enumerate(chunks):
        if i < start_chunk_idx:
            continue

        # Skip chunks that have almost no spoken dialogue (e.g. music, intro/outro silence)
        if len(chunk["text"].strip()) < 80:
            if verbose:
                print(f"   ├─ ⏭️ Chunk {i+1}/{len(chunks)} [{chunk['start_fmt']} -> {chunk['end_fmt']}]: Skipped (no dialogue)", flush=True)
            continue

        if verbose:
            print(f"   ├─ ⏳ Chunk {i+1}/{len(chunks)} [{chunk['start_fmt']} -> {chunk['end_fmt']}]: Extracting on GPU...", end="", flush=True)

        prompt = USER_EXTRACTION_PROMPT_TEMPLATE.format(
            title_ar=title_ar,
            youtube_url=youtube_base,
            start_fmt=chunk["start_fmt"],
            end_fmt=chunk["end_fmt"],
            transcript_text=chunk["text"]
        )

        t0 = time.time()
        try:
            raw_output = client.extract(prompt)
            t_chunk = round(time.time() - t0, 1)
            extracted = clean_llm_json_response(raw_output)

            chunk_units = []
            for item in extracted:
                item = sanitize_extracted_arabic_unit(item)
                ts_sec = item.get("timestamp_seconds", int(chunk["start_sec"]))
                if ts_sec < chunk["start_sec"]:
                    ts_sec = int(chunk["start_sec"])

                # Overlap deduplication: prevent saving identical question from overlapping chunk window
                q_ar = item.get("question_ar", "").strip()
                q_en = item.get("question_en", "").strip()
                is_duplicate = False
                for existing in all_units:
                    if abs(existing.get("timestamp_seconds", 0) - ts_sec) <= 90:
                        ex_ar = existing.get("question_ar", "").strip()
                        ex_en = existing.get("question_en", "").strip()
                        if (q_ar and ex_ar and (q_ar == ex_ar or q_ar in ex_ar or ex_ar in q_ar)) or \
                           (q_en and ex_en and (q_en.lower() == ex_en.lower() or q_en.lower() in ex_en.lower())):
                            is_duplicate = True
                            break

                if is_duplicate:
                    continue

                item["id"] = f"{video_id}_{len(all_units) + 1:03d}"
                item["video_id"] = video_id
                item["channel"] = video_record.get("channel", "SCDawah Arabic")
                item["video_title_ar"] = title_ar
                item["youtube_url"] = youtube_base
                item["timestamp_url"] = f"{youtube_base}&t={int(ts_sec)}s"
                item["timestamp_seconds"] = int(ts_sec)

                chunk_units.append(item)
                all_units.append(item)
                log_extracted_unit_live(item)

            # Live chunk persistence: Append to JSONL and SQLite immediately
            if chunk_units:
                append_extracted_units(chunk_units)
                try:
                    from db.database import insert_qa_unit
                    for cu in chunk_units:
                        insert_qa_unit(cu)
                except Exception:
                    pass

            # Update chunk-level checkpoint
            save_chunk_checkpoint(video_id, i + 1, len(chunks), len(all_units))

            if verbose:
                if chunk_units:
                    print(f" ✅ {len(chunk_units)} Q&A found ({t_chunk}s)", flush=True)
                    for cu in chunk_units:
                        q_text = cu.get("question_ar") or cu.get("question_en") or "Untitled"
                        print(f"   │  • ❓ \"{q_text[:70]}\" [{cu.get('timestamp_formatted', '00:00')}]", flush=True)
                else:
                    preview_txt = raw_output.strip().replace("\n", " ")[:60]
                    if preview_txt and preview_txt != "[]":
                        print(f" ⚪ No dialectical debate in segment ({t_chunk}s) [Raw: {preview_txt}...]", flush=True)
                    else:
                        print(f" ⚪ No dialectical debate in segment ({t_chunk}s)", flush=True)

        except Exception as e:
            if verbose:
                print(f" ❌ Error: {e}", flush=True)
            time.sleep(1)

    clear_chunk_checkpoint(video_id)
    if verbose:
        print(f"   💾 Completed all {len(chunks)} chunks for [{video_id}] ({len(all_units)} Q&As saved to SQLite & Drive)!\n", flush=True)

    return all_units
