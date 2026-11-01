#!/usr/bin/env python3
"""
run_colab_all_in_one.py
ONE-STEP Automated Multi-Day Extractor for Google Colab (Free T4 GPU).

Does everything in a single execution:
  1. Mounts Google Drive & navigates to project folder.
  2. Auto-installs GPU dependencies if missing.
  3. Loads Qwen 2.5 7B Instruct in 4-bit into T4 GPU VRAM.
  4. Resumes from last checkpoint automatically.
  5. Extracts dialectical Q&As and updates SQLite database live on the fly.
  6. 100% zero data loss when daily free GPU quota expires.
"""

import os
import sys
from pathlib import Path

# Prevent PyTorch CUDA memory fragmentation on Nvidia GPUs (T4 / A100)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force project root and working directory into sys.path immediately at module level
CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import json
import time


# 1. Environment & Google Drive Auto-Setup
def setup_environment():
    print("=" * 70)
    print("  🕌  DAWAH DIALECTICS: ONE-STEP LOCAL GPU PIPELINE")
    print("=" * 70)

    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    if is_colab:
        print("[1/5] Running inside Google Colab environment...")
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                print("   -> Mounting Google Drive...")
                drive.mount("/content/drive")
            
            drive_base = Path("/content/drive/MyDrive/yt-ch")
            drive_downloads = drive_base / "Downloads"
            drive_data = drive_base / "data"

            drive_data.mkdir(parents=True, exist_ok=True)
            print(f"   💾 Persistent Google Drive storage: {drive_data}")

            # Link Google Drive Downloads to local workspace if not present
            local_downloads = Path(CURRENT_DIR) / "Downloads"
            if drive_downloads.exists() and not local_downloads.exists():
                try:
                    os.symlink(drive_downloads, local_downloads)
                    print(f"   🔗 Linked subtitle corpus from Drive: {drive_downloads}")
                except Exception:
                    pass

            # Remove any stale code folders on Drive that might shadow the active Git repository
            for stale_dir in ["core", "db", "web", "scripts", "tests", "__pycache__"]:
                stale_path = drive_base / stale_dir
                if stale_path.exists() and stale_path.is_dir():
                    try:
                        import shutil
                        shutil.rmtree(stale_path)
                    except Exception:
                        pass
            for stale_file in ["run_colab_all_in_one.py", "run_app.py", "download_videos.py"]:
                stale_path = drive_base / stale_file
                if stale_path.exists():
                    try:
                        stale_path.unlink()
                    except Exception:
                        pass

            print(f"   📁 Active Code Workspace: {CURRENT_DIR}")
        except Exception as e:
            print(f"   [-] Note on Drive setup: {e}")
    else:
        print("[1/5] Running in standard environment.")

    return is_colab

    return is_colab


# 2. Auto-Install Dependencies if Missing
def ensure_dependencies():
    print("\n[2/5] Checking GPU acceleration packages...")
    packages = ["transformers", "torch", "accelerate", "bitsandbytes", "huggingface_hub", "tqdm"]
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    import subprocess
    if missing:
        print(f"   -> Installing missing packages: {', '.join(missing)}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("   -> Core packages installed successfully.")
    else:
        print("   -> Core packages already installed.")

    # Ensure pre-built CUDA wheel for llama-cpp-python (for Qwen 2.5 14B Q5_K_M GGUF + 8-bit KV Cache)
    try:
        import llama_cpp
        if not llama_cpp.llama_supports_gpu_offload():
            raise ImportError("llama_cpp lacks GPU offload")
    except Exception:
        print("   -> Installing pre-built CUDA wheel for llama-cpp-python (Q5_K_M GGUF engine)...")
        installed_wheel = False
        for cu_ver in ["cu124", "cu122", "cu121"]:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q",
                    "llama-cpp-python",
                    "--only-binary=:all:",
                    f"--extra-index-url=https://abetlen.github.io/llama-cpp-python/whl/{cu_ver}"
                ])
                installed_wheel = True
                print(f"   -> llama-cpp-python ({cu_ver}) pre-built GPU wheel installed!")
                break
            except Exception:
                continue
        if not installed_wheel:
            print("   -> Pre-built llama-cpp wheel not matched; will use Transformers 14B GPU engine.")


# 3. Main One-Step Execution Loop
def run_pipeline():
    setup_environment()
    ensure_dependencies()

    # Purge any Google Drive code paths from sys.path so they never shadow the git repo
    sys.path = [p for p in sys.path if "drive/MyDrive/yt-ch" not in p.replace("\\", "/")]
    if "/content/yt-ch" in sys.path:
        sys.path.remove("/content/yt-ch")
    if os.path.exists("/content/yt-ch"):
        sys.path.insert(0, "/content/yt-ch")
    if CURRENT_DIR in sys.path:
        sys.path.remove(CURRENT_DIR)
    sys.path.insert(0, CURRENT_DIR)

    import torch
    from core.colab_runner import (
        LocalGPUClient, process_video, append_extracted_units,
        save_checkpoint, load_checkpoints, save_checkpoints_dict, DATA_DIR
    )
    from db.database import init_db, insert_qa_unit, get_stats, DB_PATH

    # Initialize SQLite Database
    init_db()

    # Auto-enrich and sanitize any previous records in Google Drive
    try:
        from scripts.enrich_arabic_db import run_enrichment
        run_enrichment(DB_PATH)
    except Exception as e:
        print(f"   [-] Note on Arabic enrichment check: {e}")

    try:
        from scripts.sync_and_clean_607 import clean_db_glitches
        fixed_cnt = clean_db_glitches(DB_PATH)
        if fixed_cnt > 0:
            print(f"   🧹 Cleaned {fixed_cnt} records from CJK/English token splices in SQLite DB!")
    except Exception as e:
        print(f"   [-] Note on DB glitch sanitizer: {e}")

    # Verify GPU
    print("\n[3/5] Verifying GPU hardware...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   ✅ GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("   ⚠️ No GPU detected! Running on CPU (will be slower).")

    # Load Model: Qwen 2.5 14B Instruct (Q5_K_M 5-bit GGUF + 8-bit KV Cache + 8192 Context)
    print("\n[4/5] Loading Qwen 2.5 14B Instruct (Q5_K_M 5-bit / 8192 ctx / 8-bit KV Cache) into GPU memory...")
    client = LocalGPUClient(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        load_in_4bit=True,
        use_gguf_q5=True,
        context_length=8192
    )

    # Load Manifest & Checkpoints
    manifest_path = DATA_DIR / "video_manifest.json"
    if not manifest_path.exists():
        from core.metadata_linker import build_manifest
        manifest = build_manifest()
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Silently ensure all paths use forward slashes for Linux compatibility
    manifest_dirty = False
    for v in manifest:
        if "srt_file" in v and "\\" in v["srt_file"]:
            v["srt_file"] = v["srt_file"].replace("\\", "/")
            manifest_dirty = True
    if manifest_dirty:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    checkpoints = load_checkpoints()
    processed_set = set(checkpoints.get("processed_video_ids", []))
    total_qa_recorded = checkpoints.get("total_qa_extracted", 0)

    # Auto-repair: If checkpoint recorded 0 extractions across many skipped videos (e.g. from earlier path testing)
    if len(processed_set) > 50 and total_qa_recorded == 0:
        print("\n   ⚠️  Detected empty checkpoint from previous path test (videos were skipped without extraction).")
        print("   🔄 Automatically resetting checkpoint so real GPU extraction will run on all videos!")
        checkpoints = {"processed_video_ids": [], "total_qa_extracted": 0}
        processed_set = set()
        save_checkpoints_dict(checkpoints)

    # Auto-heal: If any video in checkpoint has 0 extracted units in the database, re-queue it for proper extraction
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        real_video_ids = set(r[0] for r in conn.execute("SELECT DISTINCT video_id FROM qa_units WHERE video_id IS NOT NULL").fetchall())
        conn.close()

        valid_processed = [vid for vid in checkpoints.get("processed_video_ids", []) if vid in real_video_ids]
        if len(valid_processed) < len(checkpoints.get("processed_video_ids", [])):
            diff_count = len(checkpoints.get("processed_video_ids", [])) - len(valid_processed)
            print(f"\n   🔄 Detected {diff_count} video(s) with 0 units. Resetting them for full extraction!")
            checkpoints["processed_video_ids"] = valid_processed
            save_checkpoints_dict(checkpoints)
            processed_set = set(valid_processed)
    except Exception as e:
        print(f"   [-] Checkpoint sync note: {e}")

    total_videos = len(manifest)
    completed_videos = len(processed_set)
    remaining_manifest = [v for v in manifest if v["video_id"] not in processed_set]
    # Prioritize any video currently in-progress (interrupted mid-chunk), then sort by largest transcripts
    in_progress_ids = set(checkpoints.get("chunk_progress", {}).keys())
    remaining_manifest.sort(
        key=lambda x: (x.get("video_id") in in_progress_ids, x.get("file_size", 0)),
        reverse=True
    )
    remaining_videos = len(remaining_manifest)

    print("\n[5/5] Starting Batch Dialectical Extraction...")
    print(f"   📊 Total Archive: {total_videos} videos")
    print(f"   ✅ Previously Completed: {completed_videos} videos")
    print(f"   ⏳ Remaining to Process: {remaining_videos} videos")
    print(f"   💡 Live Log Monitor: /content/drive/MyDrive/yt-ch/data/extraction_live.log")
    print("-" * 70)

    from tqdm.auto import tqdm
    pbar = tqdm(remaining_manifest, desc="Extracting Q&A Units", initial=completed_videos, total=total_videos)

    total_extracted_session = 0

    try:
        for video in pbar:
            vid = video["video_id"]
            title_preview = video.get("title_ar", "")[:35]
            pbar.set_postfix_str(f"{vid} | {title_preview}")
            sys.stdout.flush()

            t_vid_start = time.time()
            try:
                units = process_video(video, client)
                if units is None:
                    # Subtitle missing or unreadable - do not mark as completed
                    print(f"[-] Subtitle file not accessible for {vid}: {title_preview}", flush=True)
                    continue

                if units:
                    # 1. Save to JSONL in Drive
                    append_extracted_units(units)
                    # 2. Save directly to SQLite DB in Drive live
                    for u in units:
                        insert_qa_unit(u)
                    total_extracted_session += len(units)

                # 3. Save checkpoint in Drive (video was genuinely processed by model)
                save_checkpoint(vid, len(units))
                processed_set.add(vid)

                t_vid_elapsed = round(time.time() - t_vid_start, 1)
                pbar.set_postfix_str(f"+{len(units)} QA ({t_vid_elapsed}s) | Total: {total_extracted_session}")
                sys.stdout.flush()

            except Exception as e:
                print(f"\n[-] Error on video {vid}: {e}", flush=True)
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n[!] Execution paused by user. Progress safely saved.")

    print("\n" + "=" * 70)
    print(f"🎉 Session Complete! Extracted {total_extracted_session} Q&A arguments this session.")
    print("   📊 Total Library Stats:", get_stats())
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
