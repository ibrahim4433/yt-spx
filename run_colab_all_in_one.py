#!/usr/bin/env python3
"""
run_colab_all_in_one.py
ONE-STEP Automated Bootstrapper for Dawah Dialectics V2 (yt-spx) on Google Colab Free T4 GPUs.

Features:
  1. Mounts Google Drive & resolves the shared 5 TB workspace (/content/drive/MyDrive/yt-spx).
     (Leaves the legacy /content/drive/MyDrive/yt-ch V1 archive 100% untouched!)
  2. Auto-installs Deno JS runtime + yt-dlp + Stage 1 Acoustic/Diarization/Separation stack
     + Stage 2 Qwen 2.5 14B Q5_K_M CUDA GGUF wheel depending on the selected mode.
  3. Delegates to core/pipeline_orchestrator.py with subprocess CUDA isolation, M1->M6
     micro-step resume, anti-ban YouTube pacing, and multi-account Drive lock coordination.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def setup_colab_drive() -> bool:
    print("=" * 74)
    print("  DAWAH DIALECTICS V2 (YT-SPX): MULTI-STAGE ACOUSTIC & LLM PIPELINE")
    print("=" * 74)

    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    if not is_colab:
        print("[1/3] Running in standard local environment.")
        return False

    print("[1/3] Detected Google Colab runtime...")
    try:
        from google.colab import drive

        if not os.path.exists("/content/drive/MyDrive") and not os.path.exists("/content/drive/My Drive"):
            print("   -> Mounting Google Drive...")
            drive.mount("/content/drive")
    except Exception as exc:
        print(f"   [-] Note on Drive mount: {exc}")

    from core.drive_sync import SharedWorkspace

    ws = SharedWorkspace()
    print(f"   -> Shared 5 TB Workspace : {ws.root} (Drive={ws.is_drive})")
    print(f"   -> Active Code Directory : {CURRENT_DIR}")
    cookies_cand = ws.root / "cookies.txt"
    if cookies_cand.exists():
        print(f"   -> Shared cookies.txt    : Found ({cookies_cand})")
    else:
        print(
            f"   -> Shared cookies.txt    : Optional — place Netscape cookies.txt at {cookies_cand} "
            "if YouTube requests authentication."
        )
    return True


def _ensure_deno_runtime() -> None:
    """Installs Deno (<3s) so yt-dlp solves YouTube JS signature/n-parameter challenges cleanly."""
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.exists():
        os.environ["PATH"] = f"{deno_bin}:{os.environ.get('PATH', '')}"
    if shutil.which("deno") or shutil.which("node"):
        return
    print("   -> Installing Deno JS runtime for yt-dlp YouTube challenge solver...")
    subprocess.run(
        "curl -fsSL https://deno.land/install.sh | sh",
        shell=True,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if deno_bin.exists():
        os.environ["PATH"] = f"{deno_bin}:{os.environ.get('PATH', '')}"


def ensure_stage1_dependencies(constraints_path: Path) -> None:
    marker = Path("/tmp/.yt_spx_stage1_deps_ok")
    if marker.exists():
        print("   -> Stage 1 audio/diarization dependencies already verified in this runtime.")
        _ensure_deno_runtime()
        return

    _ensure_deno_runtime()

    if shutil.which("ffmpeg") is None and shutil.which("apt-get"):
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)

    c_arg = ["-c", str(constraints_path)] if constraints_path.exists() else []
    stage1_pkgs = [
        "yt-dlp>=2025.1.15",
        "curl_cffi",
        "demucs",
        "faster-whisper>=1.1.0",
        "ctc-forced-aligner",
        "deepmultilingualpunctuation",
        "soundfile",
        "scipy",
        "huggingface_hub",
        "tqdm",
    ]
    print("   -> Ensuring Stage 1 ASR, Demucs & Forced-Alignment packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + c_arg + stage1_pkgs)

    try:
        import nemo.collections.asr  # noqa: F401
    except ImportError:
        print("   -> Installing NVIDIA NeMo ASR Toolkit (MSDD + TitaNet-Large)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + c_arg + ["cython", "nemo_toolkit[asr]"]
        )

    try:
        import clearvoice  # noqa: F401
    except ImportError:
        print("   -> Installing ClearerVoice-Studio (MossFormer2_SS_16K 2-speaker overlap separator)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + c_arg + ["clearvoice"],
            check=False,
        )

    marker.write_text("ok", encoding="utf-8")


def ensure_stage2_dependencies(constraints_path: Path) -> None:
    marker = Path("/tmp/.yt_spx_stage2_deps_ok")
    if marker.exists():
        print("   -> Stage 2 Local LLM dependencies already verified in this runtime.")
        return

    c_arg = ["-c", str(constraints_path)] if constraints_path.exists() else []
    llm_pkgs = ["transformers", "accelerate", "bitsandbytes", "huggingface_hub", "tqdm"]
    print("   -> Ensuring Stage 2 LLM core packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + c_arg + llm_pkgs)

    try:
        import llama_cpp

        if not llama_cpp.llama_supports_gpu_offload():
            raise ImportError("llama_cpp lacks CUDA offload")
    except Exception:
        print("   -> Installing pre-built CUDA wheel for llama-cpp-python (Qwen 2.5 14B Q5_K_M GGUF)...")
        for cu_ver in ["cu124", "cu122", "cu121"]:
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "llama-cpp-python",
                    "--only-binary=:all:",
                    f"--extra-index-url=https://abetlen.github.io/llama-cpp-python/whl/{cu_ver}",
                ],
                check=False,
            )
            if res.returncode == 0:
                print(f"   -> Installed llama-cpp-python ({cu_ver}) GPU wheel!")
                break

    marker.write_text("ok", encoding="utf-8")


def _detect_requested_mode(argv: List[str]) -> str:
    for i, tok in enumerate(argv):
        if tok == "--mode" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--mode="):
            return tok.split("=", 1)[1]
    return "auto"


def main() -> int:
    argv = sys.argv[1:]
    mode = _detect_requested_mode(argv)
    is_colab = setup_colab_drive()

    constraints_path = Path(CURRENT_DIR) / "constraints_colab.txt"
    if is_colab and mode not in ("status", "sync-db"):
        print(f"\n[2/3] Verifying GPU packages for mode='{mode}'...")
        if mode in ("auto", "stage1", "stage1-worker"):
            ensure_stage1_dependencies(constraints_path)
        if mode in ("auto", "stage2", "stage2-worker"):
            ensure_stage2_dependencies(constraints_path)
    else:
        print(f"\n[2/3] Lightweight mode '{mode}' — skipping heavy GPU wheel checks.")

    # Ensure sys.path prioritizes /content/yt-spx and never legacy /content/drive/MyDrive/yt-ch
    sys.path = [p for p in sys.path if "yt-ch" not in p.replace("\\", "/")]
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    print(f"\n[3/3] Handing off to YT-SPX V2 Smart-Batch Orchestrator (args={argv or ['--mode', 'auto']})...\n")
    from core.pipeline_orchestrator import main as orchestrator_main

    return orchestrator_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
