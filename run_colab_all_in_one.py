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
    if ws.is_drive:
        torch_cache = ws.root / "models_cache" / "torch"
        torch_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(torch_cache))

    print(f"   -> Shared 5 TB Workspace : {ws.root} (Drive={ws.is_drive})")
    print(f"   -> Active Code Directory : {CURRENT_DIR}")
    try:
        from core.cookie_pool import CookiePool

        (ws.root / "cookies").mkdir(parents=True, exist_ok=True)
        CookiePool(ws).print_summary(indent="   -> ")
    except Exception as exc:
        print(f"   -> Cookies pool        : check skipped ({exc})")
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


def _get_pip_cache_args() -> List[str]:
    """
    Returns pip arguments pointing to /content/drive/MyDrive/yt-spx/pip_cache
    so downloaded & built wheels (ctc-forced-aligner, nemo_toolkit) are cached on 5TB Drive
    and indexed strictly by Python ABI & version constraints.
    """
    from core.drive_sync import SharedWorkspace

    ws = SharedWorkspace()
    if not ws.is_drive:
        return []
    cache_dir = ws.root / "pip_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Purge any stale flat wheels from legacy runs (e.g. numpy-1.x, ml_dtypes-0.4.x, nemo_toolkit-3.x)
    legacy_wheels_dir = cache_dir / "wheels"
    if legacy_wheels_dir.exists():
        for bad_glob in ("numpy-1.*.whl", "ml_dtypes-0.4*.whl", "nemo_toolkit-3.*.whl", "scipy-1.1[0-4]*.whl"):
            for stale_whl in legacy_wheels_dir.glob(bad_glob):
                try:
                    stale_whl.unlink()
                except OSError:
                    pass

    return [
        f"--cache-dir={cache_dir}",
        "--prefer-binary",
    ]


def _dist_version(dist_name: str) -> str:
    """
    Returns the installed distribution version WITHOUT importing the package
    (importlib.util.find_spec('a.b.c') imports parent packages and raises
    ModuleNotFoundError when 'a' is missing, so it must never be used for checks).
    """
    import importlib
    import importlib.metadata as md

    importlib.invalidate_caches()
    try:
        return md.version(dist_name)
    except md.PackageNotFoundError:
        return ""
    except Exception:
        return ""


def _nemo_2x_installed() -> bool:
    """True only for nemo_toolkit >=2.6,<3.0 (3.0 removed msdd_models.NeuralDiarizer)."""
    ver = _dist_version("nemo_toolkit")
    if not ver:
        return False
    try:
        major, minor = (int("".join(ch for ch in part if ch.isdigit()) or 0) for part in ver.split(".")[:2])
    except ValueError:
        return False
    return major == 2 and minor >= 6


def _verify_stage1_runtime_health() -> bool:
    """Fast (<0.5s) check: ml_dtypes>=0.5.4 ABI, NeMo 2.x, clearvoice, faster-whisper & demucs installed."""
    check_code = (
        "import importlib.metadata as md, importlib.util as u, ml_dtypes; "
        "assert hasattr(ml_dtypes, 'float4_e2m1fn'); "
        "v = md.version('nemo_toolkit').split('.'); "
        "assert int(v[0]) == 2 and int(''.join(c for c in v[1] if c.isdigit())) >= 6; "
        "md.version('clearvoice'); "
        "assert all(u.find_spec(m) is not None for m in ('faster_whisper', 'demucs', 'ctc_forced_aligner', 'yt_dlp'))"
    )
    res = subprocess.run(
        [sys.executable, "-c", check_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return res.returncode == 0


def ensure_stage1_dependencies(constraints_path: Path) -> None:
    marker = Path("/tmp/.yt_spx_stage1_deps_ok")
    if marker.exists():
        if _verify_stage1_runtime_health():
            print("   -> Stage 1 audio/diarization dependencies already verified in this runtime.")
            _ensure_deno_runtime()
            return
        print("   -> Detected stale or mismatched runtime packages; repairing automatically...")
        marker.unlink(missing_ok=True)

    _ensure_deno_runtime()

    if shutil.which("ffmpeg") is None and shutil.which("apt-get"):
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)

    pip_cache = _get_pip_cache_args()
    c_arg = ["-c", str(constraints_path)] if constraints_path.exists() else []
    stage1_pkgs = [
        "yt-dlp>=2025.1.15",
        "curl_cffi",
        "demucs",
        "faster-whisper>=1.1.0",
        "ctc-forced-aligner",
        "deepmultilingualpunctuation",
        "soundfile",
        "huggingface_hub",
        "tqdm",
        "ml-dtypes>=0.5.4",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ]
    print("   -> Ensuring Stage 1 ASR, Demucs & Forced-Alignment packages (cached on 5TB Drive)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pip_cache + c_arg + stage1_pkgs)

    if not _nemo_2x_installed():
        print("   -> Installing NVIDIA NeMo 2.x ASR Toolkit (MSDD NeuralDiarizer + TitaNet-Large)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"]
            + pip_cache
            + c_arg
            + ["cython", "nemo_toolkit[asr]>=2.6.0,<3.0.0", "ml-dtypes>=0.5.4"]
        )

    if not _dist_version("clearvoice"):
        print("   -> Installing ClearerVoice-Studio (MossFormer2_SS_16K 2-speaker overlap separator)...")
        # Install pure audio dependencies with constraints first, then --no-deps on clearvoice
        # so clearvoice's legacy setup.py metadata never downgrades NumPy 2.x or ml-dtypes on Python 3.13.
        clearvoice_deps = [
            "einops",
            "gdown",
            "rotary_embedding_torch",
            "python_speech_features",
            "torchinfo",
            "yamlargparse",
            "scenedetect",
            "pydub",
            "librosa",
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pip_cache + c_arg + clearvoice_deps)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + pip_cache + ["clearvoice"])

    # Final ABI guard: guarantee ml_dtypes>=0.5.4 (float4_e2m1fn required by onnx>=1.20 on Python 3.13)
    abi_check = subprocess.run(
        [sys.executable, "-c", "import ml_dtypes; assert hasattr(ml_dtypes, 'float4_e2m1fn')"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if abi_check.returncode != 0:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "ml-dtypes>=0.5.4"] + pip_cache)

    marker.write_text("ok", encoding="utf-8")


def ensure_stage2_dependencies(constraints_path: Path) -> None:
    marker = Path("/tmp/.yt_spx_stage2_deps_ok")
    if marker.exists():
        print("   -> Stage 2 Local LLM dependencies already verified in this runtime.")
        return

    pip_cache = _get_pip_cache_args()
    c_arg = ["-c", str(constraints_path)] if constraints_path.exists() else []
    llm_pkgs = ["transformers", "accelerate", "bitsandbytes", "huggingface_hub", "tqdm"]
    print("   -> Ensuring Stage 2 LLM core packages (cached on 5TB Drive)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pip_cache + c_arg + llm_pkgs)

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
                ]
                + pip_cache
                + [
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
    if is_colab and mode == "test-cookies":
        print("\n[2/3] Cookie test mode — ensuring yt-dlp + Deno only...")
        _ensure_deno_runtime()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp>=2025.1.15", "curl_cffi"], check=False)
    elif is_colab and mode not in ("status", "sync-db"):
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

    try:
        from core.audio_diarizer import preload_cuda_libraries

        preload_cuda_libraries()
    except Exception:
        pass

    print(f"\n[3/3] Handing off to YT-SPX V2 Smart-Batch Orchestrator (args={argv or ['--mode', 'auto']})...\n")
    from core.pipeline_orchestrator import main as orchestrator_main

    return orchestrator_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
