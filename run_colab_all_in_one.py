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

from __future__ import annotations
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
    # Cache model weights on local fast NVMe to keep Google Drive 100% clean of hash chunks
    local_torch_cache = Path("/root/.cache/torch") if os.path.exists("/root") else Path("/tmp/.torch_cache")
    local_torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(local_torch_cache))

    # Clean up any legacy v1 experimental model tar archive on Drive
    if ws.is_drive:
        legacy_models_tar = ws.root / ".cache" / "models" / "stage1_models.tar"
        if legacy_models_tar.exists():
            try:
                legacy_models_tar.unlink()
            except Exception:
                pass

    print(f"   -> Shared 5 TB Workspace : {ws.root} (Drive={ws.is_drive})")
    print(f"   -> Active Code Directory : {CURRENT_DIR}")
    try:
        from core.cookie_pool import CookiePool

        (ws.root / "cookies").mkdir(parents=True, exist_ok=True)
        ws.clear_youtube_circuit_breaker()
        pool = CookiePool(ws)
        pool.reset_quarantine()
        healthy = pool.healthy_cookies()
        total_discovered = len(pool.discover())
        print(f"   -> Ingestion strategy  : HLS 234/233 stream (anti-bot) (fallback cookies: {len(healthy)} healthy / {total_discovered} total)")
    except Exception as exc:
        print(f"   -> Ingestion strategy  : Anonymous first ({exc})")
    return True


STAGE1_MODEL_BUNDLE_NAME = "stage1_models_v2.tar"
STAGE1_MODEL_BUNDLE_MIN_BYTES = 1_500_000_000  # ~1.5 GB sanity threshold for full Stage 1 weights


def _get_local_stage1_cache_dirs() -> List[Tuple[str, Path]]:
    """Returns (arcname_prefix, local_dir_path) for Stage 1 model caches on fast local NVMe SSD."""
    cache_root = Path("/root/.cache") if Path("/root").exists() else (Path.home() / ".cache")
    return [
        ("huggingface/hub", cache_root / "huggingface" / "hub"),
        ("torch", cache_root / "torch"),
        ("nemo", cache_root / "nemo"),
    ]


def restore_stage1_models_from_drive(workspace: Optional[Any] = None) -> bool:
    """
    Restores Stage 1 model weights (Faster-Whisper large-v3, CTC Aligner, Punctuation,
    Demucs htdemucs, NeMo MSDD/TitaNet, MossFormer2) from the single-file uncompressed
    archive on Google Drive (.cache/models/stage1_models_v2.tar) to local NVMe SSD in ~35s.
    """
    import tarfile
    import time

    try:
        if workspace is None:
            from core.drive_sync import SharedWorkspace
            workspace = SharedWorkspace()
        if not workspace.is_drive:
            return False

        tar_path = workspace.root / ".cache" / "models" / STAGE1_MODEL_BUNDLE_NAME
        if not tar_path.exists() or tar_path.stat().st_size < STAGE1_MODEL_BUNDLE_MIN_BYTES:
            return False

        cache_root = Path("/root/.cache") if Path("/root").exists() else (Path.home() / ".cache")
        marker = cache_root / ".yt_spx_stage1_models_restored"
        if marker.exists():
            return True

        # Check if local cache already has Whisper large-v3 & torch hub populated
        hf_hub = cache_root / "huggingface" / "hub"
        if hf_hub.exists() and any(hf_hub.glob("*whisper*large-v3*")) and (cache_root / "torch").exists():
            marker.write_text("ok", encoding="utf-8")
            return True

        t0 = time.time()
        size_gb = round(tar_path.stat().st_size / (1024 ** 3), 2)
        print(f"   -> Restoring Stage 1 model bundle ({size_gb} GB) from Google Drive to fast local NVMe SSD...", flush=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(tar_path), mode="r:") as tf:
            tf.extractall(path=str(cache_root))
        marker.write_text(str(time.time()), encoding="utf-8")
        elapsed = round(time.time() - t0, 1)
        print(f"   -> [OK] Stage 1 models restored to local SSD in {elapsed}s (0 HuggingFace/NGC downloads needed)!", flush=True)
        return True
    except Exception as exc:
        print(f"   [-] Stage 1 model bundle restore note ({exc}); models will download from HuggingFace/NGC as needed.", flush=True)
        return False


def persist_stage1_models_to_drive(workspace: Optional[Any] = None) -> bool:
    """
    Packs Stage 1 local NVMe model caches (huggingface/hub, torch, nemo) into a single
    uncompressed tar archive on Google Drive (.cache/models/stage1_models_v2.tar),
    excluding any 14B GGUF files so Stage 1 bundle stays compact (~5-6 GB).
    Called automatically at the end of a Stage 1 batch if the bundle does not yet exist on Drive.
    """
    import tarfile
    import time

    try:
        if workspace is None:
            from core.drive_sync import SharedWorkspace
            workspace = SharedWorkspace()
        if not workspace.is_drive:
            return False

        models_dir = workspace.root / ".cache" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        tar_path = models_dir / STAGE1_MODEL_BUNDLE_NAME
        if tar_path.exists() and tar_path.stat().st_size >= STAGE1_MODEL_BUNDLE_MIN_BYTES:
            return True

        dirs_to_pack = [(arc, p) for arc, p in _get_local_stage1_cache_dirs() if p.exists()]
        if not dirs_to_pack:
            return False

        # Compute total eligible bytes first to ensure all Stage 1 models were actually downloaded
        total_bytes = 0
        for _, p in dirs_to_pack:
            for f in p.rglob("*"):
                if f.is_file() and not f.name.endswith(".gguf") and not f.name.endswith(".lock"):
                    try:
                        total_bytes += f.stat().st_size
                    except OSError:
                        pass

        if total_bytes < STAGE1_MODEL_BUNDLE_MIN_BYTES:
            return False

        if not workspace.acquire_lock("_stage1_models_tar", ttl_sec=900, verify_delay_sec=0.2):
            return False

        try:
            if tar_path.exists() and tar_path.stat().st_size >= STAGE1_MODEL_BUNDLE_MIN_BYTES:
                return True

            t0 = time.time()
            size_gb = round(total_bytes / (1024 ** 3), 2)
            print(f"\n[Stage 1 Cache] Saving Stage 1 model bundle ({size_gb} GB) to Google Drive ({tar_path.name}) for ~35s future startups...", flush=True)
            tmp_local_tar = Path("/tmp") / f".{STAGE1_MODEL_BUNDLE_NAME}.tmp"
            tmp_local_tar.unlink(missing_ok=True)

            def _exclude_gguf_and_locks(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
                if tarinfo.name.endswith(".gguf") or tarinfo.name.endswith(".lock"):
                    return None
                return tarinfo

            with tarfile.open(str(tmp_local_tar), mode="w:") as tf:
                for arc_prefix, local_dir in dirs_to_pack:
                    tf.add(str(local_dir), arcname=arc_prefix, filter=_exclude_gguf_and_locks)

            from core.colab_runner import _copy_large_file_chunked
            _copy_large_file_chunked(tmp_local_tar, tar_path, label="Stage 1 Models -> Drive")
            tmp_local_tar.unlink(missing_ok=True)
            elapsed = round(time.time() - t0, 1)
            print(f"[Stage 1 Cache] [OK] Cached {tar_path.name} ({size_gb} GB) on Drive in {elapsed}s! Future Colab sessions will restore it in ~35s.", flush=True)
            return True
        finally:
            workspace.release_lock("_stage1_models_tar")
    except Exception as exc:
        print(f"[Stage 1 Cache] Note while caching Stage 1 model bundle: {exc}", flush=True)
        return False


def _ensure_deno_runtime() -> None:
    """Installs Deno so yt-dlp solves YouTube challenges cleanly."""
    for d in [Path.home() / ".deno" / "bin", Path("/root/.deno/bin")]:
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{d}:{os.environ.get('PATH', '')}"
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
    for d in [Path.home() / ".deno" / "bin", Path("/root/.deno/bin")]:
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{d}:{os.environ.get('PATH', '')}"


def _get_pip_cache_args() -> List[str]:
    """
    Uses local fast NVMe disk for pip cache (/tmp/.pip_cache) so downloads & builds
    run at native datacenter speed (~100 MB/s) instead of throttling over Google Drive FUSE.
    """
    local_cache = Path("/tmp/.pip_cache")
    local_cache.mkdir(parents=True, exist_ok=True)
    return [
        f"--cache-dir={local_cache}",
        "--prefer-binary",
        "--no-warn-conflicts",
    ]


def _get_uv_binary() -> Optional[str]:
    """Locates the standalone uv executable on disk."""
    found = shutil.which("uv")
    if found:
        return found
    candidates = [
        Path(sys.prefix) / "bin" / "uv",
        Path(sys.prefix) / "Scripts" / "uv.exe",
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
        Path("/usr/bin/uv"),
    ]
    for c in candidates:
        if c.exists() and (os.access(c, os.X_OK) or sys.platform == "win32"):
            return str(c)
    return None


def _ensure_uv_runtime() -> Optional[str]:
    """Ensures Astral's uv standalone executable is available for fast parallel installs."""
    uv_bin = _get_uv_binary()
    if uv_bin:
        return uv_bin
    try:
        pip_cache = _get_pip_cache_args()
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + pip_cache + ["uv"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        uv_bin = _get_uv_binary()
        if uv_bin:
            return uv_bin
        subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _get_uv_binary()
    except Exception:
        return None


def _get_drive_wheels_dir() -> Optional[Path]:
    """Returns persistent .cache/wheels directory on Google Drive if mounted."""
    try:
        from core.drive_sync import SharedWorkspace
        ws = SharedWorkspace()
        if ws.is_drive:
            w_dir = ws.root / ".cache" / "wheels"
            w_dir.mkdir(parents=True, exist_ok=True)
            return w_dir
    except Exception:
        pass
    return None


def _fast_install(
    pkgs: List[str],
    constraints_path: Optional[Path] = None,
    no_deps: bool = False,
    extra_args: Optional[List[str]] = None,
) -> None:
    """
    Installs packages with Astral's uv (10x-50x faster parallel downloads & resolution).
    Falls back gracefully to standard pip if uv encounters any legacy metadata errors.
    """
    c_arg = ["-c", str(constraints_path)] if constraints_path and constraints_path.exists() else []
    w_dir = _get_drive_wheels_dir()
    w_args = [f"--find-links={w_dir}"] if w_dir and w_dir.exists() else []
    extra = list(extra_args or [])

    # Strategy 1: Attempt Astral's uv for ultra-fast parallel installation (using direct uv executable binary)
    uv_bin = _ensure_uv_runtime()
    if uv_bin:
        uv_cmd = [uv_bin, "pip", "install", "--system", "-q"]
        if no_deps:
            uv_cmd.append("--no-deps")
        uv_cmd.extend(c_arg + w_args + extra + pkgs)
        res = subprocess.run(
            uv_cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res.returncode == 0:
            return

    # Strategy 2: Robust standard pip fallback
    pip_cache = _get_pip_cache_args()
    pip_cmd = [sys.executable, "-m", "pip", "install", "-q"] + pip_cache
    if no_deps:
        pip_cmd.append("--no-deps")
    pip_cmd.extend(c_arg + w_args + extra + pkgs)
    subprocess.check_call(pip_cmd)


def _cache_wheel_on_drive(pkg_name: str, constraints_path: Optional[Path] = None) -> None:
    """Builds and caches a package wheel to Drive once so future runs install in 0.1s without compilation."""
    w_dir = _get_drive_wheels_dir()
    if not w_dir:
        return
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    pkg_clean = pkg_name.split("[")[0].replace("-", "_")
    existing = list(w_dir.glob(f"{pkg_clean}*{py_tag}*.whl"))
    if existing:
        return
    try:
        c_arg = ["-c", str(constraints_path)] if constraints_path and constraints_path.exists() else []
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(w_dir)] + c_arg + [pkg_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


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
        "assert all(u.find_spec(m) is not None for m in ('faster_whisper', 'demucs', 'ctc_forced_aligner', 'yt_dlp', 'yt_dlp_ejs', 'pytubefix'))"
    )
    res = subprocess.run(
        [sys.executable, "-c", check_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return res.returncode == 0


def _purge_conflicting_tensorflow() -> None:
    """
    Purges Colab's pre-installed TensorFlow (built with Protobuf 6.31).
    yt-spx exclusively uses PyTorch, CTranslate2, and GGUF/llama.cpp; removing
    broken TensorFlow prevents any Protobuf VersionError.
    """
    try:
        import importlib.util as u
        if u.find_spec("tensorflow") is not None:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "tensorflow", "tensorflow-probability"],
                check=False,
            )
    except Exception:
        pass


def ensure_stage1_dependencies(constraints_path: Path) -> None:
    # Ensure conflicting standalone nvidia-cudnn-cu12 is removed so PyTorch uses its bundled cuDNN (9.19.0)
    try:
        import importlib.util as u
        if u.find_spec("nvidia.cudnn") is not None:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "nvidia-cudnn-cu12"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass

    _purge_conflicting_tensorflow()

    if _verify_stage1_runtime_health():
        print("   -> Stage 1 audio/diarization dependencies already verified in this runtime.")
        _ensure_deno_runtime()
        restore_stage1_models_from_drive()
        return

    _ensure_deno_runtime()

    if shutil.which("ffmpeg") is None and shutil.which("apt-get"):
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)

    pip_cache = _get_pip_cache_args()
    c_arg = ["-c", str(constraints_path)] if constraints_path.exists() else []
    stage1_pkgs = [
        "yt-dlp[default]>=2025.1.15",
        "yt-dlp-ejs>=0.8.0",
        "pytubefix",
        "curl_cffi",
        "demucs",
        "faster-whisper>=1.1.0",
        "ctc-forced-aligner",
        "unidecode",
        "deepmultilingualpunctuation",
        "soundfile",
        "huggingface_hub",
        "tqdm",
        "ml-dtypes>=0.5.4",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        # NOTE: nvidia-cudnn-cu12 is deliberately omitted — PyTorch bundles its own matched cuDNN
    ]
    print("   -> Ensuring Stage 1 ASR, Demucs & Forced-Alignment packages (cached on fast local SSD)...")
    _fast_install(stage1_pkgs, constraints_path)

    # Pre-build & cache compiled C++ wheels to Drive so future Colab VMs install in 0.1s
    _cache_wheel_on_drive("ctc-forced-aligner", constraints_path)

    if not _nemo_2x_installed():
        print("   -> Installing NVIDIA NeMo 2.x ASR Toolkit (MSDD NeuralDiarizer + TitaNet-Large)...")
        _fast_install(
            ["cython", "nemo_toolkit[asr]>=2.6.0,<3.0.0", "ml-dtypes>=0.5.4"],
            constraints_path,
        )
        _cache_wheel_on_drive("antlr4-python3-runtime", constraints_path)
        _cache_wheel_on_drive("wget", constraints_path)

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
        _fast_install(clearvoice_deps, constraints_path)
        _fast_install(["clearvoice"], constraints_path, no_deps=True)

    # Final ABI guard: guarantee ml_dtypes>=0.5.4 (float4_e2m1fn required by onnx>=1.20 on Python 3.13)
    abi_check = subprocess.run(
        [sys.executable, "-c", "import ml_dtypes; assert hasattr(ml_dtypes, 'float4_e2m1fn')"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if abi_check.returncode != 0:
        _fast_install(["ml-dtypes>=0.5.4"], constraints_path, extra_args=["-U"])

    restore_stage1_models_from_drive()


def _verify_stage2_runtime_health() -> bool:
    try:
        from core.colab_runner import ensure_cuda_runtime_loaded
        ensure_cuda_runtime_loaded()
        import llama_cpp
        if not llama_cpp.llama_supports_gpu_offload():
            return False
        import transformers, accelerate, bitsandbytes
        return True
    except Exception:
        return False


def ensure_stage2_dependencies(constraints_path: Path) -> None:
    _purge_conflicting_tensorflow()

    if _verify_stage2_runtime_health():
        print("   -> Stage 2 Local LLM dependencies already verified in this runtime.")
        return

    llm_pkgs = [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "tqdm",
        "nvidia-cuda-runtime-cu12",
    ]
    print("   -> Ensuring Stage 2 LLM core packages (cached on fast local SSD)...")
    _fast_install(llm_pkgs, constraints_path)

    def _test_llama_cpp() -> bool:
        try:
            from core.colab_runner import ensure_cuda_runtime_loaded
            ensure_cuda_runtime_loaded()
            import llama_cpp
            return bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            return False

    if not _test_llama_cpp():
        print("   -> Installing pre-built CUDA wheel for llama-cpp-python (Qwen 3 / Qwen 2.5 14B Q5_K_M GGUF)...")
        w_dir = _get_drive_wheels_dir()
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if w_dir:
            cached = list(w_dir.glob(f"llama_cpp_python*{py_tag}*.whl"))
            if cached:
                print(f"   -> Found cached llama-cpp-python GPU wheel on Drive ({cached[0].name})...")
                _fast_install([str(cached[0])], no_deps=True)
                if _test_llama_cpp():
                    return
        for cu_ver in ["cu124", "cu122", "cu121"]:
            pip_cache = _get_pip_cache_args()
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--force-reinstall",
                    "--no-deps",
                ]
                + pip_cache
                + [
                    "llama-cpp-python",
                    "--only-binary=:all:",
                    f"--extra-index-url=https://abetlen.github.io/llama-cpp-python/whl/{cu_ver}",
                ],
                check=False,
            )
            if res.returncode == 0 and _test_llama_cpp():
                print(f"   -> Installed & verified llama-cpp-python ({cu_ver}) GPU wheel!")
                if w_dir:
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "wheel",
                                "-q",
                                "--no-deps",
                                "-w",
                                str(w_dir),
                                f"--extra-index-url=https://abetlen.github.io/llama-cpp-python/whl/{cu_ver}",
                                "llama-cpp-python",
                            ],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        pass
                break


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
    elif is_colab and mode == "auth-youtube":
        print("\n[2/3] Preparing YouTube Smart-TV OAuth2 environment...")
        _ensure_deno_runtime()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp>=2025.1.15", "yt-dlp-youtube-oauth2", "curl_cffi"], check=False)
        try:
            from core.yt_audio_fetcher import patch_youtube_oauth2_plugin
            patch_youtube_oauth2_plugin()
        except Exception:
            pass
    elif is_colab and mode not in ("status", "sync-db", "reset-stage1", "reset-stage2", "auth-youtube"):
        print(f"\n[2/3] Verifying GPU packages for mode='{mode}'...")
        if mode in ("auto", "stage1", "stage1-worker"):
            ensure_stage1_dependencies(constraints_path)
        if mode in ("stage2", "stage2-worker"):
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
