# Dawah Dialectics V2 (`yt-spx`)

**Multi-Stage Acoustic Diarization, Overlap Separation, Pedestrian Isolation & Local LLM Dialectical Extraction Pipeline**

> **Archive Note**: The legacy subtitle-based V1 repository remains preserved untouched at [`ibrahim4433/yt-ch`](https://github.com/ibrahim4433/yt-ch) (`/content/drive/MyDrive/yt-ch`). This repository (`yt-spx`) implements the ground-truth V2 multi-stage audio & speaker-diarization architecture using `/content/drive/MyDrive/yt-spx`.

---

## 7-Step V2 Architecture (100% Free on Google Colab T4 16 GB VRAM)

1. **`M1_DOWNLOADED` (`core/yt_audio_fetcher.py`)**: Anti-ban `yt-dlp` best-audio staging (`251` Opus / `140` M4A + `.info.json`) across all **2,640** cataloged videos (`all_channels_videos.csv` + `clean_urls.txt`), with isolated `/tmp` `cookies.txt` copies, Deno JS challenge runtime, and a shared cross-account circuit breaker.
2. **`M2_VOCALS_CLEANED` (`core/vocal_cleaner.py`)**: 15-minute chunk-checkpointed `htdemucs` (`--two-stems=vocals`) with 2s linear crossfades, pre-normalization 250ms RMS dBFS metering, `highpass=80Hz + dynaudnorm`, and 64 kbps 24kHz VoIP `.opus` archival on the 5 TB Google Drive.
3. **`M3_DIARIZED` (`core/overlap_resolver.py`)**: Overlap-aware NVIDIA NeMo multi-scale neural diarizer (`diar_msdd_telephonic` + `MarbleNet` + `TitaNet-Large` 192-D speaker embeddings) producing exact multi-speaker `.rttm` timelines and non-overlap voiceprint centroids.
4. **`M4_OVERLAPS_SEPARATED` (`core/overlap_resolver.py`)**: Surgical 2-speaker unmixing via `ClearVoice MossFormer2_SS_16K` on simultaneous cross-talk windows ($\ge 0.8\text{s}$), matched back to global speakers via `TitaNet-Large` cosine similarity.
5. **`M5_SCENARIO_BUILT` (`core/audio_diarizer.py` & `core/speaker_roles.py`)**: `faster-whisper large-v3` (`float16`) + theological vocabulary prompt + `ctc-forced-aligner` (`mms-300m-1130`) word-level boundaries + 3-tier speaker classification (`MAIN_DAEE`, `INTERLOCUTOR 1..K`, `PEDESTRIAN`) + cross-video Da'ee voiceprint registry + episode segmentation $\rightarrow$ `scenarios/{video_id}.json` & `.diarized.srt`.
6. **`M6_LLM_EXTRACTED` (`core/scenario_extractor.py` & `core/colab_runner.py`)**: Episode-checkpointed local `Qwen 2.5 14B Instruct` (`Q5_K_M` GGUF, `12,288` ctx, 8-bit KV cache) extracting structured bilingual Claim-and-Proof dossiers anchored to verified turn IDs $\rightarrow$ `extractions/{video_id}.qa.json` $\rightarrow$ compacted into `data/dawah_knowledge.db`.
7. **Interactive Scenario & Dossier Web Portal (`web/`)**: FastAPI + Dual-FTS5 SQLite (`qa_fts` & `scenario_turns_fts`) + bilingual Editorial Monograph web UI featuring Claim-and-Proof Dossiers, Dual-Speaker Exchange Timelines, Da'ee Voiceprint Filters, and the Interactive Scenario Transcript Studio with synced YouTube playback.

---

## Repository Structure

```text
yt-spx/
├── all_channels_videos.csv                # Source catalog metadata (2,640 videos)
├── clean_urls.txt                         # Canonical YouTube URL list
├── constraints_colab.txt                  # ABI-pinned Colab scientific constraints
├── run_colab_all_in_one.py                # 1-click Google Colab T4 bootstrapper
├── dawah_dialectics_v2_colab.ipynb  # Multi-account dynamic Colab T4 notebook (1-8 hrs variable)
├── download_videos.py                     # Anti-ban M1 audio staging CLI
├── run_app.py                             # Local launcher for the FastAPI web portal
│
├── core/
│   ├── drive_sync.py                      # 5 TB Drive workspace, M1->M6 state, locks & compactor
│   ├── metadata_linker.py                 # Canonical 2,640-video V2 manifest & Da'ee title hints
│   ├── yt_audio_fetcher.py                # Step 1: Anti-ban yt-dlp bestaudio + circuit breaker
│   ├── vocal_cleaner.py                   # Step 2: Chunked Demucs vocals + RMS + 64kbps Opus
│   ├── overlap_resolver.py                # Steps 3-4: NeMo MSDD + MossFormer2 + IoU alignment
│   ├── speaker_roles.py                   # Step 4: 3-tier roles, TitaNet registry & episodes
│   ├── audio_diarizer.py                  # Steps 3-5: Faster-Whisper + CTC scenario builder
│   ├── extractor_prompt.py                # Step 6: Diarized turn-anchored LLM prompts
│   ├── scenario_extractor.py              # Step 6: Episode-checkpointed Qwen 2.5 14B extractor
│   ├── colab_runner.py                    # Local T4 GPU GGUF / NF4 LLM engine
│   ├── pipeline_orchestrator.py           # Subprocess-isolated smart-batch coordinator
│   └── smart_bot.py                       # Zero-LLM bilingual dialectical assistant
│
├── db/
│   └── database.py                        # V2 SQLite + Dual-FTS5 (qa_fts & scenario_turns_fts)
│
├── scripts/
│   └── reset_for_v2_pipeline.py           # Clean-slate V2 database & manifest initializer
│
├── web/
│   ├── app.py                             # FastAPI REST endpoints (Dossiers, Scenarios, Search, Chat)
│   ├── templates/index.html               # Bilingual Editorial Monograph UI + Scenario Studio
│   └── static/
│       ├── css/style.css                  # Design tokens, RTL/LTR layout & Scenario Studio styles
│       └── js/app.js                      # Faceted search, synced YouTube player & Studio controller
│
└── tests/
    ├── test_endpoints.py                  # FastAPI V2 endpoint & FTS5 integration tests
    └── test_v2_pipeline.py                # Unit tests for M0->M6, overlap math, roles & compactor
```

---

## Quick Start

### 1. Run the Bilingual Web Portal Locally
```bash
python run_app.py
```
Then open `http://127.0.0.1:8000` (sync `dawah_knowledge.db` from `/content/drive/MyDrive/yt-spx/data/dawah_knowledge.db` as Colab batches complete).

### 2. Run Smart-Batch Extraction on Free Google Colab T4 (2–4 hrs/day)
Open `dawah_dialectics_v2_colab.ipynb` in Google Colab (T4 GPU runtime) and run **Cell 1** (`--mode auto --batch-videos 6`). It runs smoothly whether Colab grants 1 hour or 8 hours, saving micro-steps continuously. Multiple Colab accounts can share `/content/drive/MyDrive/yt-spx` via *"Add shortcut to My Drive"* and process safely in parallel.
