# Phase C — Run Golden-Result Re-Extraction on Colab (Runbook)

**When to run**: To regenerate the affected cards (`0Y7cFAUlil4, 1ZO0h38zpi4, 9XRgpFDU61Q, RE2kdyx_dmg, hiixMRVFlfk, jt4BisLcSoI, mV_oFziglRg, owPIQBEjtK8, y8vhbbKwwQU`) with the hardened Stage-2 pipeline and produce reviewable `.qa.json` shards.

## Pre-flight (do on Colab BEFORE running)

1. **Download/copy the hardened code** to the Colab workspace `/content/yt-spx` (the "Active Code Directory").
   Files that changed during this golden-fix program:
   - `core/scenario_extractor.py` (trajectory integrity, quote extension, functional-collapse hard-reject)
   - `db/database.py` (interlocator_speaker_id hydration)
2. On Colab run `python tools/scenario_backfill.py --ws /content/drive/MyDrive/yt-spx` — but wait: the real scenario JSONs for these videos ALREADY exist on the Colab drive from the previous pipeline run (`/content/drive/MyDrive/yt-spx/scenarios/{vid}.json`), so no backfill is needed for Colab. Skip backfill unless you want the GT-reconstructed scenario files (then run it, it writes to the drive).
3. Check GPU: `nvidia-smi` → Tesla T4 available.

## Run (on Colab)

```
cd /content/yt-spx
python tools/rerun_stage2_for_cards.py --ws /content/drive/MyDrive/yt-spx --force
```

Notes:
- `--force` deletes the old `extractions/{vid}.qa.json` shards first so Stage-2 re-runs (otherwise `extract_video_m6` early-returns the cached shard).
- `hiixMRVFlfk` — the hardened pipeline now **skips** it (functional diarization collapse, 0.8% inter words). Expect `[PRE-STAGE2 SKIP ...]` and 0 units — that's the desired golden behavior.
- `extract_video_m6` writes new shards to `/content/drive/MyDrive/yt-spx/extractions/{vid}.qa.json` (on Colab; or use `--ws` pointing to a sandbox for testing).
- Review copies are written to `reports/regen/{vid}.qa.json` on the local machine when you sync back.

## Post-run: Review & Ingest (on the local machine)

1. `ls reports/regen/` — new golden shards for the 9 videos (some may be 0 units — e.g., hiixMRVFlfk).
2. Review the new cards — the automated/manual audits can re-run:
   - `python tools/card_rating_system/audit_cards.py --out reports/audit_regen.json`
3. When approved, ingest (writes to the DB — on the drive or local):
   ```
   python -c "
   import json, pathlib
   from db.database import ingest_extraction_payload
   for p in pathlib.Path('reports/regen').glob('*.qa.json'):
       payload = json.loads(p.read_text(encoding='utf-8'))
       ingest_extraction_payload(payload['video_id'], payload['units'])
   "
   ```
4. Re-audit: `python tools/card_rating_system/audit_cards.py --out reports/audit_final.json`.

## Why Phase C matters

Phase C = re-extract the current cards with the hardened pipeline so the CURRENT data cards match the new golden-result guarantees (trajectory repair, GT text authority, interlocator_speaker_id hydration). Without it, the pipeline changes apply to FUTURE extractions; current cards stay as they are.

**Skip Phase C now?** Yes, you can skip it — the hardening is already live for future runs. Run it when you want to regenerate the current cards to golden.
