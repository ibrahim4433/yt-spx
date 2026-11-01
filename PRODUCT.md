# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

General seekers, skeptics, and curious readers (both Arabic and English speakers) exploring clear, rational, and evidence-backed answers to theological, philosophical, and social doubts or objections about Islam—alongside students and researchers examining real multi-speaker street-debate dialogues at Hyde Park Speakers' Corner.

## Product Purpose

Dawah Dialectics (`yt-spx` V2) is a bilingual (Arabic/English) interactive knowledge base, full-scenario debate transcript archive, and zero-latency dialectical assistant built from multi-stage acoustic vocal isolation, neural speaker diarization, 2-speaker overlap separation, and local-LLM extraction across 2,640 Speakers' Corner videos. Success means a seeker or researcher can find any theological or philosophical objection, inspect the verified dual-speaker exchange (`Interlocutor` vs `Main Da'ee`), study its deductive premises, fallacy exposed, analogy, and Socratic counter-question, or open the **Interactive Scenario Studio** to read the full multi-speaker conversation synchronized with the original YouTube video—with overlapping cross-talk cleanly separated and random pedestrian interruptions isolated.

## Positioning

Unlike generic Q&A sites, raw auto-captioned transcripts, or hallucination-prone chatbots, every entry in Dawah Dialectics V2 is grounded in acoustic speaker diarization (`NeMo MSDD` + `TitaNet-Large` voiceprints), surgical overlap separation (`MossFormer2`), and CTC phonetic word timestamps:
- Every argument attributes **who spoke what** (`Main Da'ee` voiceprint vs `Interlocutor 1..K`),
- Preserves both sides of simultaneous cross-talk instead of mangling overlapping sentences,
- Filters out random street-pedestrian interruptions while keeping them inspectable on demand,
- Decomposes each exchange into core thesis, deductive premises, fallacy exposed, real-world analogy, Socratic counter-question, verbatim quotes from both speakers, and turn-level YouTube seek links.

## Operating Context

- Used across desktop and mobile web browsers in Arabic (RTL), English (LTR), and side-by-side Parallel (`EN ⇄ عربي`) mode.
- Powered by the V2 multi-day Google Colab Free T4 pipeline writing to a shared 5 TB Google Drive workspace (`/content/drive/MyDrive/yt-spx`) across 2,640 cataloged videos (`2,033` full debates + `607` `#shorts`). Because the database starts from a clean V2 slate (zero noisy V1 subtitle records) and grows batch-by-batch as `M5_SCENARIO_BUILT` and `M6_LLM_EXTRACTED` shards sync into `data/dawah_knowledge.db`, the UI must gracefully handle both a clean-slate / early-stage database and thousands of populated records.

## Capabilities and Constraints

- **Backend stack**: FastAPI (`web/app.py`) + SQLite Dual-FTS5 (`qa_fts` & `scenario_turns_fts` in `db/database.py`) + deterministic zero-LLM bilingual dialectical assistant (`core/smart_bot.py`).
- **Full V2 data schema**:
  - **Q&A Dossiers (`qa_units`)**: `id`, `video_id`, `episode_index`, `daee_speaker_id`, `daee_profile_id`, `daee_name_en`, `daee_name_ar`, `interlocutor_speaker_id`, `interlocutor_type`, `topic_category`, `topic_subcategory`, `question_en`/`_ar`, `core_answer_summary_en`/`_ar`, `logical_premises`/`_ar`, `fallacy_or_misconception_exposed`/`_ar`, `analogy_or_metaphor`/`_ar`, `socratic_question`/`_ar`, `interlocutor_verbatim_quote`/`_ar`, `verbatim_quote`/`_ar`, `dialogue_exchange`, `source_turn_ids`, `timestamp_seconds`, `answer_timestamp_seconds`, `timestamp_url`, `tags`.
  - **Scenario Transcripts (`videos`, `video_speakers`, `scenario_turns`)**: Full turn-by-turn diarized transcripts with `speaker_role` (`MAIN_DAEE`, `CO_DAEE`, `INTERLOCUTOR`, `PEDESTRIAN`), `is_overlap`, `overlap_group_id`, `is_pedestrian_interruption`, `episode_index`, and millisecond `start_ms`/`end_ms` timestamps.
- **Required frontend capabilities**:
  - Faceted filtering by **Topic Pillar**, **Interlocutor Archetype**, and **Da'ee Voiceprint Speaker** (`/api/categories`, `/api/interlocutors`, `/api/speakers`).
  - Claim-and-Proof Dossier Cards featuring Da'ee attribution, Dual-Speaker Exchange Timelines, bilingual comparison toggle, YouTube timestamp link, and direct **"Open Scenario Context"** jump into the full diarized conversation.
  - **Scenario Transcripts** tab & **Interactive Scenario Studio `<dialog>`** with synced YouTube IFrame playback, click-any-turn seeking, parallel split-track overlap rendering, episode jump tabs, and a toggle to hide/show street pedestrian interruptions.
  - Corpus & Pipeline Ledger view surfacing V2 acoustic/LLM pipeline telemetry (`stage1_completed`, `stage2_completed`, `cleaned_hours`, `overlaps_resolved`, `pedestrian_isolated`, `speakers_identified`).

## Product Principles

1. **Clarity over Polemics**: Present philosophical and theological arguments with calm intellectual rigor.
2. **Acoustic & Speaker Ground Truth**: Always distinguish who spoke (`Main Da'ee` vs `Interlocutor` vs `Pedestrian`), surface resolved simultaneous speech clearly, and anchor every claim to exact phonetic timestamps.
3. **Complete Dialectical Anatomy**: Distinguish the interlocutor's objection, deductive premises, exposed fallacy, analogy, Socratic turn, and dual-speaker dialogue timeline.
4. **First-Class Bilingualism (AR/EN)**: Equal typography, RTL/LTR layout symmetry, and side-by-side inspection.
5. **Honest Pipeline Transparency**: Report exact live counts from `dawah_knowledge.db`—whether 0 units on a fresh V2 initialization or thousands of units after Colab smart-batch syncs—with zero fabricated metrics.
