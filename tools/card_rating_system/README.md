# Universal Apologetics Card Rating System & Quality Assurance Suite (v3.0)

> **Directory**: `tools/card_rating_system/`  
> **Project**: Dawah Dialectics (`yt-spx`)  
> **Target Database**: `dawah_knowledge.db` (`qa_units`, `scenario_turns`, `video_speakers`, `videos`)

---

## Overview

This suite provides a **re-usable, strict, agentic and automated rating system** to evaluate any batch of dialectical apologetics cards extracted from Speakers' Corner street debates.

Whenever you update extraction prompts, Stage 1 diarization, Stage 2 pipelines, or sync a new batch sample of `dawah_knowledge.db` from Google Drive, this system enables AI Agent Swarms and the CLI engine to verify:
1. Are the cards grounded 100% in acoustic and spoken ground truth without single-speaker diarization collapse (`speakers_count == 1`)?
2. Did the model leak prompt demonstration examples or hallucinate verse citations?
3. Did Round 2 Classical Arabic localization complete with full bilingual symmetry across all 8 Arabic fields, `dialogue_exchange`, and `scriptural_citations`?
4. Are the questions formulated properly from the seeker/skeptic perspective with unambiguous theological targets (never Da'ee cross-examinations, Da'ee defensive comebacks, or 1st/2nd-person street phrasing)?
5. Are `interlocutor_verbatim_quote` and `verbatim_quote` accurately separated and matched to their true speakers (`INTERLOCUTOR` vs `MAIN_DAEE`)?
6. Did the pipeline filter out police encounters, street disruptions, personal chit-chat, and video outro promotional pitches?

---

## Suite Contents

```
tools/card_rating_system/
├── RUBRIC.md            # Version 3.0.0: 5-dimension, 100-point quality rubric & 6 hard-fail gates
├── AGENT_PROMPT.md      # Manual AI Agent / Swarm rating workflow, system prompt & JSON output schema
├── audit_cards.py       # Automated Python CLI audit engine & pre-Round-2 batch diagnostic helper
└── README.md            # This documentation guide
```

---

## 1. Quick Start: Running the Automated Audit CLI

You can pre-screen the entire database or any subset with a single command:

### Audit All Cards in the Database
```bash
python tools/card_rating_system/audit_cards.py
```

### Save Detailed JSON Audit Report
```bash
python tools/card_rating_system/audit_cards.py --out reports/audit_latest.json
```

### Inspect Only Failing Cards with Verbose Defects
```bash
python tools/card_rating_system/audit_cards.py --fail-only --verbose
```

### Audit a Specific YouTube Video ID
```bash
python tools/card_rating_system/audit_cards.py --video-id KRDyUeTz_s0 --verbose
```

### Point to a Custom or Backup Database
```bash
python tools/card_rating_system/audit_cards.py --db /path/to/custom_dawah_knowledge.db
```

---

## 2. The 5 Core Rubric Dimensions (100 Pts Total)

```
┌────────────────────────────────────────────────────────────────────────┐
│             DAWAH DIALECTICS V3 EVALUATION RUBRIC (100 PTS)             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Question Formulation, Polarity & Standalone Clarity [20 Points]     │
│ 2. Grounding, Verbatim Fidelity & Speaker Separation   [25 Points]     │
│ 3. Dialectical Depth, Syllogisms & Truth Value         [20 Points]     │
│ 4. Bilingual Symmetry, Arabic Eloquence & Hygiene      [20 Points]     │
│ 5. Attribution, Timestamps & Website Presentation      [15 Points]     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Six Hard-Fail Gates (Score = 0/100)

Any card exhibiting even one of the following is an automatic **FAIL (0/100 — Tier 4 Red)**:
- `[GATE-LEAK]`: Copies prompt few-shot demonstration text (*train carriages*, *Mark 13:32*, *soldier orders*, *"whatever begins to exist"*).
- `[GATE-ARABIC]`: Missing or empty required Arabic fields (Round 2 skipped, failed, or partially localized).
- `[GATE-PERSPECTIVE]`: 1st-person confessional inversion, 1st/2nd-person street phrasing (*"How can I know..."*, *"Why can't I believe..."*), Da'ee defensive comebacks (*"...what is the problem with that?"*), or Da'ee offensive cross-examination placed in `question_en`.
- `[GATE-NOISE]`: Street drama, police scuffles, dog petting, fundraising outros, or non-apologetic personal chit-chat (*"What converted you to Islam?"*).
- `[GATE-ROLE]`: Inverted speaker roles, `daee_speaker_id == interlocutor_speaker_id`, `answer_turn_id` on `INTERLOCUTOR`, or swapped `verbatim_quote` / `interlocutor_verbatim_quote`.
- `[GATE-DIARIZATION]`: Single-speaker diarization collapse (`speakers_count == 1` on a multi-speaker debate >180s), fusing speakers and leaving `interlocutor_speaker_id` empty.

---

## 4. Re-usable AI Agent Manual Swarm Workflow

Automated Python checks cannot replace deep semantic reading of street debates. For comprehensive manual rating:
1. Read `tools/card_rating_system/RUBRIC.md` and `tools/card_rating_system/AGENT_PROMPT.md`.
2. Provide each Swarm Worker with a batch of cards alongside their `videos` metadata, `video_speakers` roster, sibling cards from the same video, and surrounding `scenario_turns` transcript window.
3. Have workers manually read and score every card one by one, recording `transcript_ground_truth_notes`, `defects`, `root_causes`, `card_level_rewrite_recommendation`, and `pipeline_fix_recommendation` in the `AGENT_PROMPT.md` JSON schema.
4. Have a verifier spot-check the worker evaluations against the source database turns before synthesizing the master root-cause fix plan.
