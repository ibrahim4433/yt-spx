# Universal Apologetics Card Rating System & Quality Assurance Suite

> **Directory**: `tools/card_rating_system/`  
> **Project**: Dawah Dialectics (`yt-spx`)  
> **Target Database**: `dawah_knowledge.db` (`qa_units`, `scenario_turns`, `videos`)

---

## Overview

This suite provides a **re-usable, strict, automated and agentic rating system** to evaluate any batch of dialectical apologetics cards extracted from Speakers' Corner street debates.

Whenever you update extraction prompts, modify Stage 2 pipelines, or sync a new batch sample of `dawah_knowledge.db` from Google Drive, this system enables you to immediately verify:
1. Are the cards grounded 100% in acoustic and spoken ground truth?
2. Did the model leak or hallucinate prompt demonstration examples?
3. Did Round 2 classical Arabic localization complete with full bilingual symmetry?
4. Are the questions formulated properly from the seeker/skeptic perspective with interrogative syntax?
5. Did the pipeline filter out police encounters, street disruptions, and video outro promotional pitches?

---

## Suite Contents

```
tools/card_rating_system/
├── RUBRIC.md            # The 5-dimension, 100-point universal quality evaluation rubric & hard-fail gates
├── AGENT_PROMPT.md      # Ready-to-use prompt template for AI Agents & Swarm Workers to evaluate cards
├── audit_cards.py       # Standalone automated Python CLI audit engine
└── README.md            # This documentation guide
```

---

## 1. Quick Start: Running the Automated Audit CLI

You can evaluate the entire database or any subset with a single command:

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
│             DAWAH DIALECTICS V2 EVALUATION RUBRIC (100 PTS)             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Question Formulation & Perspective                [15 Points]       │
│ 2. Grounding, Verbatim Fidelity & Zero-Leakage       [25 Points]       │
│ 3. Dialectical Depth, Syllogisms & Truth Value       [25 Points]       │
│ 4. Bilingual Symmetry, Arabic Eloquence & Hygiene    [20 Points]       │
│ 5. Attribution, Timestamps & Studio Integration      [15 Points]       │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Question Formulation & Perspective (15 pts)**: Complete interrogative syntax ending with `?`/`؟`. Must represent the seeker/skeptic objection, NOT a 1st-person confession (*"Why do I oppose Islam..."*).
2. **Grounding, Verbatim Fidelity & Zero-Leakage (25 pts)**: Zero prompt demonstration leakage. Verbatim quotes must match `scenario_turns` text.
3. **Dialectical Depth & Syllogism (25 pts)**: Direct convincing truth (no 3rd-person narrator summaries). Valid 3-part deductive syllogism (Major Premise, Minor Premise, Conclusion). Clear fallacy and concrete analogy.
4. **Bilingual Symmetry & Arabic Eloquence (20 pts)**: All 8 Arabic fields fully populated (*فصحى أصيلة*). Zero Russian Cyrillic, Chinese CJK, or untranslated Latin words.
5. **Attribution & Timestamps (15 pts)**: Muslim speaker attributed as Da'ee, interlocutor correctly categorized, accurate seek timestamps.

---

## 3. Automated Hard-Fail Gates (Score = 0/100)

Any card that exhibits even one of the following is an automatic **FAIL (0/100)**:
- `[GATE-LEAK]`: Copies prompt few-shot demonstration text (e.g. *train carriages*, *Mark 13:32*, *soldier orders*, *"whatever begins to exist"*).
- `[GATE-ARABIC]`: Missing or empty Arabic fields (Round 2 skipped or failed).
- `[GATE-PERSPECTIVE]`: 1st-person confessional inversion (*"Why do I hold anti-Islamic views..."*).
- `[GATE-NOISE]`: Extracted police scuffles, dog petting, street security disputes, or video outro commercial pitches.
- `[GATE-ROLE]`: Inverted speaker roles or dialogue collapsed into a single speaker monologue.

---

## 4. Re-usable AI Agent Swarm Workflow

When you want an AI Agent (or Swarm) to perform deep manual/semantic review on new cards:
1. Provide the agent with `tools/card_rating_system/RUBRIC.md`.
2. Use the system and task prompts from `tools/card_rating_system/AGENT_PROMPT.md`.
3. Pass each card alongside its corresponding `scenario_turns` context window.
4. Collect the agent's evaluation in the uniform JSON schema.
