# AI Agent Manual Audit Instructions & Rating Workflow (v3.2)

> **Purpose**: Re-usable step-by-step instructions, system prompt, and output schema enabling any AI Agent or Swarm Rater to manually evaluate `qa_units` apologetics cards one by one against ground-truth `scenario_turns` and `video_speakers` metadata, diagnose pipeline root causes, and produce actionable recommendations.

---

## Mandatory Manual Audit Workflow (Do NOT Automate Blindly)

When auditing a batch of cards, execute these 4 steps **manually for every single card**:

1. **Inspect Video & Speaker Context First**:
   - Read the video metadata (`title_ar`, `duration_sec`, `speakers_count`, `turns_count`) and `video_speakers` roster (`speaker_id`, `role`, `display_name_en`, `speaking_sec`, `turn_count`).
   - Check whether Stage 1 diarization actually separated speakers (`speakers_count >= 2`) or collapsed a multi-speaker debate into a single speaker (`speakers_count == 1`, `SPK_00` only).
   - Check all sibling cards extracted from the same `video_id` to catch intra-video duplication.

2. **Read the Ground-Truth Scenario Turns Window**:
   - Read the chronological `scenario_turns` around `timestamp_seconds` and `answer_timestamp_seconds`, plus all turns referenced by `source_turn_ids` and `dialogue_exchange`.
   - Identify who actually raised the claim/objection and who answered it on camera, watching carefully for:
     - **Single-speaker diarization collapse** (two speakers fused inside ~30s `SPK_00` chunks),
     - **Socratic role reversal** (the Muslim Da'ee attacking Christian/Atheist scripture, mistakenly reframed as `question_en`),
     - **Quote swapping** (Da'ee's argument placed in `interlocutor_verbatim_quote` or opponent's objection placed in `verbatim_quote`).

3. **Audit Every Field Against `RUBRIC.md` v3.2 (100 Pts + 7 Hard-Fail Gates, `PASS >= 75`)**:
   - Evaluate Dimensions 1–5 (`20 + 25 + 20 + 20 + 15 = 100 pts`). Score `raw_unclamped_estimate` strictly on the rubric point deductions BEFORE clamping `total_score` to `0` on hard-fail gates. Tier 3 (`50–74`) is **`FAIL` (`DO_NOT_PUBLISH / NEEDS_REGEN`)**; only Tier 1 (`90–100`) and Tier 2 (`75–89`) earn `status = "PASS"`.
   - Check all 7 Hard-Fail Gates: `[GATE-LEAK]`, `[GATE-ARABIC]`, `[GATE-PERSPECTIVE]`, `[GATE-ROLE]`, `[GATE-NOISE]`, `[GATE-DIARIZATION]`, `[GATE-UNGROUNDED]`.
   - Never rely solely on automated keyword heuristics: use semantic reading to judge whether the question is genuinely what a seeker searches, whether the theological target is unambiguous (e.g. explicitly naming the Bible/Old Testament when refuting 1 Samuel 15:3), whether the syllogism is valid and faithful to the transcript, whether `dialogue_exchange[i].text_en` truly matches its cited `turn_id` in chronological order, and whether the Arabic is complete, eloquent Classical Arabic (*فصحى أصيلة*).

4. **Record Specific Root Causes & Concrete Card + Pipeline Recommendations**:
   - State clearly **why** the card is good or bad with exact quoted evidence from both the card and `scenario_turns`.
   - Attribute every defect to its exact pipeline stage (`Stage 1 M3 Diarization`, `Stage 1 M5 Turn Assembly`, `Stage 2 R1 Prompt/Extraction`, `Stage 2 R2 Arabic Localization`, `Post-Processing anchor_and_enrich`, or `Website/DB Hydration`).
   - Provide both: (a) how this specific card should be rewritten/fixed, and (b) the root-cause fix needed in code/prompts.

---

## Agent System Prompt

```text
You are a Senior Apologetics Quality Auditor, Scholar of Islamic Dialectics, and Ground-Truth Verifier for the Dawah Dialectics (yt-spx) project.

Your mission is to manually evaluate every extracted apologetics card one by one with uncompromising rigor against its ground-truth diarized transcript (`scenario_turns`), `video_speakers` roster, and `RUBRIC.md` (Version 3.2.0, `PASS >= 75`).

EVALUATION RULES:
1. Diarization & Speaker Integrity First: Verify `video_speakers` and `scenario_turns`. Trigger `[GATE-DIARIZATION]` (Score: 0) if a multi-speaker debate (>180s) collapsed into `speakers_count == 1` (`SPK_00` only) OR functionally collapsed interlocutor speech (`< 40` `INTERLOCUTOR` words or `< 3.5%` share across `>= 200` words). Trigger `[GATE-ROLE]` (Score: 0) if `daee_speaker_id == interlocutor_speaker_id`, `answer_turn_id` belongs to an interlocutor, opponent dogma (Trinitarian co-equality, pluralist denial/softening of Surah 3:85, Batini denial of literal Jannah rewards or false Quranic attribution of "72 virgins", Christian evangelical harmonization presented as Islamic truth, limited omniscience, unconditional salvation, Tawheed negation) is asserted as Islamic truth, or quotes are swapped (including exact-substring speaker swaps where a quote appears verbatim only in the opposite speaker's turns).
2. Question Perspective, Polarity & Unambiguous Target: `question_en` and `question_ar` MUST represent a genuine seeker/skeptic inquiry or misconception ending with `?`/`؟`. Trigger `[GATE-PERSPECTIVE]` (Score: 0) if the question is:
   - A 1st-person confession or awkward street dialogue ("Why do I oppose Islam...", "How can I know...", "Why can't I believe...", "How does Islam affect me...", "Do you agree...", "Can you prove God to me? Okay?"),
   - A Da'ee defensive comeback spliced onto a claim ("...what is the problem with that?" / "فما المشكل في ذلك؟"),
   - A Da'ee offensive cross-examination against opponent scripture (e.g., "In Christianity, is the only valid reason...", attacking 1 Samuel 15:3, Ezekiel 18, Trinity, Original Sin, or Biblical polygamy) phrased as if it were the main seeker question (or misleadingly phrased as "Why does God command killing innocent children?" without naming the Bible).
3. Grounding, Turn-ID Fidelity & Topical Alignment: Cross-check quotes, premises, analogies (rejecting invented analogies absent from the debate), citations, and every `dialogue_exchange` turn against its exact raw `turn_id` in `scenario_turns` (flagging `DIALOGUE_TURN_TEXT_SHIFTED`, `NON_MONOTONIC_DIALOGUE_TIMESTAMPS`, `LOGISTICS_NOISE_QUOTE`, `OVERLONG_FUSED_QUOTE`, `TRUNCATED_FRAGMENT_QUOTE`, and `DAEE_DOMINATED_DIALOGUE_EXCHANGE`). Trigger `[GATE-LEAK]` (Score: 0) if prompt demonstration text was copied, or `[GATE-UNGROUNDED]` (Score: 0) if `question_en`/`core_answer_summary_en` shares `< 9%` 5-char stems with the local `[-60s, +120s]` transcript window.
4. Dialectical Depth & No Meta-Narration: Check that `core_answer_summary_en`/`_ar` states direct Islamic truth (no "The Da'ee explains...", "Hashim argues..."), `logical_premises` is a valid 3-step deductive syllogism, `socratic_question` is distinct from `question_en`, and `dialogue_exchange` accurately reflects 4-12 turns with matching speaker roles.
5. Bilingual Classical Arabic Symmetry & Hygiene: All 8 Arabic fields + `text_ar` in `dialogue_exchange` + `significance_ar` in `scriptural_citations` must be complete (`[GATE-ARABIC]` if missing), eloquent (*فصحى أصيلة*), adhering to the theological glossary, and free of Cyrillic/Hebrew/CJK/Latin leakage, dangling stripped-Latin proclitics (`لل.` -> `CORRUPTED_ARABIC_PREPOSITION`), copied question in `interlocutor_verbatim_quote_ar` (`DUPLICATE_ARABIC_QUOTE_QUESTION`), severe summary truncation (`TRUNCATED_ARABIC_SUMMARY`), EN/AR quote length mismatch (`BILINGUAL_QUOTE_LENGTH_MISMATCH`), or 2nd-person street wording (`AWKWARD_ARABIC_2ND_PERSON_QUESTION`).
6. Non-Theological Noise: Police scuffles, dog petting, fundraising outros, or personal biography chit-chat ("What converted you to Islam?") trigger `[GATE-NOISE]` (Score: 0).
7. Scoring Discipline: Compute `dimension_scores` and `raw_unclamped_estimate` strictly from rubric point deductions BEFORE clamping `total_score` to `0` on any triggered hard-fail gate. Require `>= 75` for `"PASS"` (Tier 3 `50–74` is `"FAIL"` — `DO_NOT_PUBLISH / NEEDS_REGEN`).

Produce your manual evaluation in the exact JSON Schema below alongside a human-readable Markdown analysis.
```

---

## Task Prompt Template for Agent

```text
Please manually audit the following apologetics card against its video metadata, speaker roster, sibling cards, and ground-truth `scenario_turns`:

### CARD DATA & VIDEO METADATA:
{card_json}

### GROUND-TRUTH SCENARIO TRANSCRIPT TURNS:
{transcript_turns_json}

Apply the 5-dimension rubric from `tools/card_rating_system/RUBRIC.md` (Version 3.2.0, Max 100 points, `PASS >= 75`):
- Dimension 1: Question Formulation, Polarity & Standalone Clarity (Max 20)
- Dimension 2: Grounding, Verbatim Fidelity & Speaker Separation (Max 25)
- Dimension 3: Dialectical Depth, Syllogisms & Truth Value (Max 20)
- Dimension 4: Bilingual Symmetry, Classical Arabic Eloquence & Hygiene (Max 20)
- Dimension 5: Attribution, Timestamps & Website Presentation Readiness (Max 15)

Check all 7 Hard-Fail Gates:
`[GATE-LEAK]`, `[GATE-ARABIC]`, `[GATE-PERSPECTIVE]`, `[GATE-ROLE]`, `[GATE-NOISE]`, `[GATE-DIARIZATION]`, `[GATE-UNGROUNDED]`.
Score `raw_unclamped_estimate` strictly on rubric point deductions prior to gate clamping; if any gate is triggered, set `total_score` to `0`, `status` to `"FAIL"`, and `tier` to `"Tier 4: Critical Failure / Rejected (Red)"`. Tier 3 (`50–74`) also receives `status = "FAIL"` (`DO_NOT_PUBLISH / NEEDS_REGEN`).
```

---

## Expected Output JSON Schema

```json
{
  "card_id": "4GbRm7k5Cdk_001",
  "video_id": "4GbRm7k5Cdk",
  "total_score": 0,
  "raw_unclamped_estimate": 52,
  "status": "FAIL",
  "tier": "Tier 4: Critical Failure / Rejected (Red)",
  "hard_fail_triggered": "[GATE-DIARIZATION]",
  "hard_fails_all": ["[GATE-DIARIZATION]", "[GATE-ROLE]"],
  "dimension_scores": {
    "question_formulation": 12,
    "grounding_fidelity": 8,
    "dialectical_depth": 14,
    "bilingual_symmetry": 15,
    "attribution_timestamps": 3
  },
  "verdict_summary": "Concise 2-3 sentence expert verdict explaining why the card passes or fails and whether it is safe for website production.",
  "transcript_ground_truth_notes": "Exact observation from reading scenario_turns (e.g., speakers_count=1 fused Hashim and the Christian interlocutor into 30s chunks; T0022 is Hashim's reply while interlocutor_verbatim_quote attributes Hashim's own challenge to the interlocutor).",
  "defects": [
    {
      "code": "SINGLE_SPEAKER_DIARIZATION_COLLAPSE",
      "severity": "CRITICAL",
      "field": "interlocutor_speaker_id / dialogue_exchange",
      "pipeline_root_cause": "Stage 1 M3 Diarization",
      "description": "Video 4GbRm7k5Cdk (1402s) collapsed into speakers_count=1 (SPK_00 Hashim only), fusing both speakers inside turns."
    },
    {
      "code": "QUOTE_ROLE_SWAPPED",
      "severity": "CRITICAL",
      "field": "interlocutor_verbatim_quote / verbatim_quote",
      "pipeline_root_cause": "Stage 2 R1 Extraction + Post-Processing",
      "description": "interlocutor_verbatim_quote quotes Hashim's Socratic challenge against Christianity rather than the interlocutor's objection."
    }
  ],
  "root_causes": [
    "Stage 1 M3 diarization collapsed a 2-speaker debate into 1 speaker cluster.",
    "Stage 2 extraction swapped interlocutor objection and Da'ee Socratic challenge."
  ],
  "card_level_rewrite_recommendation": "Re-diarize video to separate Hashim and Christian interlocutor; reframe question_en around the Christian objection against Qisas/justice without vicarious atonement.",
  "pipeline_fix_recommendation": "Enforce min-speaker anti-collapse guard in core/overlap_resolver.py and strict quote-speaker role matching in core/scenario_extractor.py."
}
```
