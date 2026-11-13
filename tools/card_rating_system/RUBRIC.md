# Dawah Dialectics (yt-spx) — Universal Apologetics Card Rating System & Quality Rubric

> **Version**: 2.0.0  
> **Target Dataset**: `dawah_knowledge.db` (`qa_units`, `scenario_turns`, `videos`)  
> **Evaluation Engine**: Automated Audit CLI & Multi-Agent Swarm  
> **Core Objective**: Ensure every apologetics card is a verified, production-ready, intellectually convincing, bilingual dialectical asset grounded 100% in acoustic and textual ground truth.

---

## 1. Executive Quality Standard & Product Philosophy

A card in *Dawah Dialectics* is **not** a raw transcript snippet, a generic article, or an AI summary. It is an **Authoritative Claim-and-Proof Dossier** designed for seekers, skeptics, and Muslims with doubts searching for rational and scriptural answers.

Every card must satisfy three uncompromising pillars:
1. **The Seeker's Natural Entrypoint**: Formulated from the perspective of an honest skeptic or inquirer asking or searching the objection in clear, grammatically sound interrogative syntax.
2. **Ground-Truth Acoustic & Spoken Veracity**: Every quote, timestamp, and speaker attribution must exist in the real video transcript (`scenario_turns`). Zero prompt leakage, zero hallucinated dialogue.
3. **Intellectual & Deductive Rigor**: The card must dismantle the objection using valid deductive syllogisms, formal fallacy exposures, real-world analogies, and decisive Socratic counter-questions, presented in eloquent Classical Arabic and academic English.

---

## 2. The 5 Core Evaluation Dimensions (100 Points Total)

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

---

### Dimension 1: Question Formulation & Perspective (15 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **1.1 Interrogative Syntax & Polish** | 5 | Must be a grammatically complete interrogative question ending with `?` (EN) and `؟` (AR). | -5 if declarative statement (e.g., "God needs a designer. No, listen.") or conversational fragment. |
| **1.2 Inquirer / Skeptic Perspective** | 6 | Must represent the false claim, theological objection, or sincere doubt from the inquirer's viewpoint. | -6 if formulated as a 1st-person confession (e.g. "Why do I oppose Islam...") or if confused with the Da'ee's Socratic question. |
| **1.3 Substantive Theological Scope** | 4 | Focuses on a real theological, philosophical, scriptural, or social doubt. | -4 if based on passing conversational filler, street logistics, or policing incidents. |

---

### Dimension 2: Grounding, Verbatim Fidelity & Zero-Leakage (25 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **2.1 Zero Prompt Leakage** | 10 | **Mandatory Hard Gate**: The card must not contain any phrases or analogies copied from prompt few-shot examples (e.g., "Mark 13:32", "whatever begins to exist has a cause", "soldier cannot fire his rifle", "freight train"). | **-25 (AUTOMATIC HARD FAIL)** if any prompt demonstration text is copied into the card. |
| **2.2 Verbatim Quote Accuracy** | 8 | `verbatim_quote` and `interlocutor_verbatim_quote` must be verbatim matches of words actually spoken in `scenario_turns` for that video. | -8 if fabricated, hallucinated, or ungrounded in the transcript. -3 if heavily paraphrased. |
| **2.3 Dialogue Trajectory Integrity** | 7 | `dialogue_exchange` turns must correspond to real turns in the video, preserving speaker roles and the sequence of objections and replies. | -7 if dialogue turns are made up or misattributed. |

---

### Dimension 3: Dialectical Depth, Syllogisms & Truth Value (25 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **3.1 Direct Convincing Truth** | 7 | `core_answer_summary_en` and `_ar` state the Islamic truth and rational proofs directly and convincingly. | -4 if written as a meta-narrative summary ("The Da'ee counters by citing...", "Hashim asks..."). |
| **3.2 Formal Deductive Syllogism** | 8 | `logical_premises` must contain a valid 3-part syllogism: Major Premise (universal axiom), Minor Premise (context/fact), Conclusion (inescapable deduction). | -8 if missing, empty, or circular. -4 if poorly structured or missing conclusion. |
| **3.3 Fallacy Exposure** | 4 | Accurately identifies the root cognitive fallacy or false assumption (e.g., Category Mistake, Strawman, False Dilemma). | -4 if generic ("bad thinking") or empty. |
| **3.4 Concrete Analogy / Metaphor** | 3 | Provides a vivid, relatable real-world comparison that clarifies the abstract metaphysical or theological concept. | -3 if absent, trivial, or incomprehensible. |
| **3.5 Socratic Counter-Question** | 3 | Formulates the Da'ee's clincher counter-question that turns the table on the debater and exposes internal contradiction. | -3 if absent or merely repeats the main question. |

---

### Dimension 4: Bilingual Symmetry, Arabic Eloquence & Hygiene (20 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **4.1 Complete Arabic Localization** | 10 | **Mandatory Gate**: All Arabic fields (`question_ar`, `core_answer_summary_ar`, `logical_premises_ar`, `fallacy_or_misconception_exposed_ar`, `analogy_or_metaphor_ar`, `socratic_question_ar`, `verbatim_quote_ar`, `interlocutor_verbatim_quote_ar`) MUST be fully populated. | **-20 (AUTOMATIC HARD FAIL)** if Arabic fields are empty/null (Round 2 missing). |
| **4.2 Eloquent Classical Arabic (فصحى أصيلة)** | 5 | Pure, scholarly Arabic prose with proper Arabic punctuation (`؟`, `،`). | -3 if literal machine translation or colloquial Arabic. |
| **4.3 Zero Foreign Script / Latin Leakage** | 5 | Clean of Russian Cyrillic, Chinese CJK punctuation, or untranslated Latin words in Arabic fields. | -5 if foreign characters or untranslated Latin terms leak into Arabic fields. |

---

### Dimension 5: Attribution, Timestamps & Studio Integration (15 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **5.1 Speaker Attribution Integrity** | 6 | `daee_name_en` and `daee_speaker_id` must point strictly to the Muslim speaker ([MAIN_DAEE] or [CO_DAEE]), never to the interlocutor. | -6 if interlocutor is credited as the Da'ee or roles are swapped. |
| **5.2 Interlocutor Archetype & Pillar** | 4 | Correct categorization (`topic_category`, `topic_subcategory`, `interlocutor_type`). | -4 if completely misclassified (e.g. Atheist debate labeled Christian Trinity). |
| **5.3 Timestamp Seek Precision** | 5 | `timestamp_seconds` and `answer_timestamp_seconds` must accurately navigate the seeker to the exact moment in the video. | -5 if timestamps are wildly inaccurate (>2 minutes off) or broken. |

---

## 3. Hard-Fail Gates (Automatic Rejection = Score 0)

Any card exhibiting **at least one** of the following conditions is assigned an automatic **FAIL (0/100)**:
1. **[GATE-LEAK] Prompt Hallucination / Leakage**: The card reproduces text, quotes, or analogies from the prompt's few-shot demonstrations rather than the actual video.
2. **[GATE-ARABIC] Missing Arabic Localization**: Key Arabic fields are `NULL` or empty, indicating Round 2 failed or was skipped.
3. **[GATE-PERSPECTIVE] 1st-Person Confessional Inversion**: The question is phrased as a personal admission of wrongdoing or anti-Islamic bias (e.g., "Why do I hate Islam?").
4. **[GATE-ROLE] Inverted Speaker Attribution**: The interlocutor's argument is labeled as the Da'ee's answer, or the opponent is credited as the Da'ee.
5. **[GATE-NOISE] Conversational Noise / Street Drama**: The card extracts a police encounter, street scuffle, dog petting, or conversational interjection rather than a theological argument.

---

## 4. Final Quality Classification Tiers

- **Tier 1: Production Ready (Gold)** [90 - 100 Points]: Flawless, authoritative, verified against audio/transcript, ready for live production seekers.
- **Tier 2: Minor Polish Needed (Silver)** [75 - 89 Points]: High value and grounded, but has slight stylistic, translation, or syntax flaws that can be auto-repaired.
- **Tier 3: Substandard / Requires Regeneration (Bronze)** [50 - 74 Points]: Grounded in the topic but has defective syllogisms, weak phrasing, or incomplete fields.
- **Tier 4: Critical Failure / Rejected (Red)** [0 - 49 Points]: Breaches hard-fail gates or suffers from hallucination, prompt leakage, or missing language tracks.
