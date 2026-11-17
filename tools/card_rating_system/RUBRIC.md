# Dawah Dialectics (`yt-spx`) — Universal Apologetics Card Rating System & Quality Rubric

> **Version**: 3.0.0  
> **Target Dataset**: `dawah_knowledge.db` (`qa_units`, `scenario_turns`, `video_speakers`, `videos`)  
> **Evaluation Engine**: Manual AI Agent Swarm + Automated Audit CLI (`audit_cards.py`)  
> **Core Objective**: Ensure every apologetics card is a verified, production-ready, intellectually convincing, bilingual dialectical dossier grounded 100% in acoustic diarization and spoken transcript truth.

---

## 1. Executive Quality Standard & Product Philosophy

A card in *Dawah Dialectics* is **not** a raw transcript snippet, a generic article, or an AI narration of a street clip. It is an **Authoritative Claim-and-Proof Dossier** built for sincere seekers, skeptics, and Muslims with doubts searching for rational, philosophical, and scriptural answers—paired with verified multi-speaker debate evidence.

Every card must satisfy four uncompromising pillars:
1. **The Seeker's Natural Entrypoint & Polarity**: The question (`question_en` / `question_ar`) must be formulated as an authentic, standalone inquiry or objection that a seeker/skeptic would genuinely ask or search—NEVER a Da'ee's offensive cross-examination against the opponent's scripture, NEVER a Da'ee's defensive comeback (*"...what is the problem with that?"*), and NEVER 1st-person street dialogue (*"How can I know..."*, *"Why can't I believe..."*, *"What converted you to Islam?"*).
2. **Unambiguous Theological Target**: When a debate involves the Da'ee critiquing another scripture (e.g., Old Testament warfare verses like 1 Samuel 15:3, Trinity, or Biblical preservation), the question must explicitly name that comparative/Biblical context rather than ambiguously asking *"Why does God command the killing of innocent children?"* (which misleads website visitors into thinking Islam teaches it).
3. **Ground-Truth Acoustic Diarization & Speaker Veracity**: Every quote, turn ID, timestamp, and speaker attribution must correspond accurately to `scenario_turns` and `video_speakers`. The Muslim Da'ee (`MAIN_DAEE`/`CO_DAEE`) and the `INTERLOCUTOR` must be cleanly separated by Stage 1 diarization (`speakers_count >= 2` for real multi-speaker debates), with zero swapped quotes or single-speaker diarization collapse.
4. **Intellectual Rigor & Bilingual Classical Arabic Symmetry**: Every card must deliver direct Islamic truth (zero 3rd-person meta-narrator summaries), a valid 3-step deductive syllogism, accurate fallacy exposure, grounded analogies/citations, a distinct Socratic counter-question, and complete, eloquent Classical Arabic (*فصحى أصيلة*) across all fields and `dialogue_exchange` turns.

---

## 2. The 5 Core Evaluation Dimensions (100 Points Total)

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

### Dimension 1: Question Formulation, Polarity & Standalone Clarity (20 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **1.1 Grammatical Interrogative Syntax** | 4 | Complete interrogative question ending with `?` (EN) and `؟` (AR); zero trailing conversational ASR filler (`"Okay?"`, `"..."`). | -4 if declarative, fragment (`"..."`), or conversational filler (`"Can you prove God to me? Okay?"`). |
| **1.2 Authentic Seeker/Skeptic Polarity** | 8 | Expresses the seeker/skeptic objection or comparative inquiry objectively. Zero Da'ee offensive cross-exam attacks, zero Da'ee defensive comebacks (*"...what is the problem with that?"*), zero 1st/2nd-person street phrasing (*"How can I know..."*, *"Do you agree..."*, *"Why can't I believe..."*). | **-20 / [GATE-PERSPECTIVE]** if Da'ee cross-exam, defensive comeback, or 1st-person confession; -5 if awkward 1st/2nd-person phrasing (`"How can I know..."`, `"Do you agree..."`). |
| **1.3 Unambiguous Theological Target** | 4 | Clear self-contained scope. If the exchange critiques Christian/Biblical/Atheist claims, the question explicitly names the Biblical/comparative/skeptical context. | -4 if misleadingly ambiguous (e.g., asking `"Why does God command killing innocent children?"` on a 1 Samuel 15:3 refutation without naming the Bible/Old Testament). |
| **1.4 Intra-Video Distinctness** | 4 | Addresses a genuinely distinct sub-argument or objection within the video rather than duplicating another card's stem in the same debate. | -4 if near-duplicate question stem repeated across multiple cards in the same video (e.g., 4–6 near-identical `"How can I know..."` cards). |

---

### Dimension 2: Grounding, Verbatim Fidelity & Speaker Separation (25 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **2.1 Zero Prompt Leakage & Fabrication** | 7 | Zero prompt demonstration clichés (e.g., `"Mark 13:32"`, `"train of freight carriages"`, `"soldier cannot fire his rifle"`, `"whatever begins to exist has a cause"`, `"lamp drawing current from a power plant"`) unless spoken verbatim on camera. | **[GATE-LEAK] (AUTOMATIC 0/100)** if prompt demonstration text is copied into the card. |
| **2.2 Interlocutor Quote Accuracy & Role** | 6 | `interlocutor_verbatim_quote` matches spoken words in `scenario_turns` spoken by the `INTERLOCUTOR` (or Da'ee framing the objection in a verified solo monologue) and represents the objection—not the Da'ee's counter-argument. | **[GATE-ROLE]** / -6 if it contains the Muslim Da'ee's counter-argument or is fabricated; -3 if fused with Da'ee speech due to diarization collapse. |
| **2.3 Da'ee Proof Quote Accuracy & Role** | 6 | `verbatim_quote` matches spoken words by `MAIN_DAEE`/`CO_DAEE` delivering proof/refutation—not the opponent's objection and not a fused 2-speaker block. | **[GATE-ROLE]** / -6 if it contains the opponent's claim/attack or is ungrounded; -3 if fused with interlocutor speech. |
| **2.4 Dialogue Trajectory Integrity** | 6 | `dialogue_exchange` preserves chronological progression across **6–12 turns** (minimum 4 turns for rich debates), with accurate `role` (`INTERLOCUTOR` vs `MAIN_DAEE`) matching who actually spoke each turn in `scenario_turns`. | -4 if `< 4` turns when the debate is long; -4 if `role` labels disagree with `scenario_turns` or collapse to a single role; -6 if dialogue turns are fabricated. |

---

### Dimension 3: Dialectical Depth, Syllogisms & Truth Value (20 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **3.1 Direct Convincing Truth (No Meta-Narration)** | 6 | `core_answer_summary_en` and `_ar` state the Islamic rational/scriptural proof directly (150–250 words target) with ZERO 3rd-person narrator phrases (`"The Da'ee explains..."`, `"Hashim argues..."`, `"In this exchange..."`). | -4 if meta-narrator phrasing appears; -3 if shallow (< 60 words) or repetitive filler. |
| **3.2 Formal 3-Step Deductive Syllogism** | 6 | `logical_premises` / `_ar` contains 3 distinct, non-circular steps: Premise 1 (Major universal axiom), Premise 2 (Minor context/fact), Conclusion (inescapable deduction). | -6 if missing/circular; -3 if fewer than 3 premises or weak connection to the spoken proof. |
| **3.3 Fallacy Exposure** | 3 | Accurately identifies and explains the root cognitive fallacy or false premise (e.g., Category Mistake, Strawman, False Dilemma, Equivocation). | -3 if empty, generic, or misidentifies the fallacy. |
| **3.4 Grounded Analogies & Scriptural Citations** | 2 | `analogy_or_metaphor` / `analogies` and `scriptural_citations` match analogies and verses actually cited or clearly implied in the video (no hallucinated verse numbers). | -2 if citations/analogies contradict or fabricate what was spoken in the transcript. |
| **3.5 Clincher Socratic Counter-Question** | 3 | `socratic_question` captures the Da'ee's decisive counter-question turning the table on the opponent's premise, distinct from `question_en`. | -3 if empty, trivial, or merely repeats `question_en`. |

---

### Dimension 4: Bilingual Symmetry, Classical Arabic Eloquence & Hygiene (20 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **4.1 Complete Arabic Localization** | 8 | All 8 Arabic fields (`question_ar`, `core_answer_summary_ar`, `logical_premises_ar`, `fallacy_or_misconception_exposed_ar`, `analogy_or_metaphor_ar`, `socratic_question_ar`, `verbatim_quote_ar`, `interlocutor_verbatim_quote_ar`) + `text_ar` in every `dialogue_exchange` turn + `significance_ar` in `scriptural_citations` are fully populated. | **[GATE-ARABIC] (AUTOMATIC 0/100)** if any required Arabic field is missing/empty; -3 if `text_ar` or `significance_ar` is missing. |
| **4.2 Eloquent Classical Arabic (فصحى أصيلة)** | 7 | Authoritative scholarly Arabic prose adhering to the Islamic apologetics glossary (`الجزيرة العربية` never `عربة`; `التزامنات` never `المؤامرات`; `بولس/متى/مرقس/لوقا/يوحنا/بطرس` never phonetic `بول/ماتيو/مارك/لوك/جون/بيتر`; proper `؟` and `،`). | -3 to -5 for glossary drift, literal calques, gender/number agreement errors, or robotic syntax. |
| **4.3 Zero Foreign Script / Latin Leakage** | 5 | Clean of Russian Cyrillic, Hebrew, Korean/CJK characters or punctuation (`。，？`), and untranslated Latin words inside Arabic fields. | -5 if Cyrillic, Hebrew, CJK, or Latin words leak into Arabic fields. |

---

### Dimension 5: Attribution, Timestamps & Website Presentation Readiness (15 Points)

| Criteria | Max Pts | Passing Standard | Failure / Deductions |
| :--- | :---: | :--- | :--- |
| **5.1 Speaker Attribution & Diarization Health** | 5 | Accurate `daee_name_en`/`_ar`, `daee_profile_id`, `daee_speaker_id`, and populated `interlocutor_speaker_id` for multi-speaker debates (`speakers_count >= 2` in `video_speakers`). | **[GATE-DIARIZATION]** if video collapsed into 1 speaker (`speakers_count == 1`) on a multi-speaker debate; -3 if `interlocutor_speaker_id` is empty despite a clear interlocutor in `scenario_turns`. |
| **5.2 Interlocutor Archetype & Topic Pillar** | 5 | Accurate `interlocutor_type` matching the real speaker in the debate (e.g., `Shiite` not `Inquiring Seeker` in a Sunni-Shia debate; `Christian / Trinitarian` not `Inquiring Seeker` when debating a Trinitarian), canonical `topic_category`, and accurate `topic_subcategory`. | -3 to -5 if `interlocutor_type` or `topic_category` misclassifies the debate. |
| **5.3 Timestamp Precision & Website UI Readiness** | 5 | `timestamp_seconds` and `answer_timestamp_seconds` accurately seek to the question and answer turns; all fields render cleanly in EN, AR, and `EN ⇄ عربي` parallel mode without missing badges or UI breakage. | -3 to -5 if timestamps are misaligned (>60s off) or `answer_timestamp_seconds < timestamp_seconds` without reason. |

---

## 3. Hard-Fail Gates (Automatic Rejection = Score 0/100)

Any card exhibiting **at least one** of the following conditions receives an automatic **FAIL (0/100 — Tier 4 Red)**:
1. **[GATE-LEAK] Prompt Hallucination / Leakage**: Reproduces text, quotes, citations, or analogies from prompt blueprints rather than the video transcript.
2. **[GATE-ARABIC] Missing Arabic Localization**: Any of the 8 core Arabic fields are `NULL` or empty, or `dialogue_exchange` lacks Arabic text.
3. **[GATE-PERSPECTIVE] Perspective / Polarity Failure**:
   - 1st-person confessional inversion (*"Why do I oppose Islam..."*) or awkward 1st/2nd-person street phrasing (*"How can I know..."*, *"Why can't I believe..."*, *"What converted you to Islam?"*).
   - Da'ee defensive comeback spliced into the question (*"...what is the problem with that?"* / *"فما المشكل في ذلك؟"*).
   - Da'ee offensive Socratic cross-examination placed into `question_en`/`question_ar` (e.g. attacking Biblical contradictions or Christian Trinity/atonement as if it were an anti-Islamic objection).
4. **[GATE-ROLE] Speaker Role or Quote Inversion**:
   - `daee_speaker_id == interlocutor_speaker_id`, `answer_turn_id` spoken by an interlocutor, or opponent dogma asserted as Islamic proof.
   - `interlocutor_verbatim_quote` contains the Da'ee's argument OR `verbatim_quote` contains the interlocutor's objection.
5. **[GATE-NOISE] Conversational Noise / Street Drama / Chit-Chat**: Extracts police scuffles, dog petting, fundraising outros, or non-apologetic personal chat (*"What converted you to Islam?"*).
6. **[GATE-DIARIZATION] Stage 1 Single-Speaker Diarization Collapse**: The source debate (>180s, clearly involving 2+ speakers in the audio/transcript) collapsed into `speakers_count = 1` (`SPK_00` only) in Stage 1 M3 diarization, fusing both speakers inside single turns and corrupting `interlocutor_speaker_id`, `dialogue_exchange`, and verbatim quotes.

---

## 4. Final Quality Classification Tiers

- **Tier 1: Production Ready (Gold)** `[90 - 100 Points]`: Flawless, authoritative, verified 1-to-1 against `scenario_turns` and `video_speakers`, ready for live website publication.
- **Tier 2: Minor Polish Needed (Silver)** `[75 - 89 Points]`: Substantive and well-grounded, with minor stylistic, Arabic, meta-narrative, or trajectory length flaws that can be repaired cleanly.
- **Tier 3: Substandard / Requires Regeneration (Bronze)** `[50 - 74 Points]`: Grounded in a real topic, but suffers from weak phrasing, awkward 1st/2nd-person question framing, intra-video duplication, wrong `interlocutor_type`, or poor quote selection.
- **Tier 4: Critical Failure / Rejected (Red)** `[0 - 49 Points]`: Triggers a Hard-Fail Gate (`[GATE-*]`) or fails due to perspective inversion, swapped quotes, diarization collapse, prompt leakage, noise, or missing Arabic.
