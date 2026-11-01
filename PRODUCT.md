# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

General seekers, skeptics, and curious readers (both Arabic and English speakers) exploring clear, rational, and evidence-backed answers to theological, philosophical, and social doubts or objections about Islam—alongside students and researchers examining real street-debate dialogues.

## Product Purpose

Dawah Dialectics is a bilingual (Arabic/English) interactive knowledge base and zero-latency dialectical assistant that distills thousands of live street debates and philosophical dialogues (primarily from SCDawah Arabic / Speaker's Corner archives) into structured, scannable argument units. Success means a seeker or skeptic can immediately find a clear objection or question, follow its step-by-step deductive premises, grasp its concrete analogy and Socratic counter-question, see the underlying fallacy exposed, read the verbatim dialogue excerpt, and jump straight to the exact second in the original YouTube video.

## Positioning

Unlike generic Islamic Q&A articles or hallucination-prone LLM chatbots, every entry in Dawah Dialectics is anchored to an authentic, timestamped real-world dialogue with an explicit interlocutor archetype (e.g., Atheist/Materialist, Christian/Trinitarian, Agnostic/Skeptic, Secular Humanist, Inquiring Seeker) and decomposed into a rigorous dialectical anatomy: core thesis, deductive premises, fallacy exposed, real-world analogy, Socratic counter-question, and verbatim transcript.

## Operating Context

- Used across desktop and mobile web browsers in both Arabic (RTL) and English (LTR), auto-detected from browser language with persistent user preference (`localStorage`) and optional in-card bilingual side-by-side comparison.
- The corpus is an active **Beta / Work-in-Progress** extraction pipeline: currently 607 pre-analyzed Q&A dialectical units across 10 topic categories are extracted from a cataloged manifest of 2,348+ subtitle files (~5.95 million spoken words). The interface should make the Beta / live-extraction progress clear and honest while making full use of the 607 extracted units.

## Capabilities and Constraints

- **Backend stack**: FastAPI (`web/app.py`) + SQLite with FTS5 unicode61 full-text search (`db/database.py`) + deterministic zero-LLM bilingual intent & token-scoring assistant (`core/smart_bot.py`).
- **Full data schema per Q&A unit**:
  - `id`, `video_id`, `channel`, `video_title_ar`, `youtube_url`, `timestamp_url`, `timestamp_seconds`, `timestamp_formatted`
  - `topic_category` (10 categories in DB, led by Prophethood & Quran, Comparative Religion & Christianity, Theology & God's Existence, Epistemology & Science, Shariah & Social Inquiries, Objective Morality & Ethics, Problem of Evil & Suffering)
  - `topic_subcategory`
  - `interlocutor_type` (Inquiring Seeker, Christian / Trinitarian, Atheist / Materialist, Secular Humanist, Agnostic / Skeptic, etc.)
  - Bilingual content fields: `question_en` / `question_ar`, `core_answer_summary_en` / `core_answer_summary_ar`, `logical_premises` / `logical_premises_ar`, `fallacy_or_misconception_exposed` / `fallacy_or_misconception_exposed_ar`, `analogy_or_metaphor` / `analogy_or_metaphor_ar`, `socratic_question` / `socratic_question_ar`, `verbatim_quote` / `verbatim_quote_ar`, `tags`.
- **Required frontend capabilities**:
  - Dynamic category AND interlocutor filtering with live counts from `/api/categories` and `/api/interlocutors`.
  - Full browsing with pagination / load-more and sorting/filtering across all 607+ Q&A units (not capped at an unreachable 50-item wall).
  - Rich dialectical cards exposing all available fields (`fallacy_or_misconception_exposed`, `topic_subcategory`, `tags`, premises, analogy, Socratic counter-question, verbatim quote, and direct timestamped YouTube trigger) plus an in-card bilingual comparison toggle.
  - Smart Dialectical Assistant with confidence indicator, structured dialectical breakdown, clickable related questions, and seamless flow from search to assistant.
  - Corpus & Extraction Beta status view showing real database metrics, category/interlocutor distributions, and pipeline transparency.

## Evidence on Hand

- `data/dawah_knowledge.db`: SQLite database containing 607 extracted dialectical Q&A units and 2,348 cataloged video records from SCDawah Arabic.
- `data/video_manifest.json`: Inventory of linked videos and subtitle metadata.
- Real YouTube video links and exact timestamp URLs (`timestamp_url`) for every Q&A unit.
- No fabricated testimonials, user counts, or commercial claims may be added.

## Product Principles

1. **Clarity over Polemics**: Present philosophical and theological arguments with calm intellectual rigor—let the deductive premises, exposed fallacies, and Socratic questions speak for themselves.
2. **Complete Dialectical Anatomy**: Never flatten a structured argument into a wall of text; distinguish the core thesis, numbered premises, fallacy exposed, analogy, Socratic question, and verbatim quote so a seeker can scan in seconds or study in depth.
3. **First-Class Bilingualism (AR/EN)**: Arabic (RTL) and English (LTR) are equal citizens—auto-detect on arrival, remember the user's choice, and let readers inspect both languages side by side on any card.
4. **Verifiable Source Truth**: Every claim traces directly to a real conversation with a visible interlocutor badge, video title, exact timestamp (`MM:SS`), and direct link to watch the moment on YouTube.
5. **Honest Beta Transparency**: Frame the 607 extracted units out of 2,348 videos accurately as an active, growing research corpus.
