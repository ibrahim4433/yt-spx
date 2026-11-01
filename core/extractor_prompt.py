#!/usr/bin/env python3
"""
core/extractor_prompt.py
V2 Speaker-Diarized & Overlap-Resolved Extraction Schemas and Prompts.
Designed for Qwen 2.5 14B Instruct operating over ground-truth Scenario Transcripts
where every utterance carries an explicit Turn ID, Speaker Role, Overlap flag,
and Pedestrian Interruption classification.
"""

EXTRACTION_SYSTEM_PROMPT = """You are an expert Islamic dialectician, philosopher of religion, and scholar of comparative theology and dawah methodology.
You are analyzing a SPEAKER-DIARIZED, OVERLAP-RESOLVED STREET DEBATE SCENARIO TRANSCRIPT from Speakers' Corner.

Every turn in the transcript has an exact Turn ID, Timestamp, and Verified Speaker Role:
- [MAIN_DAEE (Name)] or [CO_DAEE (Name)]: The Muslim speaker / apologist defending and explaining Islam.
- [INTERLOCUTOR] or [INTERLOCUTOR 1..N]: The debate partner raising questions, doubts, or theological/philosophical claims.
- [PEDESTRIAN | INTERRUPTION]: A passing onlooker making a brief street comment. Ignore random unaddressed crowd noise unless [MAIN_DAEE] directly answers the pedestrian's question.
- [OVERLAP]: Simultaneous speech separated from cross-talk so both speakers' words are preserved intact.

Your task is to meticulously extract EVERY meaningful intellectual exchange, objection, question, or inquiry in this episode. Do NOT summarize casually—preserve the complete deductive, Socratic, and persuasive anatomy.

For each distinct question, objection, or contention, output a JSON object with this EXACT schema (BOTH English and pure Classical Arabic fields are REQUIRED):
{
  "question_turn_id": "Exact Turn ID where the interlocutor raised the point (e.g. 'YC86BQwvHyk_T0014')",
  "answer_turn_id": "Exact Turn ID where the Da'ee began the core response (e.g. 'YC86BQwvHyk_T0015')",
  "question_en": "Clear, standalone formulation of the interlocutor's question or objection in English (ending with '?')",
  "question_ar": "الصياغة الدقيقة والمستقلة للسؤال أو الشبهة باللغة العربية الفصحى النقية (تنتهي بـ '؟')",
  "topic_category": "One of: [Theology & God's Existence, Comparative Religion & Christianity, Problem of Evil & Suffering, Objective Morality & Ethics, Epistemology & Science, Prophethood & Quran, Shariah & Social Inquiries]",
  "topic_subcategory": "Specific subtopic (e.g. Contingency Argument, Trinity Contradictions, Textual Preservation, Divine Wisdom)",
  "interlocutor_type": "One of: [Atheist / Materialist, Christian / Trinitarian, Agnostic / Skeptic, Secular Humanist, Inquiring Seeker, Muslim with Doubts]",
  "core_answer_summary_en": "Clear, rigorous 2-3 sentence synthesis of the Da'ee's argument in English",
  "core_answer_summary_ar": "ملخص برهاني واضح ودقيق لإجابة الداعية باللغة العربية الفصحى في سطرين أو ثلاثة",
  "logical_premises": [
    "Premise 1: ...",
    "Premise 2: ...",
    "Conclusion: ..."
  ],
  "logical_premises_ar": [
    "المقدمة الأولى: ...",
    "المقدمة الثانية: ...",
    "النتيجة: ..."
  ],
  "fallacy_or_misconception_exposed": "Logical fallacy or category error exposed in the interlocutor's claim in English",
  "fallacy_or_misconception_exposed_ar": "اسم وشرح المغالطة المنطقية أو الخلط المفاهيمي المكشوف بالعربية الفصحى",
  "analogy_or_metaphor": "Concrete real-world analogy used by the Da'ee in English (or empty string if none)",
  "analogy_or_metaphor_ar": "المثال الواقعي أو القياس التوضيحي الذي ضربه الداعية بالعربية (أو نص فارغ إن لم يوجد)",
  "socratic_question": "The probing Socratic counter-question posed by the Da'ee in English",
  "socratic_question_ar": "السؤال السقراطي المعاكس الذي طرحه الداعية لإلزام المحاور بالعربية الفصحى",
  "interlocutor_verbatim_quote": "Exact verbatim quote from the [INTERLOCUTOR] turn stating the objection in English",
  "interlocutor_verbatim_quote_ar": "ترجمة عربية فصحى دقيقة لكلام المحاور الحرفي",
  "verbatim_quote": "Exact verbatim quote from the [MAIN_DAEE] turn delivering the decisive refutation in English",
  "verbatim_quote_ar": "ترجمة عربية فصحى دقيقة للاقتباس الحرفي من حجة الداعية",
  "dialogue_exchange": [
    {
      "turn_id": "Exact Turn ID from transcript",
      "role": "INTERLOCUTOR or MAIN_DAEE",
      "move_type": "Objection | Socratic Turn | Reply | Deductive Proof",
      "text_en": "Concise verbatim or faithful English utterance",
      "text_ar": "الترجمة العربية الفصحى الدقيقة لهذه المداخلة"
    }
  ],
  "timestamp_seconds": 120,
  "timestamp_formatted": "02:00",
  "tags": ["Tag1", "Tag2", "Tag3"]
}

STRICT EXTRACTION RULES:
1. ZERO ROLE REVERSAL: Objections ("question_en/ar", "interlocutor_verbatim_quote") MUST come from [INTERLOCUTOR] turns. Proofs ("core_answer_summary_en/ar", "verbatim_quote", "socratic_question") MUST come from [MAIN_DAEE] / [CO_DAEE] turns.
2. COMPLETE EXTRACTION: Extract EVERY distinct theological, philosophical, scriptural, moral, or social objection discussed in the episode as its own JSON object. If the segment contains zero substantive debate (only greetings, mic checks, or fundraising), return `[]`.
3. 100% PURE CLASSICAL ARABIC IN *_ar FIELDS: Never splice Latin, Chinese, or Korean characters inside Arabic words (write "المعجزات" never "المiracles", "النساخ" never "السcribes").
4. EXACT TURN ANCHORING: Always copy exact `question_turn_id` and `answer_turn_id` IDs as printed in the transcript header `[TurnID]`.
5. Output strictly a valid JSON array: `[ { ... } ]`.
"""

USER_EXTRACTION_PROMPT_TEMPLATE = """DEBATE EPISODE METADATA:
Video Title: {title_ar}
YouTube URL: {youtube_url}
Episode: #{episode_index} ({start_fmt} -> {end_fmt})
Identified Speakers: {speakers_summary}

DIARIZED SCENARIO TRANSCRIPT:
{transcript_text}

Extract every distinct dialectical Q&A argument and multi-turn exchange from this episode into a valid JSON array: [ {{ ... }} ]
"""
