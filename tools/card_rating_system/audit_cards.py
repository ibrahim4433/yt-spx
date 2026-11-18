#!/usr/bin/env python3
"""
tools/card_rating_system/audit_cards.py
Universal Apologetics Card Audit & Rating Engine (v3.1) for Dawah Dialectics (yt-spx).

Evaluates cards in `qa_units` against ground-truth spoken turns in `scenario_turns`,
`video_speakers`, and `videos`. Enforces the 5 Core Rubric Dimensions (100 Pts) and
7 Hard-Fail Gates, detects defects, and outputs structured JSON & CLI reports.

Usage:
    python tools/card_rating_system/audit_cards.py
    python tools/card_rating_system/audit_cards.py --db data/dawah_knowledge.db --out report.json
    python tools/card_rating_system/audit_cards.py --video-id KRDyUeTz_s0 --verbose
    python tools/card_rating_system/audit_cards.py --fail-only --verbose
"""

import re
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from db.database import canonicalize_speaker_role as _db_canon_role
except Exception:
    _db_canon_role = None


def _canon_role(raw_role: Any, fallback: str = "") -> str:
    if _db_canon_role is not None:
        return _db_canon_role(raw_role, fallback=fallback)
    r = str(raw_role or "").strip()
    if not r:
        return fallback
    u = r.upper()
    if "PEDESTRIAN" in u or "عابرة" in r or "مارة" in r:
        return "PEDESTRIAN"
    if "CO_DAEE" in u or "CO-DAEE" in u:
        return "CO_DAEE"
    if "INTERLOCUTOR" in u or "GUEST" in u or "السائل" in r or "المستفهم" in r:
        return "INTERLOCUTOR"
    if "DAEE" in u or "الداعية" in r or "محاجج" in r:
        return "MAIN_DAEE"
    if "المحاور" in r:
        return "INTERLOCUTOR"
    return fallback

# Known prompt demonstration signatures that indicate leakage
PROMPT_LEAKAGE_SIGNATURES = [
    ("LEAK_FIRST_CAUSE_QUOTE", "whatever begins to exist has a cause"),
    ("LEAK_TRAIN_ANALOGY", "train of freight carriages"),
    ("LEAK_SOLDIER_SOCRATIC", "soldier cannot fire his rifle"),
    ("LEAK_CHILD_SURGERY_ANALOGY", "child being held down by doctors"),
    ("LEAK_MARK_13_32", "mark 13:32"),
    ("LEAK_GOD_IN_FLESH_QUOTE", "he is god in flesh. is he god or no"),
    ("LEAK_POWER_STATION_ANALOGY", "lamp drawing current from a power plant"),
]

# Non-theological noise indicators (police scuffles, dog petting, street security, donation outro)
NOISE_KEYWORDS = [
    "call the police", "police officer", "pet him", "what kind of a dog",
    "brandish", "brandished", "firearm", "police come", "911 call", "verbal dispute",
    "islamnet", "donate now", "link in the description", "fundraising", "subscribe to the channel",
]

PERSONAL_CHITCHAT_REGEX = re.compile(
    r'^\s*(?:what\s+converted\s+you\s+to\s+islam|how\s+did\s+you\s+become\s+(?:a\s+)?muslim|what\s+is\s+your\s+name)\s*[?؟!.]*$',
    re.IGNORECASE,
)

# Confessional, 1st/2nd-person street phrasing & Da'ee counter-ask perspective regexes
CONFESSIONAL_REGEX = re.compile(
    r'(\b(?:why do i|why did i|why am i|how can i believe if i|if i haven\'?t read)\b'
    r'|لماذا\s+أعارض|لماذا\s+أحمل\s+(?:مشاعر|رؤية)|لماذا\s+أكره)',
    re.IGNORECASE,
)
CONVERSATIONAL_1ST_2ND_PERSON_REGEX = re.compile(
    r'(?:\b(?:how\s+can\s+i\s+know|why\s+can\'?t\s+i\s+believe|if\s+i\s+believe\s+in|do\s+you\s+agree\s+that|'
    r'what\s+converted\s+you|how\s+does\s+islam\s+affect\s+me|can\s+you\s+prove\s+god\s+to\s+me)\b'
    r'|كيف\s+(?:يمكنني|لي\s+أن)\s+(?:أعرف|أعلم)|لماذا\s+لا\s+أستطيع\s+أن\s+أؤمن|'
    r'إذا\s+كنت\s+أؤمن\s+ب|هل\s+توافق\s+على\s+أن|كيف\s+يؤثر\s+الإسلام\s+علي|هل\s+يمكنك\s+إثبات\s+وجود\s+الله\s+لي)',
    re.IGNORECASE,
)
TRAILING_FILLER_REGEX = re.compile(r'(?:,?\s*\bokay\s*[?؟]*\s*$|،?\s*حسناً[؟?]*\s*$)', re.IGNORECASE)

DAEE_COUNTER_ASK_REGEX = re.compile(
    r'(?:what(?:\'s|\s+is|\s+would\s+be)\s+(?:the\s+)?(?:problem|issue|harm|wrong|matter)\s+(?:with|about)\s+(?:that|this|it|doing\s+(?:so|that))'
    r'|what(?:\'s|\s+is)\s+wrong\s+with\s+(?:that|this|it)'
    r'|why\s+(?:is|would)\s+(?:that|this|it)\s+(?:be\s+)?(?:a\s+problem|an\s+issue|wrong|bad)'
    r'|(?:ف|و)?ما\s+(?:هو\s+)?(?:المشكل|المشكلة|الإشكال|العيب|الضير)\s+في\s+(?:ذلك|هذا))\s*[؟?!.]*\s*$',
    re.IGNORECASE,
)
SOCRATIC_ROLE_INVERSION_REGEX = re.compile(
    r'\bshould\s+(?:god|he)\s+(?:love|approve(?:\s+of)?|reward)\s+a\s+(?:rapist|murderer|killer|liar|oppressor|hitler)\b',
    re.IGNORECASE,
)
DAEE_CROSS_EXAM_REGEX = re.compile(
    r'(?:\bif\s+god\s+is\s+(?:just|loving|all-powerful|holy)[^?]*\bwhy\s+does\s+he\s+'
    r'(?:punish\s+innocent|destroy\s+entire|command\s+the\s+killing|need\s+angels|get\s+angry)\b'
    r'|\bif\s+god\s+(?:loves|is\s+loving)\s+all\s+people[^?]*(?:hitler|rapist)'
    r'|\bif\s+jesus\s+is\s+(?:god|the\s+son\s+of\s+god)[^?]*(?:why|how)\s+(?:does|can|did|could)\s+he\b'
    r'|\bwhy\s+does\s+the\s+bible\s+allow\b'
    r'|\bif\s+the\s+bible\s+is\s+the\s+foundation\b'
    r'|\bif\s+the\s+trinity\s+is\s+central[^?]*how\s+is\s+it\b'
    r'|\bif\s+jesus\s+had\s+two\s+natures\b'
    r'|\bif\s+the\s+term\s+[\'"]?son\s+of\s+god[\'"]?\s+in\s+the\s+bible\b'
    r'|\bif\s+god\s+is\s+both\s+the\s+most\s+just\s+and\s+the\s+most\s+loving[^?]*why\s+does\s+christianity\b'
    r'|\b(?:why\s+trust\s+the\s+bible|paul\s+disagrees?\s+with\s+james|biblical\s+(?:authors?|writers?)\s+(?:disagree|contradict))\b'
    r'|\bhow\s+can\s+christians?\s+claim\b'
    r'|\bif\s+the\s+father\s+is\s+the\s+only\s+true\s+god\b'
    r'|\bif\s+the\s+father\s+and\s+(?:the\s+)?son\s+are\s+co-?equal\b'
    r'|\bas\s+suggested\s+by\s+the\s+interlocutor\b'
    r'|\baccording\s+to\s+your\s+belief\b'
    r'|\bif\s+the\s+bible\s+is\s+the\s+word\s+of\s+god\b'
    r'|\bhow\s+can\s+jesus\s+be\s+considered\s+the\s+[\'"]?son\s+of\s+god[\'"]?\b'
    r'|\bhow\s+can\s+the\s+doctrine\s+of\s+original\s+sin\b'
    r'|\bwill\s+i\s+still\s+be\s+saved\s+if\s+i\s+leave\s+christianity\b'
    r'|\bif\s+the\s+bible\s+commands\s+the\s+killing\b)',
    re.IGNORECASE,
)

# Russian, Hebrew, Korean, and Chinese CJK character sets
CYRILLIC_REGEX = re.compile(r'[\u0400-\u04FF]')
HEBREW_REGEX = re.compile(r'[\u0590-\u05FF]')
CJK_REGEX = re.compile(r'[\u4E00-\u9FFF\u3000-\u303F]')
LATIN_IN_ARABIC_REGEX = re.compile(r'\b[A-Za-z]{4,}\b')
_DANGLING_PROCLITIC_RE = re.compile(r'(?:^|[\s،(«])(?:[وف]?لل)(?:[\s.،؛؟!\)»]|$)|(?:^|\s)لل\s+ال')

META_NARRATOR_REGEX = re.compile(
    r'(?:\b(?:the\s+(?:da\'?ee|speaker|apologist|scholar|muslim|debater|narrator|presenter)'
    r'|hashim|mansur|uthman|ali|hijab|mohammed\s+hijab|hamza|adnan|subboor|zakir|shabir)\s+'
    r'(?:says|counters|explains|argues|responds|replies|clarifies|asserts|states|demonstrates|concludes|rebuts|refutes)\b'
    r'|\bin\s+this\s+(?:clip|video|exchange|dialogue),?\s+(?:he|she|the\s+\w+)\s+'
    r'(?:says|explains|argues|responds|replies|counters))',
    re.IGNORECASE,
)

_OVERLAP_STOPWORDS = frozenset({
    "that", "this", "what", "from", "with", "have", "they", "their",
    "about", "there", "which", "would", "could", "should", "where", "when",
})


def _parse_json_field(card: Dict[str, Any], raw_key: str, json_key: str) -> Any:
    """Reads `raw_key` if populated, otherwise parses `json_key` from SQLite rows."""
    val = card.get(raw_key)
    if val is None and json_key in card:
        val = card.get(json_key)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith("[") or s.startswith("{"):
            try:
                return json.loads(s)
            except Exception:
                return [val]
        return [val]
    return val if val is not None else []


def _quote_transcript_overlap(quote: str, transcript_lower: str) -> Optional[float]:
    """Computes lexical overlap ratio of >=4-char content words between `quote` and `transcript_lower`."""
    if not quote or not transcript_lower:
        return None
    sample_words = [
        w for w in re.findall(r'\b\w{4,}\b', quote.lower())
        if w not in _OVERLAP_STOPWORDS
    ]
    if not sample_words:
        return None
    matched_words = [w for w in sample_words if w in transcript_lower]
    return len(matched_words) / len(sample_words)


def _question_token_set(q: str) -> set:
    return {
        w for w in re.findall(r'\b[a-z]{4,}\b', (q or "").lower())
        if w not in _OVERLAP_STOPWORDS
    }


def _detect_opponent_bias(card: Dict[str, Any], a_turn: Optional[Dict[str, Any]]) -> Optional[str]:
    """Lazily invokes db.database.check_theological_bias (with extractor fallback) without import cycles."""
    try:
        from db.database import check_theological_bias
        return check_theological_bias(card, a_turn)
    except ImportError:
        try:
            from core.scenario_extractor import check_theological_bias
            return check_theological_bias(card, a_turn)
        except ImportError:
            return None


def audit_single_card(
    card: Dict[str, Any],
    turns: List[Dict[str, Any]],
    require_arabic: bool = True,
    video_meta: Optional[Dict[str, Any]] = None,
    sibling_questions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    card_id = card.get("id") or "UNKNOWN"
    vid = card.get("video_id") or "UNKNOWN"

    defects: List[Dict[str, str]] = []
    hard_fail: Optional[str] = None
    hard_fails_all: List[str] = []
    root_cause: Optional[str] = None
    recommendations: List[str] = []

    def _add_gate(gate_name: str, cause: str) -> None:
        nonlocal hard_fail, root_cause
        if gate_name not in hard_fails_all:
            hard_fails_all.append(gate_name)
        if hard_fail is None:
            hard_fail = gate_name
        if root_cause is None:
            root_cause = cause

    # Hydrate JSON-backed list fields cleanly (works for both DB rows and in-memory dicts)
    premises_raw = _parse_json_field(card, "logical_premises", "logical_premises_json")
    premises_ar_raw = _parse_json_field(card, "logical_premises_ar", "logical_premises_ar_json")
    analogies_raw = _parse_json_field(card, "analogies", "analogies_json")
    citations_raw = _parse_json_field(card, "scriptural_citations", "scriptural_citations_json")
    ex_turns_raw = card.get("dialogue_exchange") or card.get("dialogue_trajectory")
    if not ex_turns_raw:
        ex_turns_raw = _parse_json_field(card, "dialogue_exchange", "dialogue_exchange_json")

    # Build transcript indexes by speaker role
    transcript_en = " ".join((t.get("text_original") or "").lower() for t in turns)
    daee_transcript = " ".join(
        (t.get("text_original") or "").lower()
        for t in turns
        if _canon_role(t.get("speaker_role")) in ("MAIN_DAEE", "CO_DAEE")
    )
    inter_transcript = " ".join(
        (t.get("text_original") or "").lower()
        for t in turns
        if _canon_role(t.get("speaker_role")) == "INTERLOCUTOR"
    )
    turns_by_id: Dict[str, Dict[str, Any]] = {}
    turn_roles_by_id: Dict[str, str] = {}
    for t in turns:
        role = _canon_role(t.get("speaker_role"))
        tid = str(t.get("turn_id") or "").strip()
        if tid:
            turns_by_id[tid] = t
            turn_roles_by_id[tid] = role
        for mid in (t.get("merged_turn_ids") or []):
            m_str = str(mid or "").strip()
            if m_str:
                turns_by_id.setdefault(m_str, t)
                turn_roles_by_id.setdefault(m_str, role)

    # -------------------------------------------------------------
    # 1. HARD-FAIL GATE: [GATE-ARABIC] Missing Arabic Localization
    # -------------------------------------------------------------
    mandatory_ar_keys = [
        "question_ar",
        "core_answer_summary_ar",
        "fallacy_or_misconception_exposed_ar",
        "socratic_question_ar",
        "verbatim_quote_ar",
    ]
    if (card.get("analogy_or_metaphor") or "").strip() or (isinstance(analogies_raw, list) and analogies_raw):
        mandatory_ar_keys.append("analogy_or_metaphor_ar")
    if (card.get("interlocutor_verbatim_quote") or "").strip():
        mandatory_ar_keys.append("interlocutor_verbatim_quote_ar")

    missing_ar_keys = [k for k in mandatory_ar_keys if not (card.get(k) or "").strip()] if require_arabic else []
    if require_arabic and isinstance(premises_raw, list) and premises_raw and not (isinstance(premises_ar_raw, list) and premises_ar_raw):
        missing_ar_keys.append("logical_premises_ar")

    if require_arabic and missing_ar_keys:
        _add_gate("[GATE-ARABIC]", "Round 2 Arabic localization was omitted, failed, or partially empty.")
        defects.append({
            "code": "MISSING_ARABIC_FIELDS",
            "severity": "CRITICAL",
            "description": f"Missing required Arabic fields: {', '.join(missing_ar_keys)}",
        })
        recommendations.append("Execute Round 2 Arabic localization pipeline to populate all Arabic fields.")

    # -------------------------------------------------------------
    # 2. HARD-FAIL GATE: [GATE-LEAK] Prompt Few-Shot Demonstration Leakage
    # -------------------------------------------------------------
    citation_parts = [
        f"{c.get('reference', '')} {c.get('significance', '')}"
        for c in citations_raw if isinstance(c, dict)
    ]
    card_all_text = " ".join([
        card.get("question_en") or "",
        card.get("core_answer_summary_en") or "",
        card.get("analogy_or_metaphor") or "",
        card.get("socratic_question") or "",
        card.get("verbatim_quote") or "",
        card.get("interlocutor_verbatim_quote") or "",
        " ".join(str(x) for x in premises_raw) if isinstance(premises_raw, list) else str(premises_raw),
        " ".join(str(x) for x in analogies_raw) if isinstance(analogies_raw, list) else str(analogies_raw),
        " ".join(citation_parts),
    ]).lower()

    for leak_code, signature in PROMPT_LEAKAGE_SIGNATURES:
        if leak_code == "LEAK_MARK_13_32":
            if not re.search(r"\bmark\s*(?:chapter\s*)?13[\s:.,]+32\b", card_all_text):
                continue
            in_transcript = bool(re.search(r"\bmark\s*(?:chapter\s*)?13[\s:.,]+32\b", transcript_en))
        else:
            if signature not in card_all_text:
                continue
            in_transcript = signature in transcript_en
        if not in_transcript:
            _add_gate("[GATE-LEAK]", f"Model copied prompt demonstration example ('{signature}') not present in video transcript.")
            defects.append({
                "code": leak_code,
                "severity": "CRITICAL",
                "description": f"Prompt demonstration leakage: Found '{signature}' which does not appear in transcript.",
            })
            recommendations.append("Remove prompt few-shot demonstration text and ground quotes strictly in video turns.")

    # -------------------------------------------------------------
    # 3. HARD-FAIL GATE: [GATE-PERSPECTIVE] Confessional / Da'ee Counter-Ask / Cross-Exam
    # -------------------------------------------------------------
    q_en = (card.get("question_en") or "").strip()
    q_ar = (card.get("question_ar") or "").strip()
    if CONFESSIONAL_REGEX.search(q_en) or (q_ar and CONFESSIONAL_REGEX.search(q_ar)):
        _add_gate("[GATE-PERSPECTIVE]", "Question formulated as an awkward 1st-person confession rather than an objective seeker objection.")
        defects.append({
            "code": "CONFESSIONAL_INVERSION",
            "severity": "HIGH",
            "description": f"1st-person confession detected in question: '{q_en}'",
        })
        recommendations.append("Rephrase question_en into an objective seeker objection ending with '?'.")
    elif SOCRATIC_ROLE_INVERSION_REGEX.search(q_en):
        _add_gate("[GATE-PERSPECTIVE]", "Question is the Da'ee's polemical Socratic challenge rather than the seeker's objection.")
        defects.append({
            "code": "SOCRATIC_ROLE_INVERSION",
            "severity": "HIGH",
            "description": f"Da'ee Socratic cross-examination placed in question_en: '{q_en}'",
        })
        recommendations.append("Move Da'ee challenge to socratic_question and rewrite question_en as the seeker/interlocutor claim.")
    elif DAEE_CROSS_EXAM_REGEX.search(q_en):
        _add_gate("[GATE-PERSPECTIVE]", "Da'ee offensive cross-examination against opponent scripture/dogma was placed into question_en.")
        defects.append({
            "code": "DAEE_CROSS_EXAM_IN_QUESTION",
            "severity": "CRITICAL",
            "description": f"question_en expresses the Da'ee's critique of Christian/opponent scripture rather than the seeker's objection: '{q_en}'",
        })
        recommendations.append("Move the Da'ee's challenge into socratic_question and frame question_en around the opponent's theological claim being refuted.")

    if DAEE_COUNTER_ASK_REGEX.search(q_en) or (q_ar and DAEE_COUNTER_ASK_REGEX.search(q_ar)):
        _add_gate("[GATE-PERSPECTIVE]", "Question ends with Da'ee defensive comeback ('what is the problem with that?') instead of seeker objection.")
        defects.append({
            "code": "DAEE_COUNTER_ASK_PERSPECTIVE",
            "severity": "HIGH",
            "description": "Question ends with Da'ee defensive comeback ('what is the problem with that?' / 'فما المشكل في ذلك') instead of seeker objection.",
        })
        recommendations.append("Invert question_en/question_ar to the skeptic's objection (e.g., '..., isn't that a bad thing?').")

    if CONVERSATIONAL_1ST_2ND_PERSON_REGEX.search(q_en) or (q_ar and CONVERSATIONAL_1ST_2ND_PERSON_REGEX.search(q_ar)):
        _add_gate("[GATE-PERSPECTIVE]", "Question uses awkward 1st/2nd-person street conversation phrasing rather than standalone seeker inquiry.")
        defects.append({
            "code": "CONVERSATIONAL_1ST_2ND_PERSON_QUESTION",
            "severity": "HIGH",
            "description": f"1st/2nd-person conversational phrasing in question: '{q_en}'",
        })
        recommendations.append("Rewrite question_en/question_ar in objective 3rd-person inquiry syntax.")

    # -------------------------------------------------------------
    # 4. HARD-FAIL GATE: [GATE-NOISE] Conversational Noise / Outro Pitches / Chit-Chat
    # -------------------------------------------------------------
    q_lower = q_en.lower()
    ivq_lower = (card.get("interlocutor_verbatim_quote") or "").lower()
    vq_lower = (card.get("verbatim_quote") or "").lower()

    found_noise = [kw for kw in NOISE_KEYWORDS if (kw in q_lower or kw in ivq_lower or kw in vq_lower)]
    if found_noise:
        _add_gate("[GATE-NOISE]", "Extracted non-theological street banter, police scuffle, or promotional video outro.")
        defects.append({
            "code": "NON_THEOLOGICAL_NOISE",
            "severity": "CRITICAL",
            "description": f"Extracted non-apologetic content matching keywords: {', '.join(found_noise)}",
        })
        recommendations.append("Drop this card if it reflects street banter, police logistics, or video outro fundraising.")
    elif PERSONAL_CHITCHAT_REGEX.match(q_en):
        _add_gate("[GATE-NOISE]", "Extracted purely personal street chit-chat without theological argument.")
        defects.append({
            "code": "PERSONAL_CHITCHAT_NOISE",
            "severity": "CRITICAL",
            "description": f"Non-theological personal inquiry extracted: '{q_en}'",
        })
        recommendations.append("Filter out personal biography small talk in Stage 2 extraction.")

    # -------------------------------------------------------------
    # 5. HARD-FAIL GATE: [GATE-ROLE] Inverted Attribution / Swapped Quotes
    # -------------------------------------------------------------
    daee_id = card.get("daee_speaker_id") or ""
    inter_id = card.get("interlocutor_speaker_id") or ""
    if daee_id and inter_id and daee_id == inter_id:
        _add_gate("[GATE-ROLE]", "Da'ee and interlocutor speaker IDs are identical, collapsing dialogue into a single speaker.")
        defects.append({
            "code": "SPEAKER_COLLAPSE",
            "severity": "HIGH",
            "description": f"Speaker ID '{daee_id}' assigned to both Da'ee and Interlocutor.",
        })
        recommendations.append("Verify answer_turn_id belongs to MAIN_DAEE/CO_DAEE and question_turn_id to the interlocutor.")

    a_tid = str(card.get("answer_turn_id") or "").strip()
    if not require_arabic and turn_roles_by_id:
        if not a_tid:
            _add_gate("[GATE-ROLE]", "Missing answer_turn_id before Round 2 verification.")
            defects.append({
                "code": "MISSING_ANSWER_TURN_ID",
                "severity": "CRITICAL",
                "description": "answer_turn_id is empty and cannot be anchored to a MAIN_DAEE turn.",
            })
            recommendations.append("Anchor answer_turn_id to the substantive MAIN_DAEE response turn in the transcript.")
        elif a_tid not in turn_roles_by_id:
            _add_gate("[GATE-ROLE]", f"answer_turn_id '{a_tid}' does not exist in window transcript turns.")
            defects.append({
                "code": "HALLUCINATED_ANSWER_TURN_ID",
                "severity": "CRITICAL",
                "description": f"answer_turn_id '{a_tid}' not found in scenario turns.",
            })
            recommendations.append("Replace answer_turn_id with an exact MAIN_DAEE turn_id from the transcript.")
    if a_tid and a_tid in turn_roles_by_id and turn_roles_by_id[a_tid] not in ("MAIN_DAEE", "CO_DAEE"):
        _add_gate("[GATE-ROLE]", f"answer_turn_id '{a_tid}' points to non-Da'ee role ({turn_roles_by_id[a_tid]}).")
        defects.append({
            "code": "ANSWER_TURN_NOT_DAEE",
            "severity": "CRITICAL",
            "description": f"answer_turn_id '{a_tid}' is spoken by '{turn_roles_by_id[a_tid]}', not MAIN_DAEE/CO_DAEE.",
        })
        recommendations.append("Re-anchor answer_turn_id to the MAIN_DAEE turn delivering the core refutation.")

    bias_reason = _detect_opponent_bias(card, turns_by_id.get(a_tid))
    if bias_reason:
        _add_gate("[GATE-ROLE]", f"Opponent theological position attributed to Da'ee: {bias_reason}")
        defects.append({
            "code": "OPPONENT_BIAS_ATTRIBUTION",
            "severity": "CRITICAL",
            "description": f"Potential opponent bias attributed as Islamic proof: {bias_reason}",
        })
        recommendations.append("Separate interlocutor claims from the Da'ee's authentic Islamic refutation.")

    # Quote speaker-role swap detection (when both MAIN_DAEE and INTERLOCUTOR turns exist)
    daee_quote = (card.get("verbatim_quote") or "").strip()
    inter_quote = (card.get("interlocutor_verbatim_quote") or "").strip()
    if daee_transcript and inter_transcript:
        if daee_quote:
            vq_d_ov = _quote_transcript_overlap(daee_quote, daee_transcript) or 0.0
            vq_i_ov = _quote_transcript_overlap(daee_quote, inter_transcript) or 0.0
            if vq_i_ov >= 0.60 and vq_d_ov < 0.30 and (vq_i_ov - vq_d_ov) >= 0.30:
                _add_gate("[GATE-ROLE]", "verbatim_quote belongs to INTERLOCUTOR turns rather than MAIN_DAEE proof turns.")
                defects.append({
                    "code": "QUOTE_ROLE_SWAPPED_DAEE",
                    "severity": "CRITICAL",
                    "description": f"verbatim_quote matches INTERLOCUTOR speech ({vq_i_ov:.0%} overlap) instead of MAIN_DAEE ({vq_d_ov:.0%}).",
                })
                recommendations.append("Replace verbatim_quote with an authentic MAIN_DAEE proof turn.")
        if inter_quote:
            ivq_d_ov = _quote_transcript_overlap(inter_quote, daee_transcript) or 0.0
            ivq_i_ov = _quote_transcript_overlap(inter_quote, inter_transcript) or 0.0
            if ivq_d_ov >= 0.60 and ivq_i_ov < 0.30 and (ivq_d_ov - ivq_i_ov) >= 0.30:
                _add_gate("[GATE-ROLE]", "interlocutor_verbatim_quote belongs to MAIN_DAEE turns rather than INTERLOCUTOR turns.")
                defects.append({
                    "code": "QUOTE_ROLE_SWAPPED_INTERLOCUTOR",
                    "severity": "CRITICAL",
                    "description": f"interlocutor_verbatim_quote matches MAIN_DAEE speech ({ivq_d_ov:.0%} overlap) instead of INTERLOCUTOR ({ivq_i_ov:.0%}).",
                })
                recommendations.append("Replace interlocutor_verbatim_quote with the actual INTERLOCUTOR turn voicing the objection.")

    # -------------------------------------------------------------
    # 6. HARD-FAIL GATE: [GATE-DIARIZATION] Stage 1 Single-Speaker or Functional Collapse
    # -------------------------------------------------------------
    if video_meta and int(video_meta.get("speakers_count") or 0) == 1 and float(video_meta.get("duration_sec") or 0) > 180.0:
        _add_gate(
            "[GATE-DIARIZATION]",
            f"Video {vid} ({video_meta.get('duration_sec')}s) collapsed into speakers_count=1 (SPK_00 only) in Stage 1 M3 diarization.",
        )
        defects.append({
            "code": "SINGLE_SPEAKER_DIARIZATION_COLLAPSE",
            "severity": "CRITICAL",
            "description": f"Video {vid} has speakers_count=1 across {video_meta.get('turns_count')} turns, fusing Da'ee and interlocutor into SPK_00.",
        })
        recommendations.append("Re-run Stage 1 M3 diarization with minimum-speaker separation guard for multi-speaker debates.")
    elif video_meta and float(video_meta.get("duration_sec") or 0) > 180.0 and turns:
        total_words = sum(len((t.get("text_original") or "").split()) for t in turns)
        inter_words = sum(
            len((t.get("text_original") or "").split())
            for t in turns
            if _canon_role(t.get("speaker_role")) == "INTERLOCUTOR"
        )
        if total_words >= 200 and (inter_words < 40 or (inter_words / total_words) < 0.035):
            _add_gate(
                "[GATE-DIARIZATION]",
                f"Video {vid} ({video_meta.get('duration_sec')}s) functionally collapsed INTERLOCUTOR turns ({inter_words}/{total_words} words = {(inter_words / total_words):.1%}).",
            )
            defects.append({
                "code": "FUNCTIONAL_DIARIZATION_COLLAPSE",
                "severity": "CRITICAL",
                "description": f"Video {vid} INTERLOCUTOR speech is only {inter_words}/{total_words} words ({(inter_words / total_words):.1%}), indicating fused speakers in M3.",
            })
            recommendations.append("Re-run Stage 1 M3 diarization to separate the interlocutor voice from MAIN_DAEE.")

    # -------------------------------------------------------------
    # DIMENSIONAL SCORING (20 + 25 + 20 + 20 + 15 = 100)
    # -------------------------------------------------------------
    dim_scores = {
        "question_formulation": 20,
        "grounding_fidelity": 25,
        "dialectical_depth": 20,
        "bilingual_symmetry": 20 if require_arabic else 0,
        "attribution_timestamps": 15,
    }

    # Dim 1: Question Formulation (Max 20)
    if not q_en.endswith("?"):
        dim_scores["question_formulation"] -= 4
        defects.append({"code": "SYNTAX_NO_QUESTION_MARK", "severity": "MEDIUM", "description": "English question does not end with '?'"})
        recommendations.append("Rewrite question_en as a grammatical interrogative ending with '?'.")
    if q_en and ("..." in q_en or "…" in q_en):
        dim_scores["question_formulation"] -= 4
        defects.append({"code": "FRAGMENTED_QUESTION", "severity": "HIGH", "description": "English question contains incomplete ellipsis fragment."})
        recommendations.append("Rewrite question_en as a complete standalone question without '...'.")
    if q_en and TRAILING_FILLER_REGEX.search(q_en):
        dim_scores["question_formulation"] -= 4
        defects.append({"code": "TRAILING_ASR_FILLER", "severity": "MEDIUM", "description": f"Conversational filler at end of question: '{q_en}'"})
        recommendations.append("Remove trailing conversational filler ('Okay?') and polish into formal seeker inquiry.")
    if DAEE_COUNTER_ASK_REGEX.search(q_en) or (q_ar and DAEE_COUNTER_ASK_REGEX.search(q_ar)):
        dim_scores["question_formulation"] -= 8
    if require_arabic and q_ar and not q_ar.endswith("؟"):
        dim_scores["question_formulation"] -= 2
        defects.append({"code": "SYNTAX_NO_ARABIC_QUESTION_MARK", "severity": "LOW", "description": "Arabic question does not end with '؟'"})
    if len(q_en.split()) < 4:
        dim_scores["question_formulation"] -= 3
        defects.append({"code": "QUESTION_TOO_SHORT", "severity": "MEDIUM", "description": "Question is too short / conversational fragment."})

    # Intra-video duplicate question check
    if sibling_questions:
        q_toks = _question_token_set(q_en)
        for sib_q in sibling_questions:
            if sib_q == q_en:
                continue
            s_toks = _question_token_set(sib_q)
            if q_toks and s_toks:
                jacc = len(q_toks & s_toks) / len(q_toks | s_toks)
                if jacc >= 0.48:
                    dim_scores["question_formulation"] -= 4
                    defects.append({
                        "code": "DUPLICATE_VIDEO_QUESTION",
                        "severity": "MEDIUM",
                        "description": f"Near-duplicate question in same video ({jacc:.0%} token overlap with '{sib_q[:60]}').",
                    })
                    recommendations.append("Deduplicate overlapping intra-video cards or differentiate each card's distinct sub-argument.")
                    break

    # Dim 2: Grounding Fidelity (Max 25)
    if not daee_quote:
        dim_scores["grounding_fidelity"] -= 5
        defects.append({"code": "MISSING_DAEE_QUOTE", "severity": "MEDIUM", "description": "Missing verbatim_quote from transcript."})
        recommendations.append("Copy an exact word-for-word MAIN_DAEE proof quote from the transcript into verbatim_quote.")
    else:
        ratio = _quote_transcript_overlap(daee_quote, transcript_en)
        if ratio is not None and ratio < 0.3:
            dim_scores["grounding_fidelity"] -= 8
            defects.append({"code": "QUOTE_UNGROUNDED_DAEE", "severity": "HIGH", "description": f"Da'ee verbatim quote appears ungrounded (only {ratio:.0%} lexical overlap with transcript)."})
            recommendations.append("Replace verbatim_quote with an exact spoken quote from the Da'ee in the transcript.")

    if not inter_quote:
        dim_scores["grounding_fidelity"] -= 2
        defects.append({"code": "MISSING_INTERLOCUTOR_QUOTE", "severity": "LOW", "description": "Missing interlocutor_verbatim_quote from transcript."})
        recommendations.append("Copy an exact spoken question/objection quote from the interlocutor into interlocutor_verbatim_quote.")
    else:
        ratio_i = _quote_transcript_overlap(inter_quote, transcript_en)
        if ratio_i is not None and ratio_i < 0.3:
            dim_scores["grounding_fidelity"] -= 5
            defects.append({"code": "QUOTE_UNGROUNDED_INTERLOCUTOR", "severity": "HIGH", "description": f"Interlocutor verbatim quote appears ungrounded (only {ratio_i:.0%} lexical overlap with transcript)."})
            recommendations.append("Replace interlocutor_verbatim_quote with an exact spoken quote from the interlocutor.")

    ex_turns = ex_turns_raw if isinstance(ex_turns_raw, list) else []
    min_ex_turns = 4 if require_arabic else 2
    if len(ex_turns) < min_ex_turns:
        dim_scores["grounding_fidelity"] -= 4
        defects.append({
            "code": "THIN_DIALOGUE_EXCHANGE",
            "severity": "MEDIUM",
            "description": f"dialogue_exchange has {len(ex_turns)} turn(s) (< {min_ex_turns} target).",
        })
    if ex_turns:
        if transcript_en:
            for ex in ex_turns:
                if not isinstance(ex, dict):
                    continue
                txt_en = (ex.get("text_en") or "").strip()
                sample_tokens = [w for w in re.findall(r'\b\w{4,}\b', txt_en.lower()) if w not in _OVERLAP_STOPWORDS]
                if len(sample_tokens) >= 4:
                    t_ratio = _quote_transcript_overlap(txt_en, transcript_en)
                    if t_ratio is not None and t_ratio < 0.15:
                        dim_scores["grounding_fidelity"] -= 4
                        defects.append({"code": "UNGROUNDED_DIALOGUE_TURN", "severity": "HIGH", "description": f"Dialogue turn '{ex.get('turn_id', '?')}' appears ungrounded against transcript ({t_ratio:.0%} overlap)."})
                        recommendations.append("Ground every dialogue_exchange turn in actual transcript utterances.")
                        break
        if turns_by_id:
            for ex in ex_turns:
                if not isinstance(ex, dict):
                    continue
                ex_tid = str(ex.get("turn_id") or "").strip()
                if ex_tid in turns_by_id:
                    txt_en = (ex.get("text_en") or "").strip()
                    turn_txt = (turns_by_id[ex_tid].get("text_original") or "").lower()
                    sample_tokens = [w for w in re.findall(r'\b\w{4,}\b', txt_en.lower()) if w not in _OVERLAP_STOPWORDS]
                    if len(sample_tokens) >= 4:
                        turn_ratio = _quote_transcript_overlap(txt_en, turn_txt)
                        if turn_ratio is not None and turn_ratio < 0.35:
                            dim_scores["grounding_fidelity"] -= 5
                            defects.append({
                                "code": "DIALOGUE_TURN_TEXT_SHIFTED",
                                "severity": "HIGH",
                                "description": f"Dialogue turn '{ex_tid}' text_en matches its ground-truth scenario_turn at only {turn_ratio:.0%}.",
                            })
                            recommendations.append("Populate dialogue_exchange[i].text_en directly from turns_by_id[turn_id]['text_original'].")
                            break
        prev_ts: Optional[float] = None
        for ex in ex_turns:
            if not isinstance(ex, dict):
                continue
            ex_tid = str(ex.get("turn_id") or "").strip()
            ts_val = ex.get("timestamp_seconds")
            if ts_val is None and ex_tid in turns_by_id and turns_by_id[ex_tid].get("start_ms") is not None:
                ts_val = turns_by_id[ex_tid].get("start_ms") / 1000.0
            if ts_val is not None:
                try:
                    ts_f = float(ts_val)
                except (TypeError, ValueError):
                    continue
                if prev_ts is not None and ts_f < prev_ts - 5.0:
                    dim_scores["attribution_timestamps"] -= 3
                    defects.append({
                        "code": "NON_MONOTONIC_DIALOGUE_TIMESTAMPS",
                        "severity": "MEDIUM",
                        "description": f"dialogue_exchange turn '{ex_tid or '?'}' regresses chronologically ({ts_f:.1f}s < {prev_ts:.1f}s).",
                    })
                    recommendations.append("Sort dialogue_exchange turns in chronological timestamp order.")
                    break
                prev_ts = ts_f
        if turn_roles_by_id:
            for ex in ex_turns:
                if not isinstance(ex, dict):
                    continue
                ex_tid = str(ex.get("turn_id") or "").strip()
                claimed_role = _canon_role(ex.get("role"))
                actual_role = turn_roles_by_id.get(ex_tid) if ex_tid else None
                if actual_role and claimed_role and (
                    (actual_role == "INTERLOCUTOR" and claimed_role in ("MAIN_DAEE", "CO_DAEE"))
                    or (actual_role in ("MAIN_DAEE", "CO_DAEE") and claimed_role == "INTERLOCUTOR")
                ):
                    dim_scores["attribution_timestamps"] -= 5
                    defects.append({
                        "code": "DIALOGUE_TURN_ROLE_MISMATCH",
                        "severity": "HIGH",
                        "description": f"Turn '{ex_tid}' labeled '{claimed_role}' is actually '{actual_role}' in diarized transcript.",
                    })
                    recommendations.append("Correct swapped speaker roles in dialogue_exchange against diarized turn metadata.")
                    break
        if require_arabic and len(ex_turns) >= 2:
            roles_seen = {str(ex.get("role") or "").upper() for ex in ex_turns if isinstance(ex, dict)}
            if len(roles_seen) < 2 and inter_transcript:
                dim_scores["grounding_fidelity"] -= 3
                defects.append({
                    "code": "SINGLE_ROLE_DIALOGUE_EXCHANGE",
                    "severity": "MEDIUM",
                    "description": "dialogue_exchange collapsed into a single speaker role despite multi-speaker turns.",
                })

    # Topical Grounding Check: [GATE-UNGROUNDED]
    if len(turns) >= 4:
        card_ts = card.get("timestamp_seconds")
        anchor_ts_ms: Optional[int] = None
        if card_ts is not None:
            try:
                anchor_ts_ms = int(float(card_ts) * 1000)
            except (TypeError, ValueError):
                anchor_ts_ms = None
        if anchor_ts_ms is None and a_tid in turns_by_id:
            anchor_ts_ms = int(turns_by_id[a_tid].get("start_ms", 0))
        if anchor_ts_ms is not None:
            local_text = " ".join(
                (t.get("text_original") or "")
                for t in turns
                if int(t.get("end_ms", 0)) >= anchor_ts_ms - 60_000 and int(t.get("start_ms", 0)) <= anchor_ts_ms + 120_000
            ).lower()
            local_stems = {w[:5] for w in re.findall(r"[a-z]{5,}", local_text)}
            if len(local_stems) >= 25:
                broad_stopwords = {
                    "islam", "islamic", "muslim", "muslims", "quran", "quranic", "bible", "biblical",
                    "christian", "christians", "christianity", "allah", "jesus", "christ", "prophet",
                    "believe", "belief", "people", "claim", "claims", "objection", "because", "through",
                    "their", "there", "which", "about", "would", "could", "should", "against", "without",
                    "human", "world", "truth", "proof", "divine", "moral", "rational", "logic", "logical",
                }
                card_stems = {
                    w[:5]
                    for w in re.findall(r"[a-z]{5,}", f"{q_en} {card.get('core_answer_summary_en') or ''}".lower())
                    if w not in broad_stopwords
                }
                if len(card_stems) >= 6:
                    topic_ratio = len(card_stems & local_stems) / len(card_stems)
                    if topic_ratio < 0.09:
                        dim_scores["grounding_fidelity"] -= 12
                        _add_gate(
                            "[GATE-UNGROUNDED]",
                            f"Dossier '{card_id}' exhibits severe topical drift from its local transcript window ({len(card_stems & local_stems)}/{len(card_stems)} stems = {topic_ratio:.1%}).",
                        )
                        defects.append({
                            "code": "UNGROUNDED_DOSSIER_SUMMARY",
                            "severity": "CRITICAL",
                            "description": f"Question/summary shares only {topic_ratio:.1%} topical stems with local transcript window around {anchor_ts_ms/1000:.0f}s.",
                        })
                        recommendations.append("Drop or regenerate dossier strictly from its local scenario_turns window.")

    # Dim 3: Dialectical Depth (Max 20)
    premises_en = premises_raw if isinstance(premises_raw, list) else []
    if len(premises_en) < 2:
        dim_scores["dialectical_depth"] -= 6
        defects.append({"code": "INCOMPLETE_SYLLOGISM", "severity": "HIGH", "description": "Fewer than 2 logical premises in deductive syllogism."})
        recommendations.append("Provide a formal 3-step deductive syllogism (Premise 1, Premise 2, Conclusion) grounded in the debate.")
    elif len(premises_en) == 2:
        dim_scores["dialectical_depth"] -= 3
        defects.append({"code": "SYLLOGISM_UNDER_3_PREMISES", "severity": "LOW", "description": "Only 2 premises; rubric recommends 3 deductive steps (Major, Minor, Conclusion)."})

    sum_en = (card.get("core_answer_summary_en") or "").strip()
    if META_NARRATOR_REGEX.search(sum_en):
        dim_scores["dialectical_depth"] -= 4
        defects.append({"code": "META_NARRATOR_SUMMARY", "severity": "MEDIUM", "description": "Core summary uses 3rd-person meta-narrative instead of stating direct Islamic truth."})
        recommendations.append("Rewrite core_answer_summary_en directly without 3rd-person phrases like 'The Da'ee explains...'.")
    if len(sum_en.split()) < 60:
        dim_scores["dialectical_depth"] -= 3
        defects.append({"code": "SHALLOW_ANSWER_SUMMARY", "severity": "MEDIUM", "description": f"core_answer_summary_en is too brief ({len(sum_en.split())} words vs 150-250 target)."})
        recommendations.append("Expand core_answer_summary_en with the full rational and scriptural proof spoken in the dialogue.")

    if not (card.get("fallacy_or_misconception_exposed") or "").strip():
        dim_scores["dialectical_depth"] -= 3
        defects.append({"code": "MISSING_FALLACY", "severity": "LOW", "description": "fallacy_or_misconception_exposed is empty."})

    soc_q = (card.get("socratic_question") or "").strip()
    if not soc_q:
        dim_scores["dialectical_depth"] -= 3
        defects.append({"code": "MISSING_SOCRATIC_QUESTION", "severity": "LOW", "description": "socratic_question is empty."})
    elif q_en and soc_q.lower().rstrip("?") == q_en.lower().rstrip("?"):
        dim_scores["dialectical_depth"] -= 3
        defects.append({"code": "DUPLICATE_SOCRATIC_QUESTION", "severity": "MEDIUM", "description": "socratic_question merely repeats question_en."})

    # Dim 4: Bilingual Symmetry (Max 20) — evaluated when require_arabic=True
    if require_arabic:
        if missing_ar_keys:
            dim_scores["bilingual_symmetry"] = 0
        else:
            ar_check_keys = [
                "question_ar", "core_answer_summary_ar", "fallacy_or_misconception_exposed_ar",
                "analogy_or_metaphor_ar", "socratic_question_ar", "verbatim_quote_ar", "interlocutor_verbatim_quote_ar",
            ]
            for k in ar_check_keys:
                val = card.get(k) or ""
                if CYRILLIC_REGEX.search(val) or HEBREW_REGEX.search(val) or CJK_REGEX.search(val):
                    dim_scores["bilingual_symmetry"] -= 5
                    defects.append({"code": "FOREIGN_SCRIPT_LEAKAGE", "severity": "HIGH", "description": f"Non-Arabic script (Cyrillic/Hebrew/CJK) detected in '{k}'"})
                    break
                if LATIN_IN_ARABIC_REGEX.search(val):
                    dim_scores["bilingual_symmetry"] -= 3
                    defects.append({"code": "LATIN_LEAKAGE_IN_ARABIC", "severity": "MEDIUM", "description": f"Untranslated Latin words detected in '{k}'"})
                    break
            for k in ar_check_keys:
                val = card.get(k) or ""
                if _DANGLING_PROCLITIC_RE.search(val):
                    dim_scores["bilingual_symmetry"] -= 4
                    defects.append({
                        "code": "CORRUPTED_ARABIC_PREPOSITION",
                        "severity": "HIGH",
                        "description": f"Dangling Arabic proclitic ('لل.' / 'لل ') detected in '{k}' from stripped Latin word.",
                    })
                    recommendations.append("Re-translate or clean dangling Arabic proclitics after removing Latin tokens.")
                    break
            ivq_ar = (card.get("interlocutor_verbatim_quote_ar") or "").strip()
            if (
                len(ivq_ar.split()) >= 6
                and q_ar
                and (ivq_ar == q_ar or ivq_ar in q_ar or q_ar in ivq_ar)
                and inter_quote
                and q_en
            ):
                _dup_stop = {
                    "that", "this", "what", "when", "where", "which", "does", "only",
                    "those", "with", "from", "they", "their", "have", "about", "would",
                    "could", "should", "then",
                }
                q_stems = {w[:5] for w in re.findall(r"[a-z]{4,}", q_en.lower()) if w not in _dup_stop}
                ivq_stems = {w[:5] for w in re.findall(r"[a-z]{4,}", inter_quote.lower()) if w not in _dup_stop}
                if q_stems and ivq_stems and (len(q_stems & ivq_stems) / min(len(q_stems), len(ivq_stems))) < 0.50:
                    dim_scores["bilingual_symmetry"] -= 4
                    defects.append({
                        "code": "DUPLICATE_ARABIC_QUOTE_QUESTION",
                        "severity": "HIGH",
                        "description": "interlocutor_verbatim_quote_ar copied question_ar instead of translating interlocutor_verbatim_quote.",
                    })
                    recommendations.append("Translate interlocutor_verbatim_quote faithfully into interlocutor_verbatim_quote_ar.")
            sum_ar = (card.get("core_answer_summary_ar") or "").strip()
            if len(sum_en.split()) >= 90 and len(sum_ar.split()) < max(35, int(len(sum_en.split()) * 0.36)):
                dim_scores["bilingual_symmetry"] -= 5
                defects.append({
                    "code": "TRUNCATED_ARABIC_SUMMARY",
                    "severity": "HIGH",
                    "description": f"core_answer_summary_ar ({len(sum_ar.split())} words) is severely truncated compared to core_answer_summary_en ({len(sum_en.split())} words).",
                })
                recommendations.append("Expand core_answer_summary_ar to match the full proof detail of core_answer_summary_en.")
            if any(not (isinstance(ex, dict) and (ex.get("text_ar") or "").strip()) for ex in ex_turns):
                dim_scores["bilingual_symmetry"] -= 3
                defects.append({"code": "MISSING_DIALOGUE_TEXT_AR", "severity": "MEDIUM", "description": "One or more dialogue_exchange turns lack text_ar."})

    # Dim 5: Attribution & Timestamps (Max 15)
    ts_sec = card.get("timestamp_seconds")
    q_tid = str(card.get("question_turn_id") or "").strip()
    has_valid_turn_anchor = bool(q_tid and (not turn_roles_by_id or q_tid in turn_roles_by_id))
    if not require_arabic and turn_roles_by_id and not has_valid_turn_anchor:
        dim_scores["attribution_timestamps"] -= 5
        defects.append({"code": "UNGROUNDED_QUESTION_TURN_ID", "severity": "HIGH", "description": f"question_turn_id '{q_tid}' not found in scenario turns."})
        recommendations.append("Anchor question_turn_id to an exact turn_id in the transcript.")
    if (ts_sec is None or ts_sec < 0) and (require_arabic or not has_valid_turn_anchor):
        dim_scores["attribution_timestamps"] -= 5
        defects.append({"code": "INVALID_TIMESTAMP", "severity": "MEDIUM", "description": "Missing or negative timestamp seconds."})
    if require_arabic and not inter_id and inter_transcript:
        dim_scores["attribution_timestamps"] -= 3
        defects.append({"code": "MISSING_INTERLOCUTOR_SPEAKER_ID", "severity": "MEDIUM", "description": "interlocutor_speaker_id is empty despite INTERLOCUTOR turns existing in video."})

    # Clamp scores
    for d in dim_scores:
        dim_scores[d] = max(0, dim_scores[d])

    raw_score = sum(dim_scores.values())
    max_score = 100 if require_arabic else 80
    score_pct = int(round((raw_score / max_score) * 100)) if max_score else 0

    if hard_fail:
        raw_score = 0
        score_pct = 0
        status = "FAIL"
        tier = "Tier 4: Critical Failure / Rejected (Red)"
    else:
        status = "PASS" if score_pct >= 75 else "FAIL"
        if score_pct >= 90:
            tier = "Tier 1: Production Ready (Gold)"
        elif score_pct >= 75:
            tier = "Tier 2: Minor Polish Needed (Silver)"
        elif score_pct >= 50:
            tier = "Tier 3: Substandard / Requires Regeneration (Bronze)"
        else:
            tier = "Tier 4: Critical Failure / Rejected (Red)"

    needs_english_repair = bool(
        hard_fail
        or score_pct < 85
        or any(d.get("severity") in ("HIGH", "CRITICAL") for d in defects)
    )

    return {
        "card_id": card_id,
        "video_id": vid,
        "total_score": raw_score if require_arabic else score_pct,
        "raw_score": raw_score,
        "max_score": max_score,
        "score_pct": score_pct,
        "status": status,
        "tier": tier,
        "hard_fail_triggered": hard_fail,
        "hard_fails_all": hard_fails_all,
        "needs_english_repair": needs_english_repair,
        "dimension_scores": dim_scores,
        "defect_count": len(defects),
        "defects": defects,
        "root_cause": root_cause,
        "recommendations": recommendations,
    }


def summarize_pre_round2_batch(
    r1_units: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> Tuple[str, bool]:
    """
    Runs the Card Rating System pre-check on Round 1 English dossiers before Round 2 runs.
    Returns `(summary_text, batch_needs_repair)` where `batch_needs_repair` is False iff
    every card in `r1_units` is Status=CLEAN so Round 2 can run fast `/no_think` localization.
    """
    if not r1_units:
        return "", False

    lines = ["PRE-ROUND-2 CARD RATING & DEFECT DIAGNOSTICS (tools/card_rating_system):"]
    batch_needs_repair = False
    for idx, u in enumerate(r1_units, 1):
        ev = audit_single_card(u, turns, require_arabic=False)
        q_tid = u.get("question_turn_id", "")
        a_tid = u.get("answer_turn_id", "")
        if not ev["needs_english_repair"] and not ev["defects"]:
            lines.append(
                f"- Card {idx} ({q_tid} -> {a_tid}): Score={ev['score_pct']}/100 ({ev['tier']}) | "
                "Status=CLEAN -> Keep ALL Round 1 English fields UNCHANGED (set english_edits_needed=false) and focus on Classical Arabic localization."
            )
        else:
            batch_needs_repair = True
            status_tag = ev["hard_fail_triggered"] or ("NEEDS_REPAIR" if ev["needs_english_repair"] else "MINOR_POLISH")
            codes_hint = ", ".join(d["code"] for d in ev["defects"][:3])
            defect_strs = [f"[{d['code']}] {d['description']}" for d in ev["defects"][:4]]
            rec_strs = ev["recommendations"][:3]
            lines.append(
                f"- Card {idx} ({q_tid} -> {a_tid}): Score={ev['score_pct']}/100 | Status={status_tag}"
                + (f" [{codes_hint}]" if codes_hint else "")
            )
            if defect_strs:
                lines.append(f"  Defects: {'; '.join(defect_strs)}")
            if rec_strs:
                lines.append(f"  Targeted Fix: {' | '.join(rec_strs)}")
    return "\n".join(lines), batch_needs_repair


def format_pre_round2_rubric_summary(
    r1_units: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> str:
    """Wrapper returning the formatted pre-Round-2 rubric summary text."""
    summary_text, _ = summarize_pre_round2_batch(r1_units, turns)
    return summary_text


def run_audit(db_path: Path, video_filter: Optional[str] = None, fail_only: bool = False) -> Dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if video_filter:
        cursor.execute("SELECT * FROM qa_units WHERE video_id = ? ORDER BY rowid ASC", (video_filter,))
    else:
        cursor.execute("SELECT * FROM qa_units ORDER BY rowid ASC")
    card_rows = [dict(r) for r in cursor.fetchall()]

    print(f"[*] Loaded {len(card_rows)} cards from {db_path.name}")

    # Pre-cache video metadata, full video turns, and sibling questions per video_id
    turns_cache: Dict[str, List[Dict[str, Any]]] = {}
    video_cache: Dict[str, Dict[str, Any]] = {}
    siblings_cache: Dict[str, List[str]] = {}
    for c in card_rows:
        vid = c.get("video_id") or ""
        siblings_cache.setdefault(vid, []).append((c.get("question_en") or "").strip())

    results = []
    hard_fails_breakdown: Dict[str, int] = {}
    tier_counts = {
        "Tier 1: Production Ready (Gold)": 0,
        "Tier 2: Minor Polish Needed (Silver)": 0,
        "Tier 3: Substandard / Requires Regeneration (Bronze)": 0,
        "Tier 4: Critical Failure / Rejected (Red)": 0,
    }

    all_evals = []
    for card in card_rows:
        vid = card.get("video_id") or ""
        if vid not in video_cache:
            vrow = cursor.execute(
                "SELECT video_id, speakers_count, turns_count, duration_sec, title_ar FROM videos WHERE video_id = ?",
                (vid,),
            ).fetchone()
            video_cache[vid] = dict(vrow) if vrow else {}
        if vid not in turns_cache:
            trows = cursor.execute(
                """
                SELECT turn_id, speaker_role, speaker_label_en, speaker_label_ar,
                       start_ms, end_ms, start_fmt, end_fmt, text_original, text_ar
                FROM scenario_turns
                WHERE video_id = ?
                ORDER BY turn_index ASC
                """,
                (vid,),
            ).fetchall()
            turns_cache[vid] = [dict(t) for t in trows]

        evaluation = audit_single_card(
            card,
            turns_cache[vid],
            require_arabic=True,
            video_meta=video_cache[vid],
            sibling_questions=siblings_cache.get(vid, []),
        )
        all_evals.append(evaluation)

        tier_counts[evaluation["tier"]] = tier_counts.get(evaluation["tier"], 0) + 1
        if evaluation["hard_fail_triggered"]:
            hf = evaluation["hard_fail_triggered"]
            hard_fails_breakdown[hf] = hard_fails_breakdown.get(hf, 0) + 1

        if not fail_only or evaluation["status"] == "FAIL":
            results.append(evaluation)

    conn.close()

    total_evaluated = len(card_rows)
    passed_cards = [r for r in all_evals if r["status"] == "PASS"]
    failed_cards = [r for r in all_evals if r["status"] == "FAIL"]

    mean_all = sum(r["total_score"] for r in all_evals) / total_evaluated if total_evaluated else 0
    mean_passed = sum(r["total_score"] for r in passed_cards) / len(passed_cards) if passed_cards else 0

    synthesis = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "total_cards": total_evaluated,
            "passed_count": len(passed_cards),
            "failed_count": len(failed_cards),
            "pass_rate_pct": round(len(passed_cards) / total_evaluated * 100, 2) if total_evaluated else 0,
            "mean_score_all": round(mean_all, 2),
            "mean_score_passed": round(mean_passed, 2),
        },
        "quality_tiers": tier_counts,
        "hard_fail_gates": hard_fails_breakdown,
        "cards": results,
    }

    return synthesis


def main():
    parser = argparse.ArgumentParser(description="Universal Apologetics Card Audit & Rating Engine")
    parser.add_argument("--db", default="data/dawah_knowledge.db", help="Path to SQLite knowledge database")
    parser.add_argument("--out", default=None, help="Output JSON report file path")
    parser.add_argument("--video-id", default=None, help="Filter audit to a specific video ID")
    parser.add_argument("--fail-only", action="store_true", help="Include only failing cards in report")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-card audit results")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    synthesis = run_audit(db_path, video_filter=args.video_id, fail_only=args.fail_only)

    meta = synthesis["metadata"]
    tiers = synthesis["quality_tiers"]
    gates = synthesis["hard_fail_gates"]

    print("\n" + "=" * 70)
    print("      DAWAH DIALECTICS (YT-SPX) — APOLOGETICS CARD AUDIT SUMMARY")
    print("=" * 70)
    print(f" Database: {db_path.name} | Total Cards: {meta['total_cards']}")
    print(f" Pass Rate: {meta['pass_rate_pct']}% ({meta['passed_count']} Passed, {meta['failed_count']} Failed)")
    print(f" Mean Score (All Cards): {meta['mean_score_all']} / 100")
    print(f" Mean Score (Passed):    {meta['mean_score_passed']} / 100")
    print("-" * 70)
    print(" Quality Tiers:")
    for tier_name, cnt in tiers.items():
        pct = (cnt / meta['total_cards'] * 100) if meta['total_cards'] else 0
        print(f"   • {tier_name:<46}: {cnt:>2} ({pct:>5.1f}%)")
    print("-" * 70)
    print(" Hard-Fail Gates Breakdown:")
    for gate_name, cnt in gates.items():
        print(f"   • {gate_name:<30}: {cnt:>2} cards")
    print("=" * 70 + "\n")

    if args.verbose:
        for c in synthesis["cards"]:
            status_symbol = "[+]" if c["status"] == "PASS" else "[-]"
            gate_str = f" [{', '.join(c['hard_fails_all'])}]" if c.get("hard_fails_all") else ""
            print(f"{status_symbol} Card: {c['card_id']:<20} | Score: {c['total_score']:>3} | {c['tier']}{gate_str}")
            if c["defects"]:
                for d in c["defects"]:
                    print(f"    - [{d['severity']}] {d['code']}: {d['description']}")

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(synthesis, f, ensure_ascii=False, indent=2)
        print(f"[+] Master audit report saved to: {out_path}")


if __name__ == "__main__":
    main()
