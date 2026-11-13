#!/usr/bin/env python3
"""
tools/card_rating_system/audit_cards.py
Universal Apologetics Card Audit & Rating Engine for Dawah Dialectics (yt-spx).

Evaluates cards in `qa_units` against ground-truth spoken turns in `scenario_turns`.
Tests the 5 Core Rubric Dimensions (100 Pts), enforces Hard-Fail Gates, detects defects,
and outputs comprehensive structured reports.

Usage:
    python tools/card_rating_system/audit_cards.py
    python tools/card_rating_system/audit_cards.py --db data/dawah_knowledge.db --out report.json
    python tools/card_rating_system/audit_cards.py --video-id KRDyUeTz_s0 --verbose
    python tools/card_rating_system/audit_cards.py --fail-only
"""

import sys
import os
import re
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

# Known prompt demonstration signatures that indicate leakage
PROMPT_LEAKAGE_SIGNATURES = [
    ("LEAK_FIRST_CAUSE_QUOTE", "whatever begins to exist has a cause"),
    ("LEAK_TRAIN_ANALOGY", "train of freight carriages"),
    ("LEAK_SOLDIER_SOCRATIC", "soldier cannot fire his rifle"),
    ("LEAK_CHILD_SURGERY_ANALOGY", "child being held down by doctors"),
    ("LEAK_MARK_13_32", "mark 13:32"),
    ("LEAK_GOD_IN_FLESH_QUOTE", "he is god in flesh. is he god or no"),
    ("LEAK_POWER_STATION_ANALOGY", "lamp drawing current from a power plant")
]

# Non-theological noise indicators (police scuffles, dog petting, street security, donation outro)
NOISE_KEYWORDS = [
    "call the police", "police officer", "pet him", "what kind of a dog",
    "brandish", "brandished", "firearm", "police come", "911 call", "verbal dispute",
    "islamnet", "donate now", "link in the description", "fundraising", "subscribe to the channel"
]

# Confessional & Da'ee counter-ask perspective regexes
CONFESSIONAL_REGEX = re.compile(
    r'(\b(?:why do i|why did i|why am i|how can i believe if i|if i haven\'?t read)\b'
    r'|لماذا\s+أعارض|لماذا\s+أحمل\s+(?:مشاعر|رؤية)|لماذا\s+أكره)',
    re.IGNORECASE,
)
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

# Russian, Hebrew, Korean, and Chinese CJK character sets
CYRILLIC_REGEX = re.compile(r'[\u0400-\u04FF]')
HEBREW_REGEX = re.compile(r'[\u0590-\u05FF]')
CJK_REGEX = re.compile(r'[\u4E00-\u9FFF\u3000-\u303F]')
LATIN_IN_ARABIC_REGEX = re.compile(r'\b[A-Za-z]{4,}\b')


def audit_single_card(
    card: Dict[str, Any],
    turns: List[Dict[str, Any]],
    require_arabic: bool = True,
) -> Dict[str, Any]:
    card_id = card.get("id") or "UNKNOWN"
    vid = card.get("video_id") or "UNKNOWN"

    defects = []
    hard_fail = None
    root_cause = None
    recommendations = []

    # Concatenate transcript turns for ground-truth verification
    transcript_en = " ".join((t.get("text_original") or "").lower() for t in turns)
    turn_roles_by_id = {
        str(t.get("turn_id") or "").strip(): t.get("speaker_role")
        for t in turns if t.get("turn_id")
    }

    # -------------------------------------------------------------
    # 1. HARD-FAIL GATE: [GATE-ARABIC] Missing Arabic Localization
    # -------------------------------------------------------------
    arabic_keys = [
        "question_ar", "core_answer_summary_ar", "fallacy_or_misconception_exposed_ar",
        "analogy_or_metaphor_ar", "socratic_question_ar", "verbatim_quote_ar"
    ]
    missing_ar_keys = [k for k in arabic_keys if not (card.get(k) or "").strip()] if require_arabic else []
    if require_arabic and missing_ar_keys:
        hard_fail = "[GATE-ARABIC]"
        root_cause = "Round 2 Arabic localization was omitted, failed, or timed out."
        defects.append({
            "code": "MISSING_ARABIC_FIELDS",
            "severity": "CRITICAL",
            "description": f"Missing required Arabic fields: {', '.join(missing_ar_keys)}"
        })
        recommendations.append("Execute Round 2 Arabic localization pipeline to populate all Arabic fields.")

    # -------------------------------------------------------------
    # 2. HARD-FAIL GATE: [GATE-LEAK] Prompt Few-Shot Demonstration Leakage
    # -------------------------------------------------------------
    premises_raw = card.get("logical_premises") or []
    if isinstance(premises_raw, str):
        try:
            premises_raw = json.loads(premises_raw)
        except Exception:
            premises_raw = [premises_raw]
    analogies_raw = card.get("analogies") or []
    if isinstance(analogies_raw, str):
        try:
            analogies_raw = json.loads(analogies_raw)
        except Exception:
            analogies_raw = [analogies_raw]
    citations_raw = card.get("scriptural_citations") or []
    if isinstance(citations_raw, str):
        try:
            citations_raw = json.loads(citations_raw)
        except Exception:
            citations_raw = []
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
        " ".join(str(x) for x in premises_raw),
        " ".join(str(x) for x in analogies_raw),
        " ".join(citation_parts),
    ]).lower()

    for leak_code, signature in PROMPT_LEAKAGE_SIGNATURES:
        if signature in card_all_text:
            if signature not in transcript_en:
                hard_fail = "[GATE-LEAK]"
                root_cause = f"Model copied prompt demonstration example ('{signature}') not present in video transcript."
                defects.append({
                    "code": leak_code,
                    "severity": "CRITICAL",
                    "description": f"Prompt demonstration leakage: Found '{signature}' which does not appear in transcript."
                })
                recommendations.append("Remove prompt few-shot demonstration text and ground quotes strictly in video turns.")

    # -------------------------------------------------------------
    # 3. HARD-FAIL GATE: [GATE-PERSPECTIVE] Confessional / Da'ee Counter-Ask
    # -------------------------------------------------------------
    q_en = (card.get("question_en") or "").strip()
    q_ar = (card.get("question_ar") or "").strip()
    if CONFESSIONAL_REGEX.search(q_en) or (q_ar and CONFESSIONAL_REGEX.search(q_ar)):
        hard_fail = hard_fail or "[GATE-PERSPECTIVE]"
        root_cause = root_cause or "Question formulated as an awkward 1st-person confession rather than an objective seeker objection."
        defects.append({
            "code": "CONFESSIONAL_INVERSION",
            "severity": "HIGH",
            "description": f"1st-person confession detected in question: '{q_en}'"
        })
        recommendations.append("Rephrase question_en into an objective seeker objection ending with '?'.")
    elif SOCRATIC_ROLE_INVERSION_REGEX.search(q_en):
        hard_fail = hard_fail or "[GATE-PERSPECTIVE]"
        root_cause = root_cause or "Question is the Da'ee's polemical Socratic challenge rather than the seeker's objection."
        defects.append({
            "code": "SOCRATIC_ROLE_INVERSION",
            "severity": "HIGH",
            "description": f"Da'ee Socratic cross-examination placed in question_en: '{q_en}'"
        })
        recommendations.append("Move Da'ee challenge to socratic_question and rewrite question_en as the seeker/interlocutor claim.")

    # -------------------------------------------------------------
    # 4. HARD-FAIL GATE: [GATE-NOISE] Conversational Noise / Outro Pitches / Police
    # -------------------------------------------------------------
    q_lower = q_en.lower()
    ivq_lower = (card.get("interlocutor_verbatim_quote") or "").lower()
    vq_lower = (card.get("verbatim_quote") or "").lower()

    found_noise = [kw for kw in NOISE_KEYWORDS if (kw in q_lower or kw in ivq_lower or kw in vq_lower)]
    if found_noise:
        hard_fail = hard_fail or "[GATE-NOISE]"
        root_cause = root_cause or "Extracted non-theological street banter, police scuffle, or promotional video outro."
        defects.append({
            "code": "NON_THEOLOGICAL_NOISE",
            "severity": "CRITICAL",
            "description": f"Extracted non-apologetic content matching keywords: {', '.join(found_noise)}"
        })
        recommendations.append("Drop this card if it reflects street banter, police logistics, or video outro fundraising.")

    # -------------------------------------------------------------
    # 5. HARD-FAIL GATE: [GATE-ROLE] Inverted Attribution / Single-Speaker Collapse
    # -------------------------------------------------------------
    daee_id = card.get("daee_speaker_id") or ""
    inter_id = card.get("interlocutor_speaker_id") or ""
    if daee_id and inter_id and daee_id == inter_id:
        hard_fail = hard_fail or "[GATE-ROLE]"
        root_cause = root_cause or "Da'ee and interlocutor speaker IDs are identical, collapsing dialogue into a single speaker."
        defects.append({
            "code": "SPEAKER_COLLAPSE",
            "severity": "HIGH",
            "description": f"Speaker ID '{daee_id}' assigned to both Da'ee and Interlocutor."
        })
        recommendations.append("Verify answer_turn_id belongs to MAIN_DAEE/CO_DAEE and question_turn_id to the interlocutor.")

    a_tid = str(card.get("answer_turn_id") or "").strip()
    if a_tid and a_tid in turn_roles_by_id and turn_roles_by_id[a_tid] not in ("MAIN_DAEE", "CO_DAEE"):
        hard_fail = hard_fail or "[GATE-ROLE]"
        root_cause = root_cause or f"answer_turn_id '{a_tid}' points to non-Da'ee role ({turn_roles_by_id[a_tid]})."
        defects.append({
            "code": "ANSWER_TURN_NOT_DAEE",
            "severity": "CRITICAL",
            "description": f"answer_turn_id '{a_tid}' is spoken by '{turn_roles_by_id[a_tid]}', not MAIN_DAEE/CO_DAEE."
        })
        recommendations.append("Re-anchor answer_turn_id to the MAIN_DAEE turn delivering the core refutation.")

    # -------------------------------------------------------------
    # DIMENSIONAL SCORING
    # -------------------------------------------------------------
    dim_scores = {
        "question_formulation": 15,
        "grounding_fidelity": 25,
        "dialectical_depth": 25,
        "bilingual_symmetry": 20 if require_arabic else 0,
        "attribution_timestamps": 15,
    }

    # Dim 1: Question Formulation (Max 15)
    if not q_en.endswith("?"):
        dim_scores["question_formulation"] -= 3
        defects.append({"code": "SYNTAX_NO_QUESTION_MARK", "severity": "MEDIUM", "description": "English question does not end with '?'"})
        recommendations.append("Rewrite question_en as a grammatical interrogative ending with '?'.")
    if q_en and ("..." in q_en or "…" in q_en):
        dim_scores["question_formulation"] -= 4
        defects.append({"code": "FRAGMENTED_QUESTION", "severity": "HIGH", "description": "English question contains incomplete ellipsis fragment."})
        recommendations.append("Rewrite question_en as a complete standalone question without '...'.")
    if DAEE_COUNTER_ASK_REGEX.search(q_en) or (q_ar and DAEE_COUNTER_ASK_REGEX.search(q_ar)):
        dim_scores["question_formulation"] -= 6
        defects.append({
            "code": "DAEE_COUNTER_ASK_PERSPECTIVE",
            "severity": "HIGH",
            "description": "Question ends with Da'ee defensive comeback ('what is the problem with that?' / 'فما المشكل في ذلك') instead of seeker objection.",
        })
        recommendations.append("Invert question_en/question_ar to the skeptic's objection (e.g., '..., isn't that a bad thing?').")
    if require_arabic and q_ar and not q_ar.endswith("؟"):
        dim_scores["question_formulation"] -= 2
        defects.append({"code": "SYNTAX_NO_ARABIC_QUESTION_MARK", "severity": "LOW", "description": "Arabic question does not end with '؟'"})
    if len(q_en.split()) < 4:
        dim_scores["question_formulation"] -= 3
        defects.append({"code": "QUESTION_TOO_SHORT", "severity": "MEDIUM", "description": "Question is too short / conversational fragment."})

    # Dim 2: Grounding Fidelity (Max 25)
    daee_quote = (card.get("verbatim_quote") or "").strip()
    if not daee_quote:
        dim_scores["grounding_fidelity"] -= 5
        defects.append({"code": "MISSING_DAEE_QUOTE", "severity": "MEDIUM", "description": "Missing verbatim_quote from transcript."})
        recommendations.append("Copy an exact word-for-word MAIN_DAEE proof quote from the transcript into verbatim_quote.")
    elif transcript_en:
        sample_words = [w for w in re.findall(r'\b\w{4,}\b', daee_quote.lower()) if w not in {"that", "this", "what", "from", "with", "have"}]
        if sample_words:
            matched_words = [w for w in sample_words if w in transcript_en]
            ratio = len(matched_words) / len(sample_words)
            if ratio < 0.3:
                dim_scores["grounding_fidelity"] -= 8
                defects.append({"code": "QUOTE_UNGROUNDED_DAEE", "severity": "HIGH", "description": f"Da'ee verbatim quote appears ungrounded (only {ratio:.0%} lexical overlap with transcript)."})
                recommendations.append("Replace verbatim_quote with an exact spoken quote from the Da'ee in the transcript.")

    ex_turns = card.get("dialogue_exchange") or card.get("dialogue_trajectory") or []
    if not isinstance(ex_turns, list) or len(ex_turns) < 2:
        dim_scores["grounding_fidelity"] -= 4
        defects.append({"code": "THIN_DIALOGUE_EXCHANGE", "severity": "MEDIUM", "description": "dialogue_exchange has fewer than 2 grounded turns."})

    # Dim 3: Dialectical Depth (Max 25)
    premises_en = premises_raw if isinstance(premises_raw, list) else []
    if len(premises_en) < 2:
        dim_scores["dialectical_depth"] -= 6
        defects.append({"code": "INCOMPLETE_SYLLOGISM", "severity": "HIGH", "description": "Fewer than 2 logical premises in deductive syllogism."})
        recommendations.append("Provide a formal 3-step deductive syllogism (Premise 1, Premise 2, Conclusion) grounded in the debate.")
    elif len(premises_en) == 2:
        dim_scores["dialectical_depth"] -= 3
        defects.append({"code": "SYLLOGISM_UNDER_3_PREMISES", "severity": "LOW", "description": "Only 2 premises; rubric recommends 3 deductive steps (Major, Minor, Conclusion)."})

    sum_en = (card.get("core_answer_summary_en") or "").strip()
    if re.search(r'\b(the da\'?ee|the speaker|hashim|mansur|uthman|ali)\s+(says|counters|explains|argues|responds)\b', sum_en, re.I):
        dim_scores["dialectical_depth"] -= 3
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
        dim_scores["dialectical_depth"] -= 2
        defects.append({"code": "DUPLICATE_SOCRATIC_QUESTION", "severity": "LOW", "description": "socratic_question merely repeats question_en."})

    # Dim 4: Bilingual Symmetry (Max 20) — evaluated when require_arabic=True
    if require_arabic:
        if missing_ar_keys:
            dim_scores["bilingual_symmetry"] = 0
        else:
            for k in arabic_keys:
                val = card.get(k) or ""
                if CYRILLIC_REGEX.search(val) or HEBREW_REGEX.search(val) or CJK_REGEX.search(val):
                    dim_scores["bilingual_symmetry"] -= 5
                    defects.append({"code": "FOREIGN_SCRIPT_LEAKAGE", "severity": "HIGH", "description": f"Non-Arabic script (Cyrillic/Hebrew/CJK) detected in '{k}'"})
                    break
                if LATIN_IN_ARABIC_REGEX.search(val):
                    dim_scores["bilingual_symmetry"] -= 3
                    defects.append({"code": "LATIN_LEAKAGE_IN_ARABIC", "severity": "MEDIUM", "description": f"Untranslated Latin words detected in '{k}'"})
                    break

    # Dim 5: Attribution & Timestamps (Max 15)
    ts_sec = card.get("timestamp_seconds")
    q_tid = str(card.get("question_turn_id") or "").strip()
    has_valid_turn_anchor = bool(q_tid and (not turn_roles_by_id or q_tid in turn_roles_by_id))
    if (ts_sec is None or ts_sec < 0) and (require_arabic or not has_valid_turn_anchor):
        dim_scores["attribution_timestamps"] -= 5
        defects.append({"code": "INVALID_TIMESTAMP", "severity": "MEDIUM", "description": "Missing or negative timestamp seconds."})

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
        status = "PASS" if score_pct >= 50 else "FAIL"
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
        "needs_english_repair": needs_english_repair,
        "dimension_scores": dim_scores,
        "defect_count": len(defects),
        "defects": defects,
        "root_cause": root_cause,
        "recommendations": recommendations,
    }


def format_pre_round2_rubric_summary(
    r1_units: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> str:
    """
    Runs the Card Rating System pre-check on Round 1 English dossiers before Round 2 runs,
    generating targeted instructions telling Round 2 which cards are CLEAN (keep English untouched)
    vs which specific English defects need conditional repair alongside Arabic localization.
    """
    if not r1_units:
        return ""

    lines = ["PRE-ROUND-2 CARD RATING & DEFECT DIAGNOSTICS (tools/card_rating_system):"]
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
            status_tag = ev["hard_fail_triggered"] or ("NEEDS_REPAIR" if ev["needs_english_repair"] else "MINOR_POLISH")
            defect_strs = [f"[{d['code']}] {d['description']}" for d in ev["defects"][:4]]
            rec_strs = ev["recommendations"][:3]
            lines.append(
                f"- Card {idx} ({q_tid} -> {a_tid}): Score={ev['score_pct']}/100 | Status={status_tag}"
            )
            if defect_strs:
                lines.append(f"  Defects: {'; '.join(defect_strs)}")
            if rec_strs:
                lines.append(f"  Targeted Fix: {' | '.join(rec_strs)}")
    return "\n".join(lines)


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

    results = []
    hard_fails_breakdown = {}
    tier_counts = {"Tier 1: Production Ready (Gold)": 0, "Tier 2: Minor Polish Needed (Silver)": 0, "Tier 3: Substandard / Requires Regeneration (Bronze)": 0, "Tier 4: Critical Failure / Rejected (Red)": 0}

    for card in card_rows:
        vid = card.get("video_id")
        q_sec = card.get("timestamp_seconds") or 0
        ans_sec = card.get("answer_timestamp_seconds") or q_sec

        start_ms = max(0, (q_sec - 60) * 1000)
        end_ms = (max(q_sec, ans_sec) + 120) * 1000

        cursor.execute("""
            SELECT turn_id, speaker_role, speaker_label_en, speaker_label_ar, start_fmt, end_fmt, text_original, text_ar
            FROM scenario_turns
            WHERE video_id = ? AND start_ms >= ? AND start_ms <= ?
            ORDER BY turn_index ASC
        """, (vid, start_ms, end_ms))
        turns = [dict(t) for t in cursor.fetchall()]

        if not turns:
            cursor.execute("SELECT turn_id, speaker_role, speaker_label_en, speaker_label_ar, start_fmt, end_fmt, text_original, text_ar FROM scenario_turns WHERE video_id = ? LIMIT 20", (vid,))
            turns = [dict(t) for t in cursor.fetchall()]

        evaluation = audit_single_card(card, turns)

        tier_counts[evaluation["tier"]] = tier_counts.get(evaluation["tier"], 0) + 1
        if evaluation["hard_fail_triggered"]:
            hf = evaluation["hard_fail_triggered"]
            hard_fails_breakdown[hf] = hard_fails_breakdown.get(hf, 0) + 1

        if not fail_only or evaluation["status"] == "FAIL":
            results.append(evaluation)

    conn.close()

    total_evaluated = len(card_rows)
    passed_cards = [r for r in results if r["status"] == "PASS"]
    failed_cards = [r for r in results if r["status"] == "FAIL"]

    mean_all = sum(r["total_score"] for r in results) / total_evaluated if total_evaluated else 0
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
            "mean_score_passed": round(mean_passed, 2)
        },
        "quality_tiers": tier_counts,
        "hard_fail_gates": hard_fails_breakdown,
        "cards": results
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
            print(f"{status_symbol} Card: {c['card_id']:<20} | Score: {c['total_score']:>3} | {c['tier']}")
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
