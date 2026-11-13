#!/usr/bin/env python3
"""
core/scenario_extractor.py
Micro-Step M6 (Stage 2 Local GPU Extractor):
Processes clean Stage 1 Scenario Transcripts (scenarios/{video_id}.json) using
Qwen 2.5 14B Instruct (Q5_K_M GGUF + 8-bit KV Cache + 12,288 Context) in an isolated
GPU process on Free Colab T4.

Key V2 Advantages over V1 Subtitle Chunking:
  1. Episode-Aware Windowing: Chunks strictly along natural Interlocutor Episodes
     instead of blind 4-minute cuts—zero mid-argument truncation and ~30% faster!
  2. Per-Episode Micro-Checkpointing: Every completed episode window is saved immediately
     to Drive (extractions/partial/{vid}_win{k}.json) so a sudden Colab timeout resumes
     at the exact unfinished episode!
  3. Ground-Truth CTC Timestamp Anchoring: Resolves question_turn_id and answer_turn_id
     directly against scenario_turns millisecond timestamps—zero LLM timestamp drift!
"""

import os
import re
import json
import math
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

from core.drive_sync import SharedWorkspace, STEP_RANK
from core.extractor_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    USER_EXTRACTION_PROMPT_TEMPLATE,
    ROUND1_ENGLISH_EXTRACTION_PROMPT,
    USER_ROUND1_PROMPT_TEMPLATE,
    ROUND2_ARABIC_VERIFICATION_PROMPT,
    COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT,
    USER_ROUND2_PROMPT_TEMPLATE,
)
from core.audio_diarizer import ms_to_fmt

try:
    from db.database import (
        sanitize_text_field,
        sanitize_arabic_unit_fields,
        resolve_turn_speaker_labels,
    )
except (ImportError, ValueError):
    try:
        from ..db.database import (
            sanitize_text_field,
            sanitize_arabic_unit_fields,
            resolve_turn_speaker_labels,
        )
    except Exception:
        sanitize_text_field = None
        sanitize_arabic_unit_fields = None
        resolve_turn_speaker_labels = None

try:
    from tools.card_rating_system.audit_cards import (
        audit_single_card,
        format_pre_round2_rubric_summary,
    )
except (ImportError, ValueError):
    try:
        from ..tools.card_rating_system.audit_cards import (
            audit_single_card,
            format_pre_round2_rubric_summary,
        )
    except Exception:
        audit_single_card = None
        format_pre_round2_rubric_summary = None


def _is_dialectical_payload_item(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    keys = (
        "question_en", "question_ar", "question_turn_id", "answer_turn_id",
        "core_answer_summary_en", "core_answer_summary_ar", "question",
        "dialogue_exchange", "dialogue_trajectory",
    )
    return any(k in x for k in keys)


def _normalize_extracted_item(x: Dict[str, Any]) -> Dict[str, Any]:
    if "question" in x:
        q_val = str(x["question"] or "")
        if re.search(r"[\u0600-\u06ff]", q_val):
            x.setdefault("question_ar", q_val)
        else:
            x.setdefault("question_en", q_val)
    if "dialogue_trajectory" in x and not x.get("dialogue_exchange"):
        x["dialogue_exchange"] = x["dialogue_trajectory"]
    return x


def _strip_thinking_blocks(raw_text: str) -> str:
    """Removes Qwen3 <think>...</think> blocks (closed or truncated) before JSON array parsing."""
    if not raw_text:
        return ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    if "<think>" in cleaned.lower():
        idx_think = cleaned.lower().find("<think>")
        after_think = cleaned[idx_think + 7 :]
        json_start = re.search(r"\[\s*\{|\{\s*\"", after_think)
        if json_start:
            cleaned = (cleaned[:idx_think] + after_think[json_start.start() :]).strip()
        else:
            cleaned = cleaned[:idx_think].strip()
    return cleaned


def clean_llm_json_array(raw_text: str) -> List[Dict[str, Any]]:
    """Parses JSON arrays robustly, stripping <think> blocks and recovering completed objects even if truncated."""
    text = _strip_thinking_blocks(raw_text or "")
    if not text:
        return []
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [_normalize_extracted_item(x) for x in parsed if _is_dialectical_payload_item(x)]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return [_normalize_extracted_item(x) for x in v if _is_dialectical_payload_item(x)]
            if _is_dialectical_payload_item(parsed):
                return [_normalize_extracted_item(parsed)]
    except Exception:
        pass

    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [_normalize_extracted_item(x) for x in parsed if _is_dialectical_payload_item(x)]
        except Exception:
            pass

    # Balanced-brace stream recovery
    results: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if ch == '"' and not esc:
            in_str = not in_str
        elif ch == "\\" and in_str:
            esc = not esc
            continue
        elif not in_str:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if _is_dialectical_payload_item(obj):
                            results.append(_normalize_extracted_item(obj))
                    except Exception:
                        pass
                    start = -1
        esc = False
    return results


def sanitize_arabic_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Purges foreign scripts, Latin splices, and domain translation drift via canonical db.database cleaner."""
    if not isinstance(item, dict):
        return item
    if sanitize_arabic_unit_fields is not None:
        return sanitize_arabic_unit_fields(item)

    _fallback_re = re.compile(r"[\u0400-\u04ff\u0590-\u05ff\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]+")

    def _clean_str(val: str) -> str:
        if sanitize_text_field is not None:
            val = sanitize_text_field(val, is_arabic=True)
        return _fallback_re.sub("", val).strip()

    for k in (
        "question_ar",
        "core_answer_summary_ar",
        "fallacy_or_misconception_exposed_ar",
        "analogy_or_metaphor_ar",
        "socratic_question_ar",
        "interlocutor_verbatim_quote_ar",
        "verbatim_quote_ar",
    ):
        if isinstance(item.get(k), str) and item[k]:
            item[k] = _clean_str(item[k])

    for list_key in ("logical_premises_ar", "analogies_ar"):
        if isinstance(item.get(list_key), list):
            item[list_key] = [_clean_str(str(p)) for p in item[list_key] if _clean_str(str(p))]

    if isinstance(item.get("dialogue_exchange"), list):
        for ex in item["dialogue_exchange"]:
            if isinstance(ex, dict) and isinstance(ex.get("text_ar"), str) and ex["text_ar"]:
                ex["text_ar"] = _clean_str(ex["text_ar"])

    return item


CANONICAL_TOPIC_PILLARS = {
    "Textual Preservation & Manuscripts": [
        "Textual Preservation", "Manuscript", "Inerrancy", "Biblical Criticism",
        "Biblical Reliability", "Biblical Manuscripts", "Dead Sea Scrolls", "Codex"
    ],
    "Comparative Religion & Christianity": [
        "Comparative Religion", "Christianity", "Trinity", "Biblical",
        "Crucifixion", "Atonement", "Bible", "Christology", "Gospel"
    ],
    "Theology & God's Existence": [
        "Theology", "God's Existence", "First Cause", "Causality", "Tawheed", "Divine Unity", "Creation", "Contingency"
    ],
    "Problem of Evil & Suffering": [
        "Problem of Evil", "Suffering", "Theodicy", "Disasters", "Evil", "Ibtilaa"
    ],
    "Objective Morality & Ethics": [
        "Objective Morality", "Moral Ontology", "Ethics & Morality", "Moral Argument", "Moral", "Nihilism"
    ],
    "Epistemology & Science": [
        "Epistemology", "Science", "Evolution", "Darwin", "Scientific", "Scientism", "Consciousness", "Philosophy of Mind"
    ],
    "Prophethood & Quran": [
        "Prophethood", "Quran", "Prophet", "Miracles", "Inimitability", "Seerah"
    ],
    "Shariah & Social Inquiries": [
        "Shariah", "Social Inquiries", "Feminism", "Women", "Hijab", "Polygamy", "Inheritance", "Apostasy", "Hudood", "Slavery Context"
    ],
    "Comparative Religion & Judaism": [
        "Judaism", "Jewish", "Torah", "Talmud", "Old Testament Prophecies"
    ],
    "Comparative Religion & Eastern Traditions": [
        "Hinduism", "Buddhism", "Reincarnation", "Karma", "Eastern Traditions"
    ],
    "Hadith & Sunnah Preservation": [
        "Hadith", "Sunnah", "Isnad", "Bukhari", "Fabrication", "Narrators"
    ],
    "Islamic History & Civilizations": [
        "Islamic History", "Caliphate", "Conquests", "Andalusia", "Crusades", "Ottoman"
    ],
    "Eschatology & Free Will": [
        "Eschatology", "Free Will", "Qadr", "Destiny", "Afterlife", "Hell", "Heaven", "Resurrection"
    ],
}


def canonicalize_topic_category(raw_category: str) -> str:
    """Enforces strict mapping of ad-hoc LLM categories to the Canonical Topic Pillars."""
    if not raw_category:
        return "Theology & God's Existence"
    cat_clean = raw_category.strip()
    if cat_clean in CANONICAL_TOPIC_PILLARS:
        return cat_clean
    for pillar, variants in CANONICAL_TOPIC_PILLARS.items():
        if any(v.lower() in cat_clean.lower() for v in variants):
            return pillar
    return cat_clean


CONFESSIONAL_EN_PATTERNS = [
    (re.compile(r"^\s*why\s+do\s+i\s+oppose\s+islam\s+(?:if|when|though)\s+i\s+(?:haven't|have\s+not)\s+read\s+the\s+quran(?:\s+and\s+(?:only\s+)?have\s+a\s+superficial\s+understanding(?:\s+of\s+it)?)?[?.!]*$", re.I),
     "Can one validly critique or oppose Islam without having examined the Quran beyond a superficial understanding?"),
    (re.compile(r"^\s*why\s+do\s+i\s+hold\s+anti[- ]islamic\s+views\s+(?:if|when|though)\s+i\s+(?:haven't|have\s+not)\s+read\s+islamic\s+sources.*$", re.I),
     "Is it intellectually justified to form anti-Islamic views based on indirect impressions without examining primary Islamic sources?"),
    (re.compile(r"^\s*why\s+do\s+i\s+hold\s+anti[- ]islamic\s+views[?.!]*$", re.I),
     "Is it intellectually justified to form anti-Islamic views without examining primary Islamic sources?"),
    (re.compile(r"^\s*why\s+do\s+i\s+(?:oppose|hate|reject|criticize|critique)\s+islam\s+(?:if|when|without)\b.*$", re.I),
     "Can one reasonably critique or reject Islam without direct knowledge of its primary sources?"),
    (re.compile(r"^\s*why\s+do\s+i\s+oppose\s+islam[?.!]*$", re.I),
     "What are the rational grounds on which critics challenge Islam?"),
    (re.compile(r"^\s*why\s+do\s+i\s+", re.I),
     "Why would someone "),
    (re.compile(r"^\s*why\s+am\s+i\s+", re.I),
     "Why would a seeker be "),
    (re.compile(r"^\s*why\s+did\s+i\s+", re.I),
     "Why would one "),
    (re.compile(r"^\s*how\s+can\s+i\s+believe\s+if\s+i\s+", re.I),
     "How can a seeker believe if they "),
]

CONFESSIONAL_AR_PATTERNS = [
    (re.compile(r"^\s*لماذا\s+أعارض\s+الإسلام\s+(?:وأنا\s+لم|دون\s+أن|إذا\s+لم)\s+(?:أقرأ|أطلع\s+على)\s+القرآن(?:\s+وأملك\s+فهمًا\s+سطحيًا\s+له)?[؟?!.]*$"),
     "هل يصح للمرء أن يعارض الإسلام دون الاطلاع على القرآن الكريم والاكتفاء بفهم سطحي له؟"),
    (re.compile(r"^\s*لماذا\s+أحمل\s+(?:مشاعر|رؤية)\s+معادية\s+للإسلام\s+(?:رغم\s+أنني|وأنا)\s+لم\s+أقرأ\s+المصادر\s+الإسلامية.*$"),
     "هل من العدل والإنصاف تبني رؤية معادية للإسلام استنادًا إلى انطباعات غير مباشرة دون قراءة مصادره الأصلية؟"),
    (re.compile(r"^\s*لماذا\s+أحمل\s+(?:مشاعر|رؤية)\s+معادية\s+للإسلام[؟?!.]*$"),
     "هل من العدل والإنصاف تكوين أحكام مسبقة عن الإسلام دون الرجوع إلى تعاليمه الصحيحة؟"),
    (re.compile(r"^\s*لماذا\s+أعارض\s+الإسلام[؟?!.]*$"),
     "ما هي المنطلقات التي يبني عليها المنتقدون اعتراضاتهم على الإسلام؟"),
    (re.compile(r"^\s*لماذا\s+أعارض\s+"),
     "هل يصح الاعتراض على "),
    (re.compile(r"^\s*لماذا\s+أرفض\s+"),
     "هل يسوغ رفض "),
    (re.compile(r"^\s*لماذا\s+أحمل\s+"),
     "هل من الإنصاف تبني "),
]

CONFESSIONAL_RESIDUAL_REGEX = re.compile(
    r"(\b(?:why do i|why did i|why am i|how can i believe if i|if i haven't read|i only have a superficial)\b"
    r"|لماذا\s+أعارض|لماذا\s+أحمل\s+(?:مشاعر|رؤية)|لماذا\s+أكره|رغم\s+أنني|بل\s+استندت|وأملك\s+فهمًا|وأنا\s+لم\s+أقرأ)",
    re.IGNORECASE,
)

_EN_INTERROGATIVE_STARTERS = re.compile(
    r"^(?:who|what|where|when|why|how|which|whose|whom|is|are|am|was|were|do|does|did|"
    r"can|could|will|would|shall|should|may|might|has|have|had|if|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|can't|couldn't|wouldn't|shouldn't|won't|haven't|hasn't|suppose|assuming)\b",
    re.IGNORECASE,
)

# Da'ee defensive rejoinder clauses (e.g. "If X, what is the problem with that?" / "فما هو المشكل في ذلك؟")
_DAEE_DEFENSIVE_CLAUSE_EN = (
    r"(?:"
    r"what(?:'s|\s+is|\s+would\s+be)\s+(?:the\s+)?(?:problem|issue|harm|wrong|matter)\s+(?:with|about)\s+(?:that|this|it|doing\s+(?:so|that))"
    r"|what(?:'s|\s+is)\s+wrong\s+with\s+(?:that|this|it|doing\s+(?:so|that))"
    r"|why\s+(?:is|would)\s+(?:that|this|it)\s+(?:be\s+)?(?:a\s+problem|an\s+issue|wrong|bad|unfair|immoral|evil)"
    r"|how\s+(?:is|could)\s+(?:that|this|it)\s+(?:be\s+)?(?:a\s+problem|wrong|bad|unfair|immoral|evil)"
    r"|is\s+there\s+(?:any|a)\s+(?:problem|issue|harm|wrong)\s+(?:with|in)\s+(?:that|this|it)"
    r")"
)
_DAEE_DEFENSIVE_TAIL_EN_RE = re.compile(
    rf"^(?P<premise>.+?)(?:,\s*(?:then\s+)?(?:and\s+)?|\s*;\s*|\s*[—–]\s*|\s+(?:and|so)\s+)"
    rf"(?P<tail>{_DAEE_DEFENSIVE_CLAUSE_EN})\s*[?!.]*\s*$",
    re.IGNORECASE,
)
_BARE_DAEE_DEFENSIVE_EN_RE = re.compile(
    rf"^\s*(?:(?:so|and|then|well|okay)[,\s]+)*{_DAEE_DEFENSIVE_CLAUSE_EN}\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_DAEE_DEFENSIVE_CLAUSE_AR = (
    r"(?:"
    r"(?:ما|أي)\s+(?:هو\s+)?(?:المشكل|المشكلة|الإشكال|العيب|الضير|الخطأ|البأس|المشكله)\s+(?:في|من|ب)\s*(?:ذلك|هذا|هذا\s+الأمر)"
    r"|ما(?:ذا)?\s+(?:يعيب|يضير)\s+(?:ذلك|هذا|هذا\s+الأمر)"
    r"|(?:لماذا|لم)\s+(?:يعد|يُعد|يكون|يعتبر|يُعتبر)\s+(?:ذلك|هذا)\s+(?:مشكلة|مشكلًا|إشكالًا|أمرًا\s+سيئًا|عيبًا|خطأً)"
    r"|هل\s+(?:يوجد|هناك|في\s+(?:ذلك|هذا))\s+(?:أي\s+)?(?:مشكلة|إشكال|عيب|ضير|بأس)"
    r")"
)
_DAEE_DEFENSIVE_TAIL_AR_RE = re.compile(
    rf"^(?P<premise>.+?)(?:[،,;—–]\s*(?:ف|و|إذن\s+ف?)?|\s+(?:ف|و))"
    rf"(?P<tail>{_DAEE_DEFENSIVE_CLAUSE_AR})\s*[؟?!.]*\s*$"
)
_BARE_DAEE_DEFENSIVE_AR_RE = re.compile(
    rf"^\s*(?:ف|و)?{_DAEE_DEFENSIVE_CLAUSE_AR}\s*[؟?!.]*\s*$"
)


def _rewrite_daee_defensive_question(en: str, ar: str) -> Tuple[str, str, bool]:
    """Converts Da'ee defensive comeback endings into genuine seeker objection syntax, or rejects bare rejoinders."""
    if en:
        if _BARE_DAEE_DEFENSIVE_EN_RE.match(en):
            return en, ar, False
        m_en = _DAEE_DEFENSIVE_TAIL_EN_RE.match(en)
        if m_en:
            premise = m_en.group("premise").strip(" ,;:-—–")
            if len(premise.split()) < 4:
                return en, ar, False
            en = f"{premise}, isn't that a bad thing?"

    if ar:
        if _BARE_DAEE_DEFENSIVE_AR_RE.match(ar):
            return en, ar, False
        m_ar = _DAEE_DEFENSIVE_TAIL_AR_RE.match(ar)
        if m_ar:
            premise_ar = m_ar.group("premise").strip(" ،,;:-—–")
            if len(premise_ar.split()) < 4:
                return en, ar, False
            ar = f"{premise_ar}، أليس ذلك أمرًا سيئًا؟"

    return en, ar, True


def _has_broken_interrogative_fragment(en: str, ar: str) -> bool:
    if "..." in en or "…" in en or "..." in ar or "…" in ar:
        return True
    if re.search(r"\?\s+[a-z]", en) or re.search(r"؟\s+وأ[ا-ي]", ar):
        return True
    return False


def normalize_question_syntax_and_perspective(q_en: str, q_ar: str = "") -> Tuple[str, str, bool]:
    """
    1st-Person Confessional / Da'ee Counter-Ask Perspective & Syntax Validation Gatekeeper:
    1. Converts awkward 1st-person confessions (e.g. 'Why do I oppose Islam...') into objective seeker inquiries.
    2. Rewrites premise-backed Da'ee defensive rejoinders (e.g. '...what is the problem with that?') into seeker objections,
       and rejects bare un-rewritable defensive filler ('What is the problem with that?').
    3. Rejects truncated ellipsis fragments, un-rewritable confessional syntax, or pure raw statements lacking question form.
    4. Enforces strict interrogative syntax: '?' for English and '؟' for Arabic.
    5. Returns (normalized_q_en, normalized_q_ar, is_valid).
    """
    en = (q_en or "").strip()
    ar = (q_ar or "").strip()

    if not en and not ar:
        return "", "", False

    if _has_broken_interrogative_fragment(en, ar):
        return en, ar, False

    if en and "?" not in en and not _EN_INTERROGATIVE_STARTERS.match(en):
        return en, ar, False

    # 1. Normalize English confessional patterns
    for pat, repl in CONFESSIONAL_EN_PATTERNS:
        if pat.search(en):
            en = pat.sub(repl, en).strip()
            break

    # 2. Normalize Arabic confessional patterns
    for pat, repl in CONFESSIONAL_AR_PATTERNS:
        if pat.search(ar):
            ar = pat.sub(repl, ar).strip()
            break

    # 3. Rewrite Da'ee defensive comeback tails or reject bare defensive filler
    en, ar, ok_rejoinder = _rewrite_daee_defensive_question(en, ar)
    if not ok_rejoinder:
        return en, ar, False

    # 4. Check for any remaining confessional inversion or broken mid-sentence punctuation
    if (
        CONFESSIONAL_RESIDUAL_REGEX.search(en)
        or CONFESSIONAL_RESIDUAL_REGEX.search(ar)
        or _has_broken_interrogative_fragment(en, ar)
    ):
        return en, ar, False

    # 5. Mandatory Interrogative Syntax Enforcement
    if en:
        en = re.sub(r'[\s.,:;!?؟]+$', '', en).strip()
        if en:
            en = f"{en}?"

    if ar:
        ar = re.sub(r'[\s.,:;!?؟]+$', '', ar).strip()
        if ar:
            ar = f"{ar}؟"

    return en, ar, True


def is_substantive_apologetics_window(
    win_turns: List[Dict[str, Any]],
    window_index: int,
    total_windows: int,
) -> Tuple[bool, str]:
    """
    Apologetic Semantic Pre-Filtering Gatekeeper:
    Screens episode windows BEFORE LLM extraction to discard:
      1. YouTube video outro promotional appeals & donation pitches (Card 44).
      2. Street security incidents, police encounters & police dog petting (Cards 31, 35).
      3. Casual street greetings / logistics without theological substance (Card 9).
    Returns (is_substantive: bool, reason: str).
    """
    if not win_turns:
        return False, "Empty turn list"

    combined_text = " ".join((t.get("text_original") or "").lower() for t in win_turns)
    total_words = len(combined_text.split())

    if total_words < 25:
        return False, f"Too short for dialectical exchange ({total_words} words)"

    theological_markers = [
        "god", "allah", "jesus", "christ", "bible", "quran", "prophet", "muhammad",
        "trinity", "atheis", "creator", "creation", "first cause", "causality",
        "sin", "atonement", "salvation", "afterlife", "resurrection", "evil",
        "suffering", "morality", "objective moral", "proof", "miracle", "gospel",
        "islam", "christian", "jewish", "faith", "belief", "testament", "scriptur",
        "الله", "القرآن", "النبي", "المسيح", "عيسى", "الإسلام", "الإلحاد", "الخالق",
        "السببية", "الثالوث", "الصلب", "الفداء", "الشر", "المعاناة", "الوحي", "النبوة"
    ]
    has_theological_context = any(m in combined_text for m in theological_markers)

    # 1. Outro Promotional / Fundraising Screening
    promo_keywords = [
        "donate", "donation", "fundrais", "support our work", "support the dawah",
        "islamnet", "link in the description", "subscribe to the channel",
        "hit the bell", "patreon", "paypal", "gofundme", "merch", "leave a comment below"
    ]
    found_promo = [pk for pk in promo_keywords if pk in combined_text]
    is_last_window = (window_index >= max(1, total_windows - 1))
    strong_fundraising_markers = ("donate", "donation", "fundrais", "islamnet", "patreon", "paypal", "gofundme", "subscribe to the channel")
    has_strong_fundraising = any(m in combined_text for m in strong_fundraising_markers)
    deep_debate_markers = ("trinity", "crucif", "resurrection", "atheis", "first cause", "causality", "contradict", "manuscript", "historical")
    has_deep_debate = any(m in combined_text for m in deep_debate_markers)
    if len(found_promo) >= 2 or (is_last_window and len(found_promo) >= 1):
        if not has_theological_context or (total_words < 120 and len(found_promo) >= 1) or (has_strong_fundraising and not has_deep_debate):
            return False, f"Video outro promotional / fundraising appeal (matched: {', '.join(found_promo)})"

    # 2. Police Encounter / Street Security Incident Screening
    police_security_keywords = [
        "call the police", "police officer", "officer", "constable", "patrol car",
        "pet him", "what kind of a dog", "pet the dog", "good dog",
        "brandish", "brandished", "firearm", "police come", "911 call",
        "verbal dispute", "handcuff", "arrest", "security guard", "stay back"
    ]
    found_police = [pk for pk in police_security_keywords if pk in combined_text]
    if len(found_police) >= 2 or any(k in combined_text for k in ["what kind of a dog", "pet him", "brandish", "911 call"]):
        if not has_theological_context:
            return False, f"Street security / police incident (matched: {', '.join(found_police)})"

    # 3. Casual Street Banter Screening (Greetings / Photo requests only)
    banter_phrases = [
        "how are you doing", "what are you doing here", "nice to meet you",
        "can i take a picture", "take a photo", "take a selfie", "nice weather",
        "where are you from", "where do you live", "what is your name", "see you later"
    ]
    found_banter = [bp for bp in banter_phrases if bp in combined_text]
    if len(found_banter) >= 2 and not has_theological_context:
        return False, f"Casual street greeting / small talk without apologetics (matched: {', '.join(found_banter)})"

    return True, "Substantive debate dialogue"


def is_valid_dialectical_question(q_en: str, q_ar: str = "") -> bool:
    """
    Enforces quality threshold: discards conversational noise, rhetorical filler,
    personal street interview chit-chat, situational street drama/police incidents,
    video outro fundraising, or truncated sentence fragments.
    Allows concise, foundational theological questions (e.g. 'Is Jesus God?', 'Who created God?').
    """
    if not q_en and not q_ar:
        return False

    q_en_raw = (q_en or "").strip()
    if _has_broken_interrogative_fragment(q_en_raw, q_ar or ""):
        return False

    if q_en_raw and "?" not in q_en_raw and not _EN_INTERROGATIVE_STARTERS.match(q_en_raw):
        return False

    q_en_clean = q_en_raw.lower().rstrip("?").strip()
    words_en = q_en_clean.split()
    if len(words_en) < 3:
        return False

    # Reject Da'ee's own polemical cross-examination challenge phrased as the main question (role inversion)
    if re.search(
        r"\bshould\s+(?:god|he)\s+(?:love|approve(?:\s+of)?|reward)\s+a\s+(?:rapist|murderer|killer|liar|oppressor|hitler)\b",
        q_en_clean,
        re.IGNORECASE,
    ):
        return False

    # Reject bare un-rewritable Da'ee defensive rejoinders ("What is the problem with that?" / "فما المشكل في ذلك؟")
    if (q_en_raw and _BARE_DAEE_DEFENSIVE_EN_RE.match(q_en_raw)) or (q_ar and _BARE_DAEE_DEFENSIVE_AR_RE.match(q_ar.strip())):
        return False

    # Discard non-theological situational street drama / police / security chatter
    street_drama_patterns = [
        r"\bpolice\b", r"\bconstable\b", r"\bofficer\b", r"\bfirearm\b", r"\bweapon\b",
        r"\bbrandish\b", r"\bbrandishing\b", r"\bto the t\b", r"\bpet him\b",
        r"\bwhat kind of (a )?dog\b", r"\bpet the dog\b", r"\barrest\b",
        r"\bsecurity guard\b", r"\bhandcuff\b", r"\bdispatch\b", r"\bpatrol car\b",
        r"\b911\b", r"\bcall 911\b", r"\bverbal dispute\b"
    ]
    if any(re.search(pat, q_en_clean, re.I) for pat in street_drama_patterns):
        return False

    # Discard YouTube outro promotional / fundraising appeals
    outro_patterns = [
        r"\bdonate\b", r"\bdonation\b", r"\bfundrais\b", r"\bpatreon\b", r"\bpaypal\b",
        r"\bgofundme\b", r"\bislamnet\b", r"\bsubscribe\b", r"\bchannel\b",
        r"\blink in the description\b", r"\bsupport our (work|dawah)\b"
    ]
    if any(re.search(pat, q_en_clean, re.I) for pat in outro_patterns):
        return False

    # Discard unnormalized 1st-person confessional inversions
    if CONFESSIONAL_RESIDUAL_REGEX.search(q_en) or CONFESSIONAL_RESIDUAL_REGEX.search(q_ar or ""):
        return False

    q_ar_clean = (q_ar or "").strip().rstrip("؟").strip()
    words_ar = q_ar_clean.split()
    if q_ar_clean and len(words_ar) < 2:
        return False

    trivial_phrases_en = {
        "let her be", "what did you say", "are you sure", "excuse me",
        "can you hear me", "what is your name", "what's your name", "who are you",
        "repeat that", "are you listening", "hello", "hi there", "is that so",
        "pardon me", "listen to me", "did you hear", "where were you shot",
        "how old are you", "where do you live", "can i take a picture",
        "nice to meet you", "take a photo", "take a picture", "take a selfie",
        "pet the dog", "pet him", "good dog", "call 911", "call the police",
    }
    for tp in trivial_phrases_en:
        if q_en_clean.startswith(tp) or q_en_clean == tp:
            return False

    trivial_phrases_ar = [
        "أين أصبت", "كم عمرك", "ما اسمك", "من أنت", "هل تسمعني",
        "ماذا قلت", "عفوا", "معذرة", "أهلا", "مرحبا", "أين تسكن",
        "هل يمكن التقاط صورة", "تشرفنا", "الشرطة", "سلاح ناري",
        "التبرع", "تبرع الآن", "اشترك في القناة", "رابط الوصف"
    ]
    for tp in trivial_phrases_ar:
        if q_ar_clean.startswith(tp) or q_ar_clean == tp:
            return False

    return True


def collapse_repetition_loops(text: str) -> str:
    """Collapses ASR acoustic hallucination loops (e.g. 'Listen. Listen. Listen.' -> 'Listen.')."""
    if not text:
        return ""
    # Collapse repetitive single-word phrases repeated 3+ times
    collapsed = re.sub(r'\b(\w+[\s,.]*)\s*(?:\1){2,}', r'\1 ', text, flags=re.IGNORECASE)
    return " ".join(collapsed.split()).strip()


def merge_consecutive_speaker_turns(turns: List[Dict[str, Any]], max_gap_ms: int = 2500) -> List[Dict[str, Any]]:
    """
    Merges rapid consecutive micro-turns spoken by the same speaker.
    Prevents chopped speech fragments from confusing the LLM and enables clean verbatim quotes.
    Also suppresses Faster-Whisper repetition loops.
    """
    if not turns:
        return []
    merged: List[Dict[str, Any]] = []
    for t in turns:
        t_copy = dict(t)
        # Suppress any Whisper repetition loops in raw text
        if t_copy.get("text_original"):
            t_copy["text_original"] = collapse_repetition_loops(t_copy["text_original"])
        if (
            merged
            and merged[-1].get("speaker_id") == t_copy.get("speaker_id")
            and merged[-1].get("episode_index") == t_copy.get("episode_index")
            and not t_copy.get("is_pedestrian_interruption")
            and not merged[-1].get("is_pedestrian_interruption")
            and (int(t_copy.get("start_ms", 0)) - int(merged[-1].get("end_ms", 0))) <= max_gap_ms
        ):
            # Concatenate text
            prev_text = (merged[-1].get("text_original") or "").strip()
            curr_text = (t_copy.get("text_original") or "").strip()
            if prev_text and curr_text:
                merged[-1]["text_original"] = f"{prev_text} {curr_text}"
            elif curr_text:
                merged[-1]["text_original"] = curr_text

            prev_ar = (merged[-1].get("text_ar") or "").strip()
            curr_ar = (t_copy.get("text_ar") or "").strip()
            if prev_ar and curr_ar:
                merged[-1]["text_ar"] = f"{prev_ar} {curr_ar}"
            elif curr_ar:
                merged[-1]["text_ar"] = curr_ar

            merged[-1]["end_ms"] = t_copy.get("end_ms", merged[-1]["end_ms"])
            merged[-1]["end_fmt"] = t_copy.get("end_fmt", merged[-1]["end_fmt"])
            words_a = merged[-1].get("words") or []
            words_b = t_copy.get("words") or []
            if isinstance(words_a, list) and isinstance(words_b, list):
                merged[-1]["words"] = words_a + words_b
            if "merged_turn_ids" not in merged[-1]:
                merged[-1]["merged_turn_ids"] = [merged[-1].get("turn_id", "")]
            if t_copy.get("turn_id"):
                merged[-1]["merged_turn_ids"].append(t_copy["turn_id"])
        else:
            if t_copy.get("turn_id"):
                t_copy.setdefault("merged_turn_ids", [t_copy["turn_id"]])
            merged.append(t_copy)
    return merged


def compute_window_budget_for_context(context_length: int) -> Tuple[int, int]:
    """
    Returns (max_words_per_window, max_chars_per_window) tuned for Qwen 14B prompt + 6K-8K output headroom.
    At 32,768 ctx, up to 7,500 raw words / 72,000 formatted chars (~19,500 tokens) keeps ~8,800 output tokens free.
    """
    ctx = int(context_length or 20480)
    if ctx >= 30000:
        return 7500, 72000
    if ctx >= 26000:
        return 6500, 62000
    if ctx >= 22000:
        return 5200, 49000
    if ctx >= 19000:
        return 4000, 38000
    return 3400, 30000


def _format_transcript_line(t: Dict[str, Any]) -> str:
    tid = t.get("turn_id", "")
    s_fmt = t.get("start_fmt", "00:00")
    e_fmt = t.get("end_fmt", "00:00")
    role = t.get("speaker_role", "INTERLOCUTOR")
    label = t.get("speaker_label_en") or t.get("speaker_id", "Speaker")
    flags = []
    if role in ("MAIN_DAEE", "CO_DAEE"):
        header_role = f"{role} ({label})"
    elif role == "PEDESTRIAN":
        header_role = "PEDESTRIAN | INTERRUPTION"
    else:
        header_role = label.upper()
    if t.get("is_overlap"):
        flags.append("OVERLAP")
    flag_str = f" | {' '.join(flags)}" if flags else ""
    return f"[{tid}] [{s_fmt} -> {e_fmt}] [{header_role}{flag_str}]: {t.get('text_original', '')}"


def _pick_balanced_split_index(
    cohesive_turns: List[Dict[str, Any]],
    start: int,
    hard_end: int,
    cum_words: List[int],
    cum_chars: List[int],
    target_words: int,
    target_chars: Optional[int],
) -> int:
    """Selects a split index in (start, hard_end] preferring episode transitions, then Da'ee turn endings."""
    base_w = cum_words[start]
    base_c = cum_chars[start]
    best_ep_k = None
    best_ep_dist = None
    best_daee_k = None
    best_daee_dist = None
    closest_k = hard_end
    closest_dist = None

    for k in range(start + 1, hard_end + 1):
        w_span = cum_words[k] - base_w
        c_span = cum_chars[k] - base_c
        w_ratio = w_span / max(1, target_words)
        c_ratio = (c_span / max(1, target_chars)) if target_chars else w_ratio
        progress = max(w_ratio, c_ratio)
        dist = abs(progress - 1.0)

        if progress >= 0.70 and (closest_dist is None or dist < closest_dist):
            closest_k = k
            closest_dist = dist

        if k < len(cohesive_turns):
            prev_t = cohesive_turns[k - 1]
            curr_t = cohesive_turns[k]
            if int(curr_t.get("episode_index", 1)) != int(prev_t.get("episode_index", 1)):
                if progress >= 0.55 and (best_ep_dist is None or dist < best_ep_dist):
                    best_ep_k = k
                    best_ep_dist = dist
            elif prev_t.get("speaker_role") in ("MAIN_DAEE", "CO_DAEE"):
                if progress >= 0.70 and (best_daee_dist is None or dist < best_daee_dist):
                    best_daee_k = k
                    best_daee_dist = dist

    return best_ep_k or best_daee_k or closest_k


def build_episode_windows(
    turns: List[Dict[str, Any]],
    max_words_per_window: int = 3400,
    context_carry_turns: int = 4,
    merge_episodes: bool = False,
    max_chars_per_window: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Packs ScenarioTurns into LLM context windows after merging rapid same-speaker micro-turns.
    - When `merge_episodes=False` (legacy default): groups strictly by `episode_index`.
    - When `merge_episodes=True`: merges the full conversation across `episode_index` boundaries into
      one unified window whenever it fits within `(max_words_per_window, max_chars_per_window)`, and
      splits oversized transcripts into balanced windows preferring episode or MAIN_DAEE boundaries.
    """
    if not turns:
        return []

    cohesive_turns = merge_consecutive_speaker_turns(turns)
    if not cohesive_turns:
        return []

    if merge_episodes:
        n_turns = len(cohesive_turns)
        cum_words = [0] * (n_turns + 1)
        cum_chars = [0] * (n_turns + 1)
        for i, ct in enumerate(cohesive_turns):
            cum_words[i + 1] = cum_words[i] + len((ct.get("text_original") or "").split())
            cum_chars[i + 1] = cum_chars[i] + len(_format_transcript_line(ct)) + 1

        total_words = cum_words[n_turns]
        total_chars = cum_chars[n_turns]
        if total_words <= max_words_per_window and (max_chars_per_window is None or total_chars <= max_chars_per_window):
            return [_pack_episode_window(1, int(cohesive_turns[0].get("episode_index", 1)), cohesive_turns)]

        n_chunks_w = math.ceil(total_words / max(1, max_words_per_window))
        n_chunks_c = math.ceil(total_chars / max(1, max_chars_per_window)) if max_chars_per_window else 1
        n_chunks = max(2, n_chunks_w, n_chunks_c)
        target_words = max(1, math.ceil(total_words / n_chunks))
        target_chars = math.ceil(total_chars / n_chunks) if max_chars_per_window else None

        windows: List[Dict[str, Any]] = []
        start = 0
        while start < n_turns:
            hard_end = start + 1
            while hard_end < n_turns:
                next_w = cum_words[hard_end + 1] - cum_words[start]
                next_c = cum_chars[hard_end + 1] - cum_chars[start]
                if next_w > max_words_per_window:
                    break
                if max_chars_per_window is not None and next_c > max_chars_per_window:
                    break
                hard_end += 1

            if hard_end >= n_turns:
                batch = cohesive_turns[start:]
                windows.append(_pack_episode_window(len(windows) + 1, int(batch[0].get("episode_index", 1)), batch))
                break

            split_idx = _pick_balanced_split_index(
                cohesive_turns, start, hard_end, cum_words, cum_chars, target_words, target_chars
            )
            batch = cohesive_turns[start:split_idx]
            windows.append(_pack_episode_window(len(windows) + 1, int(batch[0].get("episode_index", 1)), batch))

            same_episode = int(cohesive_turns[split_idx].get("episode_index", 1)) == int(
                cohesive_turns[split_idx - 1].get("episode_index", 1)
            )
            carry = min(context_carry_turns, len(batch) - 1) if (same_episode and context_carry_turns > 0) else 0
            start = max(start + 1, split_idx - carry)

        return windows

    by_ep: Dict[int, List[Dict[str, Any]]] = {}
    for t in cohesive_turns:
        ep = int(t.get("episode_index", 1))
        by_ep.setdefault(ep, []).append(t)

    windows = []
    for ep_idx in sorted(by_ep.keys()):
        ep_turns = by_ep[ep_idx]
        cur_batch: List[Dict[str, Any]] = []
        cur_words = 0
        cur_chars = 0

        for t in ep_turns:
            w_cnt = len((t.get("text_original") or "").split())
            c_cnt = len(_format_transcript_line(t)) + 1
            exceeds_w = (cur_words + w_cnt) > max_words_per_window
            exceeds_c = max_chars_per_window is not None and (cur_chars + c_cnt) > max_chars_per_window
            if cur_batch and (exceeds_w or exceeds_c):
                windows.append(_pack_episode_window(len(windows) + 1, ep_idx, cur_batch))
                carry = cur_batch[-context_carry_turns:] if context_carry_turns > 0 else []
                cur_batch = list(carry) + [t]
                cur_words = sum(len((x.get("text_original") or "").split()) for x in cur_batch)
                cur_chars = sum(len(_format_transcript_line(x)) + 1 for x in cur_batch)
            else:
                cur_batch.append(t)
                cur_words += w_cnt
                cur_chars += c_cnt

        if cur_batch:
            windows.append(_pack_episode_window(len(windows) + 1, ep_idx, cur_batch))

    return windows


def _pack_episode_window(win_num: int, ep_idx: int, win_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    lines = [_format_transcript_line(t) for t in win_turns]

    return {
        "window_index": win_num,
        "episode_index": ep_idx,
        "start_ms": int(win_turns[0]["start_ms"]),
        "end_ms": int(win_turns[-1]["end_ms"]),
        "start_fmt": win_turns[0]["start_fmt"],
        "end_fmt": win_turns[-1]["end_fmt"],
        "turn_count": len(win_turns),
        "word_count": sum(len((x.get("text_original") or "").split()) for x in win_turns),
        "formatted_transcript": "\n".join(lines),
        "turns": win_turns,
    }


DANGLING_PREPOSITIONS_EN = {
    "to", "of", "in", "for", "with", "at", "by", "from", "on", "about",
    "into", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among",
    "and", "or", "but", "because", "as", "if", "when", "than", "that",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "well",
}


def clean_dangling_quote(quote: str) -> str:
    """Strips trailing dangling prepositions and hanging conjunctions from a quote."""
    if not quote:
        return ""
    words = quote.strip().split()
    while words:
        last_word = words[-1].lower().rstrip(".,!?:;\"'؟،")
        if last_word in DANGLING_PREPOSITIONS_EN:
            words.pop()
        else:
            break
    cleaned = " ".join(words).strip()
    return cleaned if cleaned else quote.strip()


def check_theological_bias(unit: Dict[str, Any], a_turn: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Theological Guardrail:
    Detects whether an extracted QA unit or candidate answer turn expresses opponent theological bias
    (e.g., defending/justifying incest, claiming the Spirit of Truth is the Holy Spirit not Muhammad,
    asserting biblical laws are void/abolished, or asserting Jesus is God / Trinity as the truth).
    Returns reason string if rejected, or None if sound.
    """
    bias_patterns = [
        # 1. Defending / justifying incest
        (r'\b(defend(ing|s)?|justif(y|ying|ies)|permit(ting|s)?|allow(ing|s)?)\s+(incest)\b', 'Defending incest', False),
        (r'\b(incest\s+(is|was)\s+(justified|permissible|allowed|moral|acceptable|necessary))\b', 'Justifying incest', False),
        (r'(تبرير|إباحة|جواز|الدفاع عن)\s+زنا المحارم', 'تبرير زنا المحارم', False),
        (r'زنا المحارم\s+(مباح|جائز|حلال|مبرر)', 'زنا المحارم مباح', False),

        # 2. Spirit of Truth is Holy Spirit (denying it is Prophet Muhammad)
        (r'\b(spirit of truth|comforter|paraclete)\s+(is|means)\s+(the\s+)?holy spirit\b', 'Claiming Spirit of Truth is Holy Spirit', True),
        (r'\b(spirit of truth|comforter)\s+is\s+not\s+(prophet\s+)?muhammad\b', 'Denying Comforter is Muhammad', False),
        (r'(روح الحق|المعزي|البارقليط)\s+(هو|يعني)\s+الروح القدس', 'روح الحق هو الروح القدس', True),
        (r'(روح الحق|المعزي|البارقليط)\s+ليس\s+(هو\s+)?محمد', 'المعزي ليس محمدا', False),

        # 3. Biblical laws are void / abolished
        (r'\b(biblical|old testament|mosaic)\s+laws?\s+(are|is)\s+(void|abolished|nullified)\b', 'Biblical laws void', True),
        (r'\b(the\s+law\s+is\s+(void|abolished|null and void))\b', 'Law is void', True),
        (r'(الناموس|شريعة العهد القديم|الشريعة الموسوية)\s+(باطل|ملغى|منسوخ بالكلية لإسقاط الأخلاق)', 'الناموس باطل', True),

        # 4. Asserting Christ is God or Trinity as the Da'ee's position
        (r'\b(jesus|christ)\s+is\s+(the\s+)?(true\s+)?god\b', 'Christ is God asserted as truth', True),
        (r'(المسيح|يسوع|عيسى)\s+هو\s+الله', 'المسيح هو الله', True),
        (r'(الثالوث|التثليث)\s+حق', 'الثالوث حق', True),

        # 5. Asserting vicarious crucifixion / atonement as the Da'ee's position
        (r'\b(jesus|christ)\s+died\s+for\s+(our\s+)?sins\b', 'Crucifixion atonement asserted as truth', True),
        (r'(المسيح|يسوع)\s+مات\s+(تكفيراً|فداءً)\s+عن\s+خطايانا', 'المسيح مات فداء عن خطايانا', True),
    ]

    texts_to_check = [
        str(unit.get("core_answer_summary_en") or ""),
        str(unit.get("core_answer_summary_ar") or ""),
        str(unit.get("verbatim_quote") or ""),
        str(unit.get("verbatim_quote_ar") or ""),
    ]

    for pk in ("logical_premises", "logical_premises_ar"):
        val = unit.get(pk)
        if isinstance(val, list):
            texts_to_check.extend(str(x) for x in val)

    if a_turn:
        texts_to_check.append(str(a_turn.get("text_original") or ""))
        texts_to_check.append(str(a_turn.get("text_ar") or ""))

    def _is_refutation(text: str, match_start: int) -> bool:
        window = text[max(0, match_start - 80):match_start].lower()
        refutation_markers = [
            "refut", "reject", "deni", "deny", "false", "fallacy", "misconception",
            "claim", "dogma", "notion", "myth", "alleg", "argument that", "oppose",
            "disprov", "counter", "refuting", "rejecting", "denying", "refutation",
            "زعم", "دعوى", "شبهة", "باطل", "بطلان", "رد", "دحض", "إبطال", "نفي", "إنكار",
            "ليس", "لا يمكن", "يستحيل", "كيف يكون", "كيف يعقل", "تناقض", "مغالطة"
        ]
        return any(marker in window for marker in refutation_markers)

    combined_text = " ".join(texts_to_check)
    for pattern, reason, check_refutation in bias_patterns:
        m = re.search(pattern, combined_text, re.IGNORECASE)
        if m:
            if check_refutation and _is_refutation(combined_text, m.start()):
                continue
            return reason
    return None


def anchor_and_enrich_extracted_unit(
    raw_unit: Dict[str, Any],
    scenario: Dict[str, Any],
    episode_index: int,
    window_start_ms: int,
    unit_seq_num: int,
    cohesive_turns: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Anchors LLM-extracted turn IDs and speaker identities directly against ground-truth
    cohesive turns and diarized Speaker records.
    Returns None if the unit fails theological validation, perspective integrity, or attributes opponent bias.
    """
    item = sanitize_arabic_fields({k: v for k, v in raw_unit.items() if not str(k).startswith("_")})
    item.pop("rubric_audit", None)
    vid = scenario["video_id"]
    yt_url = scenario.get("youtube_url") or f"https://www.youtube.com/watch?v={vid}"

    # 1. Question Syntax & 1st-Person Confessional Normalization
    raw_q_en = str(item.get("question_en") or "").strip()
    raw_q_ar = str(item.get("question_ar") or "").strip()
    norm_q_en, norm_q_ar, is_valid_perspective = normalize_question_syntax_and_perspective(raw_q_en, raw_q_ar)
    if not is_valid_perspective or not is_valid_dialectical_question(norm_q_en, norm_q_ar):
        return None
    item["question_en"] = norm_q_en
    if norm_q_ar:
        item["question_ar"] = norm_q_ar

    if cohesive_turns is None:
        cohesive_turns = merge_consecutive_speaker_turns(scenario.get("turns", []))

    cohesive_turns_by_id: Dict[str, Dict[str, Any]] = {}
    for ct in cohesive_turns:
        if "turn_id" in ct:
            cohesive_turns_by_id[ct["turn_id"]] = ct
        for mid in ct.get("merged_turn_ids", []):
            cohesive_turns_by_id[mid] = ct

    raw_turns_by_id = {t["turn_id"]: t for t in scenario.get("turns", []) if "turn_id" in t}
    speakers = scenario.get("speakers", [])
    main_daee = next((s for s in speakers if s.get("role") == "MAIN_DAEE"), None)

    q_tid = str(item.get("question_turn_id") or "").strip()
    a_tid = str(item.get("answer_turn_id") or "").strip()
    q_turn = cohesive_turns_by_id.get(q_tid) or raw_turns_by_id.get(q_tid)
    a_turn = cohesive_turns_by_id.get(a_tid) or raw_turns_by_id.get(a_tid)

    if q_turn is not None:
        ts_sec = int(q_turn["start_ms"] // 1000)
        q_role = q_turn.get("speaker_role", "")
        if q_role in ("MAIN_DAEE", "CO_DAEE"):
            inter_spk_id = ""  # Monologue / anticipated question voiced by Da'ee
        else:
            inter_spk_id = q_turn.get("speaker_id", "")
    else:
        ts_sec = max(int(window_start_ms // 1000), int(item.get("timestamp_seconds") or 0))
        inter_spk_id = ""

    if a_turn is None or a_turn.get("speaker_role") not in ("MAIN_DAEE", "CO_DAEE"):
        # Strictly reject / drop unit if candidate answer turn is missing or belongs to interlocutor
        return None

    # Substantive Answer Turn Recovery:
    # If the LLM selected a tiny filler turn (< 8 cleaned words, e.g. "Can" or "What do you think about that? That deep? Well,")
    # or a turn before the question turn, re-anchor a_turn to the first substantive MAIN_DAEE/CO_DAEE turn (>= 8 words).
    a_cleaned_text = clean_dangling_quote((a_turn.get("text_original") or "").strip())
    a_orig_words = [w for w in re.split(r"\s+", a_cleaned_text) if w]
    q_start_ms = int(q_turn["start_ms"]) if q_turn is not None else int(window_start_ms)
    if len(a_orig_words) < 8 or (q_turn is not None and int(a_turn.get("start_ms", 0)) < q_start_ms):
        raw_ex_candidates = item.get("dialogue_exchange") or item.get("dialogue_trajectory") or []
        replacement_turn = None
        if isinstance(raw_ex_candidates, list):
            for ex in raw_ex_candidates:
                if not isinstance(ex, dict):
                    continue
                cand_tid = str(ex.get("turn_id") or "").strip()
                cand_turn = cohesive_turns_by_id.get(cand_tid) or raw_turns_by_id.get(cand_tid)
                if cand_turn and cand_turn.get("speaker_role") in ("MAIN_DAEE", "CO_DAEE"):
                    cand_words = [w for w in re.split(r"\s+", clean_dangling_quote((cand_turn.get("text_original") or "").strip())) if w]
                    if len(cand_words) >= 8 and int(cand_turn.get("start_ms", 0)) >= q_start_ms:
                        replacement_turn = cand_turn
                        break
        if replacement_turn is None:
            for ct in cohesive_turns:
                if ct.get("speaker_role") in ("MAIN_DAEE", "CO_DAEE"):
                    ct_start = int(ct.get("start_ms", 0))
                    ct_words = [w for w in re.split(r"\s+", clean_dangling_quote((ct.get("text_original") or "").strip())) if w]
                    if ct_start >= q_start_ms and (ct_start - q_start_ms) <= 180_000 and len(ct_words) >= 8:
                        replacement_turn = ct
                        break
        if replacement_turn is not None:
            a_turn = replacement_turn
            a_tid = str(a_turn.get("turn_id") or a_tid)
            item["answer_turn_id"] = a_tid

    ans_sec = int(a_turn["start_ms"] // 1000)
    daee_spk_id = a_turn.get("speaker_id", main_daee["speaker_id"] if main_daee else "")

    # Speaker Role Integrity Guard: daee_speaker_id and interlocutor_speaker_id CANNOT be identical!
    if daee_spk_id and inter_spk_id and daee_spk_id == inter_spk_id:
        if q_turn and q_turn.get("speaker_role") in ("MAIN_DAEE", "CO_DAEE"):
            inter_spk_id = ""
        else:
            distinct_inter = next(
                (s["speaker_id"] for s in speakers
                 if s["speaker_id"] != daee_spk_id and s.get("role") == "INTERLOCUTOR"),
                ""
            )
            inter_spk_id = distinct_inter
    if daee_spk_id and inter_spk_id and daee_spk_id == inter_spk_id:
        inter_spk_id = ""

    matched_daee = next(
        (s for s in speakers if s["speaker_id"] == daee_spk_id and s.get("role") in ("MAIN_DAEE", "CO_DAEE")),
        main_daee
    )
    if not matched_daee or matched_daee.get("role") not in ("MAIN_DAEE", "CO_DAEE"):
        return None

    daee_name_en = matched_daee.get("display_name_en") or (main_daee.get("display_name_en") if main_daee else "Main Da'ee")
    daee_name_ar = matched_daee.get("display_name_ar") or (main_daee.get("display_name_ar") if main_daee else "الداعية المحاور")
    daee_profile_id = matched_daee.get("profile_id") or (main_daee.get("profile_id") if main_daee else "")

    # Theological Guardrail: Strictly reject any unit that fell into an interlocutor
    if daee_name_ar in ("المحاور", "المحاور 1", "المحاور 2", "Interlocutor", "Interlocutor 1", "Interlocutor 2"):
        return None

    if not daee_name_ar and main_daee and main_daee.get("display_name_ar"):
        daee_name_ar = main_daee["display_name_ar"]
        daee_name_en = main_daee.get("display_name_en") or "Main Da'ee"
        daee_profile_id = main_daee.get("profile_id") or ""

    # Theological Guardrail: Validate no heresy or opponent bias is being asserted as truth
    bias_reason = check_theological_bias(item, a_turn)
    if bias_reason:
        return None

    # Collect ordered source turn IDs referenced in dialogue_exchange + q_tid + a_tid
    raw_exchange = item.get("dialogue_exchange") or item.get("dialogue_trajectory") or []
    if not isinstance(raw_exchange, list):
        raw_exchange = []

    src_tids: List[str] = []
    for cand_id in [q_tid, a_tid] + [
        str(ex.get("turn_id") or "").strip()
        for ex in raw_exchange
        if isinstance(ex, dict)
    ]:
        if cand_id and (cand_id in cohesive_turns_by_id or cand_id in raw_turns_by_id) and cand_id not in src_tids:
            src_tids.append(cand_id)

    enriched_exchange = []
    for ex in raw_exchange[:12]:
        if not isinstance(ex, dict):
            continue
        etid = str(ex.get("turn_id") or "").strip()
        eturn = cohesive_turns_by_id.get(etid) or raw_turns_by_id.get(etid)
        role = ex.get("role") or (eturn.get("speaker_role") if eturn else "MAIN_DAEE")
        if resolve_turn_speaker_labels is not None:
            spk_label_en, spk_label_ar = resolve_turn_speaker_labels(role, eturn, daee_name_en, daee_name_ar)
        else:
            is_daee = role in ("MAIN_DAEE", "CO_DAEE")
            turn_matches = (eturn is not None and (eturn.get("speaker_role") in ("MAIN_DAEE", "CO_DAEE")) == is_daee)
            spk_label_en = ((eturn.get("speaker_label_en") if turn_matches else None) or (daee_name_en if is_daee else "Interlocutor"))
            spk_label_ar = ((eturn.get("speaker_label_ar") if turn_matches else None) or (daee_name_ar if is_daee else "المحاور"))

        enriched_exchange.append({
            "turn_id": etid,
            "role": role,
            "speaker_label_en": spk_label_en,
            "speaker_label_ar": spk_label_ar,
            "timestamp_seconds": int(eturn["start_ms"] // 1000) if eturn else ts_sec,
            "timestamp_formatted": eturn["start_fmt"] if eturn else ms_to_fmt(ts_sec * 1000),
            "move_type": ex.get("move_type", "Turn"),
            "text_en": ex.get("text_en") or (eturn.get("text_original") if eturn else ""),
            "text_ar": ex.get("text_ar") or (eturn.get("text_ar") if eturn else ""),
        })

    # Dialogue Trajectory Role Integrity: Require distinct roles in multi-turn debate
    if len(enriched_exchange) >= 2:
        roles_in_ex = {ex.get("role") for ex in enriched_exchange if ex.get("role")}
        if len(roles_in_ex) < 2 and inter_spk_id:
            for ex in enriched_exchange:
                mv = str(ex.get("move_type") or "").lower()
                if any(k in mv for k in ["objection", "reaction", "clarification", "inquiry"]):
                    ex["role"] = "INTERLOCUTOR"
                    ex["speaker_label_en"] = "Interlocutor"
                    ex["speaker_label_ar"] = "المحاور"
                else:
                    ex["role"] = "MAIN_DAEE"
                    ex["speaker_label_en"] = daee_name_en
                    ex["speaker_label_ar"] = daee_name_ar

    # Analogies: support multiple analogies list + single analogy field
    raw_an = item.get("analogies")
    if isinstance(raw_an, list) and raw_an:
        clean_analogies = [str(a).strip() for a in raw_an if str(a).strip()]
    elif item.get("analogy_or_metaphor"):
        clean_analogies = [str(item["analogy_or_metaphor"]).strip()]
    else:
        clean_analogies = []

    raw_an_ar = item.get("analogies_ar")
    if isinstance(raw_an_ar, list) and raw_an_ar:
        clean_analogies_ar = [str(a).strip() for a in raw_an_ar if str(a).strip()]
    elif item.get("analogy_or_metaphor_ar"):
        clean_analogies_ar = [str(item["analogy_or_metaphor_ar"]).strip()]
    else:
        clean_analogies_ar = []

    if clean_analogies and not item.get("analogy_or_metaphor"):
        item["analogy_or_metaphor"] = clean_analogies[0]
    if clean_analogies_ar and not item.get("analogy_or_metaphor_ar"):
        item["analogy_or_metaphor_ar"] = clean_analogies_ar[0]

    # Scriptural citations
    raw_citations = item.get("scriptural_citations")
    clean_citations = raw_citations if isinstance(raw_citations, list) else []

    raw_cat = str(item.get("topic_category") or "").strip()
    canonical_cat = canonicalize_topic_category(raw_cat)
    if not item.get("topic_subcategory") and raw_cat != canonical_cat:
        item["topic_subcategory"] = raw_cat
    item["topic_category"] = canonical_cat

    # Architectural Acoustic Merge Wiring for Verbatim Quotes:
    all_transcript = " ".join([(t.get("text_original") or "").strip() for t in cohesive_turns])
    all_transcript_lower = all_transcript.lower()

    # Anti-Leakage Gate: Discard any card with prompt demonstration signatures not in transcript
    prompt_leak_signatures = [
        "whatever begins to exist has a cause", "train of freight carriages",
        "train cannot move on its own", "train engine without wheels",
        "soldier cannot fire his rifle", "soldier cannot pull the trigger",
        "child being held down by doctors", "surgeon operating on an unknowing child",
        "he is god in flesh. is he god or no", "lamp drawing current from a power plant",
        "lamp drawing electric current", "power plant itself",
        "surah at-tur 52:35-36", "mark 13:32",
    ]
    citation_text_parts = []
    for c in clean_citations:
        if isinstance(c, dict):
            citation_text_parts.append(str(c.get("reference") or ""))
            citation_text_parts.append(str(c.get("significance") or ""))
    card_all_text = " ".join([
        item.get("question_en", ""), item.get("core_answer_summary_en", ""),
        item.get("analogy_or_metaphor", ""), item.get("socratic_question", ""),
        item.get("verbatim_quote", ""), item.get("interlocutor_verbatim_quote", ""),
        " ".join(str(x) for x in clean_analogies),
        " ".join(str(x) for x in (item.get("logical_premises") or [])),
        " ".join(citation_text_parts),
    ]).lower()
    for sig in prompt_leak_signatures:
        if sig in card_all_text and sig not in all_transcript_lower:
            return None

    # Anti-Noise Gate: Discard cards referencing street incidents, dog petting, or promotional outro pitches
    unconditional_noise_signatures = [
        "pet him", "what kind of a dog", "pet the dog", "911 call",
        "verbal dispute", "islamnet", "donate now", "link in the description",
        "fundraising", "subscribe to the channel",
    ]
    for nsig in unconditional_noise_signatures:
        if nsig in card_all_text:
            return None

    # Police/weapon chatter must not trigger on rhetorical theological analogies in core_answer_summary/analogies
    spoken_and_q_text = " ".join([
        item.get("question_en", ""),
        item.get("verbatim_quote", ""),
        item.get("interlocutor_verbatim_quote", ""),
        " ".join(str(ex.get("text_en") or "") for ex in enriched_exchange),
    ]).lower()
    for nsig in ("call the police", "police officer", "brandish", "brandished", "firearm", "police come"):
        if nsig in spoken_and_q_text:
            return None

    # Architectural Acoustic Merge Wiring for Verbatim Quotes:
    # Read from cohesive_turns instead of raw micro-turns, ensuring >= 8 words and no dangling prepositions.
    raw_vq = (item.get("verbatim_quote") or "").strip()
    vq_words = [w for w in re.split(r'\s+', raw_vq) if w]
    has_dangling_vq = bool(vq_words and vq_words[-1].lower().rstrip(".,!?:;\"'؟،") in DANGLING_PREPOSITIONS_EN)
    is_short_vq = len(vq_words) < 8

    overlap_low = False
    if raw_vq and raw_vq not in all_transcript:
        w_vq = set(re.findall(r'\w+', raw_vq.lower()))
        w_all = set(re.findall(r'\w+', all_transcript.lower()))
        overlap = len(w_vq & w_all) / max(len(w_vq), 1) if w_vq else 0.0
        if overlap < 0.45:
            overlap_low = True

    if (overlap_low or is_short_vq or has_dangling_vq) and a_turn and a_turn.get("text_original"):
        a_orig = a_turn["text_original"].strip()
        a_words = [w for w in re.split(r'\s+', a_orig) if w]
        if len(a_words) >= 8:
            item["verbatim_quote"] = clean_dangling_quote(a_orig)
            if a_turn.get("text_ar"):
                item["verbatim_quote_ar"] = a_turn["text_ar"].strip()
        else:
            # Fallback to the longest substantive Da'ee turn in enriched_exchange
            daee_ex_turns = [
                ex for ex in enriched_exchange
                if ex.get("role") in ("MAIN_DAEE", "CO_DAEE") and len((ex.get("text_en") or "").split()) >= 8
            ]
            if daee_ex_turns:
                best_ex = max(daee_ex_turns, key=lambda x: len((x.get("text_en") or "").split()))
                item["verbatim_quote"] = clean_dangling_quote(best_ex["text_en"].strip())
                if best_ex.get("text_ar") and not item.get("verbatim_quote_ar"):
                    item["verbatim_quote_ar"] = best_ex["text_ar"].strip()
            elif raw_vq:
                item["verbatim_quote"] = clean_dangling_quote(raw_vq)
    elif raw_vq:
        item["verbatim_quote"] = clean_dangling_quote(raw_vq)

    if not item.get("verbatim_quote_ar"):
        for ex in enriched_exchange:
            if str(ex.get("turn_id") or "") == str(a_tid) and ex.get("text_ar"):
                item["verbatim_quote_ar"] = clean_dangling_quote(ex["text_ar"].strip())
                break
        if not item.get("verbatim_quote_ar"):
            for ex in enriched_exchange:
                if ex.get("role") in ("MAIN_DAEE", "CO_DAEE") and ex.get("text_ar"):
                    item["verbatim_quote_ar"] = clean_dangling_quote(ex["text_ar"].strip())
                    break

    raw_ivq = (item.get("interlocutor_verbatim_quote") or "").strip()
    ivq_words = [w for w in re.split(r'\s+', raw_ivq) if w]
    has_dangling_ivq = bool(ivq_words and ivq_words[-1].lower().rstrip(".,!?:;\"'؟،") in DANGLING_PREPOSITIONS_EN)
    is_short_ivq = len(ivq_words) < 8

    ivq_overlap_low = False
    if raw_ivq and raw_ivq not in all_transcript:
        w_ivq = set(re.findall(r'\w+', raw_ivq.lower()))
        w_all = set(re.findall(r'\w+', all_transcript.lower()))
        ivq_overlap = len(w_ivq & w_all) / max(len(w_ivq), 1) if w_ivq else 0.0
        if ivq_overlap < 0.45:
            ivq_overlap_low = True

    if (ivq_overlap_low or is_short_ivq or has_dangling_ivq) and q_turn and q_turn.get("text_original"):
        q_orig = q_turn["text_original"].strip()
        q_words = [w for w in re.split(r'\s+', q_orig) if w]
        if len(q_words) >= 8:
            item["interlocutor_verbatim_quote"] = clean_dangling_quote(q_orig)
            if q_turn.get("text_ar"):
                item["interlocutor_verbatim_quote_ar"] = q_turn["text_ar"].strip()
        elif raw_ivq:
            item["interlocutor_verbatim_quote"] = clean_dangling_quote(raw_ivq)
    elif raw_ivq:
        item["interlocutor_verbatim_quote"] = clean_dangling_quote(raw_ivq)

    if not item.get("interlocutor_verbatim_quote_ar"):
        for ex in enriched_exchange:
            if str(ex.get("turn_id") or "") == str(q_tid) and ex.get("text_ar"):
                item["interlocutor_verbatim_quote_ar"] = clean_dangling_quote(ex["text_ar"].strip())
                break
        if not item.get("interlocutor_verbatim_quote_ar"):
            for ex in enriched_exchange:
                if ex.get("role") == "INTERLOCUTOR" and ex.get("text_ar"):
                    item["interlocutor_verbatim_quote_ar"] = clean_dangling_quote(ex["text_ar"].strip())
                    break

    # Topical Grounding Guard: reject fleeting <=5-word remarks whose topic was never actually answered by the Da'ee
    final_ivq_words = (item.get("interlocutor_verbatim_quote") or "").split()
    if len(final_ivq_words) <= 5:
        topic_stopwords = {
            "what", "when", "where", "which", "that", "this", "with", "from", "about",
            "does", "have", "their", "they", "would", "could", "should", "muslims",
            "muslim", "islam", "islamic", "christian", "christians", "christianity",
            "bible", "biblical", "quran", "allah", "god", "jesus", "prophet", "people",
        }
        q_content_tokens = set(re.findall(r"[a-z]{5,}", (item.get("question_en") or "").lower())) - topic_stopwords
        if len(q_content_tokens) >= 2:
            daee_spoken = " ".join(
                [item.get("verbatim_quote", ""), str(a_turn.get("text_original") or "")]
                + [str(ex.get("text_en") or "") for ex in enriched_exchange if ex.get("role") in ("MAIN_DAEE", "CO_DAEE")]
            ).lower()
            daee_tokens = set(re.findall(r"[a-z]{5,}", daee_spoken))
            if not (q_content_tokens & daee_tokens):
                return None

    resolved_episode_index = int(
        (q_turn.get("episode_index") if q_turn else None)
        or (a_turn.get("episode_index") if a_turn else None)
        or episode_index
    )

    item.update({
        "id": f"{vid}_{unit_seq_num:03d}",
        "video_id": vid,
        "channel": scenario.get("channel", "SCDawah Arabic"),
        "video_title_ar": scenario.get("title_ar", ""),
        "youtube_url": yt_url,
        "timestamp_seconds": ts_sec,
        "timestamp_formatted": ms_to_fmt(ts_sec * 1000),
        "timestamp_url": f"{yt_url}&t={ts_sec}s",
        "answer_timestamp_seconds": ans_sec,
        "answer_timestamp_formatted": ms_to_fmt(ans_sec * 1000),
        "episode_index": resolved_episode_index,
        "daee_speaker_id": daee_spk_id,
        "daee_profile_id": daee_profile_id,
        "daee_name_en": daee_name_en,
        "daee_name_ar": daee_name_ar,
        "interlocutor_speaker_id": inter_spk_id,
        "dialogue_exchange": enriched_exchange,
        "dialogue_trajectory_turns_count": len(enriched_exchange),
        "analogies": clean_analogies,
        "analogies_ar": clean_analogies_ar,
        "scriptural_citations": clean_citations,
        "source_turn_ids": src_tids,
    })
    return item


_CONDITIONAL_EN_SCALAR_KEYS = (
    "question_en",
    "core_answer_summary_en",
    "fallacy_or_misconception_exposed",
    "analogy_or_metaphor",
    "socratic_question",
    "topic_category",
    "topic_subcategory",
    "interlocutor_type",
    "question_turn_id",
    "answer_turn_id",
    "verbatim_quote",
    "interlocutor_verbatim_quote",
)


def merge_round1_and_round2_units(
    r1_units: List[Dict[str, Any]],
    r2_units: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deterministically zipper-merges Round 1 (pure English dialectical reasoning)
    with Round 2 (bounded-thinking Rubric Verification, conditional EN repairs, & Arabic localization).

    Protects against:
    - Out-of-order LLM response generation in Round 2 (matches by turn IDs or question similarity).
    - Forced/needless English rewrites when Round 1 already passed quality checks (`english_edits_needed=false`).
    - Dropping English dialogue turns if Round 2 truncates dialogue turns.
    - Wiping out English scriptural citation source/significance if Round 2 only provides Arabic significance.
    - Wiping out multiple English analogies.
    """
    if not r1_units:
        return []
    if not r2_units:
        return list(r1_units)

    used_r2_indices = set()
    final_merged = []

    for r1_idx, r1_u in enumerate(r1_units):
        r1_qtid = str(r1_u.get("question_turn_id") or "").strip()
        r1_atid = str(r1_u.get("answer_turn_id") or "").strip()
        r1_qen = (r1_u.get("question_en") or "").strip().lower()

        best_r2_idx = None

        # 1. Match by question_turn_id
        if r1_qtid:
            for r2_idx, r2_cand in enumerate(r2_units):
                if r2_idx in used_r2_indices:
                    continue
                r2_qtid = str(r2_cand.get("question_turn_id") or "").strip()
                if r2_qtid and r2_qtid == r1_qtid:
                    best_r2_idx = r2_idx
                    break

        # 2. Match by answer_turn_id
        if best_r2_idx is None and r1_atid:
            for r2_idx, r2_cand in enumerate(r2_units):
                if r2_idx in used_r2_indices:
                    continue
                r2_atid = str(r2_cand.get("answer_turn_id") or "").strip()
                if r2_atid and r2_atid == r1_atid:
                    best_r2_idx = r2_idx
                    break

        # 3. Match by question text similarity / containment
        if best_r2_idx is None and r1_qen:
            for r2_idx, r2_cand in enumerate(r2_units):
                if r2_idx in used_r2_indices:
                    continue
                r2_qen = (r2_cand.get("question_en") or "").strip().lower()
                if r2_qen and (r2_qen == r1_qen or r1_qen in r2_qen or r2_qen in r1_qen):
                    best_r2_idx = r2_idx
                    break

        # 4. Positional fallback if index exists and is unused
        if best_r2_idx is None and r1_idx < len(r2_units) and r1_idx not in used_r2_indices:
            best_r2_idx = r1_idx

        # 5. Any first unused r2 candidate
        if best_r2_idx is None:
            for r2_idx in range(len(r2_units)):
                if r2_idx not in used_r2_indices:
                    best_r2_idx = r2_idx
                    break

        if best_r2_idx is not None:
            used_r2_indices.add(best_r2_idx)
            r2_u = r2_units[best_r2_idx]
        else:
            r2_u = {}

        merged = dict(r1_u)
        r2_audit = r2_u.get("rubric_audit") if isinstance(r2_u.get("rubric_audit"), dict) else {}
        if r2_audit.get("drop_card") is True:
            merged["_dropped_by_round2_audit"] = True
            merged["_drop_reason"] = str(r2_audit.get("hard_fail") or "Dropped by Round 2 Rubric Audit")
            final_merged.append(merged)
            continue

        english_edits_needed = bool(r2_audit.get("english_edits_needed") is True)

        # Merge Arabic scalar & list fields
        for k, v in r2_u.items():
            if (k.endswith("_ar") or k == "topic_category_ar") and v:
                merged[k] = v

        # Conditional English repairs (ONLY when explicitly needed or missing in R1)
        for en_key in _CONDITIONAL_EN_SCALAR_KEYS:
            r2_val = r2_u.get(en_key)
            if not r2_val or not str(r2_val).strip():
                continue
            r1_val = merged.get(en_key)
            if not r1_val or not str(r1_val).strip():
                merged[en_key] = r2_val
            elif english_edits_needed and str(r2_val).strip() != str(r1_val).strip():
                if en_key == "core_answer_summary_en":
                    if len(str(r2_val).split()) >= 30 or len(str(r1_val).split()) < 30:
                        merged[en_key] = r2_val
                else:
                    merged[en_key] = r2_val

        for list_key in ("logical_premises", "analogies", "tags"):
            r2_list = r2_u.get(list_key)
            if not isinstance(r2_list, list) or not r2_list:
                continue
            r1_list = merged.get(list_key) or []
            if not r1_list:
                merged[list_key] = r2_list
            elif english_edits_needed:
                if list_key == "logical_premises":
                    if len(r2_list) >= len(r1_list) or len(r2_list) >= 3:
                        merged[list_key] = r2_list
                else:
                    merged[list_key] = r2_list

        # Zipper-merge dialogue exchange
        r1_ex = list(r1_u.get("dialogue_exchange") or r1_u.get("dialogue_trajectory") or [])
        r2_ex = list(r2_u.get("dialogue_exchange") or r2_u.get("dialogue_trajectory") or [])

        if r2_ex:
            merged_exchange = []
            r2_turns_by_id = {
                str(t.get("turn_id") or "").strip(): t
                for t in r2_ex if isinstance(t, dict) and t.get("turn_id")
            }
            used_r2_turns = set()

            for t_idx, t1 in enumerate(r1_ex):
                if not isinstance(t1, dict):
                    continue
                tid1 = str(t1.get("turn_id") or "").strip()
                t2 = None
                if tid1 and tid1 in r2_turns_by_id and tid1 not in used_r2_turns:
                    t2 = r2_turns_by_id[tid1]
                    used_r2_turns.add(tid1)
                elif t_idx < len(r2_ex) and id(r2_ex[t_idx]) not in used_r2_turns:
                    t2 = r2_ex[t_idx]
                    used_r2_turns.add(id(r2_ex[t_idx]))

                turn_merged = dict(t1)
                if t2 and isinstance(t2, dict):
                    if t2.get("text_ar"):
                        turn_merged["text_ar"] = t2["text_ar"]
                    if t2.get("text_en") and (not turn_merged.get("text_en") or english_edits_needed):
                        turn_merged["text_en"] = t2["text_en"]
                    if t2.get("move_type") and not turn_merged.get("move_type"):
                        turn_merged["move_type"] = t2["move_type"]
                    if t2.get("role") and not turn_merged.get("role"):
                        turn_merged["role"] = t2["role"]
                    if t2.get("speaker_label_ar"):
                        turn_merged["speaker_label_ar"] = t2["speaker_label_ar"]
                merged_exchange.append(turn_merged)

            # If R2 supplied additional turns not in R1, preserve them
            for t2 in r2_ex:
                if not isinstance(t2, dict):
                    continue
                tid2 = str(t2.get("turn_id") or "").strip()
                if (tid2 and tid2 not in used_r2_turns) and (id(t2) not in used_r2_turns):
                    merged_exchange.append(t2)

            merged["dialogue_exchange"] = merged_exchange
        elif r1_ex:
            merged["dialogue_exchange"] = r1_ex

        # Zipper-merge scriptural citations
        r1_sc = list(r1_u.get("scriptural_citations") or [])
        r2_sc = list(r2_u.get("scriptural_citations") or [])
        if r2_sc:
            merged_citations = []
            used_r2_sc = set()
            for sc_idx, sc1 in enumerate(r1_sc):
                if not isinstance(sc1, dict):
                    merged_citations.append(sc1)
                    continue
                ref1 = str(sc1.get("reference") or "").strip().lower()
                sc2 = None
                if ref1:
                    for r2_idx, cand in enumerate(r2_sc):
                        if r2_idx in used_r2_sc or not isinstance(cand, dict):
                            continue
                        ref2 = str(cand.get("reference") or "").strip().lower()
                        if ref2 and (ref2 == ref1 or ref1 in ref2 or ref2 in ref1):
                            sc2 = cand
                            used_r2_sc.add(r2_idx)
                            break
                if sc2 is None and sc_idx < len(r2_sc) and sc_idx not in used_r2_sc and isinstance(r2_sc[sc_idx], dict):
                    sc2 = r2_sc[sc_idx]
                    used_r2_sc.add(sc_idx)

                sc_merged = dict(sc1)
                if sc2 and isinstance(sc2, dict):
                    if sc2.get("significance_ar"):
                        sc_merged["significance_ar"] = sc2["significance_ar"]
                    if not sc_merged.get("source") and sc2.get("source"):
                        sc_merged["source"] = sc2["source"]
                    if not sc_merged.get("reference") and sc2.get("reference"):
                        sc_merged["reference"] = sc2["reference"]
                    if sc2.get("significance") and (not sc_merged.get("significance") or english_edits_needed):
                        sc_merged["significance"] = sc2["significance"]
                merged_citations.append(sc_merged)

            for r2_idx, sc2 in enumerate(r2_sc):
                if r2_idx not in used_r2_sc:
                    merged_citations.append(sc2)
            merged["scriptural_citations"] = merged_citations
        elif r1_sc:
            merged["scriptural_citations"] = r1_sc

        # Preserve or conditionally update analogies
        if r2_u.get("analogies_ar"):
            merged["analogies_ar"] = r2_u["analogies_ar"]

        final_merged.append(merged)

    return final_merged


def has_valid_arabic_localization(unit: Dict[str, Any]) -> bool:
    """
    Strict Transactional Barrier: Verifies that a Q&A dossier contains genuine Classical Arabic
    in its core fields (question_ar, core_answer_summary_ar) and across all populated Arabic fields.
    Rejects half-baked English-only cards where Round 2 failed, timed out, or produced empty Arabic fields.
    """
    if not isinstance(unit, dict):
        return False

    arabic_regex = re.compile(r"[\u0600-\u06ff]")
    foreign_script_regex = re.compile(r"[\u0400-\u04ff\u0590-\u05ff\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]")

    def _is_clean_ar_str(val: str) -> bool:
        return bool(val and arabic_regex.search(val) and not foreign_script_regex.search(val))

    # 1. Core mandatory theological fields that MUST be present, contain Arabic script, and be free of foreign script
    q_ar = str(unit.get("question_ar") or "").strip()
    ans_ar = str(unit.get("core_answer_summary_ar") or "").strip()
    if not _is_clean_ar_str(q_ar) or not _is_clean_ar_str(ans_ar):
        return False

    # 2. Any populated Arabic scalar field must contain authentic Arabic script and no foreign script
    for f in [
        "fallacy_or_misconception_exposed_ar",
        "analogy_or_metaphor_ar",
        "socratic_question_ar",
        "verbatim_quote_ar",
        "interlocutor_verbatim_quote_ar",
    ]:
        val = str(unit.get(f) or "").strip()
        if val and not _is_clean_ar_str(val):
            return False

    # 3. Populated Arabic premises / analogies must contain clean Arabic script
    for list_field in ("logical_premises_ar", "analogies_ar"):
        vals = unit.get(list_field)
        if isinstance(vals, str):
            try:
                vals = json.loads(vals)
            except Exception:
                vals = [vals] if vals.strip() else []
        if isinstance(vals, list) and vals:
            if not all(isinstance(p, str) and _is_clean_ar_str(p) for p in vals):
                return False

    # 4. Populated Arabic text in scriptural_citations & dialogue_exchange must be free of foreign scripts
    for cit in (unit.get("scriptural_citations") or []):
        if isinstance(cit, dict):
            sig_ar = str(cit.get("significance_ar") or "").strip()
            if sig_ar and not _is_clean_ar_str(sig_ar):
                return False
    for ex in (unit.get("dialogue_exchange") or unit.get("dialogue_trajectory") or []):
        if isinstance(ex, dict):
            txt_ar = str(ex.get("text_ar") or "").strip()
            if txt_ar and not _is_clean_ar_str(txt_ar):
                return False

    return True


def _build_round2_transcript_excerpt(
    formatted_transcript: str,
    r1_units: List[Dict[str, Any]],
    max_words: int = 1800,
) -> str:
    """
    Selects a focused transcript excerpt for Round 2 centered on the turn IDs referenced
    by Round 1 units so the prompt never overflows the LLM context window.
    """
    text = (formatted_transcript or "").strip()
    words = text.split()
    if len(words) <= max_words:
        return text

    referenced_tids = set()
    for u in r1_units:
        if not isinstance(u, dict):
            continue
        for k in ("question_turn_id", "answer_turn_id"):
            val = str(u.get(k) or "").strip()
            if val:
                referenced_tids.add(val)
        for ex in (u.get("dialogue_exchange") or u.get("dialogue_trajectory") or []):
            if isinstance(ex, dict) and ex.get("turn_id"):
                referenced_tids.add(str(ex["turn_id"]).strip())

    lines = text.splitlines()
    selected_lines = []
    word_acc = 0
    for idx, ln in enumerate(lines):
        if any(tid in ln for tid in referenced_tids):
            # Include immediate neighbors for natural flow
            start_i = max(0, idx - 1)
            end_i = min(len(lines), idx + 2)
            for sub_ln in lines[start_i:end_i]:
                if sub_ln not in selected_lines:
                    w_cnt = len(sub_ln.split())
                    if word_acc + w_cnt <= max_words:
                        selected_lines.append(sub_ln)
                        word_acc += w_cnt

    if selected_lines:
        return "\n".join(selected_lines)
    return " ".join(words[:max_words])


class Stage2ScenarioExtractor:
    """
    Runs Qwen 2.5 14B Instruct (Q5_K_M GGUF, 20,480 ctx, 8-bit KV cache) over
    Stage 1 Scenario JSON files, saving every episode window immediately to Drive.
    """

    def __init__(
        self,
        workspace: SharedWorkspace,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        context_length: int = 20480,
        llm_client: Optional[Any] = None,
    ):
        self.workspace = workspace
        self.context_length = context_length
        if llm_client is not None:
            self.client = llm_client
        else:
            from core.colab_runner import LocalGPUClient
            self.client = LocalGPUClient(
                model_name=model_name,
                load_in_4bit=True,
                use_gguf_q5=True,
                context_length=context_length,
            )

    def extract_video_m6(self, video_id: str, verbose: bool = True) -> Optional[List[Dict[str, Any]]]:
        scen_path = self.workspace.scenarios_dir / f"{video_id}.json"
        qa_shard_path = self.workspace.extractions_dir / f"{video_id}.qa.json"
        state = self.workspace.get_video_state(video_id)

        if state["step_rank"] >= STEP_RANK["M6_LLM_EXTRACTED"] and qa_shard_path.exists():
            saved = self.workspace.read_json(qa_shard_path, default={}) or {}
            saved_units = saved.get("units", []) if isinstance(saved, dict) else saved
            if isinstance(saved_units, list) and (
                not saved_units or all(has_valid_arabic_localization(u) for u in saved_units)
            ):
                return saved_units

        if not scen_path.exists():
            return None

        scenario = self.workspace.read_json(scen_path)
        if not isinstance(scenario, dict):
            return None

        # Filter to substantive speech turns (keep pedestrian lines marked so LLM has context)
        turns = scenario.get("turns", [])
        substantive_words = sum(
            len((t.get("text_original") or "").split())
            for t in turns
            if not t.get("is_pedestrian_interruption")
        )
        if substantive_words < 25:
            payload = {"video_id": video_id, "extracted_at": time.time(), "units": []}
            self.workspace.atomic_write_json(qa_shard_path, payload)
            self.workspace.advance_micro_step(video_id, "M6_LLM_EXTRACTED", metrics={"qa_count": 0})
            return []

        cohesive_turns = merge_consecutive_speaker_turns(turns)
        # Dynamically scale unified full-conversation window capacity with active LLM context length:
        # At 32K ctx (7,500 words / 72,000 chars), >99% of full debates fit into 1 single context window.
        act_ctx = getattr(self.client, "context_length", getattr(self, "context_length", 20480))
        dyn_win_words, dyn_win_chars = compute_window_budget_for_context(act_ctx)
        windows = build_episode_windows(
            turns,
            max_words_per_window=dyn_win_words,
            context_carry_turns=4,
            merge_episodes=True,
            max_chars_per_window=dyn_win_chars,
        )
        speakers_summary = ", ".join(
            f"{s['speaker_id']}={s['role']}({s['display_name_en']})"
            for s in scenario.get("speakers", [])
            if s.get("role") != "PEDESTRIAN"
        )

        if verbose:
            print(
                f"\n[Stage 2 LLM] [{video_id}] {scenario.get('title_ar', '')[:55]} "
                f"| Turns: {len(turns)} | Episode Windows: {len(windows)}",
                flush=True,
            )

        all_units: List[Dict[str, Any]] = []
        completed_win_ids = []

        for w in windows:
            w_idx = w["window_index"]
            partial_file = self.workspace.partial_extractions_dir / f"{video_id}_win{w_idx:03d}.json"
            partial_en_file = self.workspace.partial_extractions_dir / f"{video_id}_win{w_idx:03d}.en.json"

            # Pre-Extraction Gatekeeper: Screen window turns for substantive apologetics
            win_turns = w.get("turns") or []
            is_substantive, gate_reason = is_substantive_apologetics_window(
                win_turns=win_turns,
                window_index=w_idx,
                total_windows=len(windows),
            )
            if not is_substantive:
                if verbose:
                    print(f"      [Gatekeeper Window {w_idx}] Skipping non-apologetic window: {gate_reason}", flush=True)
                self.workspace.atomic_write_json(partial_file, [])
                if partial_en_file.exists():
                    try:
                        partial_en_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                completed_win_ids.append(w_idx)
                continue

            # 1. Micro-Step Resume: load already-completed episode window from Drive!
            stale_partial_en_units = None
            if partial_file.exists():
                cached_win_units = self.workspace.read_json(partial_file, default=None)
                if isinstance(cached_win_units, list):
                    if not cached_win_units:
                        # Empty list means legitimately skipped window (e.g. noise or no questions)
                        completed_win_ids.append(w_idx)
                        continue
                    elif all(has_valid_arabic_localization(cu) for cu in cached_win_units):
                        for cu in cached_win_units:
                            if not _is_duplicate_qa(cu, all_units):
                                cu["id"] = f"{video_id}_{len(all_units) + 1:03d}"
                                all_units.append(cu)
                        completed_win_ids.append(w_idx)
                        continue
                    else:
                        # Stale English-only partial window - purge corrupted partial file and treat as Round 1 units for Round 2 Arabic localization!
                        stale_partial_en_units = cached_win_units
                        partial_file.unlink(missing_ok=True)
                        if verbose:
                            print(f"      [Window {w_idx}] Self-healing missing Arabic in cached partial window ({len(cached_win_units)} units)...", flush=True)

            if w["word_count"] < 20:
                self.workspace.atomic_write_json(partial_file, [])
                if partial_en_file.exists():
                    try:
                        partial_en_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                completed_win_ids.append(w_idx)
                continue

            main_daee = next((s for s in scenario.get("speakers", []) if s.get("role") == "MAIN_DAEE"), None)
            target_name = (main_daee.get("display_name_en") if main_daee else "Main Da'ee") or "Main Da'ee"
            target_id = (main_daee.get("speaker_id") if main_daee else "SPK_00") or "SPK_00"

            # Check if client supports Two-Round extraction or is single-pass legacy
            if not (hasattr(self.client, "extract_round1_english") or hasattr(self.client, "extract_round2_arabic")):
                prompt_legacy = USER_EXTRACTION_PROMPT_TEMPLATE.format(
                    title_ar=scenario.get("title_ar", ""),
                    youtube_url=scenario.get("youtube_url", ""),
                    episode_index=w["episode_index"],
                    start_fmt=w["start_fmt"],
                    end_fmt=w["end_fmt"],
                    speakers_summary=speakers_summary or "MAIN_DAEE vs INTERLOCUTOR",
                    target_daee_name=target_name,
                    target_daee_id=target_id,
                    transcript_text=w["formatted_transcript"],
                )
                t0_sp = time.time()
                try:
                    raw_out = self.client.extract(prompt_legacy, system_prompt=EXTRACTION_SYSTEM_PROMPT)
                except TypeError:
                    raw_out = self.client.extract(prompt_legacy)
                elapsed_sp = round(time.time() - t0_sp, 1)

                parsed = clean_llm_json_array(raw_out)
                win_enriched: List[Dict[str, Any]] = []
                for raw_item in parsed:
                    if not is_valid_dialectical_question(raw_item.get("question_en", ""), raw_item.get("question_ar", "")):
                        continue
                    enriched = anchor_and_enrich_extracted_unit(
                        raw_unit=raw_item,
                        scenario=scenario,
                        episode_index=w["episode_index"],
                        window_start_ms=w["start_ms"],
                        unit_seq_num=len(all_units) + len(win_enriched) + 1,
                        cohesive_turns=cohesive_turns,
                    )
                    if enriched is not None and not _is_duplicate_qa(enriched, all_units + win_enriched):
                        win_enriched.append(enriched)
                self.workspace.atomic_write_json(partial_file, win_enriched)
                all_units.extend(win_enriched)
                completed_win_ids.append(w_idx)
                continue

            # 2. Check for intermediate Round 1 checkpoint (.en.json) or stale English-only partial
            elapsed_r1 = 0.0
            elapsed_r2 = 0.0
            elapsed = 0.0
            r1_units = stale_partial_en_units
            if r1_units is None and partial_en_file.exists():
                cached_en = self.workspace.read_json(partial_en_file, default=None)
                if isinstance(cached_en, list):
                    r1_units = cached_en
                    if verbose:
                        print(f"      [Window {w_idx}] Resuming from checkpointed Round 1 ({len(r1_units)} units)...", flush=True)

            if r1_units is None:
                prompt_r1 = USER_ROUND1_PROMPT_TEMPLATE.format(
                    title_ar=scenario.get("title_ar", ""),
                    youtube_url=scenario.get("youtube_url", ""),
                    episode_index=w["episode_index"],
                    start_fmt=w["start_fmt"],
                    end_fmt=w["end_fmt"],
                    speakers_summary=speakers_summary or "MAIN_DAEE vs INTERLOCUTOR",
                    target_daee_name=target_name,
                    target_daee_id=target_id,
                    transcript_text=w["formatted_transcript"],
                )
                t0_r1 = time.time()
                if hasattr(self.client, "extract_round1_english"):
                    raw_out_r1 = self.client.extract_round1_english(prompt_r1)
                else:
                    raw_out_r1 = self.client.extract(prompt_r1, system_prompt=ROUND1_ENGLISH_EXTRACTION_PROMPT)
                elapsed_r1 = round(time.time() - t0_r1, 1)

                parsed_r1 = clean_llm_json_array(raw_out_r1)
                r1_units = []
                for u in parsed_r1:
                    if not isinstance(u, dict):
                        continue
                    norm_en, norm_ar, ok_persp = normalize_question_syntax_and_perspective(
                        u.get("question_en", ""), u.get("question_ar", "")
                    )
                    if ok_persp and is_valid_dialectical_question(norm_en, norm_ar):
                        u_clean = dict(u)
                        u_clean["question_en"] = norm_en
                        if norm_ar:
                            u_clean["question_ar"] = norm_ar
                        r1_units.append(u_clean)
                self.workspace.atomic_write_json(partial_en_file, r1_units)

            if not r1_units:
                self.workspace.atomic_write_json(partial_file, [])
                completed_win_ids.append(w_idx)
                continue

            # 3. Round 2: Bounded-Thinking Card Rating Verification, Conditional EN Repair, & Arabic Localization
            # Sub-batch groups of up to 2 units so <=1024-token <think> + bilingual JSON fit cleanly without truncation.
            r2_batch_size = 2
            final_raw_units: List[Dict[str, Any]] = []
            elapsed_r2 = 0.0
            for b_start in range(0, len(r1_units), r2_batch_size):
                r1_batch = r1_units[b_start:b_start + r2_batch_size]
                r2_excerpt = _build_round2_transcript_excerpt(w["formatted_transcript"], r1_batch, max_words=1800)
                r1_json_str = json.dumps(r1_batch, ensure_ascii=False, indent=2)
                prompt_r2 = USER_ROUND2_PROMPT_TEMPLATE.format(
                    title_ar=scenario.get("title_ar", ""),
                    youtube_url=scenario.get("youtube_url", ""),
                    episode_index=w["episode_index"],
                    start_fmt=w["start_fmt"],
                    end_fmt=w["end_fmt"],
                    target_daee_name=target_name,
                    target_daee_id=target_id,
                    transcript_text=r2_excerpt,
                    english_units_json=r1_json_str,
                )
                if format_pre_round2_rubric_summary is not None:
                    pre_audit_summary = format_pre_round2_rubric_summary(r1_batch, w.get("turns") or [])
                    if pre_audit_summary:
                        prompt_r2 = f"{pre_audit_summary}\n\n{prompt_r2}"
                t0_r2 = time.time()
                if hasattr(self.client, "extract_round2_arabic"):
                    raw_out_r2 = self.client.extract_round2_arabic(r1_json_str, prompt_r2)
                else:
                    raw_out_r2 = self.client.extract(prompt_r2, system_prompt=ROUND2_ARABIC_VERIFICATION_PROMPT)
                elapsed_r2 = round(elapsed_r2 + (time.time() - t0_r2), 1)

                parsed_r2 = clean_llm_json_array(raw_out_r2)
                final_raw_units.extend(merge_round1_and_round2_units(r1_batch, parsed_r2))

            # 3B. Per-Unit Targeted Arabic Retry if batch Round 2 truncated or missed any unit's Arabic fields
            repaired_units: List[Dict[str, Any]] = []
            for idx_u, merged_u in enumerate(final_raw_units):
                if merged_u.get("_dropped_by_round2_audit"):
                    logger.info(
                        "Window %d unit %d dropped by Round 2 Rubric Audit (%s).",
                        w_idx, idx_u + 1, merged_u.get("_drop_reason", ""),
                    )
                    continue
                if has_valid_arabic_localization(merged_u):
                    repaired_units.append(merged_u)
                    continue

                single_r1 = [r1_units[idx_u]] if idx_u < len(r1_units) else [merged_u]
                single_json_str = json.dumps(single_r1, ensure_ascii=False, indent=2)
                single_excerpt = _build_round2_transcript_excerpt(w["formatted_transcript"], single_r1, max_words=900)
                single_prompt_r2 = USER_ROUND2_PROMPT_TEMPLATE.format(
                    title_ar=scenario.get("title_ar", ""),
                    youtube_url=scenario.get("youtube_url", ""),
                    episode_index=w["episode_index"],
                    start_fmt=w["start_fmt"],
                    end_fmt=w["end_fmt"],
                    target_daee_name=target_name,
                    target_daee_id=target_id,
                    transcript_text=single_excerpt,
                    english_units_json=single_json_str,
                )

                cur_candidate = merged_u
                for retry_attempt in range(1, 4):
                    t0_retry = time.time()
                    try:
                        if hasattr(self.client, "extract"):
                            raw_retry = self.client.extract(single_prompt_r2, system_prompt=COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT)
                        elif hasattr(self.client, "extract_round2_arabic"):
                            raw_retry = self.client.extract_round2_arabic(single_json_str, single_prompt_r2)
                        else:
                            raw_retry = "[]"
                    except TypeError:
                        if hasattr(self.client, "extract_round2_arabic"):
                            raw_retry = self.client.extract_round2_arabic(single_json_str, single_prompt_r2)
                        elif hasattr(self.client, "extract"):
                            raw_retry = self.client.extract(single_prompt_r2)
                        else:
                            raw_retry = "[]"
                    except Exception:
                        raw_retry = "[]"
                    elapsed_r2 = round(elapsed_r2 + (time.time() - t0_retry), 1)
                    parsed_retry = clean_llm_json_array(raw_retry)
                    if parsed_retry:
                        cur_candidate = merge_round1_and_round2_units([cur_candidate], parsed_retry)[0]
                        if has_valid_arabic_localization(cur_candidate):
                            break
                    if verbose:
                        print(f"      [Window {w_idx} Unit {idx_u+1}] Round 2 Arabic retry attempt {retry_attempt}/3...", flush=True)

                if has_valid_arabic_localization(cur_candidate):
                    repaired_units.append(cur_candidate)
                else:
                    logger.warning(
                        "Window %d unit %d dropped: failed complete Arabic localization after 3 attempts.",
                        w_idx, idx_u + 1
                    )

            final_raw_units = repaired_units
            elapsed = round(elapsed_r1 + elapsed_r2, 1)

            win_enriched: List[Dict[str, Any]] = []
            audit_labels_by_id: Dict[str, str] = {}
            for raw_item in final_raw_units:
                if not has_valid_arabic_localization(raw_item):
                    continue
                if not is_valid_dialectical_question(raw_item.get("question_en", ""), raw_item.get("question_ar", "")):
                    continue
                enriched = anchor_and_enrich_extracted_unit(
                    raw_unit=raw_item,
                    scenario=scenario,
                    episode_index=w["episode_index"],
                    window_start_ms=w["start_ms"],
                    unit_seq_num=len(all_units) + len(win_enriched) + 1,
                    cohesive_turns=cohesive_turns,
                )
                if enriched is None:
                    continue
                if not has_valid_arabic_localization(enriched):
                    continue
                if audit_single_card is not None:
                    card_audit = audit_single_card(enriched, w.get("turns") or [], require_arabic=True)
                    hf_gate = card_audit.get("hard_fail_triggered")
                    if hf_gate in ("[GATE-LEAK]", "[GATE-NOISE]", "[GATE-ROLE]"):
                        logger.warning(
                            "Card %s dropped by final Rubric Audit hard-fail %s: %s",
                            enriched.get("id", ""), hf_gate, card_audit.get("root_cause", ""),
                        )
                        continue
                    tier_short = card_audit["tier"].split(":")[0]
                    audit_labels_by_id[enriched["id"]] = f"{card_audit['total_score']}/100 {tier_short}"
                if not _is_duplicate_qa(enriched, all_units + win_enriched):
                    win_enriched.append(enriched)

            # Persist finalized window immediately to Drive before moving to next episode!
            self.workspace.atomic_write_json(partial_file, win_enriched)
            all_units.extend(win_enriched)
            completed_win_ids.append(w_idx)

            self.workspace.update_substep_progress(
                video_id,
                "episode_progress",
                {"completed_episodes": completed_win_ids, "total_episodes": len(windows)},
            )
            self.workspace.heartbeat_lock(f"{video_id}.stage2")

            if verbose:
                pct = int((w_idx / len(windows)) * 100)
                print(
                    f"\n   -> [Episode {w_idx}/{len(windows)} ({pct}%)] [{w['start_fmt']} -> {w['end_fmt']}] ({w['turn_count']} turns, {elapsed}s):",
                    flush=True,
                )
                if win_enriched:
                    for eu in win_enriched:
                        q_en = eu.get("question_en", "")
                        q_ar = eu.get("question_ar", "")
                        cat = eu.get("topic_category", "Theology")
                        rubric_tag = f" [Rubric: {audit_labels_by_id[eu['id']]}]" if eu.get("id") in audit_labels_by_id else ""
                        ans_en = eu.get("core_answer_summary_en", "")
                        ans_preview = ans_en[:110] + "..." if len(ans_en) > 110 else ans_en
                        print(f"      * [Card {eu['id']}] [{cat}]{rubric_tag}")
                        print(f"        Q (Seeker): \"{q_en}\"")
                        if q_ar:
                            print(f"        سؤال الباحث: \"{q_ar}\"")
                        print(f"        Truth Proof: {ans_preview}")
                else:
                    print("      (No substantive debate objections in this window - skipped)")

        # 4-Layer Ingestion Barrier: Ensure 100% Arabic Localization before advancing to M6
        unlocalized = [u for u in all_units if not has_valid_arabic_localization(u)]
        if unlocalized:
            logger.error(
                "Video %s transactional barrier breach: %d of %d cards lack complete Arabic localization! Aborting M6 commit.",
                video_id, len(unlocalized), len(all_units)
            )
            if verbose:
                print(f"\n   [TRANSACTIONAL BARRIER] Video {video_id} has {len(unlocalized)}/{len(all_units)} unlocalized cards! Halting M6 commit.")
            return None

        # Re-number sequential IDs cleanly and write final immutable shard
        for idx, u in enumerate(all_units, start=1):
            u["id"] = f"{video_id}_{idx:03d}"

        self.workspace.atomic_write_json(
            qa_shard_path,
            {
                "schema_version": "2.0",
                "video_id": video_id,
                "extracted_at": time.time(),
                "worker_id": self.workspace.worker_id,
                "units": all_units,
            },
        )

        # Clean up partial episode files now that final .qa.json shard is committed
        for p in self.workspace.partial_extractions_dir.glob(f"{video_id}_win*.json"):
            p.unlink(missing_ok=True)

        self.workspace.advance_micro_step(
            video_id,
            "M6_LLM_EXTRACTED",
            artifacts={"qa_json": str(qa_shard_path)},
            metrics={"qa_count": len(all_units)},
            extra_updates={"db_qa_synced": False},
        )

        if verbose:
            print("   " + "-" * 70)
            print(f"   >>> [STAGE 2 COMPLETE] Video {video_id} M6_LLM_EXTRACTED:")
            print(f"       * Total Extracted Seeker Cards: {len(all_units)}")
            print(f"       * Integrity: 100% VERIFIED -> Committed to Drive extractions/{video_id}.qa.json")
            print("   " + "=" * 70, flush=True)

        return all_units


def _is_duplicate_qa(candidate: Dict[str, Any], existing_list: List[Dict[str, Any]]) -> bool:
    q_en = (candidate.get("question_en") or "").strip().lower()
    q_ar = (candidate.get("question_ar") or "").strip()
    ts = int(candidate.get("timestamp_seconds", 0))
    for ex in existing_list:
        if abs(int(ex.get("timestamp_seconds", 0)) - ts) <= 75:
            ex_en = (ex.get("question_en") or "").strip().lower()
            ex_ar = (ex.get("question_ar") or "").strip()
            if (q_en and ex_en and (q_en == ex_en or q_en in ex_en or ex_en in q_en)) or (
                q_ar and ex_ar and (q_ar == ex_ar or q_ar in ex_ar or ex_ar in q_ar)
            ):
                return True
    return False
