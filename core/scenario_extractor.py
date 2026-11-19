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
    USER_ROUND2_CLEAN_TEMPLATE,
    USER_COMPACT_ROUND2_RETRY_TEMPLATE,
)
try:
    from core.colab_runner import _supports_kwarg
except (ImportError, ValueError):
    from .colab_runner import _supports_kwarg
from core.audio_diarizer import ms_to_fmt
try:
    from core.debug_tracer import (
        classify_diarization_health,
        extract_think_excerpt,
        record_video_trace_section,
    )
except (ImportError, ValueError):
    from .debug_tracer import (
        classify_diarization_health,
        extract_think_excerpt,
        record_video_trace_section,
    )

try:
    from db.database import (
        sanitize_text_field,
        sanitize_arabic_unit_fields,
        resolve_turn_speaker_labels,
        check_theological_bias as _db_check_theological_bias,
        canonicalize_speaker_role,
    )
except (ImportError, ValueError):
    try:
        from ..db.database import (
            sanitize_text_field,
            sanitize_arabic_unit_fields,
            resolve_turn_speaker_labels,
            check_theological_bias as _db_check_theological_bias,
            canonicalize_speaker_role,
        )
    except Exception:
        sanitize_text_field = None
        sanitize_arabic_unit_fields = None
        resolve_turn_speaker_labels = None
        _db_check_theological_bias = None
        canonicalize_speaker_role = None

try:
    from tools.card_rating_system.audit_cards import (
        audit_single_card,
        format_pre_round2_rubric_summary,
        summarize_pre_round2_batch,
    )
except (ImportError, ValueError):
    try:
        from ..tools.card_rating_system.audit_cards import (
            audit_single_card,
            format_pre_round2_rubric_summary,
            summarize_pre_round2_batch,
        )
    except Exception:
        audit_single_card = None
        format_pre_round2_rubric_summary = None
        summarize_pre_round2_batch = None


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


_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06ff]")
_ARABIC_LETTERS_ONLY_RE = re.compile(r"[\u0621-\u064a]")
_FOREIGN_SCRIPT_RE = re.compile(r"[\u0400-\u04ff\u0590-\u05ff\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]+")


def _is_clean_arabic_text(val: Any) -> bool:
    """Returns True when `val` has >=2 real Arabic letters and zero foreign script tokens."""
    if not isinstance(val, str):
        return False
    s = val.strip()
    return bool(s and len(_ARABIC_LETTERS_ONLY_RE.findall(s)) >= 2 and not _FOREIGN_SCRIPT_RE.search(s))


def sanitize_arabic_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Purges foreign scripts, Latin splices, and domain translation drift via canonical db.database cleaner."""
    if not isinstance(item, dict):
        return item
    if sanitize_arabic_unit_fields is not None:
        return sanitize_arabic_unit_fields(item)

    def _clean_str(val: str) -> str:
        if sanitize_text_field is not None:
            val = sanitize_text_field(val, is_arabic=True)
        cleaned = _FOREIGN_SCRIPT_RE.sub("", val).strip()
        return cleaned if (cleaned and _ARABIC_SCRIPT_RE.search(cleaned)) else ""

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

    if isinstance(item.get("scriptural_citations"), list):
        for cit in item["scriptural_citations"]:
            if isinstance(cit, dict) and isinstance(cit.get("significance_ar"), str) and cit["significance_ar"]:
                cit["significance_ar"] = _clean_str(cit["significance_ar"])

    for ex_key in ("dialogue_exchange", "dialogue_trajectory"):
        if isinstance(item.get(ex_key), list):
            for ex in item[ex_key]:
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

    # Discard pure personal chit-chat, secular statutory trivia & non-theological geopolitics
    non_theological_patterns = [
        r"^\s*(?:what\s+converted\s+you\s+to\s+islam|how\s+did\s+you\s+become\s+(?:a\s+)?muslim)\s*$",
        r"\b(?:palestinian\s+national\s+charter|jordanian\s+law|british\s+tax\s+law|legal\s+age\s+of\s+marriage\s+in\s+(?:uk|britain|jordan))\b",
        r"^\s*(?:why\s+are\s+you\s+calling\s+me\s+a\s+liar|are\s+you\s+mad|shut\s+up)\s*$",
    ]
    if any(re.search(pat, q_en_clean, re.I) for pat in non_theological_patterns):
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


def merge_consecutive_speaker_turns(
    turns: List[Dict[str, Any]],
    max_gap_ms: int = 2500,
    max_merged_words: int = 110,
) -> List[Dict[str, Any]]:
    """
    Merges rapid consecutive micro-turns spoken by the same speaker while capping merged length
    so turns do not balloon into >110-word multi-turn blobs that diverge from SQLite scenario_turns.
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
        prev_w_cnt = len((merged[-1].get("text_original") or "").split()) if merged else 0
        curr_w_cnt = len((t_copy.get("text_original") or "").split())
        if (
            merged
            and merged[-1].get("speaker_id") == t_copy.get("speaker_id")
            and merged[-1].get("episode_index") == t_copy.get("episode_index")
            and not t_copy.get("is_pedestrian_interruption")
            and not merged[-1].get("is_pedestrian_interruption")
            and (int(t_copy.get("start_ms", 0)) - int(merged[-1].get("end_ms", 0))) <= max_gap_ms
            and (prev_w_cnt + curr_w_cnt) <= max_merged_words
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


def _canon_speaker_role(raw_role: Any, fallback: str = "MAIN_DAEE") -> str:
    if canonicalize_speaker_role is not None:
        return canonicalize_speaker_role(raw_role, fallback=fallback)
    u = str(raw_role or "").strip().upper()
    if "PEDESTRIAN" in u:
        return "PEDESTRIAN"
    if "CO_DAEE" in u or "CO-DAEE" in u:
        return "CO_DAEE"
    if "INTERLOCUTOR" in u or "GUEST" in u or "محاور" in str(raw_role or ""):
        return "INTERLOCUTOR"
    if "DAEE" in u or "داعية" in str(raw_role or "") or "محاجج" in str(raw_role or ""):
        return "MAIN_DAEE"
    return fallback


_LOGISTICAL_FILLER_RE = re.compile(
    r"\b(?:your\s+time|my\s+time|started\s+my\s+time|alright,?\s+good|right,?\s+exactly|"
    r"thank\s+you,?\s+appreciate|hold\s+on|one\s+second|let\s+me\s+finish)\b",
    re.IGNORECASE,
)


def _is_logistical_filler_turn(txt: str) -> bool:
    s = (txt or "").strip()
    if not s:
        return True
    words = s.split()
    if len(words) >= 18:
        return False
    return bool(_LOGISTICAL_FILLER_RE.search(s))


def check_theological_bias(unit: Dict[str, Any], a_turn: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Theological Guardrail: delegates to canonical `db.database.check_theological_bias`
    so extraction-time and DB-ingestion checks share identical orthodox theological rules.
    """
    if _db_check_theological_bias is not None:
        return _db_check_theological_bias(unit, a_turn)
    return None


def _word_containment(needle: str, haystack_lower: str) -> float:
    w_n = set(re.findall(r"\w+", (needle or "").lower()))
    w_h = set(re.findall(r"\w+", haystack_lower or ""))
    return (len(w_n & w_h) / len(w_n)) if w_n else 0.0


_NEGATIVE_ANALOGY_RE = re.compile(
    r"^(?:none(?:\s+(?:was\s+)?(?:used|provided|mentioned|applicable|found|given|stated))?\.?|n/?a\.?|na\.?|no"
    r"|no(?:\s+(?:explicit|clear|direct|specific|concrete|real))?\s+(?:analogy|analogies|metaphor|metaphors|comparison)s?(?:\s+(?:is|was|are))?(?:\s+(?:used|provided|present|given|stated|found|employed|needed))?(?:\s+in\s+(?:the|this)\s+(?:transcript|exchange|clip|video|dialogue|debate))?\.?)$",
    re.IGNORECASE,
)


def _is_empty_or_negative_sentinel(val: Any) -> bool:
    s = str(val or "").strip()
    if not s:
        return True
    return bool(_NEGATIVE_ANALOGY_RE.match(s))


def _lookup_turn(
    tid: Any,
    cohesive_turns_by_id: Dict[str, Dict[str, Any]],
    raw_turns_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolves a turn ID by exact match, leading-underscore normalization, or unique `_Txxxx` suffix."""
    t_str = str(tid or "").strip()
    if not t_str:
        return None
    hit = cohesive_turns_by_id.get(t_str) or raw_turns_by_id.get(t_str)
    if hit is not None:
        return hit

    t_norm = t_str.lstrip("_")
    for src_map in (cohesive_turns_by_id, raw_turns_by_id):
        for k, v in src_map.items():
            if k.lstrip("_") == t_norm:
                return v

    m_suf = re.search(r"(?:^|_)(T\d+)$", t_str, re.IGNORECASE)
    if m_suf:
        suf = "_" + m_suf.group(1).upper()
        matched_objs: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for k, v in cohesive_turns_by_id.items():
            if k.upper().endswith(suf) and id(v) not in seen_ids:
                seen_ids.add(id(v))
                matched_objs.append(v)
        if len(matched_objs) == 1:
            return matched_objs[0]
        raw_objs = [v for k, v in raw_turns_by_id.items() if k.upper().endswith(suf)]
        if len(raw_objs) == 1:
            return raw_objs[0]
    return None


def _build_turn_lookups(
    scenario: Dict[str, Any],
    cohesive_turns: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Builds cohesive and raw turn lookup maps keyed by turn_id and merged_turn_ids."""
    if cohesive_turns is None:
        cohesive_turns = merge_consecutive_speaker_turns(scenario.get("turns", []))
    cohesive_turns_by_id: Dict[str, Dict[str, Any]] = {}
    for ct in cohesive_turns:
        if "turn_id" in ct:
            cohesive_turns_by_id[ct["turn_id"]] = ct
        for mid in ct.get("merged_turn_ids", []):
            cohesive_turns_by_id[mid] = ct
    raw_turns_by_id = {t["turn_id"]: t for t in scenario.get("turns", []) if "turn_id" in t}
    return cohesive_turns, cohesive_turns_by_id, raw_turns_by_id


def _is_substantive_daee_turn(cand_t: Optional[Dict[str, Any]], min_ts_ms: int) -> bool:
    if not cand_t or _canon_speaker_role(cand_t.get("speaker_role"), "") not in ("MAIN_DAEE", "CO_DAEE"):
        return False
    if int(cand_t.get("start_ms", 0)) < min_ts_ms:
        return False
    txt = clean_dangling_quote((cand_t.get("text_original") or "").strip())
    if _is_logistical_filler_turn(txt):
        return False
    return len([w for w in re.split(r"\s+", txt) if w]) >= 8


def _recover_substantive_daee_turn(
    item: Dict[str, Any],
    q_turn: Optional[Dict[str, Any]],
    a_turn: Dict[str, Any],
    a_tid: str,
    window_start_ms: int,
    cohesive_turns: List[Dict[str, Any]],
    cohesive_turns_by_id: Dict[str, Dict[str, Any]],
    raw_turns_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    """Re-anchors short, logistical-filler, or pre-question answer turns to the first substantive Da'ee reply."""
    a_cleaned_text = clean_dangling_quote((a_turn.get("text_original") or "").strip())
    a_orig_words = [w for w in re.split(r"\s+", a_cleaned_text) if w]
    q_start_ms = int(q_turn["start_ms"]) if q_turn is not None else int(window_start_ms)
    if (
        len(a_orig_words) >= 8
        and not _is_logistical_filler_turn(a_cleaned_text)
        and not (q_turn is not None and int(a_turn.get("start_ms", 0)) < q_start_ms)
    ):
        return a_turn, a_tid

    raw_ex_candidates = item.get("dialogue_exchange") or item.get("dialogue_trajectory") or []
    replacement_turn = None
    if isinstance(raw_ex_candidates, list):
        for ex in raw_ex_candidates:
            if not isinstance(ex, dict):
                continue
            cand_tid = str(ex.get("turn_id") or "").strip()
            cand_turn = _lookup_turn(cand_tid, cohesive_turns_by_id, raw_turns_by_id)
            if cand_turn and _is_substantive_daee_turn(cand_turn, q_start_ms):
                replacement_turn = cand_turn
                break
    if replacement_turn is None:
        for ct in cohesive_turns:
            if int(ct.get("start_ms", 0)) <= (q_start_ms + 180_000) and _is_substantive_daee_turn(ct, q_start_ms):
                replacement_turn = ct
                break
    if replacement_turn is not None:
        new_tid = str(replacement_turn.get("turn_id") or a_tid)
        item["answer_turn_id"] = new_tid
        return replacement_turn, new_tid
    return a_turn, a_tid


_CROWD_LOGISTICS_SENT_RE = re.compile(
    r"\b(?:stop\s+interrupting|let\s+me\s+finish|courtesy\s+to\s+let\s+me|"
    r"not\s+carry\s+on\s+the\s+conversation|don'?t\s+interrupt|hold\s+on\s+one\s+second|"
    r"calm\s+down|you'?re\s+a\s+very\s+angry|he'?s\s+doing\s+it\s+again)\b",
    re.IGNORECASE,
)


def _trim_focused_quote(text: str, anchor_text: str = "", max_words: int = 65) -> str:
    """
    Trims overlong multi-turn quotes cleanly to <= max_words at sentence boundaries,
    dropping pure crowd-control/interruption sentences when substantive proof sentences exist.
    """
    cleaned = clean_dangling_quote((text or "").strip())
    if not cleaned:
        return ""
    sents = [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", cleaned) if s.strip()]
    if len(sents) > 1:
        substantive_sents = [s for s in sents if not _CROWD_LOGISTICS_SENT_RE.search(s)]
        if substantive_sents and sum(len(s.split()) for s in substantive_sents) >= 8:
            sents = substantive_sents
            cleaned = " ".join(sents)
    words = cleaned.split()
    if len(words) <= max_words or len(sents) <= 1:
        return " ".join(words[:max_words]) if len(words) > max_words else cleaned

    anchor_toks = set(re.findall(r"[a-z]{4,}", (anchor_text or "").lower()))
    best_span = sents[0]
    best_score = -1.0
    for i in range(len(sents)):
        cur_sents: List[str] = []
        cur_words = 0
        for j in range(i, len(sents)):
            sw = len(sents[j].split())
            if cur_sents and (cur_words + sw) > max_words:
                break
            cur_sents.append(sents[j])
            cur_words += sw
            if cur_words >= 12:
                cand_txt = " ".join(cur_sents)
                if anchor_toks:
                    cand_toks = set(re.findall(r"[a-z]{4,}", cand_txt.lower()))
                    sc = len(cand_toks & anchor_toks) + min(cur_words, 45) * 0.02
                else:
                    sc = float(cur_words)
                if sc > best_score:
                    best_score = sc
                    best_span = cand_txt
    if len(best_span.split()) > max_words:
        best_span = " ".join(best_span.split()[:max_words])
    return clean_dangling_quote(best_span)


_ANALOGY_ABSTRACT_STOPWORDS = frozenset({
    "allah", "islam", "islamic", "muslim", "muslims", "quran", "quranic", "bible", "biblical",
    "jesus", "christ", "christian", "christians", "christianity", "prophet", "prophets", "messenger",
    "human", "humans", "person", "people", "someone", "anyone", "everyone", "individual", "individuals",
    "divine", "moral", "morally", "truth", "faith", "faiths", "belief", "beliefs", "world", "earth",
    "nature", "religion", "religions", "scripture", "creator", "creation", "created", "eternal",
    "justice", "mercy", "wisdom", "salvation", "paradise", "heaven", "forgiveness", "repentance",
    "mistakes", "mistake", "sinning", "sinless", "being", "beings", "similarly", "however", "therefore",
    "rather", "instead", "simply", "without", "because", "through", "against", "between", "another",
    "other", "every", "where", "which", "their", "there", "would", "could", "should", "cannot",
    "desire", "desires", "heart", "soul", "spirit", "living", "lives", "alive", "death", "dying",
    "still", "means", "meaning", "state", "form", "exact", "general", "whole", "entire", "total",
    "bound", "given", "makes", "making", "leads", "leading", "based", "follows", "following",
})


def _filter_grounded_analogies(
    analogies_en: List[str],
    analogies_ar: List[str],
    all_transcript_lower: str,
    cohesive_turns: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    Drops invented textbook analogies whose concrete imagery never appears anywhere in a long
    video transcript (`>= 4` turns and `>= 120` words), keeping both EN and AR lists aligned.
    """
    if not analogies_en or len(cohesive_turns) < 4 or len(all_transcript_lower.split()) < 120:
        return analogies_en, analogies_ar
    transcript_stems = {w[:5] for w in re.findall(r"[a-z]{5,}", all_transcript_lower)}
    kept_en: List[str] = []
    kept_ar: List[str] = []
    for idx, an in enumerate(analogies_en):
        an_s = str(an or "").strip()
        if _is_empty_or_negative_sentinel(an_s):
            continue
        concrete_stems = {
            w[:5]
            for w in re.findall(r"[a-z]{5,}", an_s.lower())
            if w not in _ANALOGY_ABSTRACT_STOPWORDS
        }
        # Only drop when analogy introduces >=2 concrete imagery terms with 0 stem hits in transcript
        if len(concrete_stems) >= 2 and not (concrete_stems & transcript_stems):
            continue
        kept_en.append(an_s)
        if idx < len(analogies_ar) and str(analogies_ar[idx] or "").strip():
            kept_ar.append(str(analogies_ar[idx]).strip())
    return kept_en, kept_ar


def _ground_dialogue_exchange(
    raw_exchange: List[Any],
    cohesive_turns_by_id: Dict[str, Dict[str, Any]],
    raw_turns_by_id: Dict[str, Dict[str, Any]],
    daee_name_en: str,
    daee_name_ar: str,
    inter_spk_id: str,
    fallback_ts_sec: int,
    repair_english_from_turn: bool = True,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Locks turn roles, snaps merged IDs to exact raw SQLite scenario_turns, sorts chronologically, and repairs drifted text_en."""
    enriched_exchange: List[Dict[str, Any]] = []
    any_diarized_turn_matched = False
    all_turns_matched = True
    seen_tids: set = set()
    for ex in raw_exchange[:12]:
        if not isinstance(ex, dict):
            continue
        etid = str(ex.get("turn_id") or "").strip()
        ex_txt = str(ex.get("text_en") or "").strip()
        if etid and etid in seen_tids:
            continue
        eturn = _lookup_turn(etid, cohesive_turns_by_id, raw_turns_by_id)

        # Snap cohesive merged turns to the best-matching exact raw scenario_turn so SQLite & Scenario Studio match 1-to-1
        raw_match = raw_turns_by_id.get(etid) if etid else None
        if eturn is not None:
            sub_ids = [sid for sid in (eturn.get("merged_turn_ids") or []) if sid in raw_turns_by_id]
            if len(sub_ids) > 1:
                if ex_txt:
                    best_sid = max(
                        sub_ids,
                        key=lambda sid: (
                            _word_containment(ex_txt, (raw_turns_by_id[sid].get("text_original") or "").lower()),
                            len((raw_turns_by_id[sid].get("text_original") or "").split()),
                        ),
                    )
                else:
                    best_sid = max(sub_ids, key=lambda sid: len((raw_turns_by_id[sid].get("text_original") or "").split()))
                raw_match = raw_turns_by_id[best_sid]
                etid = best_sid
            elif raw_match is None and eturn.get("turn_id") in raw_turns_by_id:
                etid = str(eturn["turn_id"])
                raw_match = raw_turns_by_id[etid]
            elif not (cohesive_turns_by_id.get(etid) or raw_turns_by_id.get(etid)):
                etid = str(eturn.get("turn_id") or etid)

        if raw_match is not None:
            eturn = raw_match

        if etid:
            if etid in seen_tids:
                continue
            seen_tids.add(etid)

        if eturn is not None and eturn.get("speaker_role"):
            any_diarized_turn_matched = True
            role = _canon_speaker_role(eturn["speaker_role"], fallback="MAIN_DAEE")
        else:
            if eturn is None:
                all_turns_matched = False
            role = _canon_speaker_role(ex.get("role"), fallback="MAIN_DAEE")

        if resolve_turn_speaker_labels is not None:
            spk_label_en, spk_label_ar = resolve_turn_speaker_labels(role, eturn, daee_name_en, daee_name_ar)
        else:
            is_daee = role in ("MAIN_DAEE", "CO_DAEE")
            turn_matches = (eturn is not None and (eturn.get("speaker_role") in ("MAIN_DAEE", "CO_DAEE")) == is_daee)
            spk_label_en = ((eturn.get("speaker_label_en") if turn_matches else None) or (daee_name_en if is_daee else "Interlocutor"))
            spk_label_ar = ((eturn.get("speaker_label_ar") if turn_matches else None) or (daee_name_ar if is_daee else "المحاور"))

        turn_orig = str(eturn.get("text_original") or "").strip() if eturn else ""
        if repair_english_from_turn and turn_orig:
            orig_words = set(re.findall(r"\b\w{3,}\b", turn_orig.lower()))
            ex_words = set(re.findall(r"\b\w{3,}\b", ex_txt.lower()))
            if not ex_txt:
                ex_txt = turn_orig
            elif len(ex_words) >= 4 and len(orig_words) >= 4:
                if (len(ex_words & orig_words) / len(ex_words)) < 0.55:
                    ex_txt = turn_orig
            elif len(ex_words) >= 4 and len(orig_words) < 4:
                ex_txt = turn_orig

        enriched_exchange.append({
            "turn_id": etid,
            "role": role,
            "speaker_label_en": spk_label_en,
            "speaker_label_ar": spk_label_ar,
            "timestamp_seconds": int(eturn["start_ms"] // 1000) if eturn else fallback_ts_sec,
            "timestamp_formatted": eturn["start_fmt"] if eturn else ms_to_fmt(fallback_ts_sec * 1000),
            "move_type": ex.get("move_type", "Turn"),
            "text_en": ex_txt,
            "text_ar": ex.get("text_ar") or (eturn.get("text_ar") if eturn else ""),
        })

    if all_turns_matched and len(enriched_exchange) >= 2:
        enriched_exchange.sort(key=lambda x: int(x.get("timestamp_seconds", 0)))

    if len(enriched_exchange) >= 2 and not any_diarized_turn_matched:
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
    return enriched_exchange, any_diarized_turn_matched


def _sync_role_quotes_in_item(
    item: Dict[str, Any],
    q_turn: Optional[Dict[str, Any]],
    q_tid: str,
    a_turn: Dict[str, Any],
    a_tid: str,
    enriched_exchange: List[Dict[str, Any]],
    cohesive_turns: List[Dict[str, Any]],
    inter_spk_id: str,
    sync_arabic: bool = True,
) -> None:
    """Enforces role-separated verbatim quotes against Stage-1 turns and clears stale Arabic when EN is replaced."""
    all_transcript = " ".join([(t.get("text_original") or "").strip() for t in cohesive_turns])
    all_transcript_lower = all_transcript.lower()
    daee_transcript = " ".join(
        [(t.get("text_original") or "").strip() for t in cohesive_turns if _canon_speaker_role(t.get("speaker_role"), "") in ("MAIN_DAEE", "CO_DAEE")]
    ) or all_transcript
    inter_transcript = (
        " ".join([(t.get("text_original") or "").strip() for t in cohesive_turns if _canon_speaker_role(t.get("speaker_role"), "") == "INTERLOCUTOR"])
        if inter_spk_id
        else all_transcript
    ) or all_transcript

    def _sync_single(
        quote_key: str,
        quote_ar_key: str,
        primary_turn: Optional[Dict[str, Any]],
        primary_tid: str,
        role_transcript_lower: str,
        allowed_roles: Tuple[str, ...],
        allow_exchange_fallback: bool,
    ) -> None:
        raw_q = (item.get(quote_key) or "").strip()
        q_words = [w for w in re.split(r"\s+", raw_q) if w]
        has_dangling = bool(q_words and q_words[-1].lower().rstrip(".,!?:;\"'؟،") in DANGLING_PREPOSITIONS_EN)
        is_short = len(q_words) < 8
        overlap_low = bool(
            raw_q
            and raw_q.lower() not in role_transcript_lower
            and _word_containment(raw_q, role_transcript_lower) < 0.45
        )
        sum_en_low = (item.get("core_answer_summary_en") or "").lower()
        is_summary_prose = bool(
            raw_q
            and sum_en_low
            and raw_q.lower() not in role_transcript_lower
            and _word_containment(raw_q, sum_en_low) >= 0.72
        )
        is_overlong_or_logistics = bool(
            raw_q and (len(q_words) > 90 or _CROWD_LOGISTICS_SENT_RE.search(raw_q))
        )
        anchor_ctx = f"{item.get('question_en') or ''} {item.get('core_answer_summary_en') or ''}"

        replaced_english = False
        if (overlap_low or is_summary_prose or is_short or has_dangling or is_overlong_or_logistics) and primary_turn and primary_turn.get("text_original"):
            p_orig = primary_turn["text_original"].strip()
            p_words = [w for w in re.split(r"\s+", p_orig) if w]
            if len(p_words) >= 8 or overlap_low or is_summary_prose:
                source_txt = raw_q if (is_overlong_or_logistics and not (overlap_low or is_summary_prose or is_short or has_dangling)) else p_orig
                cleaned_p = _trim_focused_quote(source_txt, anchor_text=anchor_ctx, max_words=65)
                if cleaned_p and cleaned_p != clean_dangling_quote(raw_q):
                    item[quote_key] = cleaned_p
                    replaced_english = True
                    if sync_arabic:
                        p_ar = (primary_turn.get("text_ar") or "").strip()
                        matched_ex_ar = next(
                            (
                                clean_dangling_quote(ex["text_ar"].strip())
                                for ex in enriched_exchange
                                if str(ex.get("turn_id") or "") == str(primary_tid) and ex.get("text_ar")
                            ),
                            "",
                        )
                        if p_ar or matched_ex_ar:
                            item[quote_ar_key] = p_ar or matched_ex_ar
                elif raw_q:
                    item[quote_key] = _trim_focused_quote(raw_q, anchor_text=anchor_ctx, max_words=65)
            elif allow_exchange_fallback:
                cand_turns = [
                    ex for ex in enriched_exchange
                    if ex.get("role") in allowed_roles and len((ex.get("text_en") or "").split()) >= 8
                ]
                if cand_turns:
                    best_ex = max(cand_turns, key=lambda x: len((x.get("text_en") or "").split()))
                    cleaned_best = _trim_focused_quote(best_ex["text_en"].strip(), anchor_text=anchor_ctx, max_words=65)
                    if cleaned_best != clean_dangling_quote(raw_q):
                        item[quote_key] = cleaned_best
                        replaced_english = True
                        if sync_arabic and (best_ex.get("text_ar") or "").strip():
                            item[quote_ar_key] = clean_dangling_quote((best_ex.get("text_ar") or "").strip())
                elif raw_q:
                    item[quote_key] = _trim_focused_quote(raw_q, anchor_text=anchor_ctx, max_words=65)
            elif raw_q:
                item[quote_key] = _trim_focused_quote(raw_q, anchor_text=anchor_ctx, max_words=65)
        elif raw_q:
            item[quote_key] = _trim_focused_quote(raw_q, anchor_text=anchor_ctx, max_words=65)

        if sync_arabic and not replaced_english and not item.get(quote_ar_key):
            for ex in enriched_exchange:
                if str(ex.get("turn_id") or "") == str(primary_tid) and ex.get("text_ar"):
                    item[quote_ar_key] = clean_dangling_quote(ex["text_ar"].strip())
                    break
            if not item.get(quote_ar_key):
                for ex in enriched_exchange:
                    if ex.get("role") in allowed_roles and ex.get("text_ar"):
                        item[quote_ar_key] = clean_dangling_quote(ex["text_ar"].strip())
                        break

    _sync_single("verbatim_quote", "verbatim_quote_ar", a_turn, a_tid, daee_transcript.lower(), ("MAIN_DAEE", "CO_DAEE"), True)
    _sync_single("interlocutor_verbatim_quote", "interlocutor_verbatim_quote_ar", q_turn, q_tid, inter_transcript.lower(), ("INTERLOCUTOR",), False)


def _recover_turn_by_quote(
    quote_text: str,
    cohesive_turns: List[Dict[str, Any]],
    allowed_roles: Tuple[str, ...],
) -> Optional[Dict[str, Any]]:
    """Recovers a scenario turn when the LLM altered a turn_id prefix (`_vid_Txxxx`)."""
    q_clean = (quote_text or "").strip()
    if len(q_clean.split()) < 6:
        return None
    best_turn = None
    best_score = 0.0
    for ct in cohesive_turns:
        if _canon_speaker_role(ct.get("speaker_role"), "") not in allowed_roles:
            continue
        orig = (ct.get("text_original") or "").strip().lower()
        if not orig:
            continue
        if q_clean.lower() in orig:
            return ct
        score = _word_containment(q_clean, orig)
        if score >= 0.60 and score > best_score:
            best_score = score
            best_turn = ct
    return best_turn


def _filter_grounded_citations(
    citations: List[Any],
    all_transcript_lower: str,
    cohesive_turns: List[Dict[str, Any]],
) -> List[Any]:
    """Drops fabricated verse numbers not spoken in sufficiently long scenarios."""
    if not citations or len(cohesive_turns) < 4 or len(all_transcript_lower.split()) < 120:
        return citations
    kept: List[Any] = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        ref = str(c.get("reference") or "").strip()
        sig = str(c.get("significance") or "").strip().lower()
        if "21:30" in ref and ("expansion" in sig or "expanding" in sig):
            # Al-Anbiya 21:30 is heavens/earth joined & water; universe expansion is 51:47
            continue
        pairs = re.findall(r"(\d+)\s*[:.,]\s*(\d+)", ref)
        if not pairs:
            kept.append(c)
            continue
        matched_any = False
        for ch, vs in pairs:
            if (
                re.search(rf"\b{re.escape(ch)}\s*[:.,\-]?\s*{re.escape(vs)}\b", all_transcript_lower)
                or (int(vs) >= 10 and re.search(rf"\b{re.escape(vs)}\b", all_transcript_lower))
                or (ch == "2" and vs == "255" and ("kursi" in all_transcript_lower or "throne verse" in all_transcript_lower))
            ):
                matched_any = True
                break
        if matched_any:
            kept.append(c)
    return kept


def _passes_topical_grounding_gate(
    item: Dict[str, Any],
    q_turn: Optional[Dict[str, Any]],
    a_turn: Dict[str, Any],
    cohesive_turns: List[Dict[str, Any]],
) -> bool:
    """Blocks hallucinated dossiers whose core topic does not appear in the local debate window."""
    if len(cohesive_turns) < 4:
        return True
    ref_turns = [t for t in (q_turn, a_turn) if t is not None]
    min_ms = min(int(t.get("start_ms", 0)) for t in ref_turns) - 60_000
    max_ms = max(int(t.get("end_ms", 0)) for t in ref_turns) + 120_000
    local_text = " ".join(
        (t.get("text_original") or "").strip()
        for t in cohesive_turns
        if int(t.get("end_ms", 0)) >= min_ms and int(t.get("start_ms", 0)) <= max_ms
    ).lower()
    if len(local_text.split()) < 40:
        return True

    local_stems = {w[:5] for w in re.findall(r"[a-z]{5,}", local_text)}
    if len(local_stems) < 25:
        return True

    broad_stopwords = {
        "islam", "islamic", "muslim", "muslims", "quran", "quranic", "bible", "biblical",
        "christian", "christians", "christianity", "allah", "jesus", "christ", "prophet",
        "believe", "belief", "people", "claim", "claims", "objection", "because", "through",
        "their", "there", "which", "about", "would", "could", "should", "against", "without",
        "human", "world", "truth", "proof", "divine", "moral", "rational", "logic", "logical",
    }
    raw_premises = item.get("logical_premises") or []
    premises_txt = (
        " ".join(str(p) for p in raw_premises)
        if isinstance(raw_premises, list)
        else str(raw_premises)
    )
    card_txt = f"{item.get('question_en', '')} {item.get('core_answer_summary_en', '')} {premises_txt}".lower()
    card_stems = {
        w[:5] for w in re.findall(r"[a-z]{5,}", card_txt)
        if w not in broad_stopwords
    }
    if len(card_stems) < 6:
        return True
    return (len(card_stems & local_stems) / len(card_stems)) >= 0.09


def _preground_round1_unit(
    raw_unit: Dict[str, Any],
    scenario: Dict[str, Any],
    window_start_ms: int,
    cohesive_turns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Grounds a Round 1 unit's verbatim quotes, substantive answer_turn_id, and dialogue_exchange
    against Stage-1 turns BEFORE Round 2 translates them, preventing Round 2 from translating
    mismatched/summary quotes or swapped speaker roles.
    """
    u = dict(raw_unit)
    cohesive_turns, cohesive_turns_by_id, raw_turns_by_id = _build_turn_lookups(scenario, cohesive_turns)
    q_tid = str(u.get("question_turn_id") or "").strip()
    a_tid = str(u.get("answer_turn_id") or "").strip()
    q_turn = _lookup_turn(q_tid, cohesive_turns_by_id, raw_turns_by_id)
    a_turn = _lookup_turn(a_tid, cohesive_turns_by_id, raw_turns_by_id)
    if q_turn is None and u.get("interlocutor_verbatim_quote"):
        q_turn = _recover_turn_by_quote(u["interlocutor_verbatim_quote"], cohesive_turns, ("INTERLOCUTOR",))
    if a_turn is None and u.get("verbatim_quote"):
        a_turn = _recover_turn_by_quote(u["verbatim_quote"], cohesive_turns, ("MAIN_DAEE", "CO_DAEE"))
    if q_turn is not None and q_turn.get("turn_id"):
        q_tid = str(q_turn["turn_id"])
        u["question_turn_id"] = q_tid
    if a_turn is not None and a_turn.get("turn_id"):
        a_tid = str(a_turn["turn_id"])
        u["answer_turn_id"] = a_tid

    # Normalize negative analogy sentinels and drop fabricated analogies before Round 2 sees them
    if _is_empty_or_negative_sentinel(u.get("analogy_or_metaphor")):
        u["analogy_or_metaphor"] = ""
    raw_pre_an = (
        [str(a).strip() for a in u["analogies"] if not _is_empty_or_negative_sentinel(a)]
        if isinstance(u.get("analogies"), list)
        else ([str(u["analogy_or_metaphor"]).strip()] if u.get("analogy_or_metaphor") else [])
    )
    if raw_pre_an:
        all_tr_low = " ".join((t.get("text_original") or "").strip() for t in cohesive_turns).lower()
        grounded_an, _ = _filter_grounded_analogies(raw_pre_an, [], all_tr_low, cohesive_turns)
        u["analogies"] = grounded_an
        u["analogy_or_metaphor"] = grounded_an[0] if grounded_an else ""
    elif isinstance(u.get("analogies"), list):
        u["analogies"] = []

    if a_turn is None or _canon_speaker_role(a_turn.get("speaker_role"), "") not in ("MAIN_DAEE", "CO_DAEE"):
        return u

    speakers = scenario.get("speakers", [])
    main_daee = next((s for s in speakers if _canon_speaker_role(s.get("role"), "") == "MAIN_DAEE"), None)
    daee_name_en = (main_daee.get("display_name_en") if main_daee else None) or "Main Da'ee"
    daee_name_ar = (main_daee.get("display_name_ar") if main_daee else None) or "الداعية المحاور"
    inter_spk_id = ""
    if q_turn is not None and _canon_speaker_role(q_turn.get("speaker_role"), "") not in ("MAIN_DAEE", "CO_DAEE"):
        inter_spk_id = q_turn.get("speaker_id", "")

    a_turn, a_tid = _recover_substantive_daee_turn(
        u, q_turn, a_turn, a_tid, window_start_ms, cohesive_turns, cohesive_turns_by_id, raw_turns_by_id
    )
    raw_ex = u.get("dialogue_exchange") if isinstance(u.get("dialogue_exchange"), list) else u.get("dialogue_trajectory")
    ts_sec = int(q_turn["start_ms"] // 1000) if q_turn is not None else int(window_start_ms // 1000)
    grounded_ex: List[Dict[str, Any]] = []
    if isinstance(raw_ex, list) and raw_ex:
        grounded_ex, _ = _ground_dialogue_exchange(
            raw_ex, cohesive_turns_by_id, raw_turns_by_id, daee_name_en, daee_name_ar, inter_spk_id, ts_sec
        )
        raw_ar_by_tid = {
            str(rx.get("turn_id") or "").strip(): rx.get("text_ar")
            for rx in raw_ex if isinstance(rx, dict) and rx.get("turn_id") and rx.get("text_ar")
        }
        lean_ex = []
        for idx_g, gx in enumerate(grounded_ex):
            orig_ex = raw_ex[idx_g] if idx_g < len(raw_ex) and isinstance(raw_ex[idx_g], dict) else {}
            item_ex = {
                "turn_id": gx["turn_id"],
                "role": gx["role"],
                "move_type": gx.get("move_type", "Turn"),
                "text_en": gx.get("text_en", ""),
            }
            ar_val = raw_ar_by_tid.get(gx["turn_id"]) or orig_ex.get("text_ar")
            if ar_val:
                item_ex["text_ar"] = ar_val
            lean_ex.append(item_ex)
        key_ex = "dialogue_exchange" if isinstance(u.get("dialogue_exchange"), list) else "dialogue_trajectory"
        u[key_ex] = lean_ex

    _sync_role_quotes_in_item(
        u, q_turn, q_tid, a_turn, a_tid, grounded_ex, cohesive_turns, inter_spk_id, sync_arabic=False
    )
    return u


def anchor_and_enrich_extracted_unit(
    raw_unit: Dict[str, Any],
    scenario: Dict[str, Any],
    episode_index: int,
    window_start_ms: int,
    unit_seq_num: int,
    cohesive_turns: Optional[List[Dict[str, Any]]] = None,
    drop_reason_out: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Anchors LLM-extracted turn IDs and speaker identities directly against ground-truth
    cohesive turns and diarized Speaker records.
    Returns None if the unit fails theological validation, perspective integrity, or attributes opponent bias.
    """
    def _fail(code: str) -> None:
        if drop_reason_out is not None:
            drop_reason_out.append(code)
        return None

    item = sanitize_arabic_fields({k: v for k, v in raw_unit.items() if not str(k).startswith("_")})
    item.pop("rubric_audit", None)
    vid = scenario["video_id"]
    yt_url = scenario.get("youtube_url") or f"https://www.youtube.com/watch?v={vid}"

    # 1. Question Syntax & 1st-Person Confessional Normalization
    raw_q_en = str(item.get("question_en") or "").strip()
    raw_q_ar = str(item.get("question_ar") or "").strip()
    norm_q_en, norm_q_ar, is_valid_perspective = normalize_question_syntax_and_perspective(raw_q_en, raw_q_ar)
    if not is_valid_perspective or not is_valid_dialectical_question(norm_q_en, norm_q_ar):
        return _fail("INVALID_PERSPECTIVE")
    item["question_en"] = norm_q_en
    if norm_q_ar:
        item["question_ar"] = norm_q_ar

    cohesive_turns, cohesive_turns_by_id, raw_turns_by_id = _build_turn_lookups(scenario, cohesive_turns)
    speakers = scenario.get("speakers", [])
    main_daee = next((s for s in speakers if _canon_speaker_role(s.get("role"), "") == "MAIN_DAEE"), None)

    q_tid = str(item.get("question_turn_id") or "").strip()
    a_tid = str(item.get("answer_turn_id") or "").strip()
    q_turn = _lookup_turn(q_tid, cohesive_turns_by_id, raw_turns_by_id)
    a_turn = _lookup_turn(a_tid, cohesive_turns_by_id, raw_turns_by_id)
    if q_turn is None and item.get("interlocutor_verbatim_quote"):
        q_turn = _recover_turn_by_quote(item["interlocutor_verbatim_quote"], cohesive_turns, ("INTERLOCUTOR",))
    if a_turn is None and item.get("verbatim_quote"):
        a_turn = _recover_turn_by_quote(item["verbatim_quote"], cohesive_turns, ("MAIN_DAEE", "CO_DAEE"))
    if q_turn is not None and q_turn.get("turn_id"):
        q_tid = str(q_turn["turn_id"])
        item["question_turn_id"] = q_tid
    if a_turn is not None and a_turn.get("turn_id"):
        a_tid = str(a_turn["turn_id"])
        item["answer_turn_id"] = a_tid

    if q_turn is not None:
        ts_sec = int(q_turn["start_ms"] // 1000)
        q_role = _canon_speaker_role(q_turn.get("speaker_role"), "")
        if q_role in ("MAIN_DAEE", "CO_DAEE"):
            inter_spk_id = ""  # Monologue / anticipated question voiced by Da'ee
        else:
            inter_spk_id = q_turn.get("speaker_id", "")
    else:
        ts_sec = max(int(window_start_ms // 1000), int(item.get("timestamp_seconds") or 0))
        inter_spk_id = ""

    if a_turn is None or _canon_speaker_role(a_turn.get("speaker_role"), "") not in ("MAIN_DAEE", "CO_DAEE"):
        # Strictly reject / drop unit if candidate answer turn is missing or belongs to interlocutor
        return _fail("MISSING_DAEE_TURN")

    a_turn, a_tid = _recover_substantive_daee_turn(
        item, q_turn, a_turn, a_tid, window_start_ms, cohesive_turns, cohesive_turns_by_id, raw_turns_by_id
    )

    ans_sec = int(a_turn["start_ms"] // 1000)
    daee_spk_id = a_turn.get("speaker_id", main_daee["speaker_id"] if main_daee else "")

    # Speaker Role Integrity Guard: daee_speaker_id and interlocutor_speaker_id CANNOT be identical!
    if daee_spk_id and inter_spk_id and daee_spk_id == inter_spk_id:
        if q_turn and _canon_speaker_role(q_turn.get("speaker_role"), "") in ("MAIN_DAEE", "CO_DAEE"):
            inter_spk_id = ""
        else:
            distinct_inter = next(
                (s["speaker_id"] for s in speakers
                 if s["speaker_id"] != daee_spk_id and _canon_speaker_role(s.get("role"), "") == "INTERLOCUTOR"),
                ""
            )
            inter_spk_id = distinct_inter
    if daee_spk_id and inter_spk_id and daee_spk_id == inter_spk_id:
        inter_spk_id = ""

    matched_daee = next(
        (s for s in speakers if s["speaker_id"] == daee_spk_id and _canon_speaker_role(s.get("role"), "") in ("MAIN_DAEE", "CO_DAEE")),
        main_daee
    )
    if not matched_daee or _canon_speaker_role(matched_daee.get("role"), "") not in ("MAIN_DAEE", "CO_DAEE"):
        return _fail("INVALID_DAEE_SPEAKER")

    daee_name_en = matched_daee.get("display_name_en") or (main_daee.get("display_name_en") if main_daee else "Main Da'ee")
    daee_name_ar = matched_daee.get("display_name_ar") or (main_daee.get("display_name_ar") if main_daee else "الداعية المحاور")
    daee_profile_id = matched_daee.get("profile_id") or (main_daee.get("profile_id") if main_daee else "")

    # Theological Guardrail: Strictly reject any unit that fell into an interlocutor
    if daee_name_ar in ("المحاور", "المحاور 1", "المحاور 2", "Interlocutor", "Interlocutor 1", "Interlocutor 2"):
        return _fail("INTERLOCUTOR_AS_DAEE")

    if not daee_name_ar and main_daee and main_daee.get("display_name_ar"):
        daee_name_ar = main_daee["display_name_ar"]
        daee_name_en = main_daee.get("display_name_en") or "Main Da'ee"
        daee_profile_id = main_daee.get("profile_id") or ""

    # Theological Guardrail: Validate no heresy or opponent bias is being asserted as truth
    bias_reason = check_theological_bias(item, a_turn)
    if bias_reason:
        return _fail(f"THEOLOGICAL_BIAS:{str(bias_reason)[:60]}")

    # Collect ordered source turn IDs referenced in dialogue_exchange + q_tid + a_tid
    raw_exchange = item.get("dialogue_exchange") or item.get("dialogue_trajectory") or []
    if not isinstance(raw_exchange, list):
        raw_exchange = []

    enriched_exchange, _ = _ground_dialogue_exchange(
        raw_exchange, cohesive_turns_by_id, raw_turns_by_id, daee_name_en, daee_name_ar, inter_spk_id, ts_sec
    )

    src_tids: List[str] = []
    for cand_id in [q_tid, a_tid] + [
        str(ex.get("turn_id") or "").strip()
        for ex in enriched_exchange
        if isinstance(ex, dict)
    ]:
        t_found = _lookup_turn(cand_id, cohesive_turns_by_id, raw_turns_by_id)
        if t_found is not None:
            resolved_tid = str(t_found.get("turn_id") or cand_id)
            if resolved_tid and resolved_tid not in src_tids:
                src_tids.append(resolved_tid)

    all_transcript = " ".join([(t.get("text_original") or "").strip() for t in cohesive_turns])
    all_transcript_lower = all_transcript.lower()

    # Analogies: support multiple analogies list + single analogy field, filtering negative sentinels & invented analogies
    raw_an = item.get("analogies")
    if isinstance(raw_an, list) and raw_an:
        clean_analogies = [str(a).strip() for a in raw_an if not _is_empty_or_negative_sentinel(a)]
    elif not _is_empty_or_negative_sentinel(item.get("analogy_or_metaphor")):
        clean_analogies = [str(item["analogy_or_metaphor"]).strip()]
    else:
        clean_analogies = []

    raw_an_ar = item.get("analogies_ar")
    if isinstance(raw_an_ar, list) and raw_an_ar:
        clean_analogies_ar = [str(a).strip() for a in raw_an_ar if not _is_empty_or_negative_sentinel(a)]
    elif not _is_empty_or_negative_sentinel(item.get("analogy_or_metaphor_ar")):
        clean_analogies_ar = [str(item["analogy_or_metaphor_ar"]).strip()]
    else:
        clean_analogies_ar = []

    clean_analogies, clean_analogies_ar = _filter_grounded_analogies(
        clean_analogies, clean_analogies_ar, all_transcript_lower, cohesive_turns
    )

    item["analogy_or_metaphor"] = clean_analogies[0] if clean_analogies else ""
    item["analogy_or_metaphor_ar"] = clean_analogies_ar[0] if clean_analogies_ar else ""

    # Scriptural citations grounded against transcript in long scenarios
    raw_citations = item.get("scriptural_citations")
    clean_citations = _filter_grounded_citations(
        raw_citations if isinstance(raw_citations, list) else [],
        all_transcript_lower,
        cohesive_turns,
    )

    if not _passes_topical_grounding_gate(item, q_turn, a_turn, cohesive_turns):
        return _fail("TOPICAL_DRIFT")

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
        if sig == "mark 13:32":
            if not re.search(r"\bmark\s*(?:chapter\s*)?13[\s:.,]+32\b", card_all_text):
                continue
            sig_present = bool(re.search(r"\bmark\s*(?:chapter\s*)?13[\s:.,]+32\b", all_transcript_lower))
        else:
            if sig not in card_all_text:
                continue
            sig_present = sig in all_transcript_lower
        if not sig_present:
            return _fail(f"PROMPT_LEAK:{sig}")

    # Anti-Noise Gate: Discard cards referencing street incidents, dog petting, or promotional outro pitches
    unconditional_noise_signatures = [
        "pet him", "what kind of a dog", "pet the dog", "911 call",
        "verbal dispute", "islamnet", "donate now", "link in the description",
        "fundraising", "subscribe to the channel",
    ]
    for nsig in unconditional_noise_signatures:
        if nsig in card_all_text:
            return _fail(f"NOISE:{nsig}")

    # Police/weapon chatter must not trigger on rhetorical theological analogies in core_answer_summary/analogies
    spoken_and_q_text = " ".join([
        item.get("question_en", ""),
        item.get("verbatim_quote", ""),
        item.get("interlocutor_verbatim_quote", ""),
        " ".join(str(ex.get("text_en") or "") for ex in enriched_exchange),
    ]).lower()
    for nsig in ("call the police", "police officer", "brandish", "brandished", "firearm", "police come"):
        if nsig in spoken_and_q_text:
            return _fail(f"NOISE:{nsig}")

    _sync_role_quotes_in_item(
        item, q_turn, q_tid, a_turn, a_tid, enriched_exchange, cohesive_turns, inter_spk_id, sync_arabic=True
    )

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
                return _fail("FLEETING_QUOTE")

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
    allow_implicit_en_repair: bool = False,
) -> List[Dict[str, Any]]:
    """
    Deterministically zipper-merges Round 1 (pure English dialectical reasoning)
    with Round 2 (bounded-thinking Rubric Verification, conditional EN repairs, & Arabic localization).

    Protects against:
    - Out-of-order LLM response generation in Round 2 (matches by turn IDs or question similarity).
    - Forced/needless English rewrites when Round 1 already passed quality checks (`english_edits_needed=false`).
    - Silently discarding Round 2 English repairs when Pre-Round-2 flagged defects (`allow_implicit_en_repair=True`)
      and the LLM rewrote defective English fields without toggling `rubric_audit.english_edits_needed=true`.
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

        # 3. Match by question text similarity / containment / Jaccard
        if best_r2_idx is None and r1_qen:
            r1_qwords = set(re.findall(r"\b\w{3,}\b", r1_qen))
            best_q_score = 0.0
            for r2_idx, r2_cand in enumerate(r2_units):
                if r2_idx in used_r2_indices:
                    continue
                r2_qen = (r2_cand.get("question_en") or "").strip().lower()
                if not r2_qen:
                    continue
                if r2_qen == r1_qen or r1_qen in r2_qen or r2_qen in r1_qen:
                    best_r2_idx = r2_idx
                    break
                r2_qwords = set(re.findall(r"\b\w{3,}\b", r2_qen))
                if r1_qwords and r2_qwords:
                    jacc = len(r1_qwords & r2_qwords) / len(r1_qwords | r2_qwords)
                    if jacc >= 0.55 and jacc > best_q_score:
                        best_q_score = jacc
                        best_r2_idx = r2_idx

        # 4. Positional fallback ONLY for 1-to-1 single-unit calls without conflicting turn IDs
        # Prevents cross-card Arabic contamination when Round 2 drops/reorders cards in multi-unit batches.
        if best_r2_idx is None and len(r1_units) == 1 and len(r2_units) == 1 and 0 not in used_r2_indices:
            cand0 = r2_units[0]
            c_qtid = str(cand0.get("question_turn_id") or "").strip()
            c_atid = str(cand0.get("answer_turn_id") or "").strip()

            def _ids_compatible(id_a: str, id_b: str) -> bool:
                if not id_a or not id_b or id_a == id_b or id_a.lstrip("_") == id_b.lstrip("_"):
                    return True
                ma = re.search(r"T\d+$", id_a, re.IGNORECASE)
                mb = re.search(r"T\d+$", id_b, re.IGNORECASE)
                return bool(ma and mb and ma.group(0).upper() == mb.group(0).upper())

            if _ids_compatible(r1_qtid, c_qtid) and _ids_compatible(r1_atid, c_atid):
                best_r2_idx = 0

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

        english_edits_needed = bool(
            r2_audit.get("english_edits_needed") is True
            or allow_implicit_en_repair
        )

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
            if not str(merged.get("analogy_or_metaphor_ar") or "").strip() and isinstance(r2_u["analogies_ar"], list):
                first_an_ar = str(r2_u["analogies_ar"][0] or "").strip()
                if first_an_ar:
                    merged["analogy_or_metaphor_ar"] = first_an_ar

        # Mirror verified dialogue_exchange Arabic turn text into verbatim_quote_ar before Step 3B check
        if not str(merged.get("verbatim_quote_ar") or "").strip():
            ex_list = merged.get("dialogue_exchange") or merged.get("dialogue_trajectory") or []
            if isinstance(ex_list, list):
                atid_str = str(merged.get("answer_turn_id") or "").strip()
                cand_vq_ar = next(
                    (
                        str(ex.get("text_ar") or "").strip()
                        for ex in ex_list
                        if isinstance(ex, dict)
                        and atid_str
                        and str(ex.get("turn_id") or "").strip() == atid_str
                        and _is_clean_arabic_text(str(ex.get("text_ar") or ""))
                    ),
                    "",
                )
                if not cand_vq_ar:
                    cand_vq_ar = next(
                        (
                            str(ex.get("text_ar") or "").strip()
                            for ex in ex_list
                            if isinstance(ex, dict)
                            and _canon_speaker_role(ex.get("role"), "") in ("MAIN_DAEE", "CO_DAEE")
                            and _is_clean_arabic_text(str(ex.get("text_ar") or ""))
                        ),
                        "",
                    )
                if cand_vq_ar:
                    merged["verbatim_quote_ar"] = cand_vq_ar

        final_merged.append(merged)

    return final_merged


_DANGLING_PROCLITIC_CHECK_RE = re.compile(r"(?<![\w\u0600-\u06ff])(?:[وف]?للـ?|[وف]الـ?)(?=[\s.,؛،!?؟]|$)")


def diagnose_missing_arabic_fields(unit: Dict[str, Any]) -> List[str]:
    """
    Returns descriptive, unquoted field diagnostics for every missing, foreign-script,
    dangling-preposition, duplicate-quote, or severely truncated Arabic field.
    """
    issues: List[str] = []
    if not isinstance(unit, dict):
        return ["invalid_unit"]

    q_ar = str(unit.get("question_ar") or "").strip()
    if not _is_clean_arabic_text(q_ar):
        issues.append("question_ar")

    sum_en = str(unit.get("core_answer_summary_en") or "").strip()
    sum_ar = str(unit.get("core_answer_summary_ar") or "").strip()
    if not _is_clean_arabic_text(sum_ar):
        issues.append("core_answer_summary_ar")
    elif len(sum_en.split()) >= 90 and len(sum_ar.split()) < max(35, int(len(sum_en.split()) * 0.36)):
        issues.append("truncated_summary:core_answer_summary_ar")

    for f in (
        "fallacy_or_misconception_exposed_ar",
        "analogy_or_metaphor_ar",
        "socratic_question_ar",
        "verbatim_quote_ar",
        "interlocutor_verbatim_quote_ar",
    ):
        val = str(unit.get(f) or "").strip()
        if val:
            if not _is_clean_arabic_text(val):
                issues.append(f)
            elif _DANGLING_PROCLITIC_CHECK_RE.search(val):
                issues.append(f"dangling_preposition:{f}")

    for core_f in ("question_ar", "core_answer_summary_ar"):
        c_val = str(unit.get(core_f) or "").strip()
        if c_val and _DANGLING_PROCLITIC_CHECK_RE.search(c_val):
            issues.append(f"dangling_preposition:{core_f}")

    if str(unit.get("verbatim_quote") or "").strip() and not _is_clean_arabic_text(str(unit.get("verbatim_quote_ar") or "")):
        if "verbatim_quote_ar" not in issues:
            issues.append("verbatim_quote_ar")

    en_analogy = str(unit.get("analogy_or_metaphor") or "").strip()
    if not _is_empty_or_negative_sentinel(en_analogy):
        if not _is_clean_arabic_text(str(unit.get("analogy_or_metaphor_ar") or "")):
            if "analogy_or_metaphor_ar" not in issues:
                issues.append("analogy_or_metaphor_ar")

    ivq_en = str(unit.get("interlocutor_verbatim_quote") or "").strip()
    q_en = str(unit.get("question_en") or "").strip()
    ivq_ar = str(unit.get("interlocutor_verbatim_quote_ar") or "").strip()
    if (
        len(ivq_ar.split()) >= 6
        and q_ar
        and (ivq_ar == q_ar or ivq_ar in q_ar or q_ar in ivq_ar)
        and ivq_en
        and q_en
    ):
        _dup_stop = {
            "that", "this", "what", "when", "where", "which", "does", "only",
            "those", "with", "from", "they", "their", "have", "about", "would",
            "could", "should", "then",
        }
        q_stems = {w[:5] for w in re.findall(r"[a-z]{4,}", q_en.lower()) if w not in _dup_stop}
        ivq_stems = {w[:5] for w in re.findall(r"[a-z]{4,}", ivq_en.lower()) if w not in _dup_stop}
        if q_stems and ivq_stems and (len(q_stems & ivq_stems) / min(len(q_stems), len(ivq_stems))) < 0.50:
            issues.append("duplicate_question_copied_to:interlocutor_verbatim_quote_ar")

    # Bilingual Quote Length-Symmetry Check: catches wrong-turn Arabic translations (e.g., 16w EN paired with 250w AR)
    for en_k, ar_k in (
        ("interlocutor_verbatim_quote", "interlocutor_verbatim_quote_ar"),
        ("verbatim_quote", "verbatim_quote_ar"),
    ):
        en_v = str(unit.get(en_k) or "").strip()
        ar_v = str(unit.get(ar_k) or "").strip()
        if en_v and ar_v and _is_clean_arabic_text(ar_v):
            en_w = len(en_v.split())
            ar_w = len(ar_v.split())
            if en_w >= 6 and ar_w > max(35, int(en_w * 2.3)):
                issues.append(f"length_mismatch_excess:{ar_k}")
            elif en_w >= 45 and ar_w < 3:
                issues.append(f"length_mismatch_truncated:{ar_k}")

    for list_field in ("logical_premises_ar", "analogies_ar"):
        vals = unit.get(list_field)
        if isinstance(vals, str):
            try:
                vals = json.loads(vals)
            except Exception:
                vals = [vals] if vals.strip() else []
        if isinstance(vals, list) and vals:
            if not all(isinstance(p, str) and _is_clean_arabic_text(p) for p in vals):
                issues.append(list_field)

    for cit in (unit.get("scriptural_citations") or []):
        if isinstance(cit, dict):
            sig_ar = str(cit.get("significance_ar") or "").strip()
            if sig_ar and not _is_clean_arabic_text(sig_ar):
                issues.append("scriptural_citations.significance_ar")
                break
    for ex in (unit.get("dialogue_exchange") or unit.get("dialogue_trajectory") or []):
        if isinstance(ex, dict):
            txt_ar = str(ex.get("text_ar") or "").strip()
            txt_en = str(ex.get("text_en") or "").strip()
            if txt_ar and not _is_clean_arabic_text(txt_ar):
                issues.append("dialogue_exchange.text_ar")
                break
            if txt_en and txt_ar:
                en_w = len(txt_en.split())
                ar_w = len(txt_ar.split())
                if en_w >= 8 and ar_w > max(40, int(en_w * 2.5)):
                    issues.append("length_mismatch:dialogue_exchange.text_ar")
                    break
    return issues


def has_valid_arabic_localization(unit: Dict[str, Any]) -> bool:
    """
    Strict Transactional Barrier: Verifies that a Q&A dossier contains genuine Classical Arabic
    in its core fields (question_ar, core_answer_summary_ar) and across all populated Arabic fields.
    Rejects half-baked English-only cards where Round 2 failed, timed out, or produced empty/corrupted Arabic fields.
    """
    return len(diagnose_missing_arabic_fields(unit)) == 0


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


_ARABIC_UNIT_SCALAR_FIELDS = (
    "question_ar",
    "core_answer_summary_ar",
    "fallacy_or_misconception_exposed_ar",
    "analogy_or_metaphor_ar",
    "socratic_question_ar",
    "verbatim_quote_ar",
    "interlocutor_verbatim_quote_ar",
    "topic_category_ar",
)
_ARABIC_UNIT_LIST_FIELDS = ("logical_premises_ar", "analogies_ar")


def _strip_arabic_fields(unit: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a shallow deep-copied English Round 1 seed with Arabic & internal fields stripped."""
    if not isinstance(unit, dict):
        return {}
    cleaned = {
        k: v
        for k, v in unit.items()
        if not k.startswith("_") and not k.endswith("_ar") and k != "topic_category_ar"
    }
    if isinstance(unit.get("scriptural_citations"), list):
        cleaned["scriptural_citations"] = [
            {ck: cv for ck, cv in cit.items() if ck != "significance_ar"}
            if isinstance(cit, dict)
            else cit
            for cit in unit["scriptural_citations"]
        ]
    for ex_key in ("dialogue_exchange", "dialogue_trajectory"):
        if isinstance(unit.get(ex_key), list):
            cleaned[ex_key] = [
                {ek: ev for ek, ev in ex.items() if ek not in ("text_ar", "speaker_label_ar")}
                if isinstance(ex, dict)
                else ex
                for ex in unit[ex_key]
            ]
    return cleaned


def _backfill_clean_arabic_fields(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Fills empty Arabic fields in `target` from verified clean Arabic values in `source`."""
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for field in _ARABIC_UNIT_SCALAR_FIELDS:
        if not str(target.get(field) or "").strip():
            cand = str(source.get(field) or "").strip()
            if _is_clean_arabic_text(cand):
                target[field] = cand
    for list_field in _ARABIC_UNIT_LIST_FIELDS:
        if not target.get(list_field):
            cand_list = source.get(list_field)
            if isinstance(cand_list, list) and cand_list and all(
                isinstance(p, str) and _is_clean_arabic_text(p) for p in cand_list
            ):
                target[list_field] = list(cand_list)
    tgt_cits = target.get("scriptural_citations")
    src_cits = source.get("scriptural_citations")
    if isinstance(tgt_cits, list) and isinstance(src_cits, list):
        for idx, tgt_cit in enumerate(tgt_cits):
            if idx < len(src_cits) and isinstance(tgt_cit, dict) and isinstance(src_cits[idx], dict):
                if not str(tgt_cit.get("significance_ar") or "").strip():
                    cand = str(src_cits[idx].get("significance_ar") or "").strip()
                    if _is_clean_arabic_text(cand):
                        tgt_cit["significance_ar"] = cand
    for ex_key in ("dialogue_exchange", "dialogue_trajectory"):
        tgt_ex = target.get(ex_key)
        src_ex = source.get(ex_key)
        if isinstance(tgt_ex, list) and isinstance(src_ex, list):
            src_by_tid = {
                str(ex.get("turn_id") or "").strip(): ex
                for ex in src_ex
                if isinstance(ex, dict) and str(ex.get("turn_id") or "").strip()
            }
            for idx, tgt_turn in enumerate(tgt_ex):
                if not isinstance(tgt_turn, dict):
                    continue
                tid = str(tgt_turn.get("turn_id") or "").strip()
                src_turn = src_by_tid.get(tid) if tid and tid in src_by_tid else (
                    src_ex[idx] if idx < len(src_ex) else None
                )
                if not isinstance(src_turn, dict):
                    continue
                for turn_ar_key in ("text_ar", "speaker_label_ar"):
                    if not str(tgt_turn.get(turn_ar_key) or "").strip():
                        cand = str(src_turn.get(turn_ar_key) or "").strip()
                        if _is_clean_arabic_text(cand):
                            tgt_turn[turn_ar_key] = cand


def _assess_round2_batch_thinking(
    r1_batch: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> Tuple[str, bool]:
    """Runs pre-Round-2 rubric check; enables bounded thinking only when English repairs are flagged."""
    if summarize_pre_round2_batch is not None:
        return summarize_pre_round2_batch(r1_batch, turns)
    if format_pre_round2_rubric_summary is not None:
        summary = format_pre_round2_rubric_summary(r1_batch, turns)
        needs_repair = bool(
            not summary
            or any("Status=" in ln and "Status=CLEAN" not in ln for ln in summary.splitlines())
        )
        return summary, needs_repair
    return "", True


def _invoke_round2_client(
    client: Any,
    r1_json_str: str,
    prompt_r2: str,
    enable_thinking: bool,
    max_thinking_tokens: int,
) -> str:
    """Dispatches Round 2 to extract_round2_arabic (forwarding thinking flags if accepted) or .extract()."""
    if hasattr(client, "extract_round2_arabic"):
        fn = client.extract_round2_arabic
        if _supports_kwarg(fn, "enable_thinking"):
            kwargs: Dict[str, Any] = {"enable_thinking": enable_thinking}
            if _supports_kwarg(fn, "max_thinking_tokens"):
                kwargs["max_thinking_tokens"] = max_thinking_tokens
            return fn(r1_json_str, prompt_r2, **kwargs)
        return fn(r1_json_str, prompt_r2)

    sys_prompt = ROUND2_ARABIC_VERIFICATION_PROMPT if enable_thinking else COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT
    if _supports_kwarg(client.extract, "system_prompt"):
        return client.extract(prompt_r2, system_prompt=sys_prompt)
    return client.extract(prompt_r2)


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

    def _try_recover_collapsed_turns(
        self,
        video_id: str,
        turns: List[Dict[str, Any]],
        dur_sec: float,
        spk_count: int,
        verbose: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
        """Recover collapsed speaker roles before Stage 2 gives up on a video."""
        total_words = sum(len((t.get("text_original") or "").split()) for t in turns)

        if spk_count >= 2:
            role_words: Dict[str, int] = {}
            for turn in turns:
                role = _canon_speaker_role(turn.get("speaker_role"), "")
                role_words[role] = role_words.get(role, 0) + len((turn.get("text_original") or "").split())
            co_words = role_words.get("CO_DAEE", 0)
            if total_words > 0 and co_words / total_words >= 0.08:
                promoted = []
                for turn in turns:
                    updated = dict(turn)
                    if _canon_speaker_role(updated.get("speaker_role"), "") == "CO_DAEE":
                        updated["speaker_role"] = "INTERLOCUTOR"
                    promoted.append(updated)
                inter_words = sum(
                    len((turn.get("text_original") or "").split())
                    for turn in promoted
                    if _canon_speaker_role(turn.get("speaker_role"), "") == "INTERLOCUTOR"
                )
                health = classify_diarization_health(
                    dur_sec, spk_count, total_words, inter_words, len(promoted)
                )
                if health["status"] != "COLLAPSED":
                    if verbose:
                        print(
                            f"[Stage 2 Recovery] [{video_id}] Recovered via CO_DAEE->INTERLOCUTOR "
                            f"role promotion ({inter_words}/{total_words} words).",
                            flush=True,
                        )
                    return promoted, health, "role_promotion_co_daee"

        numbered_lines = [
            f"[{idx}] {(turn.get('text_original') or '').strip()}"
            for idx, turn in enumerate(turns)
            if (turn.get("text_original") or "").strip()
        ]
        if not numbered_lines:
            return turns, classify_diarization_health(
                dur_sec, spk_count, total_words, 0, len(turns)
            ), None

        recovered = [dict(turn) for turn in turns]
        for start in range(0, len(numbered_lines), 120):
            prompt = (
                "The audio diarizer collapsed speakers in this street dawah transcript.\n"
                "Identify which turn indices [i] are spoken by the INTERLOCUTOR "
                "(non-Muslim / questioner / guest) versus MAIN_DAEE.\n"
                'Return ONLY valid JSON: {"interlocutor_indices": [int, ...]}.\n'
                "If this is genuinely a solo monologue, return "
                '{"interlocutor_indices": []}.\n\n'
                + "\n".join(numbered_lines[start:start + 120])
            )
            try:
                kwargs: Dict[str, Any] = {}
                if _supports_kwarg(self.client.extract, "enable_thinking"):
                    kwargs["enable_thinking"] = False
                if _supports_kwarg(self.client.extract, "max_thinking_tokens"):
                    kwargs["max_thinking_tokens"] = 256
                raw = self.client.extract(prompt, **kwargs)
                match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
                parsed = json.loads(match.group(0)) if match else {}
                indices = parsed.get("interlocutor_indices") or []
                for index in indices:
                    if isinstance(index, int) and 0 <= index < len(recovered):
                        recovered[index]["speaker_role"] = "INTERLOCUTOR"
                        recovered[index]["speaker_id"] = "SPEAKER_LLM_INTERLOCUTOR"
            except Exception as exc:
                if verbose:
                    print(f"[Stage 2 Recovery] [{video_id}] LLM re-diarization failed: {exc}", flush=True)

        inter_words = sum(
            len((turn.get("text_original") or "").split())
            for turn in recovered
            if _canon_speaker_role(turn.get("speaker_role"), "") == "INTERLOCUTOR"
        )
        health = classify_diarization_health(
            dur_sec, 2 if inter_words > 0 else spk_count, total_words, inter_words, len(recovered)
        )
        if health["status"] != "COLLAPSED":
            if verbose:
                print(
                    f"[Stage 2 Recovery] [{video_id}] Recovered via LLM text re-diarization "
                    f"({inter_words}/{total_words} words = {health['interlocutor_word_ratio']:.1%}).",
                    flush=True,
                )
            return recovered, health, "llm_text_rediarization"
        return turns, health, None

    def extract_video_m6(self, video_id: str, verbose: bool = True) -> Optional[List[Dict[str, Any]]]:
        t0_video = time.time()
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

        # Pre-Stage-2 Diarization Health Gate: skip collapsed debates (>180s, >=200w, <=1 spk or <3.5% inter)
        # immediately in 0.0s instead of wasting 5-8 minutes of GPU inference on cards guaranteed to fail [GATE-DIARIZATION].
        scen_metrics = scenario.get("metrics") if isinstance(scenario.get("metrics"), dict) else {}
        dur_sec_meta = float(scen_metrics.get("duration_sec") or scenario.get("duration_sec") or 0.0)
        if dur_sec_meta <= 0.0 and turns:
            dur_sec_meta = max(int(t.get("end_ms") or 0) for t in turns) / 1000.0
        spk_count_meta = int(scen_metrics.get("speakers_count") or len(scenario.get("speakers") or []))
        total_turn_words = sum(len((t.get("text_original") or "").split()) for t in turns)
        inter_turn_words = sum(
            len((t.get("text_original") or "").split())
            for t in turns
            if _canon_speaker_role(t.get("speaker_role"), "") == "INTERLOCUTOR"
        )
        diar_health = classify_diarization_health(
            duration_sec=dur_sec_meta,
            speakers_count=spk_count_meta,
            total_words=total_turn_words,
            interlocutor_words=inter_turn_words,
            turns_count=len(turns),
        )
        recovery_method = None
        if diar_health["status"] == "COLLAPSED":
            turns, diar_health, recovery_method = self._try_recover_collapsed_turns(
                video_id, turns, dur_sec_meta, spk_count_meta, verbose
            )
            if recovery_method:
                scenario["turns"] = turns
                scenario["diarization_recovery_method"] = recovery_method
                self.workspace.atomic_write_json(scen_path, scenario)

        if diar_health["status"] == "COLLAPSED":
            skip_reason = diar_health["reason"]
            zero_card_reason = f"pre_stage2_skip={skip_reason}"
            if verbose:
                print(
                    f"\n[Stage 2 LLM] [{video_id}] PRE-STAGE-2 SKIP (Diarization Collapse): {skip_reason} (0.0s LLM spent)",
                    flush=True,
                )
            self.workspace.atomic_write_json(
                qa_shard_path,
                {
                    "schema_version": "2.0",
                    "video_id": video_id,
                    "extracted_at": time.time(),
                    "worker_id": self.workspace.worker_id,
                    "units": [],
                    "pre_stage2_skip_reason": skip_reason,
                },
            )
            record_video_trace_section(
                self.workspace,
                video_id,
                "stage2_m6",
                {
                    "skipped_pre_stage2_diarization": True,
                    "pre_stage2_skip_reason": skip_reason,
                    "diarization_health": diar_health,
                    "total_elapsed_sec": round(time.time() - t0_video, 2),
                    "r1_candidates_total": 0,
                    "kept_cards_count": 0,
                    "dropped_gate_counts": {"[PRE-STAGE2-DIARIZATION-COLLAPSE]": 1},
                    "zero_card_reason": zero_card_reason,
                    "windows": [],
                },
                title_ar=scenario.get("title_ar", ""),
            )
            self.workspace.advance_micro_step(
                video_id,
                "M6_LLM_EXTRACTED",
                artifacts={"qa_json": str(qa_shard_path)},
                metrics={"qa_count": 0},
                extra_updates={"db_qa_synced": False, "zero_card_reason": zero_card_reason},
            )
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
                f"| Turns: {len(turns)} | Episode Windows: {len(windows)} "
                f"| Diarization: {diar_health['status']} ({diar_health['interlocutor_word_ratio']:.1%} inter)",
                flush=True,
            )

        all_units: List[Dict[str, Any]] = []
        completed_win_ids = []
        gatekeeper_skipped_windows = 0
        r1_total_candidates = 0
        dropped_gate_counts: Dict[str, int] = {}
        window_traces: List[Dict[str, Any]] = []

        def _record_drop(tag: str) -> None:
            dropped_gate_counts[tag] = dropped_gate_counts.get(tag, 0) + 1

        for w in windows:
            w_idx = w["window_index"]
            partial_file = self.workspace.partial_extractions_dir / f"{video_id}_win{w_idx:03d}.json"
            partial_en_file = self.workspace.partial_extractions_dir / f"{video_id}_win{w_idx:03d}.en.json"

            # Raw unmerged scenario turns covering this window (matches SQLite scenario_turns 1-to-1)
            raw_win_turns = [
                t for t in turns
                if int(t.get("end_ms") or 0) >= int(w.get("start_ms") or 0)
                and int(t.get("start_ms") or 0) <= int(w.get("end_ms") or 0)
            ] or (w.get("turns") or [])

            win_trace: Dict[str, Any] = {
                "window_index": w_idx,
                "gatekeeper_skipped": False,
                "gatekeeper_reason": "",
                "r1_elapsed_sec": 0.0,
                "r1_parsed_count": 0,
                "r1_accepted_count": 0,
                "r2_elapsed_sec": 0.0,
                "units": [],
            }

            # Pre-Extraction Gatekeeper: Screen window turns for substantive apologetics
            win_turns = w.get("turns") or []
            is_substantive, gate_reason = is_substantive_apologetics_window(
                win_turns=win_turns,
                window_index=w_idx,
                total_windows=len(windows),
            )
            if not is_substantive:
                gatekeeper_skipped_windows += 1
                win_trace["gatekeeper_skipped"] = True
                win_trace["gatekeeper_reason"] = gate_reason
                window_traces.append(win_trace)
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
                        window_traces.append(win_trace)
                        continue
                    elif all(has_valid_arabic_localization(cu) for cu in cached_win_units):
                        for cu in cached_win_units:
                            if not _is_duplicate_qa(cu, all_units):
                                cu["id"] = f"{video_id}_{len(all_units) + 1:03d}"
                                all_units.append(cu)
                        completed_win_ids.append(w_idx)
                        win_trace["resumed_from_partial"] = True
                        win_trace["kept_count"] = len(cached_win_units)
                        window_traces.append(win_trace)
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
                window_traces.append(win_trace)
                continue

            main_daee = next((s for s in scenario.get("speakers", []) if _canon_speaker_role(s.get("role"), "") == "MAIN_DAEE"), None)
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
                win_trace["r1_elapsed_sec"] = elapsed_sp
                win_trace["r1_parsed_count"] = len(parsed)
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
                window_traces.append(win_trace)
                continue

            # 2. Check for intermediate Round 1 checkpoint (.en.json) or stale English-only partial
            elapsed_r1 = 0.0
            elapsed_r2 = 0.0
            elapsed = 0.0
            r1_units = stale_partial_en_units
            r1_parsed_count = 0
            if r1_units is None and partial_en_file.exists():
                cached_en = self.workspace.read_json(partial_en_file, default=None)
                if isinstance(cached_en, list):
                    r1_units = cached_en
                    r1_parsed_count = len(cached_en)
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
                r1_parsed_count = len(parsed_r1)
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
                        r1_units.append(_preground_round1_unit(u_clean, scenario, w["start_ms"], cohesive_turns))
                self.workspace.atomic_write_json(partial_en_file, r1_units)
            else:
                r1_parsed_count = len(r1_units)
                r1_units = [
                    _preground_round1_unit(u, scenario, w["start_ms"], cohesive_turns)
                    for u in r1_units
                    if isinstance(u, dict)
                ]

            r1_total_candidates += len(r1_units)
            win_trace["r1_elapsed_sec"] = elapsed_r1
            win_trace["r1_parsed_count"] = r1_parsed_count
            win_trace["r1_accepted_count"] = len(r1_units)

            if not r1_units:
                self.workspace.atomic_write_json(partial_file, [])
                completed_win_ids.append(w_idx)
                window_traces.append(win_trace)
                continue

            win_unit_traces: List[Dict[str, Any]] = [
                {
                    "unit_index": i + 1,
                    "r1_question_en": str(u.get("question_en") or ""),
                    "r1_question_turn_id": str(u.get("question_turn_id") or ""),
                    "r1_answer_turn_id": str(u.get("answer_turn_id") or ""),
                    "pre_r2_flagged": False,
                    "pre_r2_defect_codes": [],
                    "r2_thinking_enabled": False,
                    "r2_think_excerpt": "",
                    "r2_en_repairs": {},
                    "r2_retries": [],
                    "outcome": "PENDING",
                    "drop_reason": "",
                    "final_score": None,
                    "final_tier": "",
                    "final_gates": [],
                    "final_defect_codes": [],
                }
                for i, u in enumerate(r1_units)
            ]

            # 3. Round 2: Conditional Bounded-Thinking Repair (flagged batches) or Fast /no_think Arabic Localization (CLEAN)
            # Sub-batch groups of up to 2 units so reasoning (when flagged) + bilingual JSON fit cleanly without truncation.
            r2_batch_size = 2
            final_raw_units: List[Dict[str, Any]] = []
            elapsed_r2 = 0.0
            r2_batches_total = max(1, (len(r1_units) + r2_batch_size - 1) // r2_batch_size)
            for b_idx, b_start in enumerate(range(0, len(r1_units), r2_batch_size), 1):
                r1_batch = r1_units[b_start:b_start + r2_batch_size]
                pre_audit_summary, batch_needs_thinking = _assess_round2_batch_thinking(
                    r1_batch, raw_win_turns
                )
                # CLEAN fast localization gets a focused 450-word transcript excerpt to avoid wrong-turn Arabic confusion
                r2_excerpt = _build_round2_transcript_excerpt(
                    w["formatted_transcript"],
                    r1_batch,
                    max_words=(1800 if batch_needs_thinking else 450),
                )
                r1_json_str = json.dumps(r1_batch, ensure_ascii=False, indent=2)
                r2_think_budget = 512 if batch_needs_thinking else 0
                prompt_template = USER_ROUND2_PROMPT_TEMPLATE if batch_needs_thinking else USER_ROUND2_CLEAN_TEMPLATE
                prompt_r2 = prompt_template.format(
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
                if pre_audit_summary:
                    prompt_r2 = f"{pre_audit_summary}\n\n{prompt_r2}"
                if verbose:
                    if batch_needs_thinking:
                        flag_tags = [
                            t.strip() for t in re.findall(r"Status=([^\n]+)", pre_audit_summary or "")
                            if not t.strip().startswith("CLEAN")
                        ]
                        reason_txt = flag_tags[0] if flag_tags else "EN defects"
                        mode_desc = f"Status={reason_txt} -> bounded <think> ({r2_think_budget} tok)"
                    else:
                        mode_desc = "Status=CLEAN -> fast /no_think Arabic localization"
                    print(f"      [Window {w_idx} R2 Batch {b_idx}/{r2_batches_total}] {mode_desc}...", flush=True)
                t0_r2 = time.time()
                raw_out_r2 = _invoke_round2_client(
                    self.client,
                    r1_json_str,
                    prompt_r2,
                    enable_thinking=batch_needs_thinking,
                    max_thinking_tokens=r2_think_budget,
                )
                elapsed_r2 = round(elapsed_r2 + (time.time() - t0_r2), 1)
                think_excerpt = extract_think_excerpt(raw_out_r2) if batch_needs_thinking else ""

                # Parse per-card Pre-R2 flags into unit traces
                pre_lines = [ln for ln in (pre_audit_summary or "").splitlines() if ln.strip().startswith("- Card ")]
                for local_j in range(len(r1_batch)):
                    ut = win_unit_traces[b_start + local_j]
                    ut["r2_thinking_enabled"] = bool(batch_needs_thinking)
                    if think_excerpt:
                        ut["r2_think_excerpt"] = think_excerpt
                    if local_j < len(pre_lines):
                        pline = pre_lines[local_j]
                        if "Status=CLEAN" not in pline:
                            ut["pre_r2_flagged"] = True
                            ut["pre_r2_defect_codes"] = re.findall(r"\[([A-Z0-9_\-]+)\]", pline)

                parsed_r2 = clean_llm_json_array(raw_out_r2)
                merged_batch = merge_round1_and_round2_units(
                    r1_batch, parsed_r2, allow_implicit_en_repair=batch_needs_thinking
                )
                for local_j, merged_batch_u in enumerate(merged_batch):
                    u_global_idx = b_start + local_j
                    ut = win_unit_traces[u_global_idx]
                    r1_orig = r1_batch[local_j]
                    if not merged_batch_u.get("_dropped_by_round2_audit"):
                        merged_batch_u = sanitize_arabic_fields(merged_batch_u)
                        repairs_diff: Dict[str, Dict[str, str]] = {}
                        for en_k in _CONDITIONAL_EN_SCALAR_KEYS:
                            before_v = str(r1_orig.get(en_k) or "").strip()
                            after_v = str(merged_batch_u.get(en_k) or "").strip()
                            if before_v and after_v and before_v != after_v:
                                repairs_diff[en_k] = {"before": before_v, "after": after_v}
                        if repairs_diff:
                            ut["r2_en_repairs"] = repairs_diff
                            if verbose:
                                print(
                                    f"      [R2 Repair] Unit {u_global_idx + 1}: applied English fixes on {list(repairs_diff.keys())}",
                                    flush=True,
                                )
                    merged_batch_u["_trace_idx"] = u_global_idx
                    final_raw_units.append(merged_batch_u)

            # 3B. Per-Unit Targeted Arabic Retry if batch Round 2 truncated or missed any unit's Arabic fields
            repaired_units: List[Dict[str, Any]] = []
            for idx_u, merged_u in enumerate(final_raw_units):
                t_idx = int(merged_u.get("_trace_idx", idx_u))
                ut = win_unit_traces[t_idx] if 0 <= t_idx < len(win_unit_traces) else {}
                if merged_u.get("_dropped_by_round2_audit"):
                    _record_drop("[DROP-R2-AUDIT]")
                    drop_r = str(merged_u.get("_drop_reason") or "Dropped by Round 2 Rubric Audit")
                    ut["outcome"] = "DROPPED"
                    ut["drop_reason"] = f"[DROP-R2-AUDIT] {drop_r}"
                    if verbose:
                        print(f"      [Drop Reason] Window {w_idx} Unit {idx_u + 1} dropped by R2 Rubric Audit: {drop_r}", flush=True)
                    logger.info(
                        "Window %d unit %d dropped by Round 2 Rubric Audit (%s).",
                        w_idx, idx_u + 1, drop_r,
                    )
                    continue
                if has_valid_arabic_localization(merged_u):
                    merged_u["_trace_idx"] = t_idx
                    repaired_units.append(merged_u)
                    continue

                single_r1_base = r1_units[t_idx] if 0 <= t_idx < len(r1_units) else merged_u
                clean_r1_seed = _strip_arabic_fields(single_r1_base)
                single_json_str = json.dumps([clean_r1_seed], ensure_ascii=False, indent=2)
                single_excerpt = _build_round2_transcript_excerpt(w["formatted_transcript"], [clean_r1_seed], max_words=900)
                base_single_prompt = USER_COMPACT_ROUND2_RETRY_TEMPLATE.format(
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

                cur_candidate = sanitize_arabic_fields(dict(merged_u))
                best_clean_source = dict(cur_candidate)
                # Strip specifically corrupted Arabic fields from backfill cache so retries cannot re-inject them
                for init_issue in diagnose_missing_arabic_fields(best_clean_source):
                    if ":" in init_issue:
                        bad_field = init_issue.split(":", 1)[1]
                        if bad_field in best_clean_source and isinstance(best_clean_source[bad_field], str):
                            best_clean_source[bad_field] = ""

                dropped_on_retry = False
                for retry_attempt in range(1, 4):
                    missing_diag = diagnose_missing_arabic_fields(cur_candidate)
                    if verbose:
                        print(
                            f"      [Window {w_idx} Unit {idx_u+1}] Round 2 Arabic retry attempt {retry_attempt}/3 "
                            f"(missing={', '.join(missing_diag) or 'unknown'})...",
                            flush=True,
                        )
                    feedback_line = (
                        f"RETRY DIAGNOSTIC: Populate these missing/invalid Arabic keys cleanly: {', '.join(missing_diag)}."
                        if missing_diag
                        else ""
                    )
                    t0_retry = time.time()
                    raw_retry = "[]"
                    try:
                        if retry_attempt == 3:
                            escalated_prompt = USER_ROUND2_PROMPT_TEMPLATE.format(
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
                            if feedback_line:
                                escalated_prompt = f"{feedback_line}\n\n{escalated_prompt}"
                            if hasattr(self.client, "extract_round2_arabic") and _supports_kwarg(
                                self.client.extract_round2_arabic, "enable_thinking"
                            ):
                                raw_retry = _invoke_round2_client(
                                    self.client,
                                    single_json_str,
                                    escalated_prompt,
                                    enable_thinking=True,
                                    max_thinking_tokens=384,
                                )
                            elif hasattr(self.client, "extract"):
                                if _supports_kwarg(self.client.extract, "system_prompt"):
                                    raw_retry = self.client.extract(
                                        escalated_prompt, system_prompt=ROUND2_ARABIC_VERIFICATION_PROMPT
                                    )
                                else:
                                    raw_retry = self.client.extract(escalated_prompt)
                            elif hasattr(self.client, "extract_round2_arabic"):
                                raw_retry = self.client.extract_round2_arabic(single_json_str, escalated_prompt)
                        else:
                            attempt_prompt = f"{feedback_line}\n\n{base_single_prompt}" if feedback_line else base_single_prompt
                            if hasattr(self.client, "extract"):
                                raw_retry = self.client.extract(attempt_prompt, system_prompt=COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT)
                            elif hasattr(self.client, "extract_round2_arabic"):
                                raw_retry = self.client.extract_round2_arabic(single_json_str, attempt_prompt)
                    except TypeError:
                        attempt_prompt = f"{feedback_line}\n\n{base_single_prompt}" if feedback_line else base_single_prompt
                        if hasattr(self.client, "extract_round2_arabic"):
                            raw_retry = self.client.extract_round2_arabic(single_json_str, attempt_prompt)
                        elif hasattr(self.client, "extract"):
                            raw_retry = self.client.extract(attempt_prompt)
                        else:
                            raw_retry = "[]"
                    except Exception:
                        raw_retry = "[]"
                    elapsed_r2 = round(elapsed_r2 + (time.time() - t0_retry), 1)
                    parsed_retry = clean_llm_json_array(raw_retry)
                    retry_passed = False
                    if parsed_retry:
                        retry_merged = merge_round1_and_round2_units(
                            [_strip_arabic_fields(clean_r1_seed)], parsed_retry
                        )[0]
                        if retry_merged.get("_dropped_by_round2_audit"):
                            _record_drop("[DROP-R2-AUDIT]")
                            drop_r = str(retry_merged.get("_drop_reason") or "Dropped on retry")
                            if "r2_retries" in ut:
                                ut["r2_retries"].append({
                                    "attempt": retry_attempt,
                                    "mode": "escalated_think" if retry_attempt == 3 else "compact",
                                    "missing_before": missing_diag,
                                    "passed": False,
                                })
                            ut["outcome"] = "DROPPED"
                            ut["drop_reason"] = f"[DROP-R2-AUDIT] {drop_r}"
                            logger.info(
                                "Window %d unit %d dropped on Round 2 retry by Rubric Audit (%s).",
                                w_idx, idx_u + 1, drop_r,
                            )
                            dropped_on_retry = True
                            break
                        retry_merged = sanitize_arabic_fields(retry_merged)
                        _backfill_clean_arabic_fields(retry_merged, best_clean_source)
                        _backfill_clean_arabic_fields(best_clean_source, retry_merged)
                        cur_candidate = retry_merged
                        retry_passed = has_valid_arabic_localization(cur_candidate)

                    if "r2_retries" in ut:
                        ut["r2_retries"].append({
                            "attempt": retry_attempt,
                            "mode": "escalated_think" if retry_attempt == 3 else "compact",
                            "missing_before": missing_diag,
                            "passed": retry_passed,
                        })
                    if retry_passed:
                        break

                if dropped_on_retry:
                    continue
                if has_valid_arabic_localization(cur_candidate):
                    cur_candidate["_trace_idx"] = t_idx
                    repaired_units.append(cur_candidate)
                else:
                    rem_missing = diagnose_missing_arabic_fields(cur_candidate)
                    _record_drop("[DROP-AR-RETRY]")
                    ut["outcome"] = "DROPPED"
                    ut["drop_reason"] = f"[DROP-AR-RETRY] missing={','.join(rem_missing)}"
                    if verbose:
                        print(
                            f"      [Drop Reason] Window {w_idx} Unit {idx_u+1} failed Arabic retry (missing={','.join(rem_missing)})",
                            flush=True,
                        )
                    logger.warning(
                        "Window %d unit %d dropped: failed complete Arabic localization after 3 attempts (missing=%s).",
                        w_idx, idx_u + 1, ",".join(rem_missing),
                    )

            final_raw_units = repaired_units
            elapsed = round(elapsed_r1 + elapsed_r2, 1)
            win_trace["r2_elapsed_sec"] = elapsed_r2

            win_enriched: List[Dict[str, Any]] = []
            audit_labels_by_id: Dict[str, str] = {}
            for raw_item in final_raw_units:
                t_idx = int(raw_item.get("_trace_idx", -1))
                ut = win_unit_traces[t_idx] if 0 <= t_idx < len(win_unit_traces) else {}
                if not has_valid_arabic_localization(raw_item):
                    _record_drop("[DROP-POST-AR]")
                    ut["outcome"] = "DROPPED"
                    ut["drop_reason"] = f"[DROP-POST-AR] {','.join(diagnose_missing_arabic_fields(raw_item))}"
                    continue
                if not is_valid_dialectical_question(raw_item.get("question_en", ""), raw_item.get("question_ar", "")):
                    _record_drop("[DROP-INVALID-Q]")
                    ut["outcome"] = "DROPPED"
                    ut["drop_reason"] = "[DROP-INVALID-Q]"
                    continue
                anchor_drop_reasons: List[str] = []
                enriched = anchor_and_enrich_extracted_unit(
                    raw_unit=raw_item,
                    scenario=scenario,
                    episode_index=w["episode_index"],
                    window_start_ms=w["start_ms"],
                    unit_seq_num=len(all_units) + len(win_enriched) + 1,
                    cohesive_turns=cohesive_turns,
                    drop_reason_out=anchor_drop_reasons,
                )
                if enriched is None:
                    sub_code = anchor_drop_reasons[0] if anchor_drop_reasons else "UNKNOWN"
                    base_code = sub_code.split(":")[0]
                    _record_drop(f"[DROP-ANCHOR:{base_code}]")
                    ut["outcome"] = "DROPPED"
                    ut["drop_reason"] = f"[DROP-ANCHOR:{sub_code}]"
                    if verbose:
                        print(f"      [Drop Reason] Anchor rejected unit: {sub_code}", flush=True)
                    continue
                if not has_valid_arabic_localization(enriched):
                    post_missing = diagnose_missing_arabic_fields(enriched)
                    _record_drop("[DROP-POST-AR]")
                    ut["outcome"] = "DROPPED"
                    ut["drop_reason"] = f"[DROP-POST-AR] {','.join(post_missing)}"
                    continue
                if audit_single_card is not None:
                    video_meta_audit = {
                        "speakers_count": spk_count_meta,
                        "duration_sec": dur_sec_meta,
                    }
                    card_audit = audit_single_card(
                        enriched, raw_win_turns, video_meta=video_meta_audit, require_arabic=True
                    )
                    all_gates = list(card_audit.get("hard_fails_all") or [])
                    if not all_gates and card_audit.get("hard_fail_triggered"):
                        all_gates = [card_audit["hard_fail_triggered"]]
                    final_defect_codes = [d.get("code", "") for d in (card_audit.get("defects") or []) if d.get("code")]
                    ut["final_score"] = card_audit.get("total_score")
                    ut["final_tier"] = card_audit.get("tier", "")
                    ut["final_gates"] = all_gates
                    ut["final_defect_codes"] = final_defect_codes
                    blocking_gates = [
                        g for g in all_gates
                        if g in (
                            "[GATE-LEAK]",
                            "[GATE-NOISE]",
                            "[GATE-ROLE]",
                            "[GATE-PERSPECTIVE]",
                            "[GATE-DIARIZATION]",
                            "[GATE-UNGROUNDED]",
                        )
                    ]
                    if blocking_gates:
                        for bg in blocking_gates:
                            _record_drop(bg)
                        ut["outcome"] = "DROPPED"
                        ut["drop_reason"] = f"{','.join(blocking_gates)} ({','.join(final_defect_codes[:5])})"
                        if verbose:
                            print(
                                f"      [Drop Reason] Final Rubric Audit hard-fail {','.join(blocking_gates)} "
                                f"(defects={','.join(final_defect_codes[:6])})",
                                flush=True,
                            )
                        logger.warning(
                            "Card %s dropped by final Rubric Audit hard-fail %s: %s",
                            enriched.get("id", ""), ",".join(blocking_gates), card_audit.get("root_cause", ""),
                        )
                        continue
                    tier_short = card_audit["tier"].split(":")[0]
                    audit_labels_by_id[enriched["id"]] = f"{card_audit['total_score']}/100 {tier_short}"
                if not _is_duplicate_qa(enriched, all_units + win_enriched):
                    ut["outcome"] = "KEPT"
                    win_enriched.append(enriched)
                else:
                    ut["outcome"] = "DROPPED"
                    ut["drop_reason"] = "[DROP-DUPLICATE]"
                    _record_drop("[DROP-DUPLICATE]")

            win_trace["units"] = win_unit_traces
            window_traces.append(win_trace)

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

        zero_card_reason = ""
        if len(all_units) == 0 and windows:
            gate_summary = ", ".join(f"{k}:{v}" for k, v in sorted(dropped_gate_counts.items())) or "none"
            zero_card_reason = (
                f"windows={len(windows)} gatekeeper_skipped={gatekeeper_skipped_windows} "
                f"r1_candidates={r1_total_candidates} dropped_gates=[{gate_summary}]"
            )
            logger.warning("Video %s extracted 0 Q&A cards (%s)", video_id, zero_card_reason)
            if verbose:
                print(f"   [ZERO-CARD DIAGNOSTIC] Video {video_id} produced 0 cards ({zero_card_reason})", flush=True)

        record_video_trace_section(
            self.workspace,
            video_id,
            "stage2_m6",
            {
                "skipped_pre_stage2_diarization": False,
                "pre_stage2_skip_reason": "",
                "diarization_health": diar_health,
                "total_elapsed_sec": round(time.time() - t0_video, 2),
                "r1_candidates_total": r1_total_candidates,
                "kept_cards_count": len(all_units),
                "dropped_gate_counts": dropped_gate_counts,
                "zero_card_reason": zero_card_reason,
                "windows": window_traces,
            },
            title_ar=scenario.get("title_ar", ""),
        )

        self.workspace.advance_micro_step(
            video_id,
            "M6_LLM_EXTRACTED",
            artifacts={"qa_json": str(qa_shard_path)},
            metrics={"qa_count": len(all_units)},
            extra_updates={"db_qa_synced": False, "zero_card_reason": zero_card_reason},
        )

        if verbose:
            print("   " + "-" * 70)
            print(f"   >>> [STAGE 2 COMPLETE] Video {video_id} M6_LLM_EXTRACTED:")
            print(f"       * Total Extracted Seeker Cards: {len(all_units)}")
            print(f"       * Integrity: 100% VERIFIED -> Committed to Drive extractions/{video_id}.qa.json")
            print("   " + "=" * 70, flush=True)

        return all_units


_DUP_STOPWORDS_EN = frozenset({
    "what", "when", "where", "which", "whom", "whose", "why", "how", "does", "did",
    "the", "and", "for", "with", "are", "was", "were", "that", "this", "from", "can",
    "could", "would", "should", "will", "you", "your", "about", "into", "have", "has",
    "had", "not", "but", "they", "their", "them", "then", "than", "also", "just",
    "all", "any", "even", "make", "made",
})
_DUP_STOPWORDS_AR = frozenset({
    "في", "من", "على", "إلى", "عن", "هل", "ما", "لماذا", "كيف", "متى", "أين",
    "أن", "إن", "ان", "لا", "هذا", "هذه", "التي", "الذي", "هو", "هي", "مع", "بين",
    "عند", "أو", "ثم", "بل", "كل", "قد", "كان", "تكون", "يكون",
})


def _content_tokens_en(text: str) -> set:
    return {
        t for t in re.findall(r"\b[a-z]{3,}\b", (text or "").lower())
        if t not in _DUP_STOPWORDS_EN
    }


def _content_tokens_ar(text: str) -> set:
    return {
        t for t in re.findall(r"[\u0621-\u064a]{2,}", text or "")
        if t not in _DUP_STOPWORDS_AR
    }


def _is_duplicate_qa(candidate: Dict[str, Any], existing_list: List[Dict[str, Any]]) -> bool:
    """Video-wide deduplication using exact substring match and semantic token Jaccard similarity."""
    q_en = (candidate.get("question_en") or "").strip().lower()
    q_ar = (candidate.get("question_ar") or "").strip()
    cand_nums = re.findall(r"\d+", q_en)
    cand_en_toks = _content_tokens_en(q_en)
    cand_ar_toks = _content_tokens_ar(q_ar)

    for ex in existing_list:
        ex_en = (ex.get("question_en") or "").strip().lower()
        ex_ar = (ex.get("question_ar") or "").strip()
        if (q_en and ex_en and (q_en == ex_en or q_en in ex_en or ex_en in q_en)) or (
            q_ar and ex_ar and (q_ar == ex_ar or q_ar in ex_ar or ex_ar in q_ar)
        ):
            return True

        ex_nums = re.findall(r"\d+", ex_en)
        if cand_nums and ex_nums and cand_nums != ex_nums:
            continue

        if len(cand_en_toks) >= 3:
            ex_en_toks = _content_tokens_en(ex_en)
            if len(ex_en_toks) >= 3:
                en_jacc = len(cand_en_toks & ex_en_toks) / len(cand_en_toks | ex_en_toks)
                if en_jacc >= 0.50:
                    return True

        if len(cand_ar_toks) >= 3:
            ex_ar_toks = _content_tokens_ar(ex_ar)
            if len(ex_ar_toks) >= 3:
                ar_jacc = len(cand_ar_toks & ex_ar_toks) / len(cand_ar_toks | ex_ar_toks)
                if ar_jacc >= 0.58:
                    return True

    return False
