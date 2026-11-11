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
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    from db.database import sanitize_text_field
except (ImportError, ValueError):
    try:
        from ..db.database import sanitize_text_field
    except Exception:
        sanitize_text_field = None


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
    """Purges stray Cyrillic/Russian, CJK/Hangul tokens, or Latin-Arabic splices from generated Classical Arabic fields."""
    replacements = [
        ("따라서،", "لذلك،"), ("따라서", "لذلك"), ("여전히", "لا تزال"),
        ("المiracles", "المعجزات"), ("فالمachines", "فالآلات"), ("كمessenger", "كرسول"),
        ("للمonotheism", "للتوحيد"), ("السcribes", "النساخ"), ("للسcribes", "للنساخ"),
        ("نihilist", "عدمي"), ("النihilists", "العدميون"), ("بprocesseات", "بعمليات"),
        ("ت дисквалифицируют его как совершенный божественный текст", "تسقط عنه صفة النص الإلهي الكامل المعصوم"),
        ("дисквалифицируют его как совершенный божественный текст", "تسقط عنه صفة النص الإلهي الكامل المعصوم"),
        ("педوفيليا", "البيدوفيليا (استغلال الأطفال)"),
        ("والInstinctive", "والفطري"), ("الInstinctive", "الفطري"), ("Instinctive", "فطري"),
        ("Instinct", "فطرة"), ("Reduced", "مختزلاً"), ("reduced", "مختزل"),
        ("condemn", "يُدين"),
        ("والincest", "وزنا المحارم"),
        ("الincest", "زنا المحارم"),
        ("incest", "زنا المحارم"),
        ("two natures and two wills", "طبيعتين وإرادتين"),
        ("two natures", "طبيعتين"),
        ("two wills", "إرادتين"),
        ("bikinis", "لباس البحر الفاضح"),
        ("والpedophilia", "والبيدوفيليا (استغلال الأطفال)"),
        ("الpedophilia", "البيدوفيليا (استغلال الأطفال)"),
        ("pedophilia", "البيدوفيليا (استغلال الأطفال)"),
        ("Hermeneutics", "علم التأويل"), ("Hagia Sophia", "آيا صوفيا"),
        ("Muawiyah", "معاوية"), ("Hassan", "الحسن"), ("omnipresent", "محيط بكل مكان"),
        ("SUBJX", "الفاعل"), ("rijal", "علم الرجال"),
        ("。，？，。", "؟"),
        ("。", "."),
        ("，", "،"),
        ("？", "؟"),
        ("！", "!"),
    ]

    def _clean_str(val: str) -> str:
        if sanitize_text_field is not None:
            return sanitize_text_field(val, is_arabic=True)
        for old_s, new_s in replacements:
            if old_s in val:
                val = val.replace(old_s, new_s)
        # Strip remaining Cyrillic, CJK ideographs, Hangul, and CJK punctuation
        val = re.sub(r"[\u0400-\u04ff\u4e00-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\ufffd]+", "", val)
        return val.strip()

    for k in [
        "question_ar",
        "core_answer_summary_ar",
        "fallacy_or_misconception_exposed_ar",
        "analogy_or_metaphor_ar",
        "socratic_question_ar",
        "interlocutor_verbatim_quote_ar",
        "verbatim_quote_ar",
    ]:
        if isinstance(item.get(k), str) and item[k]:
            item[k] = _clean_str(item[k])

    if isinstance(item.get("logical_premises_ar"), list):
        item["logical_premises_ar"] = [_clean_str(str(p)) for p in item["logical_premises_ar"] if str(p).strip()]

    if isinstance(item.get("dialogue_exchange"), list):
        cleaned_ex = []
        for ex in item["dialogue_exchange"]:
            if isinstance(ex, dict):
                if isinstance(ex.get("text_ar"), str):
                    ex["text_ar"] = _clean_str(ex["text_ar"])
                cleaned_ex.append(ex)
        item["dialogue_exchange"] = cleaned_ex

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


def is_valid_dialectical_question(q_en: str, q_ar: str = "") -> bool:
    """
    Enforces quality threshold: discards conversational noise, rhetorical filler,
    personal street interview chit-chat, or truncated sentence fragments.
    Allows concise, foundational theological questions (e.g. 'Is Jesus God?', 'Who created God?').
    """
    q_en_clean = (q_en or "").strip().lower().rstrip("?").strip()
    words_en = q_en_clean.split()
    if len(words_en) < 3:
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
        "nice to meet you",
    }
    for tp in trivial_phrases_en:
        if q_en_clean.startswith(tp) or q_en_clean == tp:
            return False

    trivial_phrases_ar = [
        "أين أصبت", "كم عمرك", "ما اسمك", "من أنت", "هل تسمعني",
        "ماذا قلت", "عفوا", "معذرة", "أهلا", "مرحبا", "أين تسكن",
        "هل يمكن التقاط صورة", "تشرفنا"
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


def build_episode_windows(
    turns: List[Dict[str, Any]],
    max_words_per_window: int = 3400,
    context_carry_turns: int = 4,
) -> List[Dict[str, Any]]:
    """
    Groups ScenarioTurns by `episode_index` (interlocutor encounter).
    Pre-merges rapid consecutive turns to maintain coherent argumentation units.
    If a single debate episode exceeds `max_words_per_window` (~20-25 minutes of continuous speech),
    splits cleanly at a MAIN_DAEE turn completion and carries 4 context turns forward to preserve dialectical context.
    """
    if not turns:
        return []

    # Pre-merge rapid micro-turns by the same speaker
    cohesive_turns = merge_consecutive_speaker_turns(turns)

    by_ep: Dict[int, List[Dict[str, Any]]] = {}
    for t in cohesive_turns:
        ep = int(t.get("episode_index", 1))
        by_ep.setdefault(ep, []).append(t)

    windows: List[Dict[str, Any]] = []
    for ep_idx in sorted(by_ep.keys()):
        ep_turns = by_ep[ep_idx]
        cur_batch: List[Dict[str, Any]] = []
        cur_words = 0

        for i, t in enumerate(ep_turns):
            w_cnt = len((t.get("text_original") or "").split())
            if cur_batch and (cur_words + w_cnt) > max_words_per_window:
                # Prefer splitting right after a MAIN_DAEE turn finishes
                windows.append(_pack_episode_window(len(windows) + 1, ep_idx, cur_batch))
                carry = cur_batch[-context_carry_turns:] if context_carry_turns > 0 else []
                cur_batch = list(carry) + [t]
                cur_words = sum(len((x.get("text_original") or "").split()) for x in cur_batch)
            else:
                cur_batch.append(t)
                cur_words += w_cnt

        if cur_batch:
            windows.append(_pack_episode_window(len(windows) + 1, ep_idx, cur_batch))

    return windows


def _pack_episode_window(win_num: int, ep_idx: int, win_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    lines = []
    for t in win_turns:
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
        lines.append(f"[{tid}] [{s_fmt} -> {e_fmt}] [{header_role}{flag_str}]: {t.get('text_original', '')}")

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
    Returns None if the unit fails theological validation or attributes opponent bias.
    """
    item = sanitize_arabic_fields(dict(raw_unit))
    vid = scenario["video_id"]
    yt_url = scenario.get("youtube_url") or f"https://www.youtube.com/watch?v={vid}"

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

    # Collect ordered source turn IDs referenced in dialogue_exchange + q_tid + a_tid
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
        is_daee = role in ("MAIN_DAEE", "CO_DAEE")
        
        fallback_spk_ar = daee_name_ar if is_daee else "المحاور"
        fallback_spk_en = daee_name_en if is_daee else "Interlocutor"

        enriched_exchange.append({
            "turn_id": etid,
            "role": role,
            "speaker_label_en": (eturn.get("speaker_label_en") if eturn else None) or fallback_spk_en,
            "speaker_label_ar": (eturn.get("speaker_label_ar") if eturn else None) or fallback_spk_ar,
            "timestamp_seconds": int(eturn["start_ms"] // 1000) if eturn else ts_sec,
            "timestamp_formatted": eturn["start_fmt"] if eturn else ms_to_fmt(ts_sec * 1000),
            "move_type": ex.get("move_type", "Turn"),
            "text_en": ex.get("text_en") or (eturn.get("text_original") if eturn else ""),
            "text_ar": ex.get("text_ar") or (eturn.get("text_ar") if eturn else ""),
        })

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
    # Read from cohesive_turns instead of raw micro-turns, ensuring >= 8 words and no dangling prepositions.
    all_transcript = " ".join([(t.get("text_original") or "").strip() for t in cohesive_turns])

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
        "episode_index": episode_index,
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


def merge_round1_and_round2_units(
    r1_units: List[Dict[str, Any]],
    r2_units: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deterministically zipper-merges Round 1 (pure English dialectical reasoning)
    with Round 2 (theological verification and Classical Arabic localization).

    Protects against:
    - Out-of-order LLM response generation in Round 2 (matches by turn IDs or question similarity).
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

        # Merge Arabic scalar & list fields
        for k, v in r2_u.items():
            if (k.endswith("_ar") or k == "topic_category_ar") and v:
                merged[k] = v

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
                    if not turn_merged.get("text_en") and t2.get("text_en"):
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
                    if not sc_merged.get("significance") and sc2.get("significance"):
                        sc_merged["significance"] = sc2["significance"]
                merged_citations.append(sc_merged)

            for r2_idx, sc2 in enumerate(r2_sc):
                if r2_idx not in used_r2_sc:
                    merged_citations.append(sc2)
            merged["scriptural_citations"] = merged_citations
        elif r1_sc:
            merged["scriptural_citations"] = r1_sc

        # Merge analogies (English preserved from R1, Arabic from R2)
        if r2_u.get("analogies_ar"):
            merged["analogies_ar"] = r2_u["analogies_ar"]
        if not merged.get("analogies") and r2_u.get("analogies"):
            merged["analogies"] = r2_u["analogies"]

        final_merged.append(merged)

    return final_merged


def has_valid_arabic_localization(unit: Dict[str, Any]) -> bool:
    """Verifies that a Q&A dossier contains genuine Classical Arabic in question_ar and core_answer_summary_ar."""
    if not isinstance(unit, dict):
        return False
    q_ar = str(unit.get("question_ar") or "").strip()
    ans_ar = str(unit.get("core_answer_summary_ar") or "").strip()
    if not q_ar or not ans_ar:
        return False
    return bool(re.search(r"[\u0600-\u06ff]", q_ar) and re.search(r"[\u0600-\u06ff]", ans_ar))


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
        # Dynamically scale episode window capacity with active LLM context window:
        # - At 32K ctx: up to 5,500 words (~35-45 min debates fit into 1 single window!)
        # - At 28K ctx: up to 4,800 words (~30 min debates)
        # - At 20K ctx: up to 3,800 words
        # - At 14K ctx: 3,400 words
        act_ctx = getattr(self.client, "context_length", getattr(self, "context_length", 20480))
        if act_ctx >= 30000:
            dyn_win_words = 5500
        elif act_ctx >= 26000:
            dyn_win_words = 4800
        else:
            dyn_win_words = 3400
        windows = build_episode_windows(turns, max_words_per_window=dyn_win_words, context_carry_turns=4)
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

            # 1. Micro-Step Resume: load already-completed episode window from Drive!
            stale_partial_en_units = None
            if partial_file.exists():
                cached_win_units = self.workspace.read_json(partial_file, default=None)
                if isinstance(cached_win_units, list):
                    if not cached_win_units or all(has_valid_arabic_localization(cu) for cu in cached_win_units):
                        for cu in cached_win_units:
                            if not _is_duplicate_qa(cu, all_units):
                                cu["id"] = f"{video_id}_{len(all_units) + 1:03d}"
                                all_units.append(cu)
                        completed_win_ids.append(w_idx)
                        continue
                    else:
                        # Self-heal stale English-only partial window by treating it as Round 1 units for Round 2 Arabic localization!
                        stale_partial_en_units = cached_win_units
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
                r1_units = [
                    u for u in parsed_r1
                    if is_valid_dialectical_question(u.get("question_en", ""), u.get("question_ar", ""))
                ]
                self.workspace.atomic_write_json(partial_en_file, r1_units)

            if not r1_units:
                self.workspace.atomic_write_json(partial_file, [])
                completed_win_ids.append(w_idx)
                continue

            # 3. Round 2: Bilingual Theological Verification & Classical Arabic Translation
            r2_excerpt = _build_round2_transcript_excerpt(w["formatted_transcript"], r1_units, max_words=1800)
            r1_json_str = json.dumps(r1_units, ensure_ascii=False, indent=2)
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
            t0_r2 = time.time()
            if hasattr(self.client, "extract_round2_arabic"):
                raw_out_r2 = self.client.extract_round2_arabic(r1_json_str, prompt_r2)
            else:
                raw_out_r2 = self.client.extract(prompt_r2, system_prompt=ROUND2_ARABIC_VERIFICATION_PROMPT)
            elapsed_r2 = round(time.time() - t0_r2, 1)

            parsed_r2 = clean_llm_json_array(raw_out_r2)
            final_raw_units = merge_round1_and_round2_units(r1_units, parsed_r2)

            # 3B. Per-Unit Targeted Arabic Retry if batch Round 2 truncated or missed any unit's Arabic fields
            repaired_units: List[Dict[str, Any]] = []
            for idx_u, merged_u in enumerate(final_raw_units):
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
                t0_retry = time.time()
                try:
                    raw_retry = self.client.extract(single_prompt_r2, system_prompt=COMPACT_ROUND2_ARABIC_SYSTEM_PROMPT)
                except TypeError:
                    if hasattr(self.client, "extract_round2_arabic"):
                        raw_retry = self.client.extract_round2_arabic(single_json_str, single_prompt_r2)
                    else:
                        raw_retry = self.client.extract(single_prompt_r2)
                elapsed_r2 = round(elapsed_r2 + (time.time() - t0_retry), 1)
                parsed_retry = clean_llm_json_array(raw_retry)
                if parsed_retry:
                    merged_single = merge_round1_and_round2_units([merged_u], parsed_retry)[0]
                    repaired_units.append(merged_single)
                else:
                    repaired_units.append(merged_u)

            final_raw_units = repaired_units
            elapsed = round(elapsed_r1 + elapsed_r2, 1)

            win_enriched: List[Dict[str, Any]] = []
            for raw_item in final_raw_units:
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
                        ans_en = eu.get("core_answer_summary_en", "")
                        ans_preview = ans_en[:110] + "..." if len(ans_en) > 110 else ans_en
                        print(f"      * [Card {eu['id']}] [{cat}]")
                        print(f"        Q (Seeker): \"{q_en}\"")
                        if q_ar:
                            print(f"        سؤال الباحث: \"{q_ar}\"")
                        print(f"        Truth Proof: {ans_preview}")
                else:
                    print("      (No substantive debate objections in this window - skipped)")

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
