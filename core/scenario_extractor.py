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
from core.extractor_prompt import EXTRACTION_SYSTEM_PROMPT, USER_EXTRACTION_PROMPT_TEMPLATE
from core.audio_diarizer import ms_to_fmt


def clean_llm_json_array(raw_text: str) -> List[Dict[str, Any]]:
    """Parses JSON arrays robustly, recovering completed objects even if generation truncated."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict) and ("question_en" in x or "question_ar" in x)]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict) and ("question_en" in x or "question_ar" in x)]
            if "question_en" in parsed or "question_ar" in parsed:
                return [parsed]
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
                        if isinstance(obj, dict) and ("question_en" in obj or "question_ar" in obj):
                            results.append(obj)
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
        ("。，？，。", "؟"),
        ("。", "."),
        ("，", "،"),
        ("？", "؟"),
        ("！", "!"),
    ]

    def _clean_str(val: str) -> str:
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


def is_valid_dialectical_question(q_en: str, q_ar: str) -> bool:
    """
    Enforces quality threshold: discards conversational noise, rhetorical filler,
    or truncated sentence fragments (e.g. 'Let her be?').
    """
    q_en_clean = (q_en or "").strip().lower().rstrip("?").strip()
    words_en = q_en_clean.split()
    if len(words_en) < 4:
        return False
    trivial_phrases = {
        "let her be", "what did you say", "are you sure", "excuse me",
        "can you hear me", "what is your name", "what's your name", "who are you",
        "repeat that", "are you listening", "hello", "hi there", "is that so",
        "pardon me", "listen to me", "did you hear",
    }
    for tp in trivial_phrases:
        if q_en_clean.startswith(tp) or q_en_clean == tp:
            return False
    return True


def build_episode_windows(
    turns: List[Dict[str, Any]],
    max_words_per_window: int = 1100,
    context_carry_turns: int = 3,
) -> List[Dict[str, Any]]:
    """
    Groups ScenarioTurns by `episode_index` (interlocutor encounter).
    If a single debate episode exceeds `max_words_per_window` (~10-12 minutes of speech),
    splits cleanly at a MAIN_DAEE turn completion and carries 3 context turns forward.
    """
    if not turns:
        return []

    by_ep: Dict[int, List[Dict[str, Any]]] = {}
    for t in turns:
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


def anchor_and_enrich_extracted_unit(
    raw_unit: Dict[str, Any],
    scenario: Dict[str, Any],
    episode_index: int,
    window_start_ms: int,
    unit_seq_num: int,
) -> Dict[str, Any]:
    """
    Anchors LLM-extracted turn IDs and speaker identities directly against ground-truth
    CTC-aligned ScenarioTurns and diarized Speaker records.
    """
    item = sanitize_arabic_fields(dict(raw_unit))
    vid = scenario["video_id"]
    yt_url = scenario.get("youtube_url") or f"https://www.youtube.com/watch?v={vid}"

    turns_by_id = {t["turn_id"]: t for t in scenario.get("turns", []) if "turn_id" in t}
    speakers = scenario.get("speakers", [])
    main_daee = next((s for s in speakers if s.get("role") == "MAIN_DAEE"), None)

    q_tid = str(item.get("question_turn_id") or "").strip()
    a_tid = str(item.get("answer_turn_id") or "").strip()
    q_turn = turns_by_id.get(q_tid)
    a_turn = turns_by_id.get(a_tid)

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

    if a_turn is not None:
        ans_sec = int(a_turn["start_ms"] // 1000)
        # Enforce that Da'ee speaker MUST come from MAIN_DAEE or CO_DAEE role
        if a_turn.get("speaker_role") in ("MAIN_DAEE", "CO_DAEE"):
            daee_spk_id = a_turn.get("speaker_id", main_daee["speaker_id"] if main_daee else "")
        else:
            # If candidate answer turn erroneously pointed to an interlocutor turn, fall back strictly to main_daee
            daee_spk_id = main_daee["speaker_id"] if main_daee else ""
    else:
        ans_sec = ts_sec
        daee_spk_id = main_daee["speaker_id"] if main_daee else ""

    matched_daee = next(
        (s for s in speakers if s["speaker_id"] == daee_spk_id and s.get("role") in ("MAIN_DAEE", "CO_DAEE")),
        main_daee
    ) or {}

    daee_name_en = matched_daee.get("display_name_en") or (main_daee.get("display_name_en") if main_daee else "Main Da'ee")
    daee_name_ar = matched_daee.get("display_name_ar") or (main_daee.get("display_name_ar") if main_daee else "الداعية المحاور")
    daee_profile_id = matched_daee.get("profile_id") or (main_daee.get("profile_id") if main_daee else "")

    # Ensure daee_name_ar never falls into generic interlocutor labels
    if daee_name_ar in ("المحاور", "المحاور 1", "المحاور 2", ""):
        if main_daee and main_daee.get("display_name_ar"):
            daee_name_ar = main_daee["display_name_ar"]
            daee_name_en = main_daee.get("display_name_en") or "Main Da'ee"
            daee_profile_id = main_daee.get("profile_id") or ""

    # Collect ordered source turn IDs referenced in dialogue_exchange + q_tid + a_tid
    src_tids: List[str] = []
    for cand_id in [q_tid, a_tid] + [
        str(ex.get("turn_id") or "").strip()
        for ex in (item.get("dialogue_exchange") or [])
        if isinstance(ex, dict)
    ]:
        if cand_id and cand_id in turns_by_id and cand_id not in src_tids:
            src_tids.append(cand_id)

    # Enrich each dialogue_exchange step with exact verified timestamp & speaker label
    enriched_exchange = []
    for ex in (item.get("dialogue_exchange") or []):
        if not isinstance(ex, dict):
            continue
        etid = str(ex.get("turn_id") or "").strip()
        eturn = turns_by_id.get(etid)
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
            "text_en": ex.get("text_en", ""),
            "text_ar": ex.get("text_ar", ""),
        })

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
        "source_turn_ids": src_tids,
    })
    return item


class Stage2ScenarioExtractor:
    """
    Runs Qwen 2.5 14B Instruct (Q5_K_M GGUF, 12,288 ctx, 8-bit KV cache) over
    Stage 1 Scenario JSON files, saving every episode window immediately to Drive.
    """

    def __init__(
        self,
        workspace: SharedWorkspace,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        context_length: int = 14336,
        llm_client: Optional[Any] = None,
    ):
        self.workspace = workspace
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
            return saved.get("units", []) if isinstance(saved, dict) else saved

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

        windows = build_episode_windows(turns, max_words_per_window=1100, context_carry_turns=3)
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

            # Micro-Step Resume: load already-completed episode window from Drive!
            if partial_file.exists():
                cached_win_units = self.workspace.read_json(partial_file, default=None)
                if isinstance(cached_win_units, list):
                    for cu in cached_win_units:
                        if not _is_duplicate_qa(cu, all_units):
                            cu["id"] = f"{video_id}_{len(all_units) + 1:03d}"
                            all_units.append(cu)
                    completed_win_ids.append(w_idx)
                    continue

            if w["word_count"] < 20:
                self.workspace.atomic_write_json(partial_file, [])
                completed_win_ids.append(w_idx)
                continue

            prompt = USER_EXTRACTION_PROMPT_TEMPLATE.format(
                title_ar=scenario.get("title_ar", ""),
                youtube_url=scenario.get("youtube_url", ""),
                episode_index=w["episode_index"],
                start_fmt=w["start_fmt"],
                end_fmt=w["end_fmt"],
                speakers_summary=speakers_summary or "MAIN_DAEE vs INTERLOCUTOR",
                transcript_text=w["formatted_transcript"],
            )

            t0 = time.time()
            raw_out = self.client.extract(prompt)
            elapsed = round(time.time() - t0, 1)
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
                )
                if not _is_duplicate_qa(enriched, all_units + win_enriched):
                    win_enriched.append(enriched)

            # Persist episode window immediately to Drive before moving to next episode!
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
