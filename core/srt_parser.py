#!/usr/bin/env python3
"""
core/srt_parser.py
Robust parser for YouTube subtitle (.srt) files.
Handles auto-caption rolling duplicates, timestamp conversions, and conversational chunking.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class SubtitleCue:
    index: int
    start_sec: float
    end_sec: float
    start_fmt: str
    end_fmt: str
    text: str


def time_to_seconds(time_str: str) -> float:
    """Converts '00:01:23,456' or '00:01:23.456' to float seconds."""
    time_str = time_str.strip().replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    return float(parts[0])


def seconds_to_time_fmt(seconds: float) -> str:
    """Converts seconds float to 'HH:MM:SS' or 'MM:SS'."""
    sec_int = int(seconds)
    hours = sec_int // 3600
    minutes = (sec_int % 3600) // 60
    secs = sec_int % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_srt_content(content: str) -> List[SubtitleCue]:
    """
    Parses raw SRT content into clean SubtitleCue objects,
    removing rolling auto-generated duplicates.
    """
    blocks = re.split(r"\n\s*\n", content.strip())
    raw_cues = []
    
    timestamp_re = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        
        # Look for the timestamp line
        time_match = None
        idx = 0
        time_line_idx = -1
        
        for i, line in enumerate(lines):
            m = timestamp_re.search(line)
            if m:
                time_match = m
                time_line_idx = i
                break
                
        if not time_match:
            continue
            
        start_raw, end_raw = time_match.group(1), time_match.group(2)
        start_sec = time_to_seconds(start_raw)
        end_sec = time_to_seconds(end_raw)
        
        # Text is all lines after the timestamp line
        text_lines = lines[time_line_idx + 1:]
        clean_text = " ".join(text_lines).strip()
        
        # Strip HTML tags (e.g. <font>, <c>)
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        
        if clean_text:
            raw_cues.append(
                SubtitleCue(
                    index=len(raw_cues) + 1,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_fmt=seconds_to_time_fmt(start_sec),
                    end_fmt=seconds_to_time_fmt(end_sec),
                    text=clean_text
                )
            )

    # YouTube auto-caption deduplication:
    # Auto-subs repeat previous text fragments. Clean them up.
    deduped_cues: List[SubtitleCue] = []
    for cue in raw_cues:
        if not deduped_cues:
            deduped_cues.append(cue)
            continue
            
        prev_cue = deduped_cues[-1]
        
        # Check if the text is completely identical
        if cue.text == prev_cue.text:
            # Extend duration of the previous cue
            prev_cue.end_sec = cue.end_sec
            prev_cue.end_fmt = cue.end_fmt
            continue
            
        # Check if current cue starts with previous cue text (rolling caption effect)
        if cue.text.startswith(prev_cue.text) and len(cue.text) > len(prev_cue.text):
            # Append only the new suffix
            prev_cue.text = cue.text
            prev_cue.end_sec = cue.end_sec
            prev_cue.end_fmt = cue.end_fmt
            continue

        deduped_cues.append(cue)

    return deduped_cues


def parse_srt_file(file_path: Path) -> List[SubtitleCue]:
    """Parses an SRT file from disk."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return parse_srt_content(content)


def build_continuous_transcript(cues: List[SubtitleCue], include_timestamps: bool = True) -> str:
    """Formats cues into a readable paragraph or timestamped transcript."""
    lines = []
    for cue in cues:
        if include_timestamps:
            lines.append(f"[{cue.start_fmt}] {cue.text}")
        else:
            lines.append(cue.text)
    return "\n".join(lines)


def find_natural_boundary(cues: List[SubtitleCue], target_sec: float, search_window_sec: float = 20.0) -> float:
    """Finds the nearest natural speech break (pause >= 0.8s or sentence ending) within search_window_sec of target_sec."""
    best_time = target_sec
    best_score = -1
    for i, c in enumerate(cues):
        if abs(c.end_sec - target_sec) <= search_window_sec:
            score = 0
            if c.text.rstrip().endswith((".", "?", "!", "...", "؟", ":", ";")):
                score += 5
            if i + 1 < len(cues) and (cues[i + 1].start_sec - c.end_sec >= 0.8):
                score += 10
            if score > best_score:
                best_score = score
                best_time = c.end_sec
    return best_time


def chunk_transcript_for_extraction(
    cues: List[SubtitleCue],
    chunk_duration_sec: float = 240.0,  # 4-minute window
    overlap_duration_sec: float = 60.0  # 1-minute overlap to preserve boundaries
) -> List[Dict[str, Any]]:
    """
    Splits long videos into overlapping, boundary-aware contextual windows.
    Snaps cuts to natural speech pauses and sentence completions so NO dialectical exchange
    is fragmented or lost across chunk boundaries.
    """
    if not cues:
        return []

    chunks = []
    total_duration = cues[-1].end_sec
    
    current_start = 0.0
    while current_start < total_duration:
        target_end = current_start + chunk_duration_sec
        if target_end < total_duration - 30.0:
            current_end = find_natural_boundary(cues, target_end, search_window_sec=20.0)
        else:
            current_end = total_duration

        # Ensure current_end is strictly greater than current_start
        if current_end <= current_start + 60.0:
            current_end = current_start + chunk_duration_sec
        
        # Select all cues within [current_start, current_end]
        window_cues = [
            c for c in cues 
            if (c.start_sec >= current_start and c.start_sec <= current_end)
            or (c.end_sec >= current_start and c.end_sec <= current_end)
        ]
        
        if window_cues:
            actual_start = window_cues[0].start_sec
            actual_end = window_cues[-1].end_sec
            text = build_continuous_transcript(window_cues, include_timestamps=True)
            
            chunks.append({
                "start_sec": actual_start,
                "end_sec": actual_end,
                "start_fmt": seconds_to_time_fmt(actual_start),
                "end_fmt": seconds_to_time_fmt(actual_end),
                "cue_count": len(window_cues),
                "word_count": len(text.split()),
                "text": text
            })
            
        next_start = current_end - overlap_duration_sec
        # Strictly ensure forward progress
        if next_start <= current_start:
            next_start = current_start + (chunk_duration_sec - overlap_duration_sec)

        current_start = next_start
        
        # Avoid tiny residual trailing chunks
        if current_start >= total_duration - 30.0:
            break

    return chunks
