#!/usr/bin/env python3
"""
extract-srt.py

Transcribe a video or audio file into SRT subtitles using OpenAI Whisper.

Dependencies & Install:
------------------------------------
# 1. Create & activate a virtual environment (optional but recommended):
#    python3 -m venv venv
#    source venv/bin/activate
#
# 2. Install FFmpeg (must be in your PATH):
#    Homebrew:    brew install ffmpeg
#    On OSX, run this to avoid CERTIFICATE_VERIFY_FAILED errors: /Applications/Python\ 3.12/Install\ Certificates.command
#
# 3. Install Whisper (and its dependencies):
#    pip install git+https://github.com/openai/whisper.git
#
# 4. (Optional) For CUDA GPU acceleration, install matching torch wheels:
#    pip install torch torchvision torchaudio \
#      --index-url https://download.pytorch.org/whl/cu117 \
#      
#
# Usage: ./extract-srt.py input_video.mp4 --model small --language en --max-words 8 --output subtitles.srt
"""

import argparse
import os
import sys
from datetime import timedelta

import whisper

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds (float) to SRT timestamp format: HH:MM:SS,mmm
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def segments_to_srt(segments: list[dict], max_words: int) -> str:
    """
    Given Whisper segments, build a string in .srt format.
    If a segment has more than max_words, split it into smaller chunks.
    """
    lines: list[str] = []
    idx = 1

    for seg in segments:
        words = seg["text"].strip().split()
        start_time = seg["start"]
        end_time = seg["end"]
        duration = end_time - start_time

        # If short enough, emit as-is
        if len(words) <= max_words:
            lines.append(str(idx))
            lines.append(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}")
            lines.append(" ".join(words))
            lines.append("")
            idx += 1

        else:
            # Split into sub‐segments of up to max_words
            word_duration = duration / len(words)
            for i in range(0, len(words), max_words):
                chunk = words[i : i + max_words]
                sub_start = start_time + i * word_duration
                sub_end = sub_start + len(chunk) * word_duration

                lines.append(str(idx))
                lines.append(f"{format_timestamp(sub_start)} --> {format_timestamp(sub_end)}")
                lines.append(" ".join(chunk))
                lines.append("")
                idx += 1

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio to SRT using OpenAI Whisper"
    )
    parser.add_argument(
        "input",
        help="Path to input video or audio file (e.g. .mp4, .wav)",
    )
    parser.add_argument(
        "-m", "--model",
        default="small",
        help="Whisper model to use (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Force language code (e.g. en, es). If omitted, will auto-detect.",
    )
    parser.add_argument(
        "-w", "--max-words",
        type=int,
        default=10,
        help="Maximum words per SRT caption chunk (default: 10)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output .srt file (default: <input>.<model>.srt)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    model_size = args.model
    output_path = args.output or f"{os.path.splitext(input_path)[0]}.{model_size}.srt"

    print(f"[1/3] Loading Whisper model '{model_size}'…")
    model = whisper.load_model(model_size)

    print(f"[2/3] Transcribing '{input_path}'…")
    result = model.transcribe(input_path, language=args.language)
    segments = result.get("segments", [])
    if not segments:
        print("No speech segments detected. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"[3/3] Converting {len(segments)} segments into SRT (max {args.max_words} words each) …")
    srt_content = segments_to_srt(segments, args.max_words)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"✅ Done! Subtitles written to '{output_path}'")

if __name__ == "__main__":
    main()

