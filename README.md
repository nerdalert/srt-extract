# Transcribe Video to SRT for Subtitles

Command-line tool that uses OpenAI's Whisper model to transcribe video or audio files into SRT subtitle files, with customizable chunk sizes and optional GPU acceleration.

## Features

* **Supported Formats**: Works with common video (`.mp4`, `.mkv`, etc.) and audio (`.wav`, `.mp3`, etc.) files.
* **Model Selection**: Choose from Whisper’s `tiny`, `base`, `small`, `medium`, or `large` models.
* **Language Detection**: Auto-detect language or force a specific language code (e.g., `en`, `es`).
* **Custom Chunking**: Split long segments into smaller caption chunks of up to *N* words for better readability.
* **GPU Acceleration**: Optional CUDA support for faster transcription on compatible hardware.

## Prerequisites

* **FFmpeg** in your system `PATH`:

  * macOS (Homebrew): `brew install ffmpeg`
  * Linux: install via your distro’s package manager (e.g., `apt install ffmpeg`)
* (macOS Python-only) Run `/Applications/Python \ 3.12/Install Certificates.command` to fix certificate issues.

## Installation

1. **Clone or download** this repository.
2. (Optional) **Create and activate** a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:

   ```bash
   pip install git+https://github.com/openai/whisper.git
   ```
4. (Optional) **Enable GPU acceleration** by installing matching PyTorch wheels:

   ```bash
   pip install torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cu117
   ```

## Usage

```bash
./extract-srt.py <input_file> [options]
```

### Positional Argument

* `<input_file>`: Path to your video or audio file (e.g., `lecture.mp4`, `speech.wav`).

### Options

| Flag                | Description                                                       | Default |
| ------------------- | ----------------------------------------------------------------- | ------- |
| `-m`, `--model`     | Whisper model to use (`tiny`, `base`, `small`, `medium`, `large`) | `small` |
| `-l`, `--language`  | Force language code (e.g., `en`, `es`); omit for auto-detection   | *auto*  |
| `-w`, `--max-words` | Maximum words per SRT caption chunk                               | `10`    |
| `-o`, `--output`    | Output SRT file path (defaults to `<input>.<model>.srt`)          | —       |

### Examples

1. **Basic transcription** using the small model:

   ```bash
   ./extract-srt.py lecture.mp4
   ```

2. **Specify output** and **chunk size**:

   ```bash
   ./extract-srt.py talk.mkv --model medium --max-words 8 \
     --output talk.medium.8words.srt
   ```

3. **Force Spanish** transcription:

   ```bash
   ./extract-srt.py entrevista.mp4 --language es
   ```

4. **GPU-accelerated** with large model:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ./extract-srt.py video.mp4 --model large
   ```

## How It Works

1. **Loading the Model**: Uses `whisper.load_model(model_size)` to load the specified model.
2. **Transcription**: Calls `model.transcribe(input_path, language=args.language)` to generate time-stamped segments.
3. **SRT Formatting**:

   * `format_timestamp()` converts float seconds to `HH:MM:SS,mmm` format.
   * `segments_to_srt()` builds SRT entries, splitting long segments into sub‐segments of up to *N* words, preserving timing.

