"""
Transcribes audio files to text using openai-whisper.
Results are cached to .cache/transcripts/ — re-runs are instant.

Model loading priority:
  1. models/small.pt  (bundled local file, no download needed)
  2. ~/.cache/whisper/ (openai-whisper default cache)
  3. Auto-download on first run (~461 MB)

NOTE: Whisper downloads from OpenAI's Azure CDN (openaipublic.azureedge.net),
not from HuggingFace. HF_ENDPOINT has no effect on this download.
Azure CDN is generally accessible in mainland China.
Requires: pip install openai-whisper
System dep: ffmpeg（由 /glaskuser_init 自动安装）
"""
from __future__ import annotations

from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
TRANSCRIBABLE_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "transcripts"
_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODEL_NAME = "small"


def _model_arg() -> str:
    """Return a local path if bundled, else the model name (triggers download)."""
    local = _MODELS_DIR / f"{_MODEL_NAME}.pt"
    return str(local) if local.exists() else _MODEL_NAME


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_transcribable(path: Path) -> bool:
    """True for audio and video files that Whisper can transcribe."""
    return path.suffix.lower() in TRANSCRIBABLE_EXTENSIONS


def transcribe(audio_path: Path) -> Path:
    """Return path to a cached .txt transcript; transcribes on first call."""
    cache_file = _CACHE_DIR / (audio_path.stem + ".txt")
    if cache_file.exists():
        return cache_file

    import whisper  # lazy import

    model_arg = _model_arg()
    src = "本地文件" if Path(model_arg).exists() else "下载"
    print(f"  [whisper] 转录 {audio_path.name}（模型来源：{src}）...")
    model = whisper.load_model(model_arg)
    result = model.transcribe(str(audio_path), language="zh")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result["text"], encoding="utf-8")
    print(f"  [whisper] 完成 → {cache_file}")
    return cache_file
