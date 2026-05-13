"""
Scans data/ flat (no fixed subdirectories) and groups files by user.

User ID extraction: leading alphanumeric string before the first non-alphanumeric character.
  U001.mp3                          → user_id = "U001"
  U002_interview_20240321.mp3       → user_id = "U002"
  1001_survey.csv                   → user_id = "1001"
  notes.txt                         → user_id = "notes"  (no separator → full stem)

File classification:
  Audio (.mp3/.wav/.m4a/.aac/.ogg/.flac/.webm)    → transcribe → raw_transcript
  Video (.mp4/.mov/.avi/.mkv)                      → transcribe → raw_transcript
  Text  (.txt/.pdf/.md/.docx)                      → content heuristic:
    dialogue/colloquial content                    → raw_transcript (verbatim)
    analytical/third-person content               → summary
  Table (.csv/.xlsx/.xls)                          → detect by column names:
    user_id + feature + use_count → usage_log
    question + answer             → survey
    (unrecognized columns)        → skipped with warning
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import transcriber

_TEXT_EXTENSIONS = {".txt", ".pdf", ".md"}
_DOCX_EXTENSIONS = {".docx"}
_TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Content-based classification heuristics (read first 800 chars)
_VERBATIM_PATTERNS = re.compile(
    r"嗯|啊|那个|就是|对对对|哦|呢|吧|诶|哎",
)
_VERBATIM_SPEAKER = re.compile(r"(?m)^[^\n：:]{1,10}[：:]\s")
_VERBATIM_FIRSTPERSON = re.compile(r"我")

_SUMMARY_PATTERNS = re.compile(
    r"总结|综合|整体而言|概括|分析|结论|摘要|整理"
)
_SUMMARY_ANALYTICAL = re.compile(
    r"(?:用户|受访者|该用户|被访者)(?:认为|表示|反映|感到|不感兴趣|感兴趣|偏好|使用|提到|指出|认可|拒绝|倾向)"
)


@dataclass
class FileRecord:
    filename: str
    file_type: str  # "audio", "video", "verbatim", "summary", "usage_log", "survey"
    note: str = ""


@dataclass
class UserBundle:
    user_id: str
    raw_transcript_paths: list[Path] = field(default_factory=list)
    summary_paths: list[Path] = field(default_factory=list)
    survey_rows: list[tuple[str, str]] = field(default_factory=list)
    usage_log_path: Path | None = None
    file_records: list[FileRecord] = field(default_factory=list)

    @property
    def transcript_paths(self) -> list[Path]:
        """Backward-compatible: all indexable text paths."""
        return self.raw_transcript_paths + self.summary_paths

    @property
    def survey_summary(self) -> str:
        if not self.survey_rows:
            return "（无问卷数据）"
        lines = [f"Q: {q}\nA: {a}" for q, a in self.survey_rows[:20]]
        return "\n".join(lines)


def _extract_user_id(stem: str) -> str:
    # Match leading alphanumeric-or-asterisk segment (supports anonymized IDs like 135****3824)
    m = re.match(r"[A-Za-z0-9*]+", stem)
    return m.group(0) if m else stem


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _extract_text_preview(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        import docx2txt
        return (docx2txt.process(str(path)) or "")[:800]
    return path.read_text(encoding="utf-8", errors="ignore")[:800]


def _classify_text_content(path: Path) -> str:
    """
    Reads the first 800 chars of a text file and returns 'verbatim' or 'summary'.
    Verbatim: dialogue/colloquial content (interview transcripts, Q&A logs).
    Summary: analytical third-person descriptions of user behavior.
    Defaults to 'verbatim' when evidence is ambiguous (conservative).
    """
    try:
        text = _extract_text_preview(path)
    except Exception:
        return "verbatim"

    verbatim_score = 0
    verbatim_score += len(_VERBATIM_PATTERNS.findall(text)) * 2
    verbatim_score += len(_VERBATIM_SPEAKER.findall(text)) * 3
    verbatim_score += min(len(_VERBATIM_FIRSTPERSON.findall(text)), 10)

    summary_score = 0
    summary_score += len(_SUMMARY_PATTERNS.findall(text)) * 3
    summary_score += len(_SUMMARY_ANALYTICAL.findall(text)) * 4

    return "summary" if summary_score > verbatim_score else "verbatim"


def _classify_table(path: Path, bundles: dict[str, UserBundle]) -> None:
    try:
        df = _read_table(path)
    except Exception as e:
        warnings.warn(f"无法读取 {path.name}: {e}")
        return

    cols = {c.lower() for c in df.columns}

    if {"user_id", "feature", "use_count"}.issubset(cols):
        for uid in df["user_id"].astype(str).unique():
            bundles.setdefault(uid, UserBundle(user_id=uid))
            bundles[uid].usage_log_path = path
            bundles[uid].file_records.append(FileRecord(path.name, "usage_log"))
        return

    if {"question", "answer"}.issubset(cols):
        user_id = _extract_user_id(path.stem)
        bundles.setdefault(user_id, UserBundle(user_id=user_id))
        pairs = list(zip(df["question"].astype(str), df["answer"].astype(str)))
        bundles[user_id].survey_rows.extend(pairs)
        bundles[user_id].file_records.append(FileRecord(path.name, "survey"))
        return

    warnings.warn(
        f"跳过 {path.name}：列名 {list(df.columns)} 不匹配 usage_log 或 survey 格式"
    )


def ingest(data_dir: Path, dry_run: bool = False) -> dict[str, UserBundle]:
    """
    Scan data_dir and return a dict of {user_id: UserBundle}.

    dry_run=True: skip actual audio/video transcription (mark as pending).
    Used by classify.py for classification preview without triggering Whisper.
    """
    bundles: dict[str, UserBundle] = {}

    for path in sorted(data_dir.rglob("*")):
        if any(part.startswith(".") for part in path.parts):
            continue
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if transcriber.is_transcribable(path):
            user_id = _extract_user_id(path.stem)
            bundles.setdefault(user_id, UserBundle(user_id=user_id))
            file_type = "video" if transcriber.is_video(path) else "audio"

            if dry_run:
                cache_file = (
                    Path(__file__).parent.parent / ".cache" / "transcripts" /
                    (path.stem + ".txt")
                )
                note = "已缓存转录稿" if cache_file.exists() else "待转录"
                bundles[user_id].file_records.append(
                    FileRecord(path.name, file_type, note)
                )
            else:
                txt_path = transcriber.transcribe(path)
                bundles[user_id].raw_transcript_paths.append(txt_path)
                bundles[user_id].file_records.append(FileRecord(path.name, file_type))

        elif suffix in _TEXT_EXTENSIONS:
            user_id = _extract_user_id(path.stem)
            bundles.setdefault(user_id, UserBundle(user_id=user_id))
            content_type = _classify_text_content(path)

            if not dry_run:
                if content_type == "verbatim":
                    bundles[user_id].raw_transcript_paths.append(path)
                else:
                    bundles[user_id].summary_paths.append(path)

            detect_label = "对话格式" if content_type == "verbatim" else "分析语言"
            bundles[user_id].file_records.append(
                FileRecord(path.name, content_type, f"检测：{detect_label}")
            )

        elif suffix in _DOCX_EXTENSIONS:
            user_id = _extract_user_id(path.stem)
            bundles.setdefault(user_id, UserBundle(user_id=user_id))
            content_type = _classify_text_content(path)

            if not dry_run:
                if content_type == "verbatim":
                    bundles[user_id].raw_transcript_paths.append(path)
                else:
                    bundles[user_id].summary_paths.append(path)

            detect_label = "对话格式" if content_type == "verbatim" else "分析语言"
            bundles[user_id].file_records.append(
                FileRecord(path.name, "docx", f"Word文档（{detect_label}）")
            )

        elif suffix in _TABLE_EXTENSIONS:
            _classify_table(path, bundles)

    return bundles
