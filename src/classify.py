"""
Scans data/ and prints a formatted classification report without building anything.
Audio/video files are marked as "待转录" — no Whisper runs during this preview.

Usage:
    python src/classify.py
    python src/classify.py --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ingester

_TYPE_ICONS = {
    "audio":    "🎙",
    "video":    "🎬",
    "verbatim": "📄",
    "summary":  "📋",
    "docx":     "📝",
    "survey":   "📊",
    "usage_log":"📈",
}

_TYPE_LABELS = {
    "audio":    "音频",
    "video":    "视频",
    "verbatim": "逐字稿",
    "summary":  "总结稿",
    "docx":     "Word文档",
    "survey":   "问卷",
    "usage_log":"使用日志",
}


def print_report(bundles: dict[str, ingester.UserBundle], data_dir: Path) -> None:
    total_files = sum(len(b.file_records) for b in bundles.values())
    print(f"\n扫描结果：{data_dir} 目录共 {total_files} 个文件，覆盖 {len(bundles)} 名用户\n")

    if not bundles:
        print("  （未发现任何符合格式的用户资料）")
        return

    for uid in sorted(bundles.keys()):
        bundle = bundles[uid]
        print(f"用户 {uid}：")
        if not bundle.file_records:
            print("  （无文件）")
            continue
        for record in bundle.file_records:
            icon = _TYPE_ICONS.get(record.file_type, "📁")
            label = _TYPE_LABELS.get(record.file_type, record.file_type)
            note = f"  [{record.note}]" if record.note else ""
            print(f"  {icon} {label}: {record.filename}{note}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="预览 data/ 文件分类，不触发转录或构建")
    parser.add_argument("--data-dir", default=None, help="数据目录（默认：data/）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        print(f"错误：目录不存在：{data_dir}")
        sys.exit(1)

    bundles = ingester.ingest(data_dir, dry_run=True)
    print_report(bundles, data_dir)


if __name__ == "__main__":
    main()
