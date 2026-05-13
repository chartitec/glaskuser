"""
Outputs a user's transcript content and the extraction prompt as JSON.
Claude Code reads this output, performs the analysis itself, and saves
the resulting profile to .profiles/{user_id}.json using save_profile.py.

Usage:
    python src/extract_profile.py --user 135****3824
    python src/extract_profile.py --all

Output JSON (single user):
{
  "user_id": "...",
  "has_existing_profile": true/false,
  "source_chars": 12345,
  "extract_prompt": "...full prompt with transcript embedded..."
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ingester
import profile as profile_module
from loader import DATA_DIR
from profile import get_extract_prompt

_MAX_CHARS = 80000


def _read_text(p) -> str:
    if p.suffix.lower() == ".docx":
        import docx2txt
        return docx2txt.process(str(p)) or ""
    return p.read_text(encoding="utf-8")


def _build_request(user_id: str, bundle, user_type: str = "用户") -> dict:
    texts = []
    for p in bundle.raw_transcript_paths + bundle.summary_paths:
        try:
            texts.append(_read_text(p))
        except Exception:
            pass

    if not texts:
        return {"user_id": user_id, "error": "no transcript content"}

    combined = "\n\n---\n\n".join(texts)
    source_chars = len(combined)
    if len(combined) > _MAX_CHARS:
        combined = combined[:_MAX_CHARS]

    existing = profile_module.load_profile(user_id)
    skip = existing is not None and existing.source_chars >= source_chars

    return {
        "user_id": user_id,
        "user_type": user_type,
        "has_existing_profile": existing is not None,
        "existing_version": existing.version if existing else 0,
        "source_chars": source_chars,
        "skip": skip,
        "extract_prompt": get_extract_prompt(user_type).format(transcript_text=combined),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=None, help="用户ID，或省略配合 --all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--user-type", default="用户", help="用户类型：家长/学生/老师/用户")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    bundles = ingester.ingest(data_dir)

    if not bundles:
        print(json.dumps({"error": "data/ 下未找到任何用户资料"}, ensure_ascii=False))
        sys.exit(1)

    if args.all:
        results = []
        for uid in sorted(bundles.keys()):
            results.append(_build_request(uid, bundles[uid], user_type=args.user_type))
        print(json.dumps(results, ensure_ascii=False))
    else:
        if not args.user:
            print(json.dumps({"error": "请指定 --user 或 --all"}, ensure_ascii=False))
            sys.exit(1)
        if args.user not in bundles:
            available = ", ".join(sorted(bundles.keys()))
            print(json.dumps({"error": f"未找到用户 {args.user}，可用：{available}"}, ensure_ascii=False))
            sys.exit(1)
        print(json.dumps(_build_request(args.user, bundles[args.user], user_type=args.user_type), ensure_ascii=False))


if __name__ == "__main__":
    main()
