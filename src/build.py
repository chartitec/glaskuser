"""
glaskuser_build — process data/ into queryable ChromaDB vector stores
and psychological profiles.
Called by the /glaskuser_build slash command.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ingester
import profile as profile_module
from loader import DATA_DIR, load_all_users


def _load_user_types() -> dict[str, str]:
    """Load optional data/user_types.json: {"U001": "家长", "U002": "学生"}"""
    path = DATA_DIR / "user_types.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  警告：user_types.json 读取失败：{e}")
        return {}


def main() -> None:
    print("=== GlaskUser 构建 ===\n")

    # Step 1: scan
    print("[1/4] 扫描 data/ 文件夹...")
    bundles = ingester.ingest(DATA_DIR)
    user_type_map = _load_user_types()

    if not bundles:
        print("  未找到任何用户资料。请将文件放入 data/ 文件夹后重试。")
        sys.exit(1)

    text_count = sum(
        len(b.raw_transcript_paths) + len(b.summary_paths) for b in bundles.values()
    )
    print(f"  找到 {len(bundles)} 名用户，{text_count} 份文字资料")

    # Step 2: report audio transcription
    cache_dir = Path(__file__).parent.parent / ".cache" / "transcripts"
    cached_txts = sorted(p.stem for p in cache_dir.glob("*.txt")) if cache_dir.exists() else []
    print(f"\n[2/4] 音频转录（共 {len(cached_txts)} 份）")
    for stem in cached_txts:
        print(f"  · {stem}.txt ✓")

    # Step 3: build vector indices
    print("\n[3/4] 构建向量库...")
    if user_type_map:
        print(f"  用户类型配置：{user_type_map}")
    try:
        users = load_all_users(user_type_map=user_type_map)
    except Exception as e:
        print(f"✗ 构建失败：{e}")
        sys.exit(1)

    if not users:
        print("  无可用用户（检查文字稿文件是否存在）。")
        sys.exit(1)

    print(f"  已构建 {len(users)} 名用户向量库")
    for uid, twin in sorted(users.items()):
        km = twin.knowledge_map
        new, skipped = twin.new_docs, twin.skipped_docs
        if new > 0 and skipped > 0:
            index_note = f"新增 {new} 段 / 已有 {skipped} 段保留"
        elif new > 0:
            index_note = f"新增 {new} 段"
        else:
            index_note = f"已有 {skipped} 段，无变更"
        print(f"  · 用户 {uid}（{index_note}，常用功能 {len(km.heavy)} 项，未触达 {len(km.never_used)} 项）")

    # Step 4: report profile status (extraction is done by Claude Code via skill)
    print("\n[4/4] 心理模型状态...")
    pending = []
    for uid in sorted(users.keys()):
        existing = profile_module.load_profile(uid)
        if existing:
            print(
                f"  · 用户 {uid}：已有心理模型"
                f"（v{existing.version}，综合置信度 {existing.overall_confidence():.0%}，"
                f"推断规则 {len(existing.inference_rules)} 条）"
            )
        else:
            print(f"  · 用户 {uid}：尚无心理模型 — 等待提取")
            pending.append(uid)

    print("\n=== 向量库构建完成 ===")
    print(f"已构建 {len(users)} 名用户分身（向量库）")
    if pending:
        print(f"\n待提取心理模型：{', '.join(pending)}")
        print("PROFILE_PENDING:" + json.dumps(pending, ensure_ascii=False))


if __name__ == "__main__":
    main()
