"""
Validates and saves a UserProfile JSON to .profiles/.
Called by the glaskuser_build skill after Claude Code completes the analysis.

Usage:
    python src/save_profile.py --user 135****3824 --json '{"core_values": {...}, ...}'

The JSON must contain these keys (matching profile.UserProfile structure):
  core_values, decision_framework, tech_attitude, economic_profile,
  pain_points, aspirations, inference_rules

Wraps with user_id, version, generated_at, source_chars automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import profile as profile_module
from profile import ProfileDimension, UserProfile

_REQUIRED_DIMS = ["core_values", "decision_framework", "tech_attitude", "economic_profile"]
_PARENT_DIMS = ["educational_philosophy", "child_context", "social_profile", "brand_attitude"]


def _validate(data: dict, user_type: str = "用户") -> list[str]:
    errors = []
    required = _REQUIRED_DIMS + (_PARENT_DIMS if user_type == "家长" else [])
    for dim in required:
        d = data.get(dim, {})
        if not isinstance(d, dict):
            errors.append(f"{dim} 不是对象")
            continue
        if not d.get("summary", "").strip():
            errors.append(f"{dim}.summary 缺失或为空")
        conf = d.get("confidence", -1)
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            errors.append(f"{dim}.confidence 无效（应为 0.0–1.0，得到 {conf}）")
    if len(data.get("inference_rules", [])) < 3:
        errors.append(f"inference_rules 不足 3 条（得到 {len(data.get('inference_rules', []))} 条）")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="用户ID")
    parser.add_argument("--json", required=True, help="Claude Code 提取的 JSON 字符串")
    parser.add_argument("--source-chars", type=int, default=0)
    parser.add_argument("--user-type", default="用户", help="用户类型：家长/学生/老师/用户")
    args = parser.parse_args()

    try:
        raw = args.json.strip()
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON 解析失败: {e}"}, ensure_ascii=False))
        sys.exit(1)

    errors = _validate(data, user_type=args.user_type)
    if errors:
        print(json.dumps({"error": errors}, ensure_ascii=False))
        sys.exit(1)

    existing = profile_module.load_profile(args.user)

    def _opt_dim(key: str) -> ProfileDimension | None:
        return ProfileDimension.from_dict(data[key]) if data.get(key) else None

    from datetime import datetime, timezone
    profile = UserProfile(
        user_id=args.user,
        user_type=args.user_type,
        version=(existing.version + 1) if existing else 1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_chars=args.source_chars,
        core_values=ProfileDimension.from_dict(data.get("core_values", {})),
        decision_framework=ProfileDimension.from_dict(data.get("decision_framework", {})),
        tech_attitude=ProfileDimension.from_dict(data.get("tech_attitude", {})),
        economic_profile=ProfileDimension.from_dict(data.get("economic_profile", {})),
        pain_points=data.get("pain_points", []),
        aspirations=data.get("aspirations", []),
        inference_rules=data.get("inference_rules", []),
        educational_philosophy=_opt_dim("educational_philosophy"),
        child_context=_opt_dim("child_context"),
        social_profile=_opt_dim("social_profile"),
        brand_attitude=_opt_dim("brand_attitude"),
    )

    profile_module.save_profile(profile)
    print(json.dumps({
        "status": "saved",
        "user_id": profile.user_id,
        "version": profile.version,
        "overall_confidence": profile.overall_confidence(),
        "inference_rules_count": len(profile.inference_rules),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
