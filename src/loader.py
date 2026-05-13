"""
Loads all users from data/ and builds their GlaskUser objects.
Delegates file discovery and classification to ingester.
"""
from __future__ import annotations

from pathlib import Path

from twin import GlaskUser
from knowledge_map import KnowledgeMap, build_knowledge_map
import ingester

DATA_DIR = Path(__file__).parent.parent / "data"


def load_all_users(
    data_dir: Path | None = None,
    user_type_map: dict[str, str] | None = None,
) -> dict[str, GlaskUser]:
    """
    user_type_map: {user_id: "家长"/"学生"/"老师"} — optional, defaults to "用户"
    Returns dict of {user_id: GlaskUser}
    """
    data_dir = data_dir or DATA_DIR
    user_type_map = user_type_map or {}

    bundles = ingester.ingest(data_dir)
    users: dict[str, GlaskUser] = {}

    for uid, bundle in bundles.items():
        if not bundle.raw_transcript_paths and not bundle.summary_paths:
            continue  # skip users with no usable text content
        km = (
            build_knowledge_map(uid, bundle.usage_log_path)
            if bundle.usage_log_path
            else KnowledgeMap(user_id=uid)
        )
        users[uid] = GlaskUser(
            user_id=uid,
            user_type=user_type_map.get(uid, "用户"),
            raw_transcript_paths=bundle.raw_transcript_paths,
            summary_paths=bundle.summary_paths,
            survey_summary=bundle.survey_summary,
            knowledge_map=km,
        )

    return users
