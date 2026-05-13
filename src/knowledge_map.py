"""
Builds a user's knowledge boundary map from usage logs (CSV/Excel).
Classifies each feature as: heavy / light / abandoned / never_used.
"""
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class KnowledgeMap:
    user_id: str
    heavy: list[str] = field(default_factory=list)      # >3 uses
    light: list[str] = field(default_factory=list)       # 1-2 uses
    abandoned: list[str] = field(default_factory=list)   # used then stopped
    never_used: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        lines = [f"用户 {self.user_id} 的功能接触记录："]
        if self.heavy:
            lines.append(f"- 常用（>3次）：{', '.join(self.heavy)}")
        if self.light:
            lines.append(f"- 偶尔使用（1-2次）：{', '.join(self.light)}")
        if self.abandoned:
            lines.append(f"- 曾使用后放弃：{', '.join(self.abandoned)}")
        if self.never_used:
            lines.append(f"- 从未触发：{', '.join(self.never_used)}")
        return "\n".join(lines)


def build_knowledge_map(user_id: str, log_path: Path) -> KnowledgeMap:
    """
    Expected CSV/Excel columns: user_id, feature, use_count, abandoned (bool)
    Adjust column names below to match your actual export format.
    """
    suffix = log_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(log_path)
    else:
        df = pd.read_csv(log_path)

    user_df = df[df["user_id"].astype(str) == str(user_id)]
    all_features = df["feature"].unique().tolist()
    used_features = set(user_df["feature"].tolist())

    km = KnowledgeMap(user_id=user_id)
    km.never_used = [f for f in all_features if f not in used_features]

    for _, row in user_df.iterrows():
        feature = row["feature"]
        count = int(row.get("use_count", 1))
        abandoned = bool(row.get("abandoned", False))

        if abandoned:
            km.abandoned.append(feature)
        elif count > 3:
            km.heavy.append(feature)
        else:
            km.light.append(feature)

    return km
