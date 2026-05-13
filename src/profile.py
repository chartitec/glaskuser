"""
Extracts and manages a user's psychological profile from interview transcripts.

The profile captures stable cognitive/value dimensions that persist across questions.
It is used as the primary reasoning framework in the system prompt, allowing the twin
to infer answers for scenarios not directly covered by the transcript corpus.

Storage: .profiles/{safe_user_id}.json
Versioning: profile.version increments each time a rebuild detects new source material.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROFILES_DIR = Path(__file__).parent.parent / ".profiles"

# ──────────────────────────────────────────────
# Extraction prompt
# ──────────────────────────────────────────────

EXTRACT_PROMPT = """\
你是一位资深用户研究分析师。请仔细阅读以下用户访谈逐字稿，提取该用户的稳定心理结构。

**提取原则：**
- 只提取有直接文本证据支撑的结论，绝不臆测
- 置信度（0.0–1.0）反映证据强度：
    多处反复、情绪强烈 → 0.85–1.0
    明确提及但仅一次 → 0.55–0.75
    仅间接暗示 → 0.30–0.50
- key_quotes 必须是逐字稿中的原话片段（10–40字），不得改写或概括
- inference_rules 必须从上述价值取向推导，格式：
  "当被问及[场景]，该用户倾向于[反应]，因为[价值取向]"

**维度说明：**
- core_values：最核心、最稳定的价值观（学习观/育儿观/效率观等）
- decision_framework：评估产品/功能时的主要判断标准和逻辑链
- tech_attitude：对新技术、AI功能的整体接受度与判断方式
- economic_profile：消费倾向、价格敏感度、品牌比较行为
- pain_points：明确表达的不满和痛点（每条10–30字）
- aspirations：期望改善的方向（每条10–30字）
- inference_rules：3–6条推断规则，当无直接语料时用于推导该用户对新场景的可能反应

**严格输出 JSON，不要包含任何其他文字：**
{{
  "core_values": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "decision_framework": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "tech_attitude": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "economic_profile": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "pain_points": ["...", "..."],
  "aspirations": ["...", "..."],
  "inference_rules": ["当被问及...", "..."]
}}

---

访谈逐字稿：

{transcript_text}
"""

# ──────────────────────────────────────────────
# 家长用户专项提取 Prompt（8维度）
# ──────────────────────────────────────────────

PARENT_EXTRACT_PROMPT = """\
你是一位资深用户研究分析师，专注于家长用户研究。请仔细阅读以下家长访谈逐字稿，提取该用户的稳定心理结构。

**提取原则：**
- 只提取有直接文本证据支撑的结论，绝不臆测
- 置信度（0.0–1.0）反映证据强度：
    多处反复、情绪强烈 → 0.85–1.0
    明确提及但仅一次 → 0.55–0.75
    仅间接暗示 → 0.30–0.50
- key_quotes 必须是逐字稿中的原话片段（10–40字），不得改写或概括
- inference_rules 必须能预测该家长对任意教育产品/服务的可能反应，不局限于当前访谈中提到的产品
- inference_rules 格式："当被问及[场景]，该用户倾向于[反应]，因为[核心驱动]"

**维度说明（家长专项，8个维度）：**
- core_values：育儿与家庭的核心价值导向（成绩优先/全面发展/快乐成长等，最根本的驱动信念）
- educational_philosophy：教育理念与方法论（鸡娃/素质/放养倾向；对学业压力的态度；投入与收益的预期逻辑）
- child_context：孩子学业现状与亲子关系（孩子成绩情况/学习习惯；家长参与度；亲子互动模式）
- social_profile：社会阶层感知与身份心理（阶层意识；与同圈层的比较行为；教育焦虑来源；身份认同对消费决策的影响）
- brand_attitude：品牌认知与信任逻辑（品牌选择依据：口碑/权威背书/价格信号；对广告的信任度；参照系来源）
- economic_profile：消费能力与教育消费模式（可接受价位区间；教育支出在家庭中的优先级；价格敏感触发条件）
- tech_attitude：对科技/AI产品的接受度与使用逻辑（AI辅助学习的接受程度；对电子产品的整体态度）
- decision_framework：评估教育产品时的主要决策逻辑链（关键因素排序；试用/退款/口碑验证等决策路径）
- pain_points：明确表达的不满和痛点（每条10–30字）
- aspirations：期望改善的方向（每条10–30字）
- inference_rules：3–6条跨场景推断规则（须覆盖不同教育产品品类）

**严格输出 JSON，不要包含任何其他文字：**
{{
  "core_values": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "educational_philosophy": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "child_context": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "social_profile": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "brand_attitude": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "economic_profile": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "tech_attitude": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "decision_framework": {{"summary": "...", "confidence": 0.0, "key_quotes": ["..."]}},
  "pain_points": ["...", "..."],
  "aspirations": ["...", "..."],
  "inference_rules": ["当被问及...", "..."]
}}

---

访谈逐字稿：

{transcript_text}
"""

_EXTRACT_PROMPTS: dict[str, str] = {
    "家长": PARENT_EXTRACT_PROMPT,
}


def get_extract_prompt(user_type: str) -> str:
    """Return the appropriate extraction prompt for the given user type."""
    return _EXTRACT_PROMPTS.get(user_type, EXTRACT_PROMPT)

# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class ProfileDimension:
    summary: str
    confidence: float
    key_quotes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> ProfileDimension:
        return cls(
            summary=d.get("summary", ""),
            confidence=float(d.get("confidence", 0.0)),
            key_quotes=d.get("key_quotes", []),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UserProfile:
    user_id: str
    version: int
    generated_at: str
    source_chars: int
    # Core dimensions (all user types)
    core_values: ProfileDimension
    decision_framework: ProfileDimension
    tech_attitude: ProfileDimension
    economic_profile: ProfileDimension
    # Required lists
    pain_points: list[str]
    aspirations: list[str]
    inference_rules: list[str]
    # Fields with defaults (backward-compatible)
    user_type: str = "用户"
    # Parent-specific optional dimensions
    educational_philosophy: ProfileDimension | None = None
    child_context: ProfileDimension | None = None
    social_profile: ProfileDimension | None = None
    brand_attitude: ProfileDimension | None = None

    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        def _opt_dim(key: str) -> ProfileDimension | None:
            return ProfileDimension.from_dict(d[key]) if d.get(key) else None

        return cls(
            user_id=d["user_id"],
            version=d.get("version", 1),
            generated_at=d.get("generated_at", ""),
            source_chars=d.get("source_chars", 0),
            core_values=ProfileDimension.from_dict(d.get("core_values", {})),
            decision_framework=ProfileDimension.from_dict(d.get("decision_framework", {})),
            tech_attitude=ProfileDimension.from_dict(d.get("tech_attitude", {})),
            economic_profile=ProfileDimension.from_dict(d.get("economic_profile", {})),
            pain_points=d.get("pain_points", []),
            aspirations=d.get("aspirations", []),
            inference_rules=d.get("inference_rules", []),
            user_type=d.get("user_type", "用户"),
            educational_philosophy=_opt_dim("educational_philosophy"),
            child_context=_opt_dim("child_context"),
            social_profile=_opt_dim("social_profile"),
            brand_attitude=_opt_dim("brand_attitude"),
        )

    def to_dict(self) -> dict:
        d = {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "version": self.version,
            "generated_at": self.generated_at,
            "source_chars": self.source_chars,
            "core_values": self.core_values.to_dict(),
            "decision_framework": self.decision_framework.to_dict(),
            "tech_attitude": self.tech_attitude.to_dict(),
            "economic_profile": self.economic_profile.to_dict(),
            "pain_points": self.pain_points,
            "aspirations": self.aspirations,
            "inference_rules": self.inference_rules,
        }
        for key in ("educational_philosophy", "child_context", "social_profile", "brand_attitude"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val.to_dict()
        return d

    def overall_confidence(self) -> float:
        dims = [self.core_values, self.decision_framework,
                self.tech_attitude, self.economic_profile,
                self.educational_philosophy, self.child_context,
                self.social_profile, self.brand_attitude]
        scores = [d.confidence for d in dims if d is not None and d.confidence > 0]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def to_prompt_block(self) -> str:
        """Format profile as an LLM-readable reasoning framework for system prompt."""
        lines = []

        # Core dimensions (all user types)
        core_dims = [
            ("核心价值观", self.core_values),
            ("决策框架", self.decision_framework),
            ("技术态度", self.tech_attitude),
            ("经济消费画像", self.economic_profile),
        ]
        # Parent-specific dimensions
        parent_dims = [
            ("教育理念", self.educational_philosophy),
            ("孩子与亲子关系", self.child_context),
            ("社会阶层感知", self.social_profile),
            ("品牌认知逻辑", self.brand_attitude),
        ]

        for label, dim in core_dims + parent_dims:
            if dim is None or not dim.summary:
                continue
            conf_pct = int(dim.confidence * 100)
            lines.append(f"**{label}**（置信度 {conf_pct}%）")
            lines.append(dim.summary)
            if dim.key_quotes:
                lines.append(f'  原话依据：「{dim.key_quotes[0]}」')
            lines.append("")

        if self.pain_points:
            lines.append("**核心痛点**")
            for p in self.pain_points:
                lines.append(f"• {p}")
            lines.append("")

        if self.aspirations:
            lines.append("**改善期望**")
            for a in self.aspirations:
                lines.append(f"• {a}")
            lines.append("")

        if self.inference_rules:
            lines.append("**推断规则**（当无直接访谈证据时，按此框架推导你的反应）")
            for i, rule in enumerate(self.inference_rules, 1):
                lines.append(f"{i}. {rule}")
            lines.append("")

        return "\n".join(lines)


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────

def _profile_path(user_id: str) -> Path:
    safe_id = user_id.replace("*", "X")
    return _PROFILES_DIR / f"{safe_id}.json"


def load_profile(user_id: str) -> Optional[UserProfile]:
    path = _profile_path(user_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return UserProfile.from_dict(json.load(f))
    except Exception:
        return None


def save_profile(profile: UserProfile) -> None:
    _PROFILES_DIR.mkdir(exist_ok=True)
    with open(_profile_path(profile.user_id), "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Building
# ──────────────────────────────────────────────

def build_profile(
    user_id: str,
    transcript_paths: list[Path],
    api_client=None,
    max_chars: int = 80000,
    force: bool = False,
    user_type: str = "用户",
) -> Optional[UserProfile]:
    """
    Read all transcripts, call Claude to extract psychological profile, save and return it.

    Skips rebuild if cached profile's source_chars >= current total (unless force=True).
    Returns None if no transcript content is available or extraction fails.
    """
    # Collect transcript text
    texts = []
    for p in transcript_paths:
        try:
            if p.suffix.lower() == ".docx":
                import docx2txt
                texts.append(docx2txt.process(str(p)) or "")
            else:
                texts.append(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not texts:
        return None

    combined = "\n\n---\n\n".join(texts)
    source_chars = len(combined)

    # Skip if we already have a profile with same or more source material
    existing = load_profile(user_id)
    if existing and not force and existing.source_chars >= source_chars:
        return existing

    if len(combined) > max_chars:
        combined = combined[:max_chars]

    if api_client is None:
        from anthropic import Anthropic
        api_client = Anthropic()

    prompt = get_extract_prompt(user_type).format(transcript_text=combined)
    try:
        resp = api_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
    except Exception as e:
        print(f"[profile] 提取失败 {user_id}: {e}")
        return None

    def _opt_dim(key: str) -> ProfileDimension | None:
        return ProfileDimension.from_dict(data[key]) if data.get(key) else None

    profile = UserProfile(
        user_id=user_id,
        user_type=user_type,
        version=(existing.version + 1) if existing else 1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_chars=source_chars,
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

    save_profile(profile)
    return profile
