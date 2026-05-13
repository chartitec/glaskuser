"""
CLI entry point for the glaskuser skill.
Called by SKILL.md via bash.

Usage:
  python query.py --user U001 --question "你觉得品牌权威性怎么样？"
  python query.py --user all  --question "你觉得品牌权威性怎么样？"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from anthropic import Anthropic
from loader import load_all_users


def query_single(users, user_id: str, question: str) -> None:
    if user_id not in users:
        available = ", ".join(sorted(users.keys()))
        print(f"错误：未找到用户 {user_id}。已加载的用户：{available}")
        sys.exit(1)
    answer = users[user_id].ask(question)
    km = users[user_id].knowledge_map
    print(f"\n**用户 {user_id} 的回答**\n")
    print(answer)
    print(f"\n---\n*知识边界摘要：常用功能 {len(km.heavy)} 项，从未触发 {len(km.never_used)} 项*")


def query_all(users, question: str) -> None:
    user_ids = sorted(users.keys())
    print(f"\n向 {len(user_ids)} 名用户分身提问中...\n")

    individual: dict[str, str] = {}
    for uid in user_ids:
        print(f"- 用户 {uid}...", end=" ", flush=True)
        individual[uid] = users[uid].ask(question)
        print("完成")

    # Aggregate summary — focus on cross-user decision pattern evidence
    answers_block = "\n\n".join(f"用户{uid}：{ans}" for uid, ans in individual.items())
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""\
你是一名资深用户研究分析师。以下是多名用户对同一问题的访谈回答（已由分身基于各自语料生成）。

你的核心任务是识别**底层决策驱动因子**，而不是汇总表面观点。

分析框架：
1. **跨用户一致的决策驱动**：找出多名用户共同的底层逻辑（如"成绩驱动的消费意愿"、"性价比优先"）。这些逻辑须能跨品类迁移——同一驱动因子对图书、学习机、辅导班的决策均适用。
2. **表面差异 vs 本质分歧**：区分"说法不同但底层逻辑相同"与"真正的价值观分歧"。前者统一描述，后者明确标注哪些用户有实质差异及原因。
3. **证据锚定**：每个结论必须有对应的用户回答片段作为来源依据；若某结论仅来自1名用户，明确注明"单一来源，待验证"。

输出格式：
- **群体决策驱动**（1–3个，每个附代表性用户原话）
- **本质分歧**（若有，说明哪些用户有不同底层逻辑及其可能原因）
- **对当前问题的群体倾向判断**（结论 + 证据基础 + 置信度说明）""",
        messages=[{"role": "user", "content": f"问题：{question}\n\n各用户回答（含置信层级标注）：\n{answers_block}"}],
    )
    print(f"\n**群体总结**\n\n{resp.content[0].text}")
    print("\n---\n**各用户原始回答**\n")
    for uid, ans in individual.items():
        print(f"**用户 {uid}**\n{ans}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="用户ID，或 'all' 查询全部用户")
    parser.add_argument("--question", required=True, help="要向分身提问的问题")
    parser.add_argument("--data-dir", default=None, help="数据目录路径（默认：项目根目录下的 data/）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    users = load_all_users(data_dir=data_dir)
    if not users:
        print("错误：data/ 下未找到任何可用的用户资料（文字稿、音频、问卷）")
        sys.exit(1)

    if args.user.lower() == "all":
        query_all(users, args.question)
    else:
        query_single(users, args.user, args.question)


if __name__ == "__main__":
    main()
