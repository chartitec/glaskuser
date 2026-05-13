"""
Prepares RAG context for a glaskuser query without making any API call.
Outputs JSON consumed by Claude Code, which generates the response itself.

Single user:  {"user_id": "...", "system": "...", "user_message": "..."}
All users:    [{"user_id": "...", "system": "...", "user_message": "..."}, ...]

--history accepts a JSON array of {"role": "user"/"assistant", "content": "..."} pairs
for multi-turn simulate sessions (last 5 Q&A pairs, 10 messages max).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loader import load_all_users


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="用户ID，或 'all'")
    parser.add_argument("--question", required=True)
    parser.add_argument("--history", default=None, help="JSON array of prior Q&A messages")
    parser.add_argument("--history-file", default=None, help="path to a JSON file containing prior Q&A messages")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    history = None
    if args.history_file:
        try:
            history = json.loads(Path(args.history_file).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(json.dumps({"error": f"--history-file 读取失败: {e}"}, ensure_ascii=False))
            sys.exit(1)
    elif args.history:
        try:
            history = json.loads(args.history)
        except json.JSONDecodeError:
            print(json.dumps({"error": "--history 不是合法 JSON"}, ensure_ascii=False))
            sys.exit(1)

    data_dir = Path(args.data_dir) if args.data_dir else None
    users = load_all_users(data_dir=data_dir)

    if not users:
        print(json.dumps({"error": "data/ 下未找到任何用户资料"}, ensure_ascii=False))
        sys.exit(1)

    if args.user.lower() == "all":
        results = []
        for uid in sorted(users.keys()):
            system, user_message = users[uid].prepare_prompt(args.question, history=history)
            results.append({"user_id": uid, "system": system, "user_message": user_message})
        print(json.dumps(results, ensure_ascii=False))
    else:
        if args.user not in users:
            available = ", ".join(sorted(users.keys()))
            print(json.dumps({"error": f"未找到用户 {args.user}，已加载：{available}"}, ensure_ascii=False))
            sys.exit(1)
        system, user_message = users[args.user].prepare_prompt(args.question, history=history)
        print(json.dumps({"user_id": args.user, "system": system, "user_message": user_message}, ensure_ascii=False))


if __name__ == "__main__":
    main()
