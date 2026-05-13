"""
Hybrid semantic + BM25 search over raw user corpus.
Returns matching excerpts with similarity scores and confidence assessment.
Does NOT roleplay — surfaces evidence for external analysis.

Usage:
    python src/search.py --user U001 --question "是否喜欢新上线的ai出题功能"
    python src/search.py --user all  --question "..."
    python src/search.py --user U001 --question "..." --top-k 10

Output JSON (single user):
{
  "user_id": "U001",
  "question": "...",
  "results": [
    {
      "rank": 1,
      "text": "...",
      "source": "filename",
      "rrf_score": 0.0156,
      "sem_score": 0.78,
      "confidence_level": "high|medium|low",
      "retrieval": "hybrid|semantic|bm25"
    }
  ],
  "confidence": {
    "score": 0.62,
    "level": "high|medium|low|none",
    "basis": "direct|indirect|none"
  }
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loader import load_all_users


def _confidence_level(sem_score: float, retrieval: str) -> str:
    if retrieval == "bm25":
        return "medium"  # keyword match but no semantic score
    if sem_score >= 0.75:
        return "high"
    if sem_score >= 0.45:
        return "medium"
    return "low"


def _overall_confidence(results: list[dict]) -> dict:
    if not results:
        return {"score": 0.0, "level": "none", "basis": "none"}

    # Use best semantic score across results for overall confidence
    best_sem = max((r["sem_score"] for r in results), default=0.0)

    if best_sem >= 0.75:
        basis = "direct"
        adjusted = round(best_sem, 4)
    elif best_sem >= 0.45:
        basis = "indirect"
        adjusted = round(best_sem * 0.75, 4)
    elif any(r["retrieval"] == "bm25" for r in results):
        # BM25-only hits mean keyword match but semantic distance uncertain
        basis = "indirect"
        adjusted = 0.45
    else:
        basis = "none"
        adjusted = round(best_sem * 0.4, 4)

    if adjusted >= 0.65:
        level = "high"
    elif adjusted >= 0.40:
        level = "medium"
    elif adjusted > 0.0:
        level = "low"
    else:
        level = "none"

    return {"score": adjusted, "level": level, "basis": basis}


def search_user(user_obj, question: str, top_k: int = 8) -> dict:
    nodes = user_obj.retrieve_nodes(question, top_k)

    results = []
    for i, node in enumerate(nodes, 1):
        sem_score = node["sem_score"]
        retrieval = "bm25" if sem_score == 0.0 else "hybrid"
        results.append({
            "rank": i,
            "text": node["text"],
            "source": node["source"],
            "rrf_score": node["rrf_score"],
            "sem_score": sem_score,
            "confidence_level": _confidence_level(sem_score, retrieval),
            "retrieval": retrieval,
        })

    return {
        "user_id": user_obj.user_id,
        "question": question,
        "results": results,
        "confidence": _overall_confidence(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="用户ID，或 'all'")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    users = load_all_users(data_dir=data_dir)

    if not users:
        print(json.dumps({"error": "data/ 下未找到任何用户资料"}, ensure_ascii=False))
        sys.exit(1)

    if args.user.lower() == "all":
        results = [
            search_user(users[uid], args.question, args.top_k)
            for uid in sorted(users.keys())
        ]
        print(json.dumps(results, ensure_ascii=False))
    else:
        if args.user not in users:
            available = ", ".join(sorted(users.keys()))
            print(json.dumps(
                {"error": f"未找到用户 {args.user}，已加载：{available}"},
                ensure_ascii=False,
            ))
            sys.exit(1)
        print(json.dumps(
            search_user(users[args.user], args.question, args.top_k),
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
