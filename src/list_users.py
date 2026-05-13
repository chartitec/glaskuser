"""
List all built user personas from the local ChromaDB store.
Outputs JSON for consumption by the /glaskuser_simulate and /glaskuser_search commands.

Output: [{"user_id": "U001", "docs": 42, "has_profile": true}, ...]
"""
import json
import re
import sys
from pathlib import Path

try:
    import chromadb as _cdb
    from chromadb.telemetry.product.posthog import Posthog as _ChromaPosthog
    _ChromaPosthog._direct_capture = lambda self, event: None
except Exception:
    pass

CHROMA_PATH = Path(__file__).parent.parent / ".chroma"
PROFILES_PATH = Path(__file__).parent.parent / ".profiles"
DATA_PATH = Path(__file__).parent.parent / "data"


def _build_safe_to_original() -> dict[str, str]:
    """Scan data/ to map safe_id (X replaces *) → original user_id (with *)."""
    mapping: dict[str, str] = {}
    if not DATA_PATH.exists():
        return mapping
    for p in sorted(DATA_PATH.rglob("*")):
        if any(part.startswith(".") for part in p.parts) or not p.is_file():
            continue
        m = re.match(r"[A-Za-z0-9*]+", p.stem)
        if m:
            orig = m.group(0)
            mapping[orig.replace("*", "X")] = orig
    return mapping


def main() -> None:
    if not CHROMA_PATH.exists():
        print(json.dumps([]))
        sys.exit(0)

    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=chromadb.Settings(anonymized_telemetry=False))
    safe_to_original = _build_safe_to_original()

    twins = []
    for col_name in client.list_collections():
        col_name = str(col_name)
        if col_name.startswith("user_"):
            safe_id = col_name[len("user_"):]
            display_id = safe_to_original.get(safe_id, safe_id)
            col = client.get_collection(col_name)
            has_profile = (PROFILES_PATH / f"{safe_id}.json").exists()
            twins.append({"user_id": display_id, "docs": col.count(), "has_profile": has_profile})

    twins.sort(key=lambda x: x["user_id"])
    print(json.dumps(twins, ensure_ascii=False))


if __name__ == "__main__":
    main()
