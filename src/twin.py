"""
Builds and queries a glaskuser persona for one user.
Combines transcript (RAG) + survey answers + knowledge map (constraint).
Retrieval uses hybrid search: semantic (ChromaDB) + BM25 (rank_bm25) merged via RRF.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # suppress HF tokenizers fork warning

warnings.filterwarnings("ignore", category=UserWarning)  # suppress jieba pkg_resources warning

import jieba
# jieba registers a StreamHandler on import; clear it before initializing to suppress
# "Building prefix dict..." messages that go directly to stderr regardless of sys.stderr
_jieba_logger = logging.getLogger("jieba")
_jieba_logger.handlers.clear()
_jieba_logger.addHandler(logging.NullHandler())
_jieba_logger.propagate = False
jieba.initialize()

import chromadb

# chromadb 0.6.x calls posthog.capture(user_id, event, props) but posthog ≥7 only accepts
# capture(event, **kwargs), causing noisy "Failed to send telemetry event" stderr spam.
# Patch _direct_capture to silence it entirely.
try:
    from chromadb.telemetry.product.posthog import Posthog as _ChromaPosthog
    _ChromaPosthog._direct_capture = lambda self, event: None
except Exception:
    pass
from anthropic import Anthropic
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi

from knowledge_map import KnowledgeMap
import profile as profile_module

_CHROMA_PATH = Path(__file__).parent.parent / ".chroma"
_EMBED_LOCAL_PATH = Path(__file__).parent.parent / "models" / "bge-small-zh-v1.5"
_client: Anthropic | None = None
_client_key: str | None = None
_embed_initialized = False


def _ensure_embed() -> None:
    global _embed_initialized
    if not _embed_initialized:
        model_name = (
            str(_EMBED_LOCAL_PATH)
            if _EMBED_LOCAL_PATH.exists()
            else "BAAI/bge-small-zh-v1.5"
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
        _embed_initialized = True


def _get_client() -> Anthropic:
    global _client, _client_key
    current_key = os.environ.get("ANTHROPIC_API_KEY")
    if _client is None or current_key != _client_key:
        _client = Anthropic()
        _client_key = current_key
    return _client


def _tokenize(text: str) -> list[str]:
    """Jieba word-level tokenization for Chinese BM25."""
    return [w for w in jieba.cut(text) if w.strip()]


_NO_PROFILE_NOTICE = """\
（心理模型尚未构建。请运行 /glaskuser_build 以生成基于访谈的推断框架。）
"""

SYSTEM_TEMPLATE = """\
你正在扮演用户 {user_id}，一名真实参与过用户研究的受访者（{user_type}）。

## 你的心理模型与价值框架

{profile_block}
## 你对产品功能的熟悉程度

{knowledge_block}

## 你的问卷回答摘要

{survey_summary}

---

## 回答规则（必须严格遵守）

### ⚠️ 最高优先级：无证据时必须拒绝推断

以下情况**必须**直接说"这个我真的没谈到过/没接触过，不知道"，**禁止用任何推断填充**：
1. 检索到的片段与问题完全无关（系统已标注"相关性极弱"）
2. 没有检索到任何相关片段
3. 问题涉及你从未使用过的功能（knowledge map 中 `never_used`）

这条规则优先于心理模型、inference_rules 和所有其他规则。宁可回答"我不知道"，不可编造。

### 置信层级体系
每条回答末尾必须标注以下层级之一：

- **[直接证据]**：访谈中明确说过，接近原话表达；仅来自总结稿时注明（总结稿）
- **[框架推断]**：无直接访谈语料，但有至少一个 inference_rule 或 core_values 维度可以合理推导；
  措辞须体现不确定性（"我觉得"、"应该是"、"按我的习惯"）；
  **若推断链无法从现有心理模型维度直接推导，升级为[弱推断]或拒绝，不得强行推断**
- **[弱推断]**：有一定逻辑关联但把握低于 50%；须说"这个我不太确定，不过如果硬要说..."；
  不得给出具体数字、操作步骤或明确结论

### 推理优先级
1. 有直接访谈证据且语义相关度充足 → 用，标 [直接证据]
2. 无直接证据但心理模型有对应维度/inference_rule → 推导，标 [框架推断]
3. 推理链断裂、模型维度无法覆盖，且语料覆盖极弱 → 直接拒绝，不标层级
4. 有微弱关联但把握不足 → 标 [弱推断]，显式说明不确定

### 功能边界约束（优先于置信层级规则）
- `never_used` 功能：必须说"我不太清楚这个功能"，不可推断
- `abandoned` 功能：表达困惑或不好用的感受，不提供操作细节
- `light` 功能：只给模糊印象，不提供操作步骤

### 表达风格
- 绝对不要给出产品文档级别的标准答案
- 用第一人称、口语化表达，像真实用户在访谈中说话
- 框架推断时推理过程可短暂显现（"因为我一向觉得..."），让回答有内在逻辑
- 拒绝时也要自然口语化："这个嘛，我真的不记得聊过，不太好说"
"""


_HASH_META_KEY = "_indexed_file_hashes"


def _file_hash(path: Path) -> str:
    """Return first 8 hex chars of sha256 for a file."""
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:8]


def _current_hashes(raw_paths: list[Path], summary_paths: list[Path]) -> dict[str, str]:
    result = {}
    for p in raw_paths + summary_paths:
        result[p.name] = _file_hash(p)
    return result


def _get_or_create_index(
    user_id: str,
    raw_paths: list[Path],
    summary_paths: list[Path],
) -> tuple[VectorStoreIndex, int, int]:
    """Returns (index, new_docs_count, skipped_docs_count)."""
    _ensure_embed()
    chroma_client = chromadb.PersistentClient(path=str(_CHROMA_PATH), settings=chromadb.Settings(anonymized_telemetry=False))
    safe_id = user_id.replace("*", "X")
    collection = chroma_client.get_or_create_collection(f"user_{safe_id}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    current_hashes = _current_hashes(raw_paths, summary_paths)

    # Load previously indexed file hashes from collection metadata
    stored_meta = collection.metadata or {}
    stored_hashes: dict[str, str] = json.loads(stored_meta.get(_HASH_META_KEY, "{}"))

    # Find new or modified files
    new_raw = [p for p in raw_paths if stored_hashes.get(p.name) != current_hashes[p.name]]
    new_sum = [p for p in summary_paths if stored_hashes.get(p.name) != current_hashes[p.name]]
    skipped = len(raw_paths) + len(summary_paths) - len(new_raw) - len(new_sum)

    if not new_raw and not new_sum:
        return VectorStoreIndex.from_vector_store(vector_store), 0, skipped

    all_docs = []
    for p in new_raw:
        docs = SimpleDirectoryReader(input_files=[str(p)]).load_data()
        for doc in docs:
            doc.metadata["source_type"] = "verbatim"
        all_docs.extend(docs)
    for p in new_sum:
        docs = SimpleDirectoryReader(input_files=[str(p)]).load_data()
        for doc in docs:
            doc.metadata["source_type"] = "summary"
        all_docs.extend(docs)

    if collection.count() > 0:
        # Append to existing index
        index = VectorStoreIndex.from_vector_store(vector_store)
        for doc in all_docs:
            index.insert(doc)
    else:
        index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_ctx)

    # Persist updated hashes
    stored_hashes.update(current_hashes)
    collection.modify(metadata={**stored_meta, _HASH_META_KEY: json.dumps(stored_hashes)})

    return index, len(all_docs), skipped


def _build_bm25(user_id: str) -> tuple[dict, BM25Okapi]:
    """Fetch all chunks from ChromaDB and build an in-memory BM25 index."""
    _ensure_embed()
    chroma_client = chromadb.PersistentClient(path=str(_CHROMA_PATH), settings=chromadb.Settings(anonymized_telemetry=False))
    safe_id = user_id.replace("*", "X")
    collection = chroma_client.get_or_create_collection(f"user_{safe_id}")
    docs = collection.get(include=["documents", "metadatas"])

    texts = docs.get("documents") or []
    if not texts:
        return docs, BM25Okapi([[""]])

    tokenized = [_tokenize(t) for t in texts]
    return docs, BM25Okapi(tokenized)


def _rrf_merge(
    sem_hits: list[tuple[str, float]],   # [(doc_id, sem_score), ...]
    bm25_hits: list[tuple[str, float]],  # [(doc_id, bm25_score), ...]
    top_k: int,
    k: int = 60,
) -> list[tuple[str, float, float]]:
    """Reciprocal Rank Fusion. Returns [(doc_id, rrf_score, sem_score), ...]."""
    rrf: dict[str, float] = {}
    sem_score_map: dict[str, float] = {doc_id: score for doc_id, score in sem_hits}

    for rank, (doc_id, _) in enumerate(sem_hits):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_hits):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
    return [(doc_id, rrf_score, sem_score_map.get(doc_id, 0.0)) for doc_id, rrf_score in ranked]


class GlaskUser:
    def __init__(
        self,
        user_id: str,
        user_type: str,
        raw_transcript_paths: list[Path],
        summary_paths: list[Path],
        survey_summary: str,
        knowledge_map: KnowledgeMap,
    ):
        self.user_id = user_id
        self.user_type = user_type
        self.survey_summary = survey_summary
        self.knowledge_map = knowledge_map
        self._index, self.new_docs, self.skipped_docs = _get_or_create_index(
            user_id, raw_transcript_paths, summary_paths
        )
        self._bm25_docs, self._bm25 = _build_bm25(user_id)
        self._profile = profile_module.load_profile(user_id)

    def retrieve_nodes(self, question: str, top_k: int = 8) -> list[dict]:
        """
        Hybrid retrieval: semantic (ChromaDB) + BM25, merged via RRF.
        Returns list of dicts with: text, source, source_type, rrf_score, sem_score.
        """
        fetch_k = top_k * 2  # over-fetch before RRF pruning

        # — Semantic retrieval —
        sem_nodes = self._index.as_retriever(similarity_top_k=fetch_k).retrieve(question)
        sem_hits = [
            (node.node.id_, float(node.score) if node.score is not None else 0.0)
            for node in sem_nodes
        ]
        sem_node_map = {node.node.id_: node.node for node in sem_nodes}

        # — BM25 retrieval —
        texts = self._bm25_docs.get("documents") or []
        ids = self._bm25_docs.get("ids") or []
        metas = self._bm25_docs.get("metadatas") or [{}] * len(texts)

        bm25_hits: list[tuple[str, float]] = []
        if texts:
            scores = self._bm25.get_scores(_tokenize(question))
            top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:fetch_k]
            bm25_hits = [(ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

        # — RRF merge —
        merged = _rrf_merge(sem_hits, bm25_hits, top_k=top_k)

        results = []
        for doc_id, rrf_score, sem_score in merged:
            if doc_id in sem_node_map:
                node = sem_node_map[doc_id]
                text = node.text
                meta = node.metadata
            else:
                # BM25-only hit: look up in raw docs
                try:
                    idx = ids.index(doc_id)
                    text = texts[idx]
                    meta = metas[idx] if idx < len(metas) else {}
                except ValueError:
                    continue

            results.append({
                "text": text,
                "source": meta.get("file_name", "逐字稿"),
                "source_type": meta.get("source_type", "verbatim"),
                "rrf_score": round(rrf_score, 6),
                "sem_score": round(sem_score, 4),
            })

        return results

    def _retrieve_context(self, question: str, top_k: int = 5, nodes: list[dict] | None = None) -> str:
        if nodes is None:
            nodes = self.retrieve_nodes(question, top_k)
        if not nodes:
            return "（访谈记录中未找到直接相关内容）"
        chunks = []
        for i, node in enumerate(nodes, 1):
            src = node["source"]
            type_label = "逐字稿" if node["source_type"] == "verbatim" else "总结稿"
            chunks.append(f"[片段{i} · {src} · {type_label}]\n{node['text']}")
        return "\n\n".join(chunks)

    def _corpus_quality_note(self, nodes: list[dict]) -> str:
        """Generate a self-awareness note about retrieval quality for this question."""
        if not nodes:
            return (
                "⚠️ 【系统提示：语料覆盖 = 无】"
                "访谈记录中未找到任何相关片段。"
                "根据回答规则，必须拒绝推断，直接告知：这个我没有谈到过，真的不知道。"
            )
        top_sem = nodes[0]["sem_score"]
        count = len(nodes)
        if top_sem >= 0.60 and count >= 3:
            return f"【系统提示：语料覆盖 = 充足】检索到 {count} 个高度相关片段，可以直接引用。"
        if top_sem >= 0.40 or count >= 2:
            return (
                f"【系统提示：语料覆盖 = 有限】检索到 {count} 个相关片段，"
                "相关度中等。优先引用片段原话，不足部分可用心理模型补充，标 [框架推断]。"
            )
        return (
            f"⚠️ 【系统提示：语料覆盖 = 极弱】检索到 {count} 个片段但语义相关性很低（分数 < 0.40）。"
            "根据回答规则，不得强行推断。如无法从心理模型直接推导，"
            "须如实告知：这个我真的没怎么聊过，不好说。"
        )

    def prepare_prompt(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> tuple[str, str]:
        """Returns (system_prompt, user_message) without calling the API.

        history: list of {"role": "user"/"assistant", "content": "..."} for multi-turn context.
        Only the last 5 turns are included to keep prompt size bounded.
        """
        nodes = self.retrieve_nodes(question, top_k=5)
        context = self._retrieve_context(question, nodes=nodes)
        quality_note = self._corpus_quality_note(nodes)

        profile_block = (
            self._profile.to_prompt_block()
            if self._profile
            else _NO_PROFILE_NOTICE
        )

        system = SYSTEM_TEMPLATE.format(
            user_id=self.user_id,
            user_type=self.user_type,
            profile_block=profile_block,
            knowledge_block=self.knowledge_map.to_prompt_block(),
            survey_summary=self.survey_summary,
        )

        history_block = ""
        if history:
            recent = history[-10:]  # keep last 5 Q&A pairs (10 messages)
            lines = []
            for msg in recent:
                role_label = "研究员" if msg["role"] == "user" else "分身"
                lines.append(f"{role_label}：{msg['content']}")
            history_block = (
                "\n\n【本轮对话历史（供参考，不作为主要检索依据）】\n"
                + "\n".join(lines)
            )

        user_message = (
            f"以下是来自你的访谈记录中检索到的相关片段（直接证据，优先参考）：\n\n"
            f"{context}\n\n"
            f"{quality_note}"
            f"{history_block}\n\n"
            f"问题：{question}"
        )
        return system, user_message

    def ask(self, question: str, history: list[dict] | None = None) -> str:
        system, user_message = self.prepare_prompt(question, history=history)
        response = _get_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
