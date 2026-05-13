# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本文件为 Claude Code 在此代码库工作时提供指引。

## 项目概述

GlaskUser——一个 AI 驱动的质性研究工具，将访谈录音、文字稿、问卷、使用日志构建为可对话的 AI 分身。研究员可反复提问；分身的回答受该用户真实认知边界约束，不会给出产品文档级标准答案。

分身推理依赖两层结构：**语料检索（RAG）** 提供直接访谈证据，**心理模型（UserProfile）** 在无直接证据时提供推断框架。

## 环境配置

```bash
pip install -r requirements.txt   # Python 3.10 / 3.11
```

**只需要 `ANTHROPIC_API_KEY`**，Claude Code 运行时自动注入，无需 `.env` 文件。  
向量嵌入使用本地 HuggingFace 模型（`BAAI/bge-small-zh-v1.5`），不需要 OpenAI API key。

## Skill 命令（主要使用方式）

| 命令 | 说明 |
|------|------|
| `/glaskuser_init` | 一键安装所有依赖：pip 包、ffmpeg（自动安装）、Whisper 模型、embedding 模型 |
| `/glaskuser_build` | 扫描 `data/`，转录音频/视频，构建 ChromaDB 向量库，提取心理模型 |
| `/glaskuser_simulate` | CLI 对话模式：Claude Code 直接扮演分身作答，**不调 Anthropic API** |
| `/glaskuser_search` | 原始语料语义搜索：返回匹配片段 + 置信度评分，**不扮演用户**，研究员视角呈现证据链 |

## 直接运行（调试用）

```bash
# 预览 data/ 文件分类，不触发转录或构建
python src/classify.py

# 构建向量库 + 心理模型
python src/build.py

# 列出已构建的分身（输出 JSON：user_id / docs / has_profile）
python src/list_users.py

# 准备 RAG 上下文（不调 API，输出 JSON）
python src/context.py --user U001 --question "你怎么看待这个功能？"
python src/context.py --user U001 --question "..." --history '[{"role":"user","content":"..."}]'
python src/context.py --user U001 --question "..." --history-file /tmp/history.json

# 语义搜索原始语料，返回匹配片段 + 置信度（不调 API）
python src/search.py --user U001 --question "是否喜欢新上线的AI出题功能" --top-k 8
python src/search.py --user all  --question "..."   # 跨用户聚合搜索

# 心理模型提取（输出 transcript + prompt JSON，由 Claude Code 分析后调用 save_profile.py 保存）
python src/extract_profile.py --user U001 --user-type 家长   # 家长：8 维度
python src/extract_profile.py --user U001 --user-type 用户   # 通用：4 维度（默认）
python src/extract_profile.py --all

# 保存 Claude Code 提取的心理模型 JSON
python src/save_profile.py --user U001 --user-type 家长 --json '{"core_values": {...}, ...}' --source-chars 12345

# 直接调用 Anthropic API 提问（非 skill 路径，用于独立测试）
python src/query.py --user U001 --question "你怎么看待这个功能？"
```

## 数据格式

所有文件平铺放入 `data/`，无需子目录。**文件名必须以用户 ID 开头**（字母数字均可，如 `U001`、`1001`）：

| 扩展名 | 识别为 | 说明 |
|--------|--------|------|
| `.txt` `.pdf` `.md` `.docx` | 逐字稿 或 总结稿 | 内容自动判断：对话/口语体→逐字稿；分析/第三人称→总结稿（.docx 需 docx2txt） |
| `.mp3` `.wav` `.m4a` `.aac` `.ogg` `.flac` `.webm` | 音频→自动转录 | 需要 ffmpeg；转录结果缓存于 `.cache/transcripts/` |
| `.mp4` `.mov` `.avi` `.mkv` | 视频→自动转录 | 同上 |
| `.csv` `.xlsx` 含 `question`/`answer` 列 | 问卷 | 合并进 persona 提示词 |
| `.csv` `.xlsx` 含 `user_id`/`feature`/`use_count` 列 | 使用日志 | 驱动功能边界逻辑 |

同一用户 ID 的多份文件自动合并。**用户 ID 提取规则：** `re.match(r"[A-Za-z0-9*]+", filename_stem)`，支持脱敏格式如 `135****3824`。ChromaDB 集合名内部将 `*` 替换为 `X`（`twin.py`）；`list_users.py` 在输出时通过扫描 `data/` 目录映射还原原始 `****` 格式，展示侧不受影响。

**用户类型配置（可选）：** 在 `data/user_types.json` 中声明每位用户的类型，`build.py` 和 `extract_profile.py` 据此选择对应的提取维度：

```json
{"U001": "家长", "U002": "学生", "U003": "用户"}
```

代码仅对 `家长` 有专项处理（8 维度 `PARENT_EXTRACT_PROMPT`）；`学生`、`老师`、`用户` 及其他所有类型均走相同的 4 维度通用 `EXTRACT_PROMPT`，行为完全一致。`/glaskuser_build` 向导会在首次构建时通过 AskUserQuestion 创建此文件。

**文字文件分类（`ingester._classify_text_content`）：** 读取文件前 800 字，按口语词汇、说话人标记、第一人称密度（逐字稿信号）与分析性词汇、第三人称用户描述（总结稿信号）打分，默认保守归为逐字稿。

## 架构

```
src/
├── ingester.py          # 扫描 data/，按用户 ID 分组文件，分类类型 → UserBundle
│                        #   新增：video 支持；文字内容自动归 verbatim/summary；dry_run 模式
├── transcriber.py       # 本地 Whisper small 转录，结果缓存到 .cache/；支持音频 + 视频
├── knowledge_map.py     # KnowledgeMap，按使用日志将功能分级（heavy/light/abandoned/never_used）
├── profile.py           # UserProfile 数据模型（16 字段）+ 持久化；get_extract_prompt(user_type) 分发
│                        #   家长：PARENT_EXTRACT_PROMPT（8 维度）；其他：EXTRACT_PROMPT（4 维度）
├── loader.py            # 编排 ingester → KnowledgeMap → GlaskUser；支持 user_type_map
├── twin.py              # GlaskUser 类，核心混合检索（语义 + BM25/RRF）+ system prompt 构建
│                        #   profile 注入、多轮 history、置信层级 [直接证据]/[框架推断]/[弱推断]
├── context.py           # RAG 上下文准备，不调 API，输出 JSON；支持 --history / --history-file
├── search.py            # 混合语料语义搜索，返回片段 + 置信度 JSON（/glaskuser_search 入口）
├── list_users.py        # 读取 ChromaDB 列出已构建分身，输出 JSON
├── classify.py          # 预览 data/ 文件分类，不触发转录（dry_run 模式）
├── extract_profile.py   # 读取逐字稿，输出含 extract_prompt 的 JSON 供 Claude Code 分析
│                        #   字段：user_id / user_type / source_chars / skip / extract_prompt（transcript 内嵌于 prompt）
├── save_profile.py      # 校验并持久化 Claude Code 提取的 UserProfile JSON 到 .profiles/
├── build.py             # /glaskuser_build 入口（4 步：扫描→转录报告→向量库→心理模型状态）
├── init.py              # /glaskuser_init 入口（下载模型）
└── query.py             # CLI 独立入口，直接调 Anthropic API（非 skill 主路径）
```

**数据流：**  
`ingester.py` 扫描 `data/` → 音频/视频经 `transcriber.py` 转录 → 文字文件按内容分为逐字稿/总结稿 → 分组为 `UserBundle` → `loader.py` 构建 `KnowledgeMap` + `GlaskUser` → ChromaDB 向量检索 + BM25 via RRF 混合排序 → `profile.py` 加载心理模型 → Claude Code 生成受约束的分身回答

**心理模型提取流程（skill 路径）：**  
`extract_profile.py` 读取逐字稿输出 prompt JSON → Claude Code 分析 → `save_profile.py` 校验并保存 `.profiles/{user_id}.json` → 下次 simulate 时 `twin.py` 自动加载注入 system prompt

**三种推理模式：**
- **Skill 模拟模式（simulate）：** `context.py` 做混合检索输出 JSON，Claude Code 扮演分身作答，全程不调 Anthropic API
- **Skill 搜索模式（search）：** `search.py` 做语义检索输出原始片段 + 置信度，Claude Code 以研究员视角分析证据链，不扮演用户
- **直接调用模式：** `query.py` 经由 `GlaskUser.ask()` 直接调 Anthropic API，用于独立测试

**置信层级体系（system prompt 约束）：**

最高优先级：语料覆盖为"无"或"极弱"时，分身必须拒绝作答，不得用心理模型填充空白。
- `[直接证据]`：访谈中明确说过，接近原话
- `[框架推断]`：无直接证据，但可从 UserProfile 的 core_values / inference_rules 直接推导；措辞须体现不确定性
- `[弱推断]`：逻辑有关联但把握不足（<50%），须显式说明，不得给出具体数字或操作步骤

推理链断裂、模型维度无法覆盖时直接拒绝，不升级为 [弱推断]。

**群体查询（`query.py --user all`）：** 聚合分析不汇总表面观点，而是提取跨用户、跨产品品类（图书/学习机/辅导班等）一致的底层决策驱动因子，区分"说法不同但底层逻辑相同"与"真正的价值观分歧"，每个结论附用户原话来源。

**功能边界（`knowledge_map.py`）：** 按使用次数分为 `heavy`（>3 次）、`light`（1–2 次）、`abandoned`（曾用后放弃）、`never_used`，注入 system prompt 中的认知约束规则。"从未触发"的功能分身不可推断。

## 心理模型（UserProfile）

存储于 `.profiles/{safe_user_id}.json`，共 16 字段（5 元数据字段：`user_id` / `user_type` / `version` / `generated_at` / `source_chars`；8 内容维度；3 列表：`pain_points` / `aspirations` / `inference_rules`）。所有用户均包含 4 个通用维度，家长用户额外包含 4 个专项维度（置信度 0–1）：

**通用维度（所有用户类型）**

| 维度 | 说明 |
|------|------|
| `core_values` | 最核心、最稳定的价值观（学习观/育儿观/效率观等） |
| `decision_framework` | 评估产品/功能时的主要判断标准和逻辑链 |
| `tech_attitude` | 对新技术、AI 功能的整体接受度 |
| `economic_profile` | 消费倾向、价格敏感度、品牌比较行为 |

**家长专项维度（`user_type == 家长` 时必须提取）**

| 维度 | 说明 |
|------|------|
| `educational_philosophy` | 教育理念与方法论（鸡娃/素质/放养倾向；投入产出预期） |
| `child_context` | 孩子学业现状与亲子关系（成绩/学习习惯/家长参与/互动模式） |
| `social_profile` | 社会阶层感知（阶层意识/比较行为/焦虑来源/身份认同对消费的影响） |
| `brand_attitude` | 品牌认知逻辑（口碑/权威背书/价格信号；广告信任度；参照系来源） |

以及 `pain_points`、`aspirations`（列表），和 `inference_rules`（3–6 条跨品类推断规则，当无直接语料时推导该用户对新场景的可能反应）。

每条维度附 `key_quotes`：必须是逐字稿中 10–40 字的原话片段。家长专项维度字段类型为 `Optional`，非家长用户默认 `null`，向后兼容。

## 模型离线分发

| 模型 | 文件路径 | 大小 | 分发方式 |
|------|----------|------|----------|
| Whisper small | `models/small.pt` | ~461MB | 随 zip 打包；缺失时自动下载到 `~/.cache/whisper/` |
| BGE embedding | `models/bge-small-zh-v1.5/` | ~92MB | 随 zip 打包；缺失时由 `/glaskuser_init` 从 HuggingFace 镜像下载并保存 |

`transcriber.py` 和 `twin.py` 均优先从 `models/` 路径加载，本地文件存在时完全离线运行，无需网络访问模型服务。

## 关键设计约束

- ChromaDB 持久化在 `.chroma/`；集合非空时跳过重建，增量索引仅处理变更文件（sha256 hash 比对）
- 心理模型持久化在 `.profiles/`；`source_chars` 不变时跳过重建，重复运行安全
- `data/`、`.cache/`、`.chroma/`、`.profiles/` 均在 `.gitignore` 中，不提交用户数据
- `Settings.embed_model` 在 `twin.py` 模块顶层设置，影响整个 llama-index 进程
- `classify.py` 使用 `dry_run=True` 调用 `ingester.ingest()`，不触发 Whisper 转录

## 已知行为说明（测试验证）

**build.py 输出术语**

`[3/4]` 进度行中 "已有 X 段" 的 X 是**源文件数**（每份音频或文字文件计 1），不是 ChromaDB 切片数。真正的切片数通过 `list_users.py` 的 `docs` 字段获取（实测：1 份访谈录音 → 6–37 个切片）。

`[1/4]` 输出的 "X 份文字资料" 在纯音频数据集中，指的是 `.cache/transcripts/` 下已有的缓存转录文件数（不是 `data/` 中的原始文字文件）。

**依赖兼容性**

chromadb 0.6.x 与 posthog ≥7.0 存在 API 不兼容（`capture()` 参数签名变更），导致每次创建 PersistentClient 时产生 `Failed to send telemetry event` stderr 噪音。已通过 monkey-patch `twin.py` 中的 `_ChromaPosthog._direct_capture` 修复。

jieba 0.42.x 在 initialize 时会向 stderr 直接写日志（StreamHandler 绕过 sys.stderr 重定向）。已通过在 import jieba 后清除其 handlers 并替换为 NullHandler 修复，同时屏蔽 jieba._compat 的 `pkg_resources` UserWarning。

**已验证正常工作的链路**（Python 3.11, chromadb 0.6.3, llama-index-core 0.12.10）
- `classify.py`：干跑扫描，输出用户/文件分类，支持已缓存转录标注
- `build.py`：向量库构建（增量）+ 心理模型状态报告，输出 `PROFILE_PENDING` 标记供 skill 解析
- `list_users.py`：读取 ChromaDB 输出 JSON，自动扫描 `data/` 映射还原原始 user_id（含 `*`）
- `context.py`：混合检索（语义 + BM25/RRF）输出 system prompt + user_message JSON，不调 API
- `search.py`：原始语料语义搜索，返回 rrf_score / sem_score / confidence_level
- `extract_profile.py`：输出含逐字稿的 extract_prompt JSON，供 Claude Code 分析（无单独 transcript 字段）
- `init.py`：依赖安装 + 模型加载，已有文件自动跳过
