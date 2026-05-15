# GlaskUser — Turn Historical User Research into an Always-Available AI User Panel

[中文版](README.zh-CN.md)

The value of historical user interview data is often vastly underutilized: archived once complete, hard to reuse. To continuously mine user insights, you either need sustained budget or must rely on partial, speculative inference. GlaskUser addresses these scenarios:

- Lacking a user research budget, but wanting to extract actionable insights from existing recordings or transcripts
- Having new questions for the same users, but can no longer contact them
- Accumulating a large volume of interview material, yet struggling to locate a specific opinion
- Wanting to distill group consensus across multiple interviews, but unsure if AI summaries are trustworthy

**Core approach: transform raw interview records into queryable AI personas using empirical user research frameworks.** Every persona response is anchored to that user's actual corpus. Psychological models are extracted from interviews, encompassing core values, decision frameworks, and transferable inference rules. When no direct evidence exists, the persona explicitly says "We didn't discuss this — I can't say" — refusing to fabricate.

---

## Comparison with Existing Solutions

|           | GlaskUser                                         | AI user research platforms (atypica.ai, etc.) | Prompting LLMs directly    | Enterprise IM + Knowledge Base |
| --------- | ------------------------------------------------- | --------------------------------------------- | -------------------------- | ------------------------------ |
| **Source**  | Your own interview recordings / transcripts         | Public web data or synthetic profiles           | Model training data (general) | Internal documents              |
| **Users**   | Real users you actually interviewed                  | AI-constructed average users or uploaded profiles | Imaginary                    | Internal docs; real if from interviews |
| **Evidence** | Cites original interview excerpts, confidence-labeled; derived answers based on user psychological model + corpus reasoning | Untraceable                                    | Untraceable                 | Returns document snippets        |
| **Uncertainty** | Explicitly says "We didn't discuss this, I don't know" | Fluent fabrication with obvious "curse of knowledge" | Fluent fabrication with obvious "curse of knowledge" | No results or low-relevance snippets |
| **Follow-up** | Supported, like conducting another interview         | Supported, unbounded                           | Supported, unbounded         | Not supported                    |
| **Cross-user analysis** | Extracts consistent underlying decision drivers across users | Supported                                       | Supported, no evidence       | Not supported                    |
| **Data Privacy** | Voice transcription and semantic search run fully offline; corpus is passed to Anthropic API via Claude Code for questioning | Interview materials uploaded to third-party platforms, risk of leakage | Uploaded to model provider   | Internal network                |

---

## Use Cases

- **Distill group insights**: Find shared decision logic and underlying drivers across users
- **Validate product decisions**: After drafting new features, query user personas to quickly collect feedback rooted in historical user base, decision frameworks, and product similarity
- **Compare individual differences**: The same question, how do multiple user personas each respond

---

## Prerequisites

- [Claude Code] installed, with Anthropic API Key configured
- Interview recordings or transcripts for at least one user; the more the better — broader coverage yields more accurate answers

---

## First-Time Setup

### Step 1: Unzip & Add Interview Materials

Unzip `glaskuser_dist.zip` to any location, then place interview files into the `data/` folder. **Filenames must start with a user ID**:

```
data/
├── U001_interview.mp3
├── U001_summary.docx          ← Multiple files for the same user are auto-merged
├── U002_interview.m4a
├── U003_transcript.txt
└── ...
```

User IDs can be any alphanumeric combination; anonymized formats (e.g., `135****3824`) are also supported.

Supported formats:

| Type                                    | Formats                                              |
| --------------------------------------- | ---------------------------------------------------- |
| Interview audio                         | `.mp3` `.wav` `.m4a` `.aac` `.ogg` `.flac` `.webm` |
| Interview video                         | `.mp4` `.mov` `.avi` `.mkv`                         |
| Transcripts (verbatim or summary)       | `.txt` `.pdf` `.md` `.docx`                         |
| Surveys (with question/answer columns)  | `.csv` `.xlsx` `.xls`                               |
| Usage logs (with user_id/feature/use_count columns) | `.csv` `.xlsx` `.xls`                               |

---

### Step 2: Open the Folder with Claude Code

Navigate to the folder in your terminal and launch Claude Code; or open Claude Code first, then drag the folder in.

---

### Step 3: Run the Initialization Command

```
/glaskuser_init
```

Claude will guide you through the entire process — no manual operations needed:

**Phase 1: Environment Setup** (automatic)

1. Check Python version (3.10+ required)
2. Install all pip dependencies
3. Auto-install ffmpeg (macOS via brew, Windows via winget or choco, Linux via apt)
4. Load the Whisper speech recognition model (~461MB)
5. Load the semantic search model bge-small-zh-v1.5 (~92MB)

If both models are included in the zip package, they load directly; otherwise they are auto-downloaded from a mirror.

**Phase 2: Interactive Wizard** (a few confirmations needed)

6. Ask about your data types, display corresponding naming conventions
7. Wait for you to place files in `data/`, then scan and preview classification results
8. Build the vector knowledge base (audio auto-transcribed, text auto-indexed)
9. If new users are detected without psychological models: ask for user type, auto-extract psychological models
10. Guide you into search or persona conversation mode

**About transcription speed**: First-time audio processing is relatively slow (~30 seconds per minute of recording). Transcription results are cached — subsequent runs complete almost instantly.

---

## Ongoing Usage

### When the Knowledge Base Is Unchanged, Ask Directly

```
/glaskuser_simulate   ← Talk to a single user persona (first-person, supports multi-turn follow-up)
/glaskuser_search     ← Search raw interview excerpts (researcher perspective, view original evidence directly)
```

### After Adding New Interview Materials

Place new files into `data/`, then run:

```
/glaskuser_build
```

The system only processes new or changed files; already-transcribed and already-indexed content is auto-skipped, typically completing within seconds. Afterward, use `/glaskuser_simulate` or `/glaskuser_search` to ask questions as usual.

---

## Optional Configuration: User Types

`/glaskuser_build` (and the build step of `/glaskuser_init`) will ask for user type and auto-create `data/user_types.json` when new users are detected without psychological models. You can also manually edit this file:

```json
{"U001": "parent", "U002": "student", "U003": "user"}
```

User type determines the extraction dimensions of the psychological model:

| Type                          | Dimensions | Specialized Dimensions                                       |
| ----------------------------- | ---------- | ------------------------------------------------------------ |
| parent                        | 8          | Educational philosophy, parent-child relationship, social class perception, brand cognition (+ 4 general dimensions) |
| student / teacher / user / others | 4          | Core values, decision framework, tech attitude, economic profile |

If users in the same batch have different types, modify the JSON per entry. When the file does not exist, all users default to "user" (4 dimensions).

---

## Notes

- Voice transcription and semantic search run fully offline; corpus is passed to Anthropic API via Claude Code for questioning, consuming API call quota
- The richer the user corpus, the more accurate persona responses — operate on sufficient interview material whenever possible
- Building personas consumes significant tokens — verify your model choice and quota availability
- Surveys and usage logs are optional; personas can be built with only recordings or transcripts
