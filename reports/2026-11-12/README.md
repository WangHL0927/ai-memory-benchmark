English | [简体中文](README.zh-CN.md)

# Memory Ingestion Quality Report (2026-11-12)

This report evaluates memory ingestion. A **system under test** is a product that receives conversations and exports memories. A **memory policy** defines what should be retained; a **memory-worthy unit** is independently usable information selected under that policy. A **claim** is one verifiable fact, preference, decision, or plan in a memory. **Annotation** identifies memory-worthy units before a **judge model** applies **split judging**: separate checks of recall coverage and claim accuracy.

| Item | Setting |
|---|---|
| Date | 2026-11-12 |
| Scope | Memory ingestion from individual conversations: misses and false memories |
| Systems under test | OpenViking 0.5.4.1, ai-memory 3.4.0, claude-mem 15.17.0, Basic Memory 1.0.0 + a fixed agent |
| Datasets | HaluMem-Medium: 30 conversations from 6 source users; Trace Commons: 31 segments, 18 judged for every system |
| Judge model | `gpt-6.1-sol`, reasoning effort low; no human calibration performed |
| Framework | [Evaluation framework](../../docs/framework.md) v0.6, split judging (Section 6.4) |

## Summary

The four systems use the same LLM and retain product-default generation parameters.

- **HaluMem-Medium** (30 conversations): miss rates range from 31% to 49%; only 4%–11% of memory-worthy units are completely uncovered, the rest are partially covered. Claim-based false-memory rates range from 12% to 21%.
- **Trace Commons** (18 coding-agent segments judged for every system): miss rates range from 6% (OpenViking) to 52% (ai-memory); false-memory rates range from 7% to 16%.
- OpenViking's low miss rate on Trace Commons comes from the verbatim chat logs kept in its memories, which make up 96% of its memory text. Judged on its summaries alone, its miss rate is 60%.
- claude-mem writes the most: about 700 claims per Trace Commons segment, with input tokens about 17 times the transcript length. Its miss rate there is 16%.

## 1. Evaluation setup

### 1.1 Data

| Dataset | Content | Pinned revision | Source of memory-worthy units |
|---|---|---|---|
| [HaluMem-Medium](../../datasets/halumem.json) | Multi-turn everyday conversations | `cb04336` | Supplied memory points converted under [halumem-v1](../../policies/halumem-v1.md) |
| [Trace Commons](../../datasets/trace-commons.json) | Coding-agent conversations | `112ebd4` | Blind model annotation with `gpt-6.1-sol`, reasoning effort high, under [default-v1](../../policies/default-v1.md) |

HaluMem-Medium uses the first 30 conversations in sample-ID order, from 6 source users and containing 251 memory-worthy units. Trace Commons has 31 segments after splitting long conversations.

### 1.2 Systems under test

| System | Configuration | Details |
|---|---|---|
| OpenViking 0.5.4.1 | Official wizard configuration for a custom OpenAI-compatible endpoint | Temperature 0.0; reproduces the product's disabled-thinking behavior when connecting directly to Bailian |
| ai-memory 3.4.0 | Single-page consolidation with default generation parameters | Temperature 0.2; assistant replies are not captured by default |
| claude-mem 15.17.0 | `openai-compatible` with default generation parameters | Temperature 0.3; maximum output 4096 tokens; vector functionality disabled |
| Basic Memory 1.0.0 + a fixed agent | Minimal agent with the official `memory-capture` system prompt | Basic Memory does not perform extraction; the result represents the product-agent combination |

All systems use `deepseek-v4.1-flash-1017`. OpenViking, ai-memory, and Basic Memory use `qwen3.7-text-embedding` with 1024 dimensions. Vector functionality is disabled for claude-mem, which does not use that embedding model. The parameters actually sent on each call are retained in run records. See [product configuration](../../docs/systems.md) for the complete settings and source references.

Each conversation starts with an empty memory store. All stored memories are exported after ingestion.

### 1.3 Metrics

- **Miss rate:** the proportion of memory-worthy units that cannot be fully recovered from all exported memories jointly. “Completely uncovered” includes none and distorted; partial coverage is counted separately.
- **False-memory rate:** unsupported or distorted claims divided by all claims. Partially supported claims do not enter the numerator, and each claim belongs to only one category. Attached verbatim conversation excerpts count as 1 claim in total.

Intervals are 95% bootstrap intervals from 10,000 resamples with seed 20261112. The clustering key is `gold.session_id`. The HaluMem converter assigns the same identifier to sessions from the same source user, so resampling uses **6 source-user clusters**, not 30 independent conversations.

## 2. Results

### 2.1 HaluMem-Medium

#### Products, listed by name

| System | Miss rate (95% interval) | Completely uncovered | False-memory rate (95% interval) | Claims per conversation | Memories per conversation |
|---|---|---|---|---|---|
| ai-memory | 49% (39%–61%) | 9% | 19% (18%–20%) | 33 | 1.0 |
| claude-mem | 47% (37%–61%) | 11% | 15% (12%–19%) | 179 | 12.1 |
| OpenViking | 41% (33%–47%) | 9% | 12% (8%–16%) | 30 | 7.5 |

#### Fixed-agent combination

| System | Miss rate (95% interval) | Completely uncovered | False-memory rate (95% interval) | Claims per conversation | Memories per conversation |
|---|---|---|---|---|---|
| Basic Memory + a fixed agent | 31% (24%–37%) | 4% | 21% (19%–24%) | 40 | 1.0 |

Each system has 30 valid samples and 251 memory-worthy units, with no failed samples excluded. Some intervals overlap; no paired significance test was computed, and no overall ranking is given. Basic Memory's result represents the product combined with the fixed agent.

### 2.2 Trace Commons

Results on the 18 segments that were judged for every system (340 memory-worthy units):

| System | Miss rate | Completely uncovered | False-memory rate (claims) | Claims per segment | Memories per segment |
|---|---|---|---|---|---|
| OpenViking | 6% (3%–9%) | 0.6% | 9% (6%–13%) | 85 | 12.0 |
| claude-mem | 16% (11%–22%) | 5% | 16% (13%–18%) | 696 | 64.6 |
| Basic Memory + agent | 48% (41%–53%) | 11% | 9% (6%–11%) | 84 | 1.0 |
| ai-memory | 52% (43%–60%) | 18% | 7% (5%–10%) | 78 | 1.0 |
| OpenViking (summaries only) | 60% (52%–68%) | 19% | 10% (7%–15%) | 71 | 12.0 |

Valid segments per system across all 31: OpenViking 29, ai-memory 30, Basic Memory + agent 30, claude-mem 19. For claude-mem, 4 segments could not be replayed because tool calls and results in the source transcript do not pair up, and 8 were not judged. Segments without a valid result for every system are excluded from the table above so that all rows cover the same conversations.

## 3. Observations

### 3.1 Product behavior and configuration

**OpenViking attaches source text.** OpenViking appends the original chat log to its memory files. On HaluMem-Medium, 114 of 224 memories (51%) contain such excerpts; on Trace Commons, verbatim text makes up 96% of its memory text, and its total memory text is about 0.72 times the length of the transcripts. Judged on summaries alone, its miss rate is 49% (42%–54%) on HaluMem-Medium and 60% (52%–68%) on Trace Commons, compared with 41% and 6% when the attached text is included. This comparison removes attached source text from the same exports and judges the result again; it is not a separate product run. Each conversation in this batch has 1 LLM call.

**OpenViking temperature and thinking settings.** The official custom-endpoint wizard writes a temperature of 0.0. When connecting directly to Bailian, the product sends a domain-specific flag to disable thinking. The local proxy changes the visible domain, so the configuration explicitly supplies `enable_thinking:false` to reproduce direct-connection behavior.

**ai-memory does not capture assistant replies.** Its default is `capture_assistant=false`. It exports 1 page per conversation in this batch. The judgments include errors that describe uncaptured assistant replies as “the conversation had no assistant replies.” This difference between captured input and the actual conversation can affect generated memories.

**claude-mem records observations and summaries.** The adapter submits tool events and conversation-summary events. This batch averages 179 claims per conversation, with a median of 22 LLM calls. The default `reasoning:{enabled:false}` does not disable thinking on the evaluation endpoint; recorded endpoint measurements are described in [systems.md](../../docs/systems.md). The per-call output limit is 4096 tokens. Native skips, drops, and output truncations are recorded in run metadata and must be inspected per sample.

**claude-mem vector configuration.** Its OpenAI embedding option uses the fixed model `text-embedding-ada-002`, which cannot be configured as the required qwen model. This report uses vector functionality disabled; the results do not establish behavior with vectors enabled.

**Basic Memory and the agent.** Basic Memory provides storage and retrieval; the agent decides what to write. This batch exports 1 note per conversation. The combination has the lowest miss-rate point estimate and the highest false-memory-rate point estimate in this sample.

### 3.2 Evaluation definitions

**Most misses are partial coverage.** Completely uncovered units account for 4%–11% of all memory-worthy units. The miss rate also includes partial units whose subject is retained but key conditions or details are missing.

**The counting unit affects the false-memory rate.** The whole-memory measure counts a memory as erroneous when `grounded=no` or `distorted=true`. In this batch, that rate is 90% for ai-memory, 93% for Basic Memory + the fixed agent, 16% for OpenViking, and 70% for claude-mem. Claim-based rates range from 12% to 21%. Claim segmentation and grouping verbatim excerpts still affect comparisons.

**Judging conventions need to match.** Judge-model selection compared different conventions for dates and distortion. Whole-memory false-memory rates vary with these conventions and cannot directly calibrate the claim-based measure. See [judge-model selection](../../docs/judge-selection.md).

## 4. Model usage

Median LLM usage per conversation (Trace Commons: the 18 common segments):

| System | Calls (HaluMem / Trace) | Input tokens (HaluMem / Trace) | Input tokens ÷ transcript characters |
|---|---|---|---|
| OpenViking | 1 / 1 | 9.9K / 47.3K | 0.6 / 0.4 |
| ai-memory | 1 / 1 | 6.3K / 25.2K | 0.4 / 0.2 |
| Basic Memory + agent | 3 / 3 | 37.4K / 161.3K | 2.5 / 1.3 |
| claude-mem | 22 / 58 | 243.3K / 1,946.6K | 14.7 / 16.7 |

Token counts come from the local proxy. One claude-mem Trace segment has no recorded usage and is excluded from its median.

## 5. Limitations

- Judgments have not been calibrated against independent human labels; the reliability of claim segmentation and counting has not been separately validated.
- A shared LLM and external embedding model, and disabled vectors for claude-mem, differ from default installations. Results may differ with each product's default models.
- Each conversation starts with an empty store. The evaluation does not cover cross-conversation updates, deduplication, long-term maintenance, or retrieval quality.
- Memory-worthy units depend on the memory policy, which is not given to the products. Original HaluMem points may also contain multiple facts.
- Selection uses sample-ID order and includes only 30 conversations from 6 source-user clusters; it does not establish performance across the full dataset.

## 6. Reproduction

1. Download data using the [dataset manifests](../../datasets/README.md), verifying revisions and hashes.
2. Follow [Running an evaluation](../../README.md#running-an-evaluation) for conversion, annotation, product runs, judging, and scoring. HaluMem has converted gold labels, so blind annotation can be skipped.
3. Inspect per-sample run and judgment records for models, reasoning effort, prompt hashes, and actual request parameters. Unavailable information remains marked “未记录” (“not recorded”).

Publication of Trace Commons derivatives follows the manifest's CC-BY-4.0 attribution and redaction requirements. HaluMem is marked `scores_only`; only aggregate scores and artifact hashes are published.
