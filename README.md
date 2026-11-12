English | [简体中文](README.zh-CN.md)

# AI Memory Benchmark

A method and toolchain for evaluating AI memory systems. The current scope is **ingestion quality**: given a conversation, what does a system fail to retain, and what does it store incorrectly?

## Contents

- [Background](#background)
- [What is evaluated](#what-is-evaluated)
- [Data and report: 2026-11-12](#data-and-report-2026-11-12)
- [Quick start](#quick-start)
- [Running an evaluation](#running-an-evaluation)
- [Methods](#methods)
- [Datasets and licensing](#datasets-and-licensing)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [License](#license)

## Background

Assistants with long-term memory, including coding agents with memory plugins, select information from conversations and store it for later use. Missing information limits later recall; incorrect memories can affect subsequent answers.

Most existing memory benchmarks evaluate retrieval: whether stored information supports correct answers later. Fewer examine the stored content itself, and those that evaluate ingestion use different definitions of what should be remembered. See [related work](docs/related-work.md).

## What is evaluated

Each **system under test** is a memory product configured using the ordinary installation procedure and documented defaults, with the model settings described below. All systems receive the same conversations. Two primary metrics are reported.


A **memory policy** defines what should be retained. A **memory-worthy unit** is independently reusable information selected from the conversation during **annotation**, the process of labeling that information under the policy. A **claim** is one verifiable fact, preference, decision, or plan in a memory. Reports also include claims per conversation, memories per conversation, and average characters per memory. JSON metrics include numerators, denominators, and bootstrap confidence intervals clustered by `gold.session_id`.

| Metric | Definition |
|---|---|
| **Miss rate** | The proportion of memory-worthy units that cannot be fully recovered from the system's combined memories |
| **False-memory rate** | The proportion of claims in the stored memories that lack conversation support or distort the original meaning |



The design follows three principles:

- **Separate evaluation mechanics from retention criteria.** The [framework](docs/framework.md) defines how to evaluate; replaceable [memory policies](policies/) define what should be remembered. Each evaluation identifies its policy.
- **Use a common interface.** All systems share the same input/output contract, judging procedure, and statistics.
- **Annotate before judging.** Memory-worthy units are fixed before any system output is examined.

Results are in the [2026-11-12 evaluation report](reports/2026-11-12/README.md).

## Data and report: 2026-11-12

| Component | Coverage |
|---|---|
| [Framework](docs/framework.md) v0.6 | Split judging: separate recall and precision checks (Section 6.4) |
| Memory policies [default-v1](policies/default-v1.md) and [halumem-v1](policies/halumem-v1.md) | Project conversations and HaluMem memory points, respectively |
| HaluMem-Medium | The report covers 30 conversations and 251 memory-worthy units |
| Trace Commons | 31 segments; 18 judged for every system |
| Adapters: OpenViking, ai-memory, claude-mem, Basic Memory | Implemented |
| Judge-model selection (the model that assesses exported memories) | Documented in [judge-selection.md](docs/judge-selection.md) |
| [2026-11-12 report](reports/2026-11-12/README.md) | HaluMem-Medium and Trace Commons results |
| Independent human checks of annotation and judging | Not yet performed |

## Quick start

Requirements: Python 3, using only the standard library; tested on Python 3.14.

```sh
git clone <repository-url> && cd <repository-directory>

# Run unit tests with synthetic data, without external models or products
python3 -m unittest

# Generate a report from the bundled synthetic example
python3 -m tools.score_split --data-dir tests/fixtures/evaluation --systems alpha beta --output tmp/demo/report
python3 -m tools.report_split_html tmp/demo/report.json --output tmp/demo/report.html
# Produces tmp/demo/report.md, report.json, and report.html
```

## Running an evaluation

A complete evaluation requires:

- A model endpoint compatible with the OpenAI API for the systems under test.
- Codex CLI or Claude Code CLI for annotation and judging.
- The runtime environment for each memory product.

**1. Configure**

Copy `.env.example` to `.env` and set the endpoint, key, model names, and embedding dimensions. Git ignores `.env`.

**2. Prepare data**

Download the datasets listed in [datasets/README.md](datasets/README.md), verify the pinned revisions and SHA-256 hashes, and convert the data into `transcript.md` files with numbered turns:

```sh
python3 -m tools.prepare.halumem tmp/datasets/halumem/raw/HaluMem-Medium.jsonl
python3 -m tools.prepare.extract <conversation-file> <sample-id> --source claude
python3 -m tools.prepare.segment --data-dir <data-directory>   # Split long conversations
```

**3. Annotate without seeing system outputs**

The annotation model labels memory-worthy units under the selected policy before seeing any system output:

```sh
python3 -m tools.model.annotate --data-dir <data-directory> --annotator codex --model gpt-6.1-sol --reasoning-effort high
```

**4. Run the systems under test**

Each system starts with an empty memory store and exports all memories after ingestion:

```sh
python3 -m tools.run.openviking --data-dir <data-directory>
python3 -m tools.run.ai_memory  --data-dir <data-directory>
python3 -m tools.run.claude_mem --data-dir <data-directory>
python3 -m tools.run.basic_memory --data-dir <data-directory>
```

**5. Judge**

```sh
python3 -m tools.model.judge_split --data-dir <data-directory> --annotator codex --model gpt-6.1-sol --reasoning-effort low
```

**6. Score and render**

```sh
python3 -m tools.score_split --data-dir <data-directory> --output <output-prefix>
python3 -m tools.report_split_html <output-prefix>.json --output <output-prefix>.html
```

Batch jobs can resume: annotation skips existing gold files; judging reuses valid results whose input hashes match; product runs skip existing results unless `--force` is supplied. Progress logs are written to `tmp/logs/`. Dispatch stops after 3 consecutive failures by default, or 10 for claude-mem. See [tools/README.md](tools/README.md) for arguments and file formats.

For other conversation sources, see [evaluating your own data](docs/evaluate-your-data.md).

## Methods

**Annotation and judging** prepare the evaluation data; they are not part of the memory product's output:

- **Annotation:** memory-worthy units are labeled before system outputs are seen. Every unit includes conversation evidence. The program checks quotations against the transcript and removes evidence that cannot be matched.
- **Split judging** uses two independent checks performed by a **judge model**, the model evaluating the system's exported memories:
  - Recall judging checks whether each memory-worthy unit can be recovered from all memories jointly. It determines the miss rate.
  - Precision judging splits each memory into claims and checks their support and meaning. It determines the false-memory rate.

  Claim-based counting reduces the effect of different numbers of exported memories. Claim segmentation and excerpt counting still affect the result: verbatim conversation excerpts attached to a memory count as 1 claim in total. See [judgment fields](tools/README.md#判定结果).

| Role | Model | Reasoning effort |
|---|---|---|
| Shared LLM for the systems under test | `deepseek-v4.1-flash-1017`; other generation parameters follow product defaults | — |
| Shared embedding model | `qwen3.7-text-embedding`; vector functionality is disabled for claude-mem because its embedding model cannot be selected | — |
| Annotation | `gpt-6.1-sol` through Codex CLI | high |
| Judging | `gpt-6.1-sol` through Codex CLI | low |

See [judge-model selection](docs/judge-selection.md).

All model requests from the systems under test pass through a local proxy. It validates model names and forwards request fields unchanged. It records the parameters actually sent, usage, requests, and responses. The real API key remains in the proxy process and is not passed to the products.

**Product configuration.** Generation parameters follow product defaults. Systems using external embeddings share the qwen model; claude-mem runs with vector functionality disabled. See [defaults and evaluation configuration](docs/systems.md). Two distinctions affect interpretation:

- Basic Memory does not extract memories itself. An external agent decides what to write, so its result represents “Basic Memory + a fixed agent and prompt” and is identified separately from standalone products.
- Products are not given the evaluation's memory policy. Whether a memory is worth retaining is therefore a reference measure, not a primary metric.

**Sources.** The framework identifies borrowed methods for judging, denominators, blind annotation, sample size, and statistics in [Appendix A](docs/framework.md#附录-a外部引用) and the accompanying source markers.

## Datasets and licensing

| Dataset | Content | License | Published artifacts |
|---|---|---|---|
| [HaluMem-Medium](datasets/halumem.json) | Multi-turn everyday conversations with supplied gold labels; see the [usage guide](docs/halumem.md) | CC-BY-NC-ND-4.0 | Scores and hashes only |
| [Trace Commons](datasets/trace-commons.json) | Coding-agent conversations | CC-BY-4.0 | Processed annotations, system outputs, and judgments with attribution |

- Raw data is not bundled. Download it into the ignored `tmp/datasets/` directory and verify hashes against the manifests.
- Private content belongs in `tmp/`. The tools enforce output-path restrictions for private conversations and reports containing source text.
- Only processed artifacts permitted by their source license and checked for personal information are eligible for publication under `data/` and `reports/`.

## Reproducibility

- **Data:** each dataset has a pinned revision and file hashes.
- **Models:** records include the available model version, parameters, input size, truncation information, prompt version, and hash. Unavailable fields are marked “未记录” (“not recorded”). Prompt files are versioned under `tools/model/prompts/`.
- **Cache:** model results are addressed by content. Reuse requires matching model, parameters, prompt, and input; changes produce a different cache key. Raw model responses are retained for auditing.
- **Intermediate artifacts:** where licensing permits, annotations, system outputs, and judgments are published with the report so others can inspect or rescore them without repeating model calls.

## Repository layout

```
CHANGELOG.md   Notable project changes
docs/          Framework, product configuration, judge selection, HaluMem guide, related work, data workflow
policies/      Memory policies: Markdown descriptions and machine-readable JSON
datasets/      Dataset manifests: sources, revisions, hashes, and licenses
tools/         Standard-library Python implementation; see tools/README.md
  prepare/     Conversion and segmentation
  model/       Model annotation, judging, and versioned prompts
  run/         Product adapters and the shared model proxy
tests/         Unit tests using synthetic data
data/          Processed artifacts eligible for publication
tmp/           Git-ignored private data, dataset downloads, caches, and logs
```

## Roadmap

**Near term**

- Complete the four-system Trace Commons results and their static HTML report.
- Run independent human checks of sampled annotations and judgments, reporting agreement and the thresholds used.

**Later**

- Increase sample coverage and add publicly accessible datasets that do not require an access application.
- Add systems such as Graphiti / Zep and LangMem.
- Improve the interface for independent human checks.

## License

- Project-owned code in `tools/` and `tests/`: [Apache-2.0](LICENSE); see also [NOTICE](NOTICE). Third-party source material retains its upstream license.
- Documentation, the evaluation framework, memory policies, and reports: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
- Published artifacts under `data/`: CC-BY-4.0 with source attribution. Trace Commons derivatives credit its contributors. The HaluMem manifest permits publication of scores and hashes only.
