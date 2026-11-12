# Changelog

This file records notable project changes using the [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) format.

## Unreleased

### Added

- A memory-ingestion evaluation framework with policy-defined memory-worthy units, negatives (examples that should not be retained), split judging, and clustered bootstrap intervals.
- The `default-v1` and `halumem-v1` memory policies, dataset manifests with pinned revisions and hashes, and a [HaluMem usage guide](docs/halumem.md).
- Data conversion, blind annotation, four product adapters, a shared model proxy, scoring, HTML reports, independent human-check forms, judge-model comparison, and publication tools.
- Documentation of [product configuration](docs/systems.md) and [judge-model selection](docs/judge-selection.md).
- A [2026-11-12 report](reports/2026-11-12/README.md) covering HaluMem-Medium; Trace Commons results remain pending.
- English primary versions and Chinese counterparts for the repository README and the 2026-11-12 report README.

### Changed

- Precision judging counts atomic claims. The primary false-memory rate uses unsupported or distorted claims; the whole-memory rate remains an auxiliary metric.
- The model proxy preserves request fields and records the parameters actually sent by each product.
- Report headers summarize each system's model, prompt source, and total call counts; detailed identities and prompt hashes remain in JSON.
- Documentation distinguishes HaluMem's 30 selected conversations from its 6 source-user bootstrap clusters and states the configuration exception for claude-mem embeddings.

### Removed

- The retired item-to-unit mapping judge and report route, baseline adapters, and interactive annotation/judging pages.
