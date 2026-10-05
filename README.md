# Hexagon 64-byte cache-maintenance stride evidence

Evidence for [PR #29977](https://github.com/ggml-org/llama.cpp/pull/29977), source commit [69813de](https://github.com/flyingdoog/tandem/commit/69813deaefbb429750807e9f44cd488bc2879316).

[Download the review archive](https://github.com/flyingdoog/tandem/raw/refs/heads/codex/hexagon-cpy-stale-cache-evidence/hexagon-stride-cache-review.zip) and read `REVIEW-NOTES.md`, `REPRODUCE.md`, and `VALIDATION.md`.

The source PR changes only `hex-utils.h`: `dccleaninva` instruction steps become 64 bytes, with 128-byte range alignment and 512-byte blocks preserved. The previous CPY-specific mitigation is removed.

The archive contains the generic patch, earlier CPY patch and diagnostic variants, corrected reproduction scripts, MIT-licensed test text, public Qwen3.5 per-run metrics and timing records, and fresh build and v81 CPY/CONCAT/SSM_CONV checks. Recorded model comparisons use `2ca15f54`; fresh builds use `a3a1c47` plus the submitted patch. Matching reported metrics do not establish raw-logit bitwise equality. The cache-level and byte-level stale-read mechanism has not been directly captured.

Models, binaries, private-model records, proprietary SDK objects or disassembly, credentials, device identifiers and network addresses are excluded.

Archive SHA-256: `d79ef327fb06b5fdfbf8e5e044c7752c4c83eb8c0f8ea6a1fcfd633e9da87423`.
