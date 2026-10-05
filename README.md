# Hexagon CPY stale-cache reproduction evidence

Evidence for the single-file fix in [d2f5295](https://github.com/flyingdoog/tandem/commit/d2f5295213fb4b094bb5e06c01b1296896df068d).

[Download the review archive](https://github.com/flyingdoog/tandem/raw/refs/heads/codex/hexagon-cpy-stale-cache-evidence/hexagon-cpy-stale-cache-review.zip) and read `REVIEW-NOTES.md`, `REPRODUCE.md`, and `VALIDATION.md`.

The archive includes the proposed patch, the diagnostic per-op full-flush patch, portable reproduction scripts, MIT-licensed test text, public Qwen3.5-4B per-run metrics, and fresh build and CPY test records. The prior PPL/KLD runs use `2ca15f54`; fresh v73/v75/v79/v81 builds and 138/138 v81 CPY tests use `a3a1c47` plus the proposed patch. Matching displayed metrics do not establish bitwise raw-logit equality.

The source PR contains only `cpy-ops.c`. This evidence branch has no model weights, binaries, private-model records, credentials, device serials or network addresses.

Archive SHA-256: `5189bb5cfa426308deb59250c324feadfc1ac185436b64e55d3f5e5dfb86a4df`.

AI assistance was used for review, testing and evidence preparation at the contributor's request.
