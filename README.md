# Hexagon work-queue reproduction artifacts

Evidence for the single-file fix in commit [f4994f8](https://github.com/flyingdoog/llama.cpp/commit/f4994f8be38cbd1cf652d5a15fe600e52ec765e2).

[Download the review archive](https://github.com/flyingdoog/llama.cpp/raw/refs/heads/codex/hexagon-work-queue-evidence/hexagon-work-queue-pr-review.zip), extract it, and read `REPRODUCE.md` and `VALIDATION.md`.

The archive includes the original and fixed reproduction hooks, a portable build driver, a deterministic pthread/QURT host regression (`bash validation-host/run.sh` on Linux with GCC), and sanitized validation logs. The original host regression is expected to exit 42 on a duplicate callback; the fixed one must exit 0. The host shim is not DSP hardware validation. Hardware and compilation results are identified separately in the validation record.

Archive SHA-256: `2e87fa5b5648a094c48e821e1f3a7cd7cda594124584c00f84efd3b92ceb3a98`.

This evidence-only branch is separate from the proposed source change. No models, private requests, credentials, or runtime binaries are included.
