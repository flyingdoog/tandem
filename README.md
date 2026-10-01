# Tandem: serving hybrid linear-attention GUI agents on mobile NPUs

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Based on llama.cpp b11200](https://img.shields.io/badge/based%20on-llama.cpp%20b11200-informational)](https://github.com/ggml-org/llama.cpp)

Tandem serves hybrid linear-attention models, such as Qwen3.5 with its Gated DeltaNet layers, as the policy of a GUI
agent on the Hexagon NPU of Snapdragon phones. It is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) and a
**drop-in replacement for `llama-server`**: the HTTP API is unchanged, so an agent switches to Tandem by starting a
different binary.

On a Snapdragon 8 Elite phone serving the fine-tuned Qwen3.5-4B policy of a deployed GUI agent, compared with
llama.cpp's NPU backend in its fastest configuration, Tandem

- serves text steps with **24% lower latency** (1.49 s against 1.97 s per step) and screenshot steps with **16% lower
  latency** (4.87 s against 5.80 s),
- uses **26% less energy** per step,
- cuts the standby power of a phone with the model loaded **3.4×**, to the level of a phone without a model server,
- and is **3.3–14× faster** than llama.cpp and MNN on the phone's GPU and CPU.

## Why agents need their own serving runtime

Engines built for chat leave three costs in every agent step of a hybrid model. Tandem treats the agent step as its unit
of work and removes them:

| | What goes wrong | llama.cpp on the NPU | Tandem |
|---|---|---|---|
| **Checkpoint gap** | The recurrent state resumes only at stored checkpoints, and a chat server stores them where a chat would continue | 539 new prompt tokens per step, where a KV cache would need 428 | **443**, with checkpoints where consecutive prompts diverge |
| **Rollback tax** | Speculative decoding has to roll the recurrent state back, which pushes the prefill off the NPU's fast kernel | speculation makes a step **0.58 s slower** | speculation makes a step **0.30 s faster** |
| **Residency tax** | The NPU session holds its power votes for as long as the model is loaded | 0.76 W standby with the model loaded | **0.22 W**, as little as without a model server |

## Results

![Server time per agent step: Tandem 1.49 s per text step and 4.87 s per screenshot step; llama.cpp on the NPU 1.97 and 5.80 s; with speculation 2.56 and 7.60 s; MNN on the GPU 4.98 and 19.1 s; llama.cpp on the GPU 5.08 and 22.8 s; llama.cpp on the CPU 11.2 and 69.8 s](docs/tandem/step-time.svg)

Server time per agent step on a OnePlus 13T (Snapdragon 8 Elite, Hexagon v79) for a fine-tuned Qwen3.5-4B with 4-bit
weights (Q4_K_M), replaying the agent's real traffic: 61 text steps and 21 steps with a screenshot. Every engine runs in
its fastest configuration unless marked. MNN's screenshot step is the mean of 4 of the 21 steps; llama.cpp's CPU
screenshot step is from a separate run with speculation.

- **Energy.** On battery, a text step costs 11.1 J above idle power under Tandem against 15.0 J under llama.cpp's NPU
  backend.
- **Standby power.** With the model loaded and the screen off, the phone draws 0.22 W under Tandem instead of 0.76 W.
  The next request pays a few milliseconds to restore the power votes.
- **Hybrid on par with full attention.** llama.cpp needs 38% more time per step for the hybrid Qwen3.5-4B than for a
  full-attention model of the same size (Qwen3-VL-4B); under Tandem the two are equally fast.
- **Same answers.** The kernels, the draft head and idle power release leave every output bit unchanged; the
  checkpoint policy and speculation change only the floating-point rounding, which reworded at most one free-text
  summary per trace and changed no action.

![Server time per text step, llama.cpp on the NPU against Tandem: Qwen3.5-2B 0.97 and 0.77 s, Qwen3.5-4B 1.81 and 1.35 s, Qwen3.5-9B 3.04 and 2.15 s, the agent policy on the Snapdragon 8 Elite 1.97 and 1.49 s and on the Snapdragon 8 Elite Gen 5 1.69 and 1.28 s, AndroidControl 1.58 and 1.21 s](docs/tandem/text-steps.svg)

The gains hold for the stock Qwen3.5 models, where they grow with the model (21–29% less time per text step), on the
next NPU generation (OnePlus 15, Snapdragon 8 Elite Gen 5, Hexagon v81; with screenshots 3.95 s against 4.67 s per
step, 15% less), and on the public AndroidControl benchmark (its first 300 test steps, with identical answers).
llama.cpp runs in its fastest configuration, on the NPU without speculation.

In these comparisons, Tandem runs with checkpoints at the divergence point but keeps llama.cpp's tail pass. Dropping it
as well, as `--agent-checkpoints` does, shortens a text step by about another 0.09 s.

## What Tandem changes

| Mechanism | Where | What it does | Switch (default) |
|---|---|---|---|
| Divergence-point checkpoints | server | checkpoints the recurrent state where the prompt stops matching the previous one, so the next step resumes there | `--agent-checkpoints` (or `LLAMA_CKPT_DIVERGE=1`) |
| No chat-only passes | server | drops the extra pass over the last 4 prompt tokens and the re-snapshot of a restored state, which only a chat's next turn uses | `--agent-checkpoints` (or `LLAMA_CKPT_TAIL=0`) |
| Checkpoint before an image | server | a checkpoint right before the screenshot, for re-queries of the same screen | `LLAMA_CKPT_BEFORE_IMAGE=2` (off) |
| Rollback-aware recurrent prefill | NPU | keeps the rollback snapshots of speculative decoding on the chunked matrix-unit kernel | on |
| Column-shared verification kernel | NPU | verifies up to 4 tokens at little more than the cost of one: each weight block is unpacked once for all columns | on |
| Trimmed draft head | model file | the MTP head drafts over 32,768 rows of the output matrix instead of the full vocabulary | `tools/tandem/make_draft_head.py` |
| Channel-major convolution, DMA state gathers, fused SwiGLU | NPU | less data movement per prefill pass | on |
| DDR performance vote | NPU | raises the memory controller's performance mode while requests run | `GGML_HEXAGON_PWR=1` |
| Idle power release | NPU | relaxes the session's power votes after an idle period and restores them before the next batch | `GGML_HEXAGON_IDLE_MS=3000` |
| Work-queue race fix, watchdog | NPU | the server runs for thousands of requests; an NPU that stops answering becomes a restartable crash | on, `GGML_HEXAGON_WATCHDOG=60` |

The NPU paths can be switched off for comparison with `GGML_HEXAGON_TANDEM_OFF`, a bitmask: 1 verification kernel,
2 rollback-aware prefill, 4 and 8 conv-state gathers, 16 fused SwiGLU. With all of Tandem's switches off, the build runs
llama.cpp's stock policy and kernels.

## Quick start

**Requirements.** A phone with a Snapdragon 8 Elite (Hexagon v79) or Snapdragon 8 Elite Gen 5 (Hexagon v81); Tandem
was tested on a OnePlus 13T and a OnePlus 15, and root access is not needed. The build also produces libraries for v73
and v75, which were not tested. A Qwen3.5 model with its multi-token prediction (MTP) layer, and optionally its vision
tower (`mmproj`); tested: Qwen3.5 2B, 4B and 9B. On other hybrid models (tested: LFM2, Granite 4.0-H) the checkpoint
policy and idle power release apply, while the Gated DeltaNet kernels and the MTP mechanisms do not.

**Build** exactly like llama.cpp for Snapdragon, with the toolchain container that includes the Android NDK and the
Hexagon SDK ([docs/backend/snapdragon](docs/backend/snapdragon/README.md)). This installs `llama-server` and its
libraries to `/data/local/tmp/llama.cpp` on the connected phone:

```bash
./scripts/snapdragon/build.py --target adb --push
```

**Prepare a model.** Convert the Hugging Face checkpoint with `convert_hf_to_gguf.py`, which keeps the MTP layer,
quantize it (for example to Q4_K_M with `llama-quantize`), and add the trimmed draft head. The corpus, prompts and
answers of your workload, chooses which tokens the head can draft; without it, the tool takes the special, byte and
most common CJK tokens and then the lowest token ids.

```bash
python tools/tandem/make_draft_head.py model-Q4_K_M.gguf model-Q4_K_M-dh.gguf \
    --tokenizer tokenizer.json --corpus my_prompts_and_answers.jsonl
```

**Run** on the phone, then send the agent's requests to the OpenAI-compatible endpoint (`/v1/chat/completions`) as you
would to `llama-server`:

```bash
cd /data/local/tmp/llama.cpp
LD_LIBRARY_PATH=$PWD/lib ./bin/llama-server -m model-Q4_K_M-dh.gguf --mmproj mmproj.gguf \
    --device HTP0 --mmproj-device HTP0 -ngl 99 -t 4 -tb 4 \
    --ctx-size 8192 --ctx-checkpoints 8 --cache-ram 512 --no-warmup \
    --spec-type draft-mtp --agent-checkpoints --port 8080
```

**Notes.**

- `--agent-checkpoints` is meant for stateless clients that send the whole prompt at every step, as agents do. A chat
  client still gets correct answers, but it resumes from an earlier checkpoint.
- Bound the prompt cache (`--cache-ram`): llama.cpp's default of 8 GiB can exhaust a phone's memory.
- Requests that ask for log-probabilities make the server compute full-vocabulary probabilities, which costs time on
  every generated token without speculation.

## License

MIT, like llama.cpp; Tandem's changes are released under the same license. Tandem is built on llama.cpp and its
Hexagon backend; upstream's README is in [README-llama.cpp.md](README-llama.cpp.md).
