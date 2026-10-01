# Tandem

Tandem serves hybrid linear-attention models, such as Qwen3.5 with its Gated DeltaNet layers, as the policy of a GUI
agent on the Hexagon NPU of Snapdragon phones. It is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
(release b11200) and a drop-in replacement for `llama-server`: the HTTP API is unchanged, so an agent switches to Tandem
by starting a different binary.

Engines built for chat leave three costs in every agent step of a hybrid model. The recurrent state resumes only at
stored checkpoints, which chat-oriented servers place where a chat would continue, so each step recomputes much of the
prompt it shares with the previous step. Speculative decoding has to roll the recurrent state back, which on the NPU
makes it slower rather than faster. And an idle model keeps the NPU powered up between tasks. Tandem treats the agent
step as its unit of work and removes these costs; the paper describes the design and its evaluation.

## Results

Server time per agent step on a OnePlus 13T (Snapdragon 8 Elite, Hexagon v79 NPU) for the policy of a deployed GUI
agent, a fine-tuned Qwen3.5-4B with 4-bit weights (Q4_K_M), replaying the agent's real traffic: 61 text steps and 21
steps with a screenshot. Every engine runs in its fastest configuration unless marked. In parentheses: the time
relative to Tandem.

| Engine | Text step | Screenshot step |
|---|---|---|
| **Tandem (NPU)** | **1.49 s** | **4.87 s** |
| llama.cpp (NPU) | 1.97 s (1.32×) | 5.80 s (1.19×) |
| llama.cpp (NPU, with MTP speculation) | 2.56 s (1.71×) | 7.60 s (1.56×) |
| MNN (GPU, OpenCL) | 4.98 s (3.3×) | 19.1 s (3.9×)¹ |
| llama.cpp (GPU, OpenCL) | 5.08 s (3.4×) | 22.8 s (4.7×) |
| llama.cpp (CPU, 8 threads) | 11.2 s (7.5×) | 69.8 s (14×)² |

¹ 4 of the 21 screenshot steps. ² With MTP speculation, from a separate run.

- **Energy:** on battery, a text step costs 11.1 J above idle power under Tandem against 15.0 J under llama.cpp's
  NPU backend, 26% less.
- **Standby power:** with the model loaded and the screen off, the phone draws 0.22 W under Tandem instead of 0.76 W,
  3.4× less and as little as without a model server. The next request pays a few milliseconds to restore the votes.
- **Hybrid versus full attention:** llama.cpp serves the hybrid Qwen3.5-4B 1.38× slower than a full-attention model
  of the same size (Qwen3-VL-4B); under Tandem the two are equally fast.
- **Model sizes:** on the stock Qwen3.5 models, Tandem's text steps are 1.27× (2B), 1.34× (4B) and 1.42× (9B) faster
  than llama.cpp's NPU backend.
- **Next NPU generation:** on a OnePlus 15 (Snapdragon 8 Elite Gen 5, Hexagon v81), 1.28 s against 1.69 s per text
  step (1.32×) and 3.95 s against 4.67 s per screenshot step (1.18×).
- **Public benchmark:** on the first 300 test steps of AndroidControl, 1.21 s against 1.58 s per step (1.30×), with
  identical answers.

In these comparisons Tandem runs with divergence-point checkpoints but keeps the tail pass; dropping it as well, as
`--agent-checkpoints` does, shortens a text step by about another 0.09 s.

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

## Requirements

- A phone with a Snapdragon 8 Elite (Hexagon v79) or Snapdragon 8 Elite Gen 5 (Hexagon v81). Tandem was tested on a
  OnePlus 13T and a OnePlus 15; root access is not needed. The build also produces libraries for v73 and v75, which
  were not tested.
- A Qwen3.5 model with its multi-token prediction (MTP) layer, in GGUF format, and optionally its vision tower
  (`mmproj`). Tested: Qwen3.5 2B, 4B and 9B. On other hybrid models (tested: LFM2, Granite 4.0-H) the checkpoint
  policy and idle power release apply; the Gated DeltaNet kernels and the MTP mechanisms do not.

## Build

Tandem builds exactly like llama.cpp for Snapdragon, with the toolchain container that includes the Android NDK and the
Hexagon SDK ([docs/backend/snapdragon](docs/backend/snapdragon/README.md)):

```bash
./scripts/snapdragon/build.py --target adb --push
```

This installs `llama-server` and the libraries to `/data/local/tmp/llama.cpp` on the connected phone.

## Prepare a model

1. Convert the Hugging Face checkpoint with `convert_hf_to_gguf.py`, which keeps the MTP layer, and quantize it, for
   example to Q4_K_M with `llama-quantize`.
2. Add the trimmed draft head:

   ```bash
   python tools/tandem/make_draft_head.py model-Q4_K_M.gguf model-Q4_K_M-dh.gguf \
       --tokenizer tokenizer.json --corpus my_prompts_and_answers.jsonl
   ```

   The corpus, prompts and answers of your workload, chooses which tokens the head can draft. Without `--corpus`, the
   tool takes the special, byte and most common CJK tokens and then the lowest token ids, which works but drafts less
   well for a specific workload.

## Run

On the phone:

```bash
cd /data/local/tmp/llama.cpp
LD_LIBRARY_PATH=$PWD/lib ./bin/llama-server -m model-Q4_K_M-dh.gguf --mmproj mmproj.gguf \
    --device HTP0 --mmproj-device HTP0 -ngl 99 -t 4 -tb 4 \
    --ctx-size 8192 --ctx-checkpoints 8 --cache-ram 512 --no-warmup \
    --spec-type draft-mtp --agent-checkpoints --port 8080
```

The agent then sends its requests to the OpenAI-compatible endpoint (`/v1/chat/completions`) as it would to
`llama-server`.

Notes:

- `--agent-checkpoints` is meant for stateless clients that send the whole prompt at every step, as agents do. A chat
  client still gets correct answers, but it resumes from an earlier checkpoint.
- Bound the prompt cache (`--cache-ram`): llama.cpp's default of 8 GiB can exhaust a phone's memory.
- Requests that ask for log-probabilities make the server compute full-vocabulary probabilities, which costs time on
  every generated token without speculation.
- With idle power release, the first request after an idle period restores the power votes first, which takes a few
  milliseconds.
- The kernels, the draft head and idle power release leave every output bit unchanged. The checkpoint policies and
  speculative decoding change how the prompt is split into batches and thus the floating-point rounding.

## License

MIT, like llama.cpp. Tandem's changes are released under the same license.
