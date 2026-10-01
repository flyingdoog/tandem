#!/usr/bin/env python3
"""Add a trimmed MTP draft head to a Qwen3.5 GGUF that has an MTP (nextn) layer, for Tandem's speculative decoding.

The MTP layer drafts each token through the model's full output matrix (248,320 rows for Qwen3.5) and a top-k over the
whole vocabulary. This tool copies K of those rows (default 32,768) into a separate draft head together with their token
ids. The runtime then drafts over these rows only, while verification still uses the full vocabulary: a token outside
the draft head is never drafted, but it is still produced whenever the model chooses it, so the output does not change.

Rows are copied as stored, without requantization: from output.weight when the model has its own (untied, e.g.
Qwen3.5-9B), else from token_embd.weight (tied, e.g. Qwen3.5-2B and 4B).

Token choice, in this order until K tokens are chosen:
  1. control and user-defined tokens, and the 256 byte tokens
  2. the first --cjk pure CJK tokens by id
  3. the most frequent tokens of a corpus of your workload (--corpus, needs --tokenizer)
  4. the remaining tokens in id order (the BPE merge order roughly follows frequency)

usage:
  python make_draft_head.py in.gguf out.gguf [--k 32768] [--cjk 8000] [--tokenizer tokenizer.json --corpus a.jsonl b.txt]
  python make_draft_head.py in.gguf out.gguf --ids-from other-with-draft-head.gguf
Corpus files: .txt (one text) or .jsonl (per line a "text" field, chat "messages", or the line itself). --tokenizer is
the model's Hugging Face tokenizer.json (pip install tokenizers).
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf  # noqa: E402

CJK = re.compile("^[　-〿㐀-䶿一-鿿＀-￯]+$")
TOKEN_CONTROL, TOKEN_USER_DEFINED = 3, 4  # tokenizer.ggml.token_type values


def bytes_to_unicode():
    """The byte-to-character table of byte-level BPE (GPT-2, Qwen)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_DECODER = {c: b for b, c in bytes_to_unicode().items()}


def token_text(s):
    try:
        return bytes(BYTE_DECODER[ch] for ch in s).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def corpus_texts(paths):
    for path in paths:
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        yield line
                        continue
                    if isinstance(r, dict) and isinstance(r.get("text"), str):
                        yield r["text"]
                    elif isinstance(r, dict) and isinstance(r.get("messages"), list):
                        yield "\n".join(m.get("content", "") for m in r["messages"] if isinstance(m.get("content"), str))
                    else:
                        yield line
        else:
            with open(path, encoding="utf-8") as f:
                yield f.read()


def pick_tokens(reader, k, n_cjk, n_vocab, tokenizer, corpus):
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    types = reader.fields["tokenizer.ggml.token_type"].contents() if "tokenizer.ggml.token_type" in reader.fields else [1] * len(tokens)
    byte_level = reader.fields["tokenizer.ggml.model"].contents() == "gpt2" if "tokenizer.ggml.model" in reader.fields else False
    keep, seen = [], set()

    def add(i):
        if 0 <= i < n_vocab and i not in seen:
            seen.add(i)
            keep.append(i)

    for i, t in enumerate(types):
        if t in (TOKEN_CONTROL, TOKEN_USER_DEFINED):
            add(i)
    if byte_level:
        for i, s in enumerate(tokens):
            if len(s) == 1 and s in BYTE_DECODER:
                add(i)
    else:
        print("note: not a byte-level BPE vocabulary, skipping the byte and CJK token rules")

    n_cjk_added = 0
    if byte_level:
        for i, s in enumerate(tokens):
            if n_cjk_added >= n_cjk:
                break
            text = token_text(s)
            if text and CJK.match(text):
                add(i)
                n_cjk_added += 1

    freq = collections.Counter()
    if corpus:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer)
        for text in corpus_texts(corpus):
            freq.update(tok.encode(text).ids)
        for tid, _ in freq.most_common():
            if len(keep) >= k:
                break
            add(tid)

    for i in range(n_vocab):
        if len(keep) >= k:
            break
        add(i)
    print(f"draft vocabulary: {len(keep)} tokens ({n_cjk_added} CJK by id, {len(freq)} distinct tokens in the corpus)")
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="GGUF with an MTP (nextn) layer")
    ap.add_argument("dst", help="output GGUF: the same model plus the draft head")
    ap.add_argument("--k", type=int, default=32768, help="draft vocabulary size (default 32768)")
    ap.add_argument("--cjk", type=int, default=8000, help="pure CJK tokens taken by id (default 8000, 0 = none)")
    ap.add_argument("--tokenizer", default=None, help="the model's tokenizer.json, needed for --corpus")
    ap.add_argument("--corpus", nargs="*", default=[], help=".txt or .jsonl files of the workload")
    ap.add_argument("--ids-from", default=None, help="reuse the draft token ids of a GGUF that already has a draft head "
                    "(same tokenizer), e.g. to give a requantized model exactly the same draft vocabulary")
    a = ap.parse_args()
    if a.corpus and not a.tokenizer:
        ap.error("--corpus needs --tokenizer")

    reader = gguf.GGUFReader(a.src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    names = [t.name for t in reader.tensors]
    nextn = [int(m.group(1)) for n in names for m in [re.match(r"blk\.(\d+)\.nextn\.", n)] if m]
    if not nextn:
        sys.exit(f"{a.src} has no MTP (nextn) layer: convert the checkpoint with its MTP weights")
    if any(n.endswith(".nextn.draft_head.weight") for n in names):
        sys.exit(f"{a.src} already has a draft head")
    nextn_blk = max(nextn)

    emb = next(t for t in reader.tensors if t.name == "token_embd.weight")
    lm = next((t for t in reader.tensors if t.name == "output.weight"), emb)  # untied LM head if present, else tied
    n_vocab = int(emb.shape[1])
    print(f"{arch}: rows of {lm.name} ({lm.tensor_type.name}), vocabulary {n_vocab}, MTP layer {nextn_blk}")

    if a.ids_from:
        prev = gguf.GGUFReader(a.ids_from)
        ids = np.array(next(t for t in prev.tensors if t.name.endswith(".nextn.draft_ids.weight")).data, dtype=np.int32)
    else:
        ids = np.array(sorted(pick_tokens(reader, min(a.k, n_vocab), a.cjk, n_vocab, a.tokenizer, a.corpus)), dtype=np.int32)

    head = np.ascontiguousarray(lm.data[ids])  # [K, row bytes]: quantized rows are independent of each other
    writer = gguf.GGUFWriter(a.dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)

    extra = [(f"blk.{nextn_blk}.nextn.draft_head.weight", head, lm.tensor_type),
             (f"blk.{nextn_blk}.nextn.draft_ids.weight", ids, gguf.GGMLQuantizationType.I32)]
    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, data, ttype in extra:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for name, data, ttype in extra:
        writer.write_tensor_data(data)
    writer.close()
    print(f"wrote {a.dst}: draft head {head.nbytes / 1e6:.1f} MB for {len(ids)} tokens (full head {lm.n_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
