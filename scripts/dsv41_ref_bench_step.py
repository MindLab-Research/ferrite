#!/usr/bin/env python3
"""Official-reference decode step-time harness (ferrite accept/步时 calibration).

WHAT IT MEASURES
  * prefill latency  -- one forward over the whole prompt
  * per-decode-step latency (ms) for the official single-token loop
  * aggregate: mean / p50 / p90 / min / max step ms, and decode tok/s

WHY IT EXISTS
  The official ``generate.py`` emits exactly ONE token per forward pass. It does
  NOT run the DSpark / MTP draft head: ``model.py`` implements ``forward_spec``
  but ``generate.py`` never calls it, and model.py's own comment says the
  speculative-decoding loop is "out of scope for this repo". There is therefore
  NO official accept-rate number -- only an authoritative per-step time for a
  pure single-step decode engine.

  This file reproduces ``generate.generate()``'s loop *verbatim* and only adds
  CUDA-event timers; it does not modify any official file.

USAGE (on 43.202.208.136, official inference dir is the cwd)
  cd /opt/dlami/nvme/models/DeepSeek-V4.1-Flash/inference
  MP=8 torchrun --nproc-per-node 8 ~/ref_oracle/dsv41_ref_bench_step.py \
      --ckpt-path  /opt/dlami/nvme/models/V41-demo-TP8 \
      --config     ~/ref_oracle/config_fp8.json \
      --input-file ~/ref_oracle/p_sh.json \
      --max-new-tokens 256 --temperature 0 \
      --out ~/ref_oracle/bench_step.json

  temperature 0 => greedy, which is the same decoding mode the accept harness
  compares against.
"""

import argparse
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import AutoTokenizer

# --- import the OFFICIAL model + encoding code ---------------------------------
# generate.py already inserts ../encoding on sys.path at import time, so importing
# it also gives us the exact prompt-encoding path (no reimplementation).
OFFICIAL = "/opt/dlami/nvme/models/DeepSeek-V4.1-Flash/inference"
if not os.path.exists(os.path.join(OFFICIAL, "model.py")):
    OFFICIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref_inference")
sys.path.insert(0, os.path.abspath(OFFICIAL))

from model import Transformer, ModelArgs  # noqa: E402
import generate as gen  # noqa: E402


@torch.inference_mode()
def timed_decode(model, prompt_tokens, max_new_tokens, eos_id):
    """generate.generate()'s loop verbatim, plus CUDA-event timers.

    Iteration 0 prefills the whole prompt (min(prompt_lens) tokens); every later
    iteration decodes exactly one token, which is what we want to time.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    prefill_ms = None
    step_ms = []
    gen_ids = []
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    for cur_pos in range(min(prompt_lens), total_len):
        torch.cuda.synchronize()
        ev0.record()
        next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)[0]
        ev1.record()
        torch.cuda.synchronize()
        dt = ev0.elapsed_time(ev1)  # ms

        if prefill_ms is None:
            prefill_ms = dt
        else:
            step_ms.append(dt)

        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        gen_ids.append(int(next_token[0].item()))
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break

    return prefill_ms, step_ms, gen_ids


def build_prompt(tokenizer, margs, input_file, thinking_mode):
    if input_file.endswith(".json"):
        cases = gen.load_cases(input_file)
        raw_prompts = [gen.to_json(c["messages"]) for c in cases]
    else:
        with open(input_file) as f:
            raw = f.read().rstrip("\n").split("\n\n")
        cases = [{"messages": [{"role": "user", "content": gen.parse_tagged_text(p)}]} for p in raw]
        raw_prompts = raw
    _, tokens, _, images = gen.prepare_case(cases[0], thinking_mode, tokenizer, margs)
    if any(images):
        print("[bench] WARN: image prompt; text path only is measured", flush=True)
    return tokens, raw_prompts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"])
    ap.add_argument("--out", default="")
    cli = ap.parse_args()

    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *a, **k: None

    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    with open(cli.config) as f:
        margs = ModelArgs(**json.load(f))
    margs.temperature = cli.temperature

    tokenizer = AutoTokenizer.from_pretrained(cli.ckpt_path)
    print("[bench] build model", flush=True)
    with torch.device("cuda"):
        model = Transformer(margs, tokenizer)
    print("[bench] load model", flush=True)
    load_model(model, os.path.join(cli.ckpt_path, f"model{rank}-mp{world}.safetensors"))
    torch.set_default_device("cuda")

    prompt_tokens, raw_prompt = build_prompt(tokenizer, margs, cli.input_file, cli.thinking_mode)
    print(f"[bench] prompt tokens = {len(prompt_tokens)}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    prefill_ms, step_ms, gen_ids = timed_decode(
        model, [prompt_tokens], cli.max_new_tokens, tokenizer.eos_token_id
    )
    text = tokenizer.decode(gen_ids)

    if rank == 0:
        n = len(step_ms)
        s = sorted(step_ms)

        def pct(p):
            return s[min(n - 1, int(p * n))] if n else float("nan")

        mean = statistics.fmean(step_ms) if n else float("nan")
        result = {
            "prompt_tokens": len(prompt_tokens),
            "temperature": cli.temperature,
            "prefill_ms": round(prefill_ms, 3),
            "decode_steps": n,
            "step_ms_mean": round(mean, 3),
            "step_ms_p50": round(pct(0.5), 3),
            "step_ms_p90": round(pct(0.9), 3),
            "step_ms_min": round(s[0], 3) if n else None,
            "step_ms_max": round(s[-1], 3) if n else None,
            "decode_tok_per_s": round(1000.0 / mean, 3) if n and mean else None,
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        print("\n===== OFFICIAL STEP-TIME BENCH =====", flush=True)
        for k, v in result.items():
            print(f"{k:>18}: {v}", flush=True)
        print(f"\n{'-' * 64}\nPROMPT: {raw_prompt}\n{'-' * 64}", flush=True)
        print(text, flush=True)
        print(f"{'-' * 64}", flush=True)
        if cli.out:
            with open(cli.out, "w") as f:
                json.dump({"result": result, "text": text}, f, ensure_ascii=False, indent=2)
            print(f"[bench] wrote {cli.out}", flush=True)
        print("BENCHDONE", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
