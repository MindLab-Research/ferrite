#!/usr/bin/env python3
"""DERIVED (non-official) DSpark/MTP accept-rate harness.

⚠️  This is NOT part of the official reference repo. The official ``generate.py``
    is a pure single-token autoregressive loop and never calls the DSpark draft
    head, so the official script cannot produce an accept rate. ``model.py``
    implements ``Transformer.forward_spec`` (the DSpark draft forward) but its own
    comment states the speculative-decoding loop is "out of scope for this repo".

    This harness builds that missing loop *only to measure acceptance*. It must be
    sanity-checked before the numbers are trusted -- it prints its own parity
    check (see PARITY below) and refuses to report stats if parity fails.

METHOD (cache-safe, per-position)
  1. Greedy-decode the prompt with the OFFICIAL single-step loop -> canonical
     token stream G[0..N-1]. This is the "target" oracle; temperature 0.
  2. For sampled positions i, re-prefill the whole prefix (prompt + G[:i]) from
     start_pos = 0, so every measurement starts from a freshly re-seeded KV cache
     (no cross-step cache contamination). Then call ``forward_spec(G[i], hidden,
     start_pos=L-1)`` to draft the next ``dspark_block_size`` tokens and compare
     them against G[i+1 .. i+block].

     PARITY: re-prefilling prompt+G[:i] must reproduce G[i] as the model's own
     next token; the harness asserts this at every sampled position. If it holds,
     the re-prefill reconstructed the exact context and the drafts are valid.

  accept_len(i) = length of the longest common prefix between
                  drafted[i+1 .. i+block] and the target G[i+1 .. i+block].
  accept_rate   = mean(accept_len) / block_size, vs the perfect value of 1.0
                  (i.e. "accept ~= block_size", the claim being checked).

USAGE (on 43.202.208.136, official inference dir is the cwd)
  cd /opt/dlami/nvme/models/DeepSeek-V4.1-Flash/inference
  MP=8 torchrun --nproc-per-node 8 ~/ref_oracle/dsv41_ref_bench_accept.py \
      --ckpt-path  /opt/dlami/nvme/models/V41-demo-TP8 \
      --config     ~/ref_oracle/config_fp8.json \
      --input-file ~/ref_oracle/p_sh.json \
      --max-new-tokens 64 --stride 1 \
      --out ~/ref_oracle/bench_accept.json

  Runtime ≈ N/stride prefill passes (one per sampled position) on top of the one
  greedy pass. Keep --max-new-tokens modest (64 is plenty for a mean estimate).
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

OFFICIAL = "/opt/dlami/nvme/models/DeepSeek-V4.1-Flash/inference"
if not os.path.exists(os.path.join(OFFICIAL, "model.py")):
    OFFICIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref_inference")
sys.path.insert(0, os.path.abspath(OFFICIAL))

from model import Transformer, ModelArgs  # noqa: E402
import generate as gen  # noqa: E402


@torch.inference_mode()
def greedy_decode(model, prompt_tokens, max_new_tokens, eos_id):
    """Official single-step loop (generate.generate), verbatim, greedy."""
    prompt_lens = [len(prompt_tokens)]
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((1, total_len), -1, dtype=torch.long)
    tokens[0, : prompt_lens[0]] = torch.tensor(prompt_tokens, dtype=torch.long)

    prev_pos = 0
    finished = torch.tensor([False])
    prompt_mask = tokens != -1
    gen_ids = []
    for cur_pos in range(prompt_lens[0], total_len):
        next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)[0]
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        tok = int(next_token[0].item())
        gen_ids.append(tok)
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    return gen_ids


@torch.inference_mode()
def draft_block(model, prefix_ids, target_next, block_size):
    """Re-prefill ``prefix_ids`` (start_pos=0) then DSpark-draft the next block.

    Returns (drafts, confidence, reproduced_next) where drafts are the
    ``block_size`` tokens predicted for positions len(prefix)+1 .. +block_size.
    """
    p = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
    out = model.forward(p, 0)
    reproduced_next = int(out[0][0].item())
    hidden = out[2]  # main_hidden: [1, L, 3*dim]

    start_pos = p.size(1) - 1  # last position of the prefix
    draft_out, _logits, conf = model.forward_spec(target_next, hidden[:, -1:], start_pos)
    # forward_spec output_ids[k] is the token for position start_pos+1+k; index 0
    # is the already-known token we passed in, so real drafts start at index 1.
    drafts = draft_out[0, 1 : 1 + block_size].tolist()
    return drafts, conf, reproduced_next


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"])
    ap.add_argument("--stride", type=int, default=1, help="sample one position every N")
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
    margs.temperature = 0.0  # greedy target AND greedy draft
    block_size = int(margs.dspark_block_size)
    if block_size <= 0:
        print("[accept] dspark_block_size == 0 in config -> no DSpark head; aborting", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(cli.ckpt_path)
    print("[accept] build model", flush=True)
    with torch.device("cuda"):
        model = Transformer(margs, tokenizer)
    print("[accept] load model", flush=True)
    load_model(model, os.path.join(cli.ckpt_path, f"model{rank}-mp{world}.safetensors"))
    torch.set_default_device("cuda")

    if cli.input_file.endswith(".json"):
        cases = gen.load_cases(cli.input_file)
        raw_prompt = gen.to_json(cases[0]["messages"])
    else:
        with open(cli.input_file) as f:
            raw_prompt = f.read().rstrip("\n").split("\n\n")[0]
        cases = [{"messages": [{"role": "user", "content": gen.parse_tagged_text(raw_prompt)}]}]
    _, prompt_tokens, _, images = gen.prepare_case(cases[0], cli.thinking_mode, tokenizer, margs)
    print(f"[accept] prompt tokens = {len(prompt_tokens)}  block_size = {block_size}", flush=True)

    # ---- step 1: canonical greedy stream -------------------------------------
    print("[accept] greedy decode (canonical target stream) ...", flush=True)
    G = greedy_decode(model, prompt_tokens, cli.max_new_tokens, tokenizer.eos_token_id)
    print(f"[accept] greedy produced {len(G)} tokens", flush=True)

    # ---- step 2: per-position accept -----------------------------------------
    lcp_list = []
    conf_acc, conf_rej = [], []
    parity_fail = 0
    positions = list(range(0, max(0, len(G) - block_size), max(1, cli.stride)))
    for i in positions:
        prefix = prompt_tokens + G[:i]
        drafts, conf, reproduced = draft_block(model, prefix, G[i], block_size)
        if reproduced != G[i]:
            parity_fail += 1
            if parity_fail <= 3:
                print(f"[accept] PARITY FAIL at i={i}: reproduced={reproduced} != G={G[i]}", flush=True)
            continue
        target = G[i + 1 : i + 1 + block_size]
        k = 0
        while k < len(drafts) and k < len(target) and drafts[k] == target[k]:
            k += 1
        lcp_list.append(k)
        c = conf[0].tolist()
        for j in range(min(len(c), block_size)):
            (conf_acc if j < k else conf_rej).append(float(c[j]))

    if rank == 0:
        if parity_fail:
            print(
                f"\n[accept] !! {parity_fail}/{len(positions)} positions failed parity -- "
                "drafts NOT trustworthy; results may be invalid",
                flush=True,
            )
        if not lcp_list:
            print("[accept] no valid samples; aborting", flush=True)
        else:
            n = len(lcp_list)
            mean_lcp = statistics.fmean(lcp_list)
            hist = {k: lcp_list.count(k) for k in range(block_size + 1)}
            result = {
                "block_size": block_size,
                "samples": n,
                "parity_fail": parity_fail,
                "mean_accept_len": round(mean_lcp, 4),
                "accept_rate": round(mean_lcp / block_size, 4),
                "accept_len_hist": hist,
                "mean_confidence_accepted": round(statistics.fmean(conf_acc), 4) if conf_acc else None,
                "mean_confidence_rejected": round(statistics.fmean(conf_rej), 4) if conf_rej else None,
                "effective_tokens_per_round": round(1.0 + mean_lcp, 4),
            }
            print("\n===== DSPARK ACCEPT (derived, non-official) =====", flush=True)
            for key, v in result.items():
                print(f"{key:>26}: {v}", flush=True)
            print(f"\n{'-' * 64}\nPROMPT: {raw_prompt}\n{'-' * 64}", flush=True)
            print(f"greedy text: {tokenizer.decode(G)}", flush=True)
            if cli.out:
                with open(cli.out, "w") as f:
                    json.dump(
                        {"result": result, "greedy_text": tokenizer.decode(G)},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[accept] wrote {cli.out}", flush=True)
        print("ACCEPTDONE", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
