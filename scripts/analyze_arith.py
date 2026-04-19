from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product

import torch

from noema.arith import ArithBatcher
from noema.latent import LatentGPT
from noema.model import GPTConfig
from noema.tokenizer import ArithTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    tok = ArithTokenizer()
    cfg = GPTConfig(**ckpt["model_cfg"])
    model = LatentGPT(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_terms = ckpt["n_terms"]
    n_thoughts = ckpt["eval_n_thoughts"]
    print(f"ckpt step={ckpt.get('step')}  n_terms={n_terms}  n_thoughts={n_thoughts}  saved_acc={ckpt.get('acc')}")

    batcher = ArithBatcher(tok, n_terms, n_thoughts, seed=0)
    seq_len = batcher.seq_len
    ans_pos = batcher.answer_position()

    all_inputs = list(product(range(10), repeat=n_terms))
    correct_map: dict[tuple[int, ...], bool] = {}
    pred_map: dict[tuple[int, ...], int] = {}

    with torch.no_grad():
        for chunk_start in range(0, len(all_inputs), 256):
            chunk = all_inputs[chunk_start : chunk_start + 256]
            B = len(chunk)
            input_ids = torch.full((B, seq_len), tok.pad_id, dtype=torch.long)
            thought_mask = torch.zeros((B, seq_len), dtype=torch.bool)
            for b, ops in enumerate(chunk):
                prompt = tok.encode("+".join(str(o) for o in ops) + "=")
                ans = sum(ops) % 10
                t = 0
                input_ids[b, t : t + batcher.prompt_len] = torch.tensor(prompt)
                t += batcher.prompt_len
                input_ids[b, t] = tok.bot_id
                t += 1
                bot_end = t
                t += n_thoughts
                input_ids[b, t] = tok.eot_id
                t += 1
                input_ids[b, t] = tok.encode(str(ans))[0]
                thought_mask[b, bot_end : bot_end + n_thoughts] = True

            input_ids = input_ids.to(args.device)
            thought_mask = thought_mask.to(args.device)
            logits, _ = model.forward_latent(input_ids, thought_mask)
            preds = logits[:, ans_pos - 1].argmax(dim=-1).cpu()
            true = input_ids[:, ans_pos].cpu()
            for b, ops in enumerate(chunk):
                pred_tok = preds[b].item()
                true_tok = true[b].item()
                correct_map[ops] = pred_tok == true_tok
                pred_digit = pred_tok - tok.stoi["0"] if 0 <= pred_tok - tok.stoi["0"] <= 9 else -1
                pred_map[ops] = pred_digit

    total = len(correct_map)
    right = sum(correct_map.values())
    print(f"\noverall exhaustive accuracy: {right}/{total} = {right / total:.4f}")

    print("\naccuracy by true answer (a+b+c) mod 10:")
    by_ans: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for ops, ok in correct_map.items():
        a = sum(ops) % 10
        by_ans[a][0] += int(ok)
        by_ans[a][1] += 1
    for ans in range(10):
        r, t = by_ans[ans]
        print(f"  ans={ans}: {r:3d}/{t:3d} = {r / t:.3f}")

    if n_terms >= 2:
        print("\naccuracy by a+b (pre-mod, first two operands):")
        by_ab: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for ops, ok in correct_map.items():
            by_ab[ops[0] + ops[1]][0] += int(ok)
            by_ab[ops[0] + ops[1]][1] += 1
        for s in sorted(by_ab):
            r, t = by_ab[s]
            mark = "  carry" if s >= 10 else ""
            print(f"  a+b={s:2d}: {r:3d}/{t:3d} = {r / t:.3f}{mark}")

    if n_terms == 3:
        print("\naccuracy by c (last operand):")
        by_c: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for ops, ok in correct_map.items():
            by_c[ops[-1]][0] += int(ok)
            by_c[ops[-1]][1] += 1
        for c in range(10):
            r, t = by_c[c]
            print(f"  c={c}: {r:3d}/{t:3d} = {r / t:.3f}")

    print("\nmost frequent wrong predictions:")
    wrong: dict[tuple[int, int], int] = defaultdict(int)
    for ops, ok in correct_map.items():
        if not ok:
            true_ans = sum(ops) % 10
            wrong[(true_ans, pred_map[ops])] += 1
    for (t_ans, p_ans), n in sorted(wrong.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  true={t_ans} pred={p_ans:2d}: {n} cases")


if __name__ == "__main__":
    main()
