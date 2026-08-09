#!/usr/bin/env python
"""分叉位置 logits 分析 v2：多 run 生成找分叉，在分叉前一步对比 eager/sdpa 的
logits（差异量级 + top1-top2 竞争激烈度 + argmax）
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    content = tok.apply_chat_template(
        [{"role": "user", "content": "Who are you?"}],
        add_generation_prompt=True, tokenize=False)
    input_ids = tok.encode(content, return_tensors="pt").cuda()
    prompt_n = input_ids.shape[1]

    models = {}
    for impl in ["eager", "sdpa"]:
        models[impl] = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=impl, trust_remote_code=True)

    def gen_to(model, n):
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=n,
                                 eos_token_id=None, pad_token_id=tok.eos_token_id)
        return out[0][prompt_n:].tolist()

    def get_logits(model, seq):
        with torch.no_grad():
            out = model(torch.tensor([seq]).cuda())
        return out.logits[0, -1, :].float()

    # 多 run 找分叉
    found = None
    for r in range(args.runs):
        te = gen_to(models["eager"], args.length)
        ts = gen_to(models["sdpa"], args.length)
        if te != ts:
            d = next(i for i, (x, y) in enumerate(zip(te, ts)) if x != y)
            print(f"run{r+1}: 分叉于 {d} (eager={te[d]} sdpa={ts[d]})")
            found = (r, d, te, ts)
            break
        print(f"run{r+1}: 一致")
    if found is None:
        print(f"{args.runs} 次均一致（概率性，本次未触发）")
        return

    r, d, te, ts = found
    # 分叉前一步：用 eager 的前 d 个 token（两实现一致）forward
    full = input_ids[0].tolist() + te[:d]
    lg_e = get_logits(models["eager"], full)
    lg_s = get_logits(models["sdpa"], full)
    diff = (lg_e - lg_s).abs()
    print(f"分叉前一步({d} 个生成后) logits: max_abs={diff.max().item():.3e} "
          f"mean_abs={diff.mean().item():.3e}")
    for name, lg in [("eager", lg_e), ("sdpa", lg_s)]:
        top2 = torch.topk(lg, 2)
        gap = (top2.values[0] - top2.values[1]).item()
        print(f"{name}: argmax={top2.indices[0].item()} "
              f"top1-top2={gap:.6f} top1={top2.values[0].item():.4f}")
        # EOS 与 argmax 的差距
        eos_score = lg[151643].item()
        print(f"      EOS 分数={eos_score:.4f} (与 top1 差 {top2.values[0].item()-eos_score:.6f})")


if __name__ == "__main__":
    main()
