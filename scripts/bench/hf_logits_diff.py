#!/usr/bin/env python
"""分叉位置 logits 差异量级：HF eager vs sdpa 在分叉前一步的 logits 对比
设计：分叉于 206（第 206 个生成 token 不同）-> 前 205 个一致 -> 用一致序列
forward 第 206 步，对比两个实现的 logits 向量（量级）+ top-2 竞争激烈度。
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--divergence", type=int, default=206)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    content = tok.apply_chat_template(
        [{"role": "user", "content": "Who are you?"}],
        add_generation_prompt=True, tokenize=False)
    input_ids = tok.encode(content, return_tensors="pt").cuda()
    prompt_n = input_ids.shape[1]

    def gen_to(model, n):
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=n,
                                 eos_token_id=None, pad_token_id=tok.eos_token_id)
        return out[0][prompt_n:].tolist()

    def get_logits(model, seq):
        with torch.no_grad():
            out = model(torch.tensor([seq]).cuda())
        return out.logits[0, -1, :].float()

    results = {}
    for impl in ["eager", "sdpa"]:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=impl, trust_remote_code=True)
        results[impl] = (model, gen_to(model, args.divergence - 1))

    a, b = results["eager"][1], results["sdpa"][1]
    assert a == b, "前序 token 不一致，无法对比 logits（分叉提前发生）"
    print(f"前 {args.divergence - 1} 个生成 token 两实现一致: True")

    full = input_ids[0].tolist() + a
    lg_e = get_logits(results["eager"][0], full)
    lg_s = get_logits(results["sdpa"][0], full)
    diff = (lg_e - lg_s).abs()
    print(f"logits 差异: max_abs={diff.max().item():.3e} "
          f"mean_abs={diff.mean().item():.3e} "
          f"p99_abs={torch.quantile(diff, 0.99).item():.3e}")
    for name, lg in [("eager", lg_e), ("sdpa", lg_s)]:
        top2 = torch.topk(lg, 2)
        gap = (top2.values[0] - top2.values[1]).item()
        print(f"{name}: argmax={top2.indices[0].item()} "
              f"top1={top2.values[0].item():.4f} top2={top2.values[1].item():.4f} "
              f"top1-top2={gap:.4f}")
    e_arg, s_arg = lg_e.argmax().item(), lg_s.argmax().item()
    print(f"argmax 一致: {e_arg == s_arg} (eager={e_arg} sdpa={s_arg})")


if __name__ == "__main__":
    main()
