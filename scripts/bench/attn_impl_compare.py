#!/usr/bin/env python
"""HF 侧对照实验：不同 attention 实现（eager/sdpa/flash）在长序列下是否互相分叉
目的：验证"attention 实现差异 -> 长序列 token 分叉"是否普遍现象（我们的实现 vs HF 同理）
"""
import argparse
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--prompt", default="Who are you?")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    content = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=False)
    input_ids = tok.encode(content, return_tensors="pt")

    def run(attn_impl):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=attn_impl, trust_remote_code=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=args.length, eos_token_id=None,
                pad_token_id=tok.eos_token_id)
        dt = time.perf_counter() - t0
        return dt, out[0][input_ids.shape[1]:].tolist()

    print(f"length={args.length} impls=[eager, sdpa, flash_attention_2]")
    results = {}
    for impl in ["eager", "sdpa", "flash_attention_2"]:
        try:
            dt, toks = run(impl)
            results[impl] = (dt, toks)
            print(f"{impl}: {dt:.3f}s gen={len(toks)}")
        except Exception as e:
            print(f"{impl}: FAILED {type(e).__name__}: {e}")

    # 两两对比
    impls = list(results.keys())
    for i in range(len(impls)):
        for j in range(i + 1, len(impls)):
            a, b = impls[i], impls[j]
            ta, tb = results[a][1], results[b][1]
            if ta == tb:
                print(f"{a} vs {b}: identical")
            else:
                d = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), -1)
                print(f"{a} vs {b}: MISMATCH at {d} ({ta[d]} vs {tb[d]})")

if __name__ == "__main__":
    main()
