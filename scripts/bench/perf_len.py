#!/usr/bin/env python
"""LLASYS 长度扫描：per-token vs 生成长度（强制生成到 N 不截断 + 逐 token 一致验证）
用法: python perf_len.py --model <path> --device nvidia --lengths 64,256,1024,2048
"""
import argparse
import ctypes
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="nvidia")
    ap.add_argument("--lengths", default="64,256,1024,2048")
    ap.add_argument("--prompt", default="Who are you?")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ---- HF 参照 ----
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="cuda" if args.device == "nvidia" else "cpu",
        trust_remote_code=True)
    content = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=False)
    input_ids = tok.encode(content, return_tensors="pt").to(hf_model.device)
    prompt_n = input_ids.shape[1]

    # ---- llaisys ----
    from llaisys.models.qwen2 import Qwen2
    from llaisys.libllaisys import LIB_LLAISYS, DeviceType
    dev = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU
    model = Qwen2(args.model, dev)
    ll_inputs = input_ids[0].tolist()
    n_ll = len(ll_inputs)

    def ll_gen_n(n):
        """自写循环：Reset -> prefill -> decode 到 n（忽略 EOS），返回 (耗时, 生成 token 列表)"""
        LIB_LLAISYS.llaisysQwen2ModelReset(model._model)
        arr = (ctypes.c_int64 * n_ll)(*ll_inputs)
        t0 = time.perf_counter()
        nxt = int(LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, arr, n_ll))
        if nxt < 0:
            raise RuntimeError("prefill Infer failed")
        tokens = [nxt]
        while len(tokens) < n:
            a1 = (ctypes.c_int64 * 1)(nxt)
            nxt = int(LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, a1, 1))
            if nxt < 0:
                raise RuntimeError("decode Infer failed")
            tokens.append(nxt)
        dt = time.perf_counter() - t0
        return dt, tokens

    def hf_gen_n(n):
        with torch.no_grad():
            t0 = time.perf_counter()
            out = hf_model.generate(
                input_ids, max_new_tokens=n, eos_token_id=None,
                pad_token_id=tok.eos_token_id)
            dt = time.perf_counter() - t0
        return dt, out[0][prompt_n:].tolist()

    print(f"device={args.device} prompt_tokens={prompt_n} lengths={lengths} runs={args.runs}")
    for n in lengths:
        print(f"===== length={n} =====")
        for r in range(args.runs):
            dt_l, tok_l = ll_gen_n(n)
            dt_h, tok_h = hf_gen_n(n)
            # 逐 token 一致验证：长度 + 逐位
            len_ok = (len(tok_l) == n and len(tok_h) == n)
            same = tok_l == tok_h
            if not same:
                first_diff = next((i for i, (a, b) in enumerate(zip(tok_l, tok_h)) if a != b), -1)
                print(f"  MISMATCH at {first_diff}: llaisys={tok_l[first_diff]} hf={tok_h[first_diff]}")
            print(f"run{r+1} llaisys={dt_l:.3f}s HF={dt_h:.3f}s "
                  f"len_ok={len_ok} token_identical={same}")


if __name__ == "__main__":
    main()
