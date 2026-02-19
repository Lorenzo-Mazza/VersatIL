import torch
import time
import numpy as np
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F


def benchmark(name: str, seq_len: int = 2048, batch: int = 4, heads: int = 32, head_dim: int = 64, runs: int = 30):
    print(f"\n=== {name} (bs={batch} seq={seq_len} heads={heads} dim={head_dim}) ===")

    # All in bf16 — realistic for modern training
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Warmup
    for _ in range(10):
        if name == "flash_attention_2":
            out = flash_attn_func(q, k, v, causal=True)
        elif name == "sdpa_default":
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        elif name == "sdpa_forced_flash":
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        elif name == "sdpa_math":
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)

        loss = out.sum()
        loss.backward()
        q.grad.zero_();
        k.grad.zero_();
        v.grad.zero_()

    torch.cuda.synchronize()

    times = []
    torch.cuda.reset_peak_memory_stats()

    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if name == "flash_attention_2":
            out = flash_attn_func(q, k, v, causal=True)
        elif name == "sdpa_default":
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        elif name == "sdpa_forced_flash":
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        elif name == "sdpa_math":
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)

        loss = out.sum()
        loss.backward()
        q.grad.zero_();
        k.grad.zero_();
        v.grad.zero_()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"Mean fwd+bwd : {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"Peak memory  : {peak_gb:.2f} GB")


if __name__ == "__main__":
    print(f"Torch {torch.__version__} | CUDA {torch.version.cuda} | GPU {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision('high')

    benchmark("sdpa_math")  # slow baseline
    benchmark("sdpa_default")  # what HF "sdpa" actually uses
    benchmark("sdpa_forced_flash")  # forced native Flash via context manager
    benchmark("flash_attention_2")  # Dao's package (your current one)