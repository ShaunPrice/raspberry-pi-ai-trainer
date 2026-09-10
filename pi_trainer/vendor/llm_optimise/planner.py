"""Transparent capacity estimates for conventional dense decoder transformers."""

from .config import finite, positive


def estimate_memory(
    parameters_b,
    weight_bits,
    layers,
    kv_heads,
    head_dim,
    context,
    *,
    concurrency=1,
    kv_bits=16,
    overhead_gib=0.5,
    unified=False,
    gpu_fraction=1.0,
):
    for key, value in dict(
        parameters_b=parameters_b,
        weight_bits=weight_bits,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        context=context,
        concurrency=concurrency,
        kv_bits=kv_bits,
    ).items():
        positive(value, key)
    finite(gpu_fraction, "gpu_fraction")
    finite(overhead_gib, "overhead_gib")
    if not 0 <= gpu_fraction <= 1 or overhead_gib < 0:
        raise ValueError("gpu_fraction must be in [0,1]; overhead must be nonnegative")
    weight = parameters_b * 1e9 * weight_bits / 8 / 1024**3
    kv = 2 * layers * kv_heads * head_dim * context * concurrency * kv_bits / 8 / 1024**3
    return {
        "weight_payload_gib": weight,
        "kv_payload_gib": kv,
        "overhead_allowance_gib": overhead_gib,
        "estimated_total_gib": weight + kv + overhead_gib,
        "estimated_device_gib": None
        if unified
        else (weight + kv) * gpu_fraction + (overhead_gib if gpu_fraction else 0),
        "memory_kind": "shared unified pool" if unified else "separate host and device pools",
        "evidence": "estimate only; not a fit guarantee",
        "assumptions": [
            "Dense transformer with uniform full attention; GQA uses KV heads, not query heads.",
            "Quantisation scales, mixed tensor dtypes, graphs, padding and allocator overhead may exceed the allowance.",
            "Does not model MoE, hybrid/SSM, MLA, sliding-window attention, training or file-backed host copies.",
            "Do not add Apple GPU allocation to system RAM: they share one physical pool.",
        ],
    }
