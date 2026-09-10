"""Schema-derived Soup recipes; heavy training stacks stay optional and isolated."""

from pathlib import Path

from .config import positive
from .runner import write_json


def training_recipe(engine, model, data, output, *, max_length=512, rank=8):
    if engine not in ("soup-stream", "soup-qlora", "soup-mlx"):
        raise ValueError("engine must be soup-stream, soup-qlora or soup-mlx")
    for value in (model, data, output):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("model, data and output must be nonempty strings")
    positive(max_length, "max_length", integer=True)
    positive(rank, "rank", integer=True)
    training = {
        "epochs": 1,
        "lr": 0.0002,
        "batch_size": 1,
        "lora": {"r": rank, "alpha": rank * 2},
        "quantization": "4bit",
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 1,
    }
    config = {
        "base": model,
        "task": "sft",
        "backend": "mlx" if engine == "soup-mlx" else "transformers",
        "data": {"train": data, "format": "alpaca", "val_split": 0.1, "max_length": max_length},
        "training": training,
        "output": output,
    }
    notes = [
        "Recipe prepared from upstream Soup schema; training has not run.",
        "Use Python 3.10–3.12 in a separate Soup environment. Retain a held-out task test set.",
        "Measure peak memory and task quality after export; fitting training is not proof of inference quality.",
    ]
    if engine == "soup-stream":
        training.update(stream_layers=True, stream_source="auto", stream_buffers=2)
        notes += [
            "Layer streaming is beta. CUDA is the primary path; MPS is experimental; MLX cannot combine with stream_layers.",
            "RAM/NVMe offload trades transfer time for VRAM. Validate against a resident control when possible.",
        ]
    elif engine == "soup-mlx":
        notes += [
            "Apple Silicon MLX SFT; use a compatible pre-quantised MLX model for 4-bit training.",
            "MLX resume restores adapter weights, not the optimiser state.",
        ]
    return {"config": config, "notes": notes}


def save_recipe(recipe, path):
    path = Path(path).resolve()
    # JSON is valid YAML: no PyYAML needed in the lightweight core.
    write_json(path, recipe["config"])
    return {**recipe, "path": str(path), "command": ["soup", "train", "--config", str(path)]}
