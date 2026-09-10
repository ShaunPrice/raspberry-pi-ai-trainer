# LLM-Optimise reuse

The existing source and MIT licence were inspected on 10 September 2026. Three small modules are copied verbatim with attribution, source commit and SHA-256 records in `pi_trainer/vendor/llm_optimise/PROVENANCE.json`. This is bounded source reuse, not a network connection to a running LLM-Optimise application. The source project remains unchanged.

| Capability | Current implementation | Boundary |
|---|---|---|
| MLX, QLoRA and layer-streamed host recipes | Reused `training.py` saves Soup-compatible configuration | These recipes do not launch Soup; model and engine are external |
| Saved-response task-quality scoring | Reused `quality.py`: exact, JSON, JSON subset and numeric gates | Scores supplied responses, not live inference; JSON-schema gates are rejected in this subset |
| Dense-transformer memory estimate | Reused `planner.py` with Hailo-specific caveat | Generic inference payload estimate, not accelerator allocation or training-memory prediction |
| Host training execution | Project-owned PyTorch classifier and Transformers/PEFT LoRA workers | Real framework/model dependencies required; no Hailo runtime result implied |
| Group-aware dataset separation | Project-owned immutable annotation revisions and group splits | Requires correct user-provided group identity; no semantic duplicate detector |
| Bounded optimisation | Project-owned real training sweeps with validation-quality/artifact-size Pareto selection | Host metrics only; Hailo latency and HEF size require compilation/device evidence |
| Process supervision and remote compute | Project-owned durable queue, cancellation/logs and Linux SSH workers | Registration is not a connection test; no unattended retries of external writes |
| Native packaging / remote MCP | Native Tk application and stdio MCP | Standalone packages require release validation; no HTTP/OAuth MCP service |

The reused utilities are available through the shared service and interfaces. `Store.llm_recipe` snapshots supplied Alpaca JSONL and passes only training records into the generated recipe; designated validation/test records remain excluded. This recipe path is distinct from executable `train` jobs.

For executable Hailo-10H LLM host fine-tuning, choose a `train` job with `engine: "transformers-peft"`, a local compatible model directory, selected Python environment and reviewed dataset. Downloads are disabled by default. The resulting adapter still needs a vendor-qualified model-specific export/compile recipe before it can become a Hailo deployment. Neither a generic adapter nor a GGUF file proves HEF compatibility.

Examples for the retained utilities:

```sh
python3.11 -m pi_trainer --data-dir .trainer llm-recipe PROJECT_ID --engine soup-mlx --model /path/to/compatible-model --data examples/llm/alpaca.jsonl
python3.11 -m pi_trainer --data-dir .trainer score examples/llm/tasks.jsonl examples/llm/outputs.json
```

These synthetic examples verify recipe/scoring plumbing only. A score of 1.0 on supplied correct fixture responses is not a trained-model benchmark. Direct `python -m pi_trainer.llm_tools recipe ...` remains a utility that expects an already-separated training file.
