# Smoke fixtures

`llm/workshop.jsonl` contains ten original synthetic records for testing import and split plumbing. It is not a useful training corpus or a quality benchmark. No model weights are included.

`llm/alpaca.jsonl` is synthetic instruction/output data for recipe preparation. `tasks.jsonl` and `outputs.json` form a two-task scoring fixture with supplied correct outputs; a 100% score on these is not a model-quality result.
