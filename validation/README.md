# Verification record

The following checks were recorded on 10 September 2026. Raw logs, screenshots, environment listings and local state are retained in the development workspace and excluded from the public repository because they contain machine-specific paths. The test sources and reproducible smoke scripts are published.

| Check | Recorded result | Scope |
|---|---|---|
| macOS core suite | 80 tests, 3 skipped | Optional providers disabled |
| macOS provider suite | 80 tests, 1 skipped | Real CPU vision training and Docker ext4 insertion |
| Linux container suite | 80 tests, 2 skipped | Core operations and direct ext4 insertion |
| Native Aqua | Project creation, LLM and workflow windows passed | Tk 9.0.3 |
| Standalone Mac console/MCP | Detached export and 29 MCP tools passed | Tested outside source directory |
| Vision training | Real checkpoint, ONNX and calibration outputs | Synthetic colour fixtures only |
| LLM training | Real LoRA training and adapter reload | Locally generated tiny random GPT2 fixture |
| Video capture | Three PNG frames verified | Generated local video; no camera accessed |
| Browser | Real training and optimisation jobs, no console errors | Synthetic data; responsive layout checked |
| Image customisation | Injection, filesystem checks and hash readback passed | Synthetic ext4, original image unchanged |
| Mac archive signature | Extracted archive passed strict ad-hoc verification | Not Developer ID signed or notarized |

These are overlapping suites. Synthetic fixtures validate software behavior, not useful model accuracy, Hailo compatibility or Raspberry Pi bootability. Hardware inference, licensed SDK compilation, real SSH workers, Windows execution and Linux native desktop acceptance remain unverified by these local records. Hosted CI reports its own current results separately.

## Reproduce

```sh
python3.11 -m unittest discover -s tests -v
docker build -t pi-trainer-image-worker:local -f docker/image-worker.Dockerfile docker
docker run --rm --network none --cap-drop ALL --mount "type=bind,src=$PWD,dst=/app,readonly" -w /app pi-trainer-image-worker:local python -m unittest discover -s tests -v
```

Set `PI_TRAINER_TEST_DOCKER=1` on the host to enable its Docker image-backend test. Install the optional training dependencies to enable real CPU/ONNX tests. `native_smoke.py` requires a desktop session; `release_smoke.py` requires the locally built macOS console bundle.
