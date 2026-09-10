# Testing 10H-targeted models without a board

The portable harness exercises real host training and inference using a synthetic colour classifier and a locally generated tiny random GPT2 model. It verifies ONNX output against the reloaded PyTorch checkpoint, calibration shape/range, dataset split separation, LoRA reload, changed logits, adapter merge equivalence and bounded generation.

A passing fixture validates the host pipeline. It does not establish useful model quality, Hailo quantisation accuracy, a valid compiled HEF, accelerator performance or device memory fit. A deployment candidate requires its own pinned checkpoint and representative held-out evaluation data.

## Mac or Windows Python

Install the optional training dependencies described in [training setup](training.md), then run:

```sh
python validation/host_10h_smoke.py --output validation/host-test-1 --device cpu
```

On a compatible Mac, `--device mps` tests Apple Metal. With a compatible CUDA-enabled PyTorch installation, `--device cuda` tests NVIDIA GPU training. Use an empty output directory for every run so previous evidence remains intact. The generated report explicitly separates successful host checks from unavailable vendor and hardware gates. Native Windows CPU and RTX 5090 Laptop CUDA runs are recorded in the [verification record](../validation/README.md).

The manual GitHub workflow `Hailo-10H Windows host validation` uses a standard Windows runner, installs CPU dependencies, runs the full suite and the portable harness, and uploads results. POSIX-only tests for the Pi-side release lock are skipped on Windows; Windows host transport tests remain enabled. Hosted Windows results do not establish access to a user's Windows workstation or validate its GPU configuration.

## Linux Docker

```sh
docker build -t pi-trainer-host-validation -f docker/host-validation.Dockerfile docker
mkdir -p validation/linux-retest
docker run --rm --network none --cap-drop ALL --memory 4g --cpus 2 \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --mount "type=bind,src=$PWD/validation/linux-retest,dst=/output" \
  pi-trainer-host-validation \
  python validation/host_10h_smoke.py --output /output/model-test --device cpu
```

The build downloads framework dependencies; execution is offline and does not access a camera. Container architecture follows the Docker host: ARM64 Linux tests on Apple Silicon are not an x86-64 Hailo compiler environment.

## Hailo emulation and compilation

These require a compatible Hailo DFC installation and supported model-specific flow. They are marked blocked when absent. Hailo's vision Model Zoo documents numerical emulation; that does not establish a universal full-pipeline LLM emulator. Running the compiled GenAI HEF and measuring device performance requires access to a 10H.

- [Hailo Model Zoo evaluation](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/GETTING_STARTED.rst)
- [Hailo GenAI prerequisites](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/USAGE.rst)
