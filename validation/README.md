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

These are overlapping suites. Synthetic fixtures validate software behavior, not useful model accuracy, Hailo compatibility or Raspberry Pi bootability. This initial baseline did not test hardware inference, licensed SDK compilation or Windows execution. Subsequent sections record those later checks, including physical 8L inference and settings preservation. Linux native desktop acceptance remains separate.

## Reproduce

```sh
python3.11 -m unittest discover -s tests -v
docker build -t pi-trainer-image-worker:local -f docker/image-worker.Dockerfile docker
docker run --rm --network none --cap-drop ALL --mount "type=bind,src=$PWD,dst=/app,readonly" -w /app pi-trainer-image-worker:local python -m unittest discover -s tests -v
```

Set `PI_TRAINER_TEST_DOCKER=1` on the host to enable its Docker image-backend test. Install the optional training dependencies to enable real CPU/ONNX tests. `native_smoke.py` requires a desktop session; `release_smoke.py` requires the locally built macOS console bundle.

## Subsequent board-free host checks — 10 September 2026

The portable `host_10h_smoke.py` completed on Mac ARM64 CPU, Mac Apple Metal, Linux ARM64 Docker CPU and GitHub-hosted Windows x64 CPU. All four verified ONNX/PyTorch output agreement, train-only calibration and disjoint splits, real LoRA training, adapter reload/merge and bounded generation. These used synthetic fixtures, not a supplied deployment checkpoint.

- [Windows model run and downloadable evidence](https://github.com/ShaunPrice/raspberry-pi-ai-trainer/actions/runs/34437019856): model harness passed. The workflow initially failed on a separate temporary-directory cleanup race, subsequently fixed by explicitly waiting for test-owned worker processes to exit.
- [Corrected cross-platform unit regression](https://github.com/ShaunPrice/raspberry-pi-ai-trainer/actions/runs/34437344850): all six OS/Python jobs passed. Windows Python 3.11 and 3.12 each ran 80 tests with eight documented platform/optional-dependency skips. This lightweight CI does not install the model frameworks; the separate Windows model run supplies that evidence.
- Local Mac framework-enabled core run: 80 tests, two skips, no failures. Linux Docker framework-enabled run: 80 tests, one skip, no failures.

No Hailo DFC SDK was installed in these model-test environments. Those host-only runs did not test numerical emulation, compilation or HEF device execution; the subsequent vendor SDK and physical 8L sections record those later results. A real deployment-model run awaits the model ID/checkpoint and any required base model. Windows native application packaging remains unverified. See [reproduction instructions](../docs/host-validation.md).

## Native Windows CPU and NVIDIA GPU — 10 September 2026

Source commit `5be79afe267cdfd5a6023bae47230e19c0623b55` was tested on a physical Windows 11 workstation through SSH, in a fresh isolated Python 3.12.10 environment. PyTorch 2.14.0+cu130 executed a CUDA matrix operation on an RTX 5090 Laptop GPU (compute capability 12.0). The full suite passed: 80 tests, seven platform/optional-feature skips, no failures.

Separate CPU and CUDA runs of `host_10h_smoke.py` passed vision training, ONNX comparison, train-only calibration, split separation, LoRA training, adapter reload/merge and generation. Maximum ONNX differences were 7.45e-9 (CPU) and 1.49e-8 (CUDA); maximum adapter merge differences were 6.71e-8 and 8.94e-8. Both generated eight tokens. These tiny synthetic fixtures establish pipeline operation, not useful model accuracy or comparative GPU performance. Downloaded evidence archives and model artifact hashes were verified; raw evidence remains local.

Ubuntu 24.04 x86-64 under WSL was also checked for architecture and GPU visibility. No Hailo SDK was found in the environments checked. This was a WSL preflight only; the Linux model tests above ran in the ARM64 Docker container. Successful PyTorch CUDA execution does not establish Hailo SDK GPU compatibility, numerical emulation, HEF compilation or 10H execution.

## Vendor SDK execution — 10 September 2026

User-supplied vendor Docker images were loaded into the physical Windows workstation's Linux x86-64 Docker engine. All three targets completed the real application compiler worker with a generated Conv/ReLU/pooling ONNX fixture and 64 synthetic calibration samples, followed by native and quantised SDK emulation:

| Target | DFC | HEF bytes | Native maximum absolute error vs NumPy | Quantised maximum absolute error vs NumPy |
|---|---|---:|---:|---:|
| Hailo-8L | 3.34.0 | 219,343 | 2.38e-7 | 0.001378 |
| Hailo-8 | 3.34.0 | 259,257 | 2.38e-7 | 0.001378 |
| Hailo-10H | 5.4.0 | 69,632 | 3.87e-7 | 0.001378 |

Both emulation modes returned finite outputs with the expected shape. Native results passed the independent NumPy tolerance; quantised error was measured without an application-quality threshold. HailoRT's offline `parse-hef` also reported the expected compatibility: HAILO8L, HAILO8, and HAILO15H/HAILO10H respectively. Downloaded archive and HEF/output hashes were verified. The tested compiler worker SHA256 was `2cbe1a9c7196e3dc67c6f81c48b96ef333b15237117d694952bcb25e4123dbe0`.

The first 10H run compiled successfully but exposed an emulation harness API mismatch. A complete fresh run passed after using the installed SDK's `ClientRunner.infer` API; the original failure evidence was preserved. All successful test containers exited normally and the SDK images remain available. No PCIe drivers were installed and no accelerator hardware was exercised.

These results supersede the earlier missing-SDK blocker for these synthetic vision fixtures. They do not qualify a deployment model or custom LLM. DFC 5.4 documents LoRA compilation from Hailo-provided pre-optimised GenAI HARs, but explicitly does not support GenAI numerical emulation. A matching candidate, vendor assets and adequate compute resources are still required. See [SDK setup and reproduction](../docs/hailo-sdk-setup.md). Proprietary SDKs, manuals and generated model artifacts remain local and excluded from Git.

## Physical Hailo-8L check — 10 September 2026

The generated 8L vision fixture was tested on a Raspberry Pi 5 with a physical HAILO8L accelerator and HailoRT/firmware 4.23.0. The DFC 3.34.0 HEF hash matched the compiler evidence, the installed runtime parsed it as HAILO8L, and `hailortcli run` completed 64 synthetic input frames with exit code zero. This establishes a real load/inference check for this fixture and runtime combination, not useful model accuracy or a representative throughput benchmark.

Before testing, 64 application/settings files were backed up on the Pi and a hash-verified backup copy was downloaded to the Mac. The existing detector's command, working directory and environment were saved privately. The detector was briefly stopped to release the accelerator and restarted by the test supervisor's cleanup block. All 64 file hashes remained unchanged. A subsequent check confirmed the detector owned the accelerator again, its stream port was listening, and the original application process and HailoRT service remained active. No driver, firmware or system configuration changes were made. Backups and detailed device evidence remain local and excluded from the public repository.


## Physical Hailo-8 check — 10 September 2026

T1’s vision Pi reported HAILO8 with HailoRT/firmware 4.24.0. The verified DFC 3.34.0 fixture HEF completed 64 synthetic frames with exit zero. An initial attempt using the default multi-process group was refused as unavailable; using the perception stack’s existing `SHARED` group succeeded. This validates fixture execution, not deployment-model accuracy or representative throughput.

Before testing, 129 tracked/application/runtime configuration files were archived privately on the Pi and a checksum-verified copy downloaded to the Mac. Every file hash remained unchanged afterwards. The vision service, HailoRT service and perception process retained their original PIDs and accelerator ownership throughout. No service was stopped, no driver/firmware/settings were changed, and no motor or autonomy commands were issued. No restoration was necessary because the original state was preserved. Private backups and raw evidence remain excluded from Git.
