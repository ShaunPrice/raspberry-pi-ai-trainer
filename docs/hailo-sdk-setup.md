# Local Hailo SDK setup

The proprietary Hailo SDK is installed separately from the MIT-licensed workbench. Keep downloaded archives and drivers in `Hailo_SDK/` or `Hailo-SDK/`; both directories are excluded from Git.

## Supplied packages

The September 2026 setup uses these user-supplied downloads:

| Package | Intended role |
|---|---|
| `hailo8_ai_sw_suite_2026-07_docker.zip` | DFC 3.34.0 / Model Zoo 2.19.0 / HailoRT 4.24.0; 8/8L targets accepted |
| `hailo_ai_sw_suite_2026-08_docker.zip` | DFC 5.4.0 / Model Zoo 5.4.0 / HailoRT 5.4.0; `hailo10h` target accepted |
| `hailort-pcie-driver_4.24.0_all.deb` | HailoRT 4.x PCIe kernel driver |
| `hailort-pcie-driver_5.4.0_all.deb` | HailoRT 5.x PCIe kernel driver |

Archive release dates and driver versions do not establish the embedded compiler version. Record the actual package versions, image digest and successful target checks before selecting a worker.

## Compiler host

Use a Linux x86-64 Docker engine. Windows Docker Desktop can supply this engine even when Docker integration is disabled inside a particular WSL distribution. The Apple Silicon Docker host used for the earlier Linux training tests is ARM64 and is not the native compiler host.

Load each vendor image separately and retain separate environments for the compiler families. For board-free CPU validation, use a container with only the test source and output directory mounted. The vendor desktop launcher also enables device access and broad host mounts; those are unnecessary for this headless compiler check.

PCIe drivers are for a host with an attached supported accelerator. They are not required to parse, optimise or compile a model without a board. Driver installation and physical Hailo execution are separate from compiler setup.

A successful vision fixture compilation does not qualify custom LLM compilation. A 10H LLM still needs a supported checkpoint, export/quantisation recipe and the appropriate GenAI compiler flow.

## Verified environment discovery

The 2026-08 image loaded on the x86-64 Windows Docker engine with image ID `sha256:22ddb630a4913ed8ea1bfaf34110f0f7e7688964e8a85df75915ce8f57943c31`. Its SDK interpreter is `/local/workspace/hailo_virtualenv/bin/python` (Python 3.10.12). DFC 5.4.0 accepts `hailo10h`; it rejects `hailo8` and `hailo8l` and directs those targets to DFC 3.x. The 2026-07 8-family image has ID `sha256:962aeda88f612f7fdf9e42cd65ffd883dbd5cd4a9449968718b9a05744f49416`, Python 3.10.12 and DFC 3.34.0. It accepts 8/8L and rejects 10H, directing it to DFC 5.x. Target discovery alone is not compilation evidence; see the [validation record](../validation/README.md) for actual test results.

The headless fixture runs as the image's non-root user, with no network, all capabilities dropped, no new privileges, four CPUs and a 24 GiB memory limit. Test source is read-only; only the dedicated result directory is writable.

## GenAI boundary confirmed in DFC 5.4.0

The locally supplied DFC 5.4.0 User Guide, sections 4.8 and 6, documents LoRA compilation starting from a Hailo-provided pre-optimised GenAI HAR. The workflow loads matching Safetensors adapter weights, applies the model's scripts and per-adapter calibration, optimises, and compiles a HEF. It does not support arbitrary custom base models. The tutorial's adapter settings and target modules are model-specific qualification inputs; the workbench's generic LoRA defaults are not proof of compatibility.

The same guide explicitly excludes GenAI from every DFC numerical-emulation mode. Vision emulation results must not be presented as LLM emulation. GenAI execution requires the target hardware and a matching HailoRT version.

The guide specifies substantial GenAI resources: at least 64 GB RAM and 200 GB disk, with a 24 GB NVIDIA GPU; release notes warn that optimisation can require up to 160 GB RAM. The current Docker engine exposes about 47 GiB memory. A small vision fixture passing within a 24 GiB limit does not clear this GenAI resource requirement. Select the deployment checkpoint, matching vendor HAR, calibration data and compatible compute resources before attempting that build.

The licensed guides and SDK assets remain local and are not redistributed in this repository.

## Reproduce the vendor fixture

After loading the vendor image into a Linux x86-64 Docker engine, create an empty output directory and run the published `validation/hailo_sdk_smoke.py` inside that SDK image, with these arguments:

```sh
python /source/hailo_sdk_smoke.py --target hailo10h \
  --compiler-source /source/compiler.py --output /results/hailo10h-run-1
```

Mount only `validation/hailo_sdk_smoke.py` and `pi_trainer/compiler.py` into `/source` read-only, and the dedicated output directory at `/results` read-write. Use the SDK interpreter identified above. For 8/8L, choose the separate 8-family image and corresponding `--target`. The script requires an empty result directory and defaults to a 20-minute limit per compiler/emulation stage. It uses CPU execution and fixture-only optimisation settings, writes the real HEF and HAR files, and checks native emulation against an independent NumPy reference. Quantised differences are reported without treating them as application accuracy.
