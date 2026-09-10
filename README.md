# Pi Trainer

A local workbench for training models off-Pi, compiling them for Hailo, packaging deployments, customising an existing Raspberry Pi OS image, and testing a Raspberry Pi over a trusted network connection. Native desktop, web, CLI and stdio MCP share one project store and durable job queue.

Hailo-8L and Hailo-8 support vision projects. Hailo-10H supports vision and LLM projects now; physical 10H qualification is deferred until hardware is available. The owner's Raspberry Pi 5 with Hailo-8L has passed a physical 64-frame synthetic inference test, with its original detector restored and all 64 backed-up file hashes unchanged. T1’s Hailo-8 also passed 64 synthetic frames using its existing sharing group, with 129 backed-up files unchanged and its perception services continuously running. Hailo-10H hardware remains untested.

Vision compilation and native/quantised SDK emulation passed for all three targets: DFC 3.34.0 for 8L/8 and DFC 5.4.0 for 10H. These checks validate synthetic fixtures, not deployment-model accuracy. See the [verification record](validation/README.md) and [SDK setup](docs/hailo-sdk-setup.md).

## Included capabilities

- Dataset imports/uploads, immutable snapshots, classification labels and group annotations, dataset export, and bounded host-camera capture.
- Executable off-Pi PyTorch classification training with ONNX export, and Transformers/PEFT LoRA fine-tuning for LLMs.
- Bounded host optimisation sweeps, saved-response quality scoring, training recipes and memory estimates reused from LLM-Optimise.
- Vision compilation through an installed compatible Hailo DFC; a separate explicit vendor-recipe contract for custom 10H LLM compilation.
- Deployment ZIPs with hashes and target/runtime metadata; SSH staging, release selection, rollback and explicit device benchmarks.
- Existing `.img`/`.img.xz` customisation: copy the image and add model/scripts, validate the filesystem and produce a checksummed `.img`.
- Persistent jobs, progress, logs and cancellation; registered Linux SSH workers for training, optimisation, compilation and image work.

**Implemented providers still need their real dependencies.** Training frameworks, licensed Hailo SDKs, compatible base models, Docker and Pi runtimes are not bundled. Missing dependencies fail explicitly. A 10H compiler recipe must be qualified for the selected model; there is no generic GGUF-to-HEF conversion. Host tests do not establish hardware performance or bootability.

## Get started

```sh
git clone https://github.com/ShaunPrice/raspberry-pi-ai-trainer.git
cd raspberry-pi-ai-trainer
python3.11 -m pi_trainer --data-dir .trainer web
```

Open the loopback address printed by the server. See [training setup](docs/training.md) for optional frameworks and compiler environments.

To build native application bundles on your own operating system:

```sh
python3.11 -m pip install pyinstaller==6.22.2
python3.11 packaging/build.py
```

On macOS the output is `dist/native/Pi Trainer.app`, with a companion CLI at `dist/native/pi-trainer/pi-trainer`. The convenience `.command` launchers expect these locally built bundles and a training environment at `.venv-training`. Build outputs, trained models and local data are not committed. Native bundles contain the controller; training frameworks and vendor SDKs use separately selected Python environments. Developer ID signing/notarization is not configured.

## Run from source

Requires Python 3.11+. The core uses the standard library; the native desktop needs Tk. Compute providers use separately selected Python environments. From this directory:

```sh
python3.11 -m pi_trainer --data-dir .trainer web
python3.11 -m pi_trainer --data-dir .trainer desktop
python3.11 -m pi_trainer --data-dir .trainer doctor
```

The web server prints its loopback address, normally `http://127.0.0.1:8765`. Keep it local: remote authenticated web hosting is not included. Use the same absolute `--data-dir` across all four interfaces to share projects and jobs. On Windows use `py -3.11`; on Linux use the installed Python 3.11+ executable and install Tk if packaged separately. Signed/notarized distribution and Windows/Linux installers still require separate release verification.

Create a project and import real data:

```sh
python3.11 -m pi_trainer --data-dir .trainer create --name "Workshop vision" --target hailo8l --task vision
python3.11 -m pi_trainer --data-dir .trainer import PROJECT_ID /absolute/path/to/class-folders
python3.11 -m pi_trainer --data-dir .trainer datasets PROJECT_ID
```

For LLM work, create `--target hailo10h --task llm` and import reviewed Alpaca JSONL. Files under `examples/` are synthetic smoke fixtures, not training-quality datasets.

## Run and inspect jobs

The native workflow dialog and browser expose editable job specifications. CLI uses the same provider contract. Save a training specification as `train.json`, selecting a Python environment with the required frameworks:

```json
{"engine":"pytorch","python":"/absolute/path/to/training-python","epochs":2,"image_size":32,"batch_size":8}
```

```sh
python3.11 -m pi_trainer --data-dir .trainer run PROJECT_ID train --spec train.json --wait
python3.11 -m pi_trainer --data-dir .trainer workflow-jobs
python3.11 -m pi_trainer --data-dir .trainer job JOB_ID
python3.11 -m pi_trainer --data-dir .trainer logs JOB_ID
python3.11 -m pi_trainer --data-dir .trainer cancel JOB_ID
```

Without `--wait`, submission returns a durable job ID and the worker continues independently of the interface. A stopped worker is recorded as interrupted; external operations are not silently replayed. Available kinds are `train`, `compile`, `package`, `image`, `deploy`, `benchmark`, `rollback`, `capture`, `export_dataset` and `optimise`.

Use `invoke OPERATION --arguments args.json` for the shared operations, including device/worker enrollment. Configure trusted SSH aliases and host keys first. Enrollment records configuration; it is not a successful connection test. The dependency inspector similarly reports installed tools, not proven functionality.

## Existing OS images

Image work **adds a deployment to a copy of an existing OS image**. It does not generate a distribution, resize partitions, install Hailo packages or erase media. Use an image with sufficient free space and the matching runtime. Mac/Windows require the Linux Docker worker:

```sh
docker build -t pi-trainer-image-worker:local -f docker/image-worker.Dockerfile docker
```

Submit an `image` job with `{"base_image":"/path/raspios.img.xz","bundle_path":"/path/deployment.zip"}`. Prepare an offline Imager catalogue or explicitly open Raspberry Pi Imager from the workbench. Neither action selects storage or writes media; choose and confirm the removable target inside Imager. See [image setup and verification](docs/images.md).

## MCP

Configure a local MCP client with an absolute interpreter and shared data directory:

```json
{
  "mcpServers": {
    "pi-trainer": {
      "command": "/absolute/path/to/python3.11",
      "args": ["-m", "pi_trainer", "--data-dir", "/absolute/path/to/trainer-data", "mcp"],
      "env": {"PYTHONPATH": "/absolute/path/to/Raspberry Pi AI 2+ Trainer"}
    }
  }
}
```

Alternatively install with `python -m pip install .` in a chosen environment. MCP exposes the shared operations over stdio; it does not expose an unrestricted terminal. Hosted MCP HTTP/OAuth is outside this delivery.

## Documentation and evidence

- [Architecture and provider boundaries](docs/architecture.md)
- [Training and compiler setup](docs/training.md)
- [Host validation without a 10H board](docs/host-validation.md)
- [Deployment and Pi helper](docs/deployment.md)
- [Existing image customisation](docs/images.md)
- [LLM-Optimise reuse](docs/llm-optimise-reuse.md)
- [Acceptance gates](docs/acceptance.md) and [verification record](validation/README.md)

```sh
python3.11 -m unittest discover -s tests -v
```

The integrated suite currently runs 80 tests with 3 dependency/opt-in skips. Real synthetic vision training, local tiny-model LoRA training, ONNX export, native Aqua workflow windows and a browser-submitted durable training job have passed. Synthetic ext4 injection has passed in Linux Docker and through the Mac-to-Docker path. See the verification record for fixture limitations and subsequent run updates. Subsequent hosted CI and CPU-model tests passed on Windows; Mac CPU/Metal and Linux Docker host-model checks also passed. See the verification record for the exact runs and fixture limitations. Windows native packaging, real licensed Hailo compilation, network workers and physical Pi inference still require qualification.

For registered Linux workers, the registered Python runs the controller and needs Python 3.11+. Set `provider_python` in the job specification to a separate absolute worker-local SDK/training Python path, for example `/opt/hailo-sdk/bin/python`. This permits vendor SDKs requiring an older Python without changing the controller. If omitted, the registered worker Python is used. Local jobs use `python`.

## License

[MIT](LICENSE), copyright 2026 Shaun Price. Reused LLM-Optimise utilities retain their [MIT license and attribution](pi_trainer/vendor/llm_optimise/LICENSE). Third-party frameworks, vendor SDKs, models and operating system images retain their respective licenses.
