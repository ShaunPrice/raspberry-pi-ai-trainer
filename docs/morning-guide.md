# Start here in the morning

The workbench now runs real host training and durable workflows. Double-click **`Open Pi Trainer.command`** to open the packaged Mac app with the existing demo projects and configured training Python. Alternatively open **`dist/native/Pi Trainer.app`** with its separate default application data directory and choose the training/deployment workflow window. The native window, web interface, CLI and MCP share the same storage when given the same data directory.

## Quick start on this Mac

From the project directory, the companion executable can inspect dependencies or start the web interface:

```sh
./dist/native/pi-trainer/pi-trainer --data-dir .trainer doctor
./dist/native/pi-trainer/pi-trainer --data-dir .trainer web --port 8876
```

Open the loopback address printed by the server. If another workbench is already serving that port, use that existing window or choose another port.

The desktop/CLI bundles contain the controller. For training, select the prepared interpreter at `.venv-training/bin/python` using its absolute path in the job specification. This environment contains the locally tested frameworks; the packaged controller's internal interpreter is not the training environment. SDK compilation can use a separate compatible interpreter on a registered Linux worker.

## First useful workflow

1. Create a Hailo-8L or Hailo-8 vision project. Import reviewed images arranged in class folders, or upload files and add labels/groups. Use source/session groups for related images to prevent splitting adjacent frames across training and test sets.
2. Select `train`, load its template, set the external Python path, then run explicitly. The current vision trainer is a small classifier. Monitor status, logs and result artifacts; closing a UI does not cancel its durable job.
3. Review the exported ONNX, calibration data and host validation/test metrics. The existing synthetic smoke projects demonstrate plumbing, not useful recognition quality.
4. Configure the compatible licensed Linux Hailo DFC environment, then run `compile` with the ONNX and representative calibration `.npy`. On Apple Silicon, use a registered compatible Linux compiler worker.
5. Package the resulting target-specific HEF and reviewed scripts with the **actual** installed Pi runtime version. Enroll a trusted SSH alias before deploying or benchmarking.
6. For an OS image, select `image` with an existing `.img`/`.img.xz` and deployment ZIP. It copies the image and inserts model/scripts. Prepare/open Imager for a separate user-confirmed media write.

CLI job example after saving a specification to `train.json`:

```sh
./dist/native/pi-trainer/pi-trainer --data-dir .trainer run PROJECT_ID train --spec train.json --wait
./dist/native/pi-trainer/pi-trainer --data-dir .trainer workflow-jobs
./dist/native/pi-trainer/pi-trainer --data-dir .trainer logs JOB_ID
```

Registered Linux workers support training, optimisation, compilation and image customisation. Registration does not establish a successful SSH connection; dependency discovery likewise reports installation rather than actual SDK operation.

## Hailo-10H path

Create a 10H LLM project, import reviewed Alpaca JSONL and select `transformers-peft` with a compatible local model directory. LoRA training is implemented and has run against a locally generated tiny fixture model. That fixture is not a useful language model.

A custom Hailo LLM build still requires a vendor-qualified recipe for the exact model family. The adapter deliberately fails if that recipe is missing; it does not manufacture a HEF from GGUF or an arbitrary adapter. Physical 10H inference remains deferred until you have the hardware.

## What was actually checked

The core suite ran 80 tests with three skips; the training/Docker-enabled Mac suite ran 79 with one Linux-only skip, and the Linux container suite ran 79 with two optional-provider skips. Native Aqua project creation and LLM/workflow windows passed. A browser-submitted synthetic vision training job completed with real ONNX output and no browser console errors. Real filesystem insertion passed in Linux Docker and through Mac-to-Docker. The extracted Mac release archive passed strict ad-hoc code-signature verification; it is not Developer ID signed or notarized. See [validation evidence](../validation/README.md) for the exact fixture limitations and later full-provider runs.

Licensed vision compilation and native/quantised SDK emulation have now passed for 8L, 8 and 10H on the Omen. A live Pi 5/Hailo-8L completed 64 synthetic inference frames, followed by detector restart and verification of unchanged backed-up settings. T1’s Hailo-8 also passed 64 synthetic frames with its existing perception processes and 129 backed-up files unchanged. The application's registered-worker workflow and customised OS image boot remain separate unverified gates. See the [current validation record](../validation/README.md). Windows/Linux package builds are configured but have not run. The image worker does not resize partitions or install the Hailo runtime: use a base image with the matching packages and enough free filesystem space. Imager preparation/opening never selects or writes a disk.

For registered Linux workers, the registered Python runs the controller and needs Python 3.11+. Set `provider_python` in the job specification to a separate absolute worker-local SDK/training Python path, for example `/opt/hailo-sdk/bin/python`. This permits vendor SDKs requiring an older Python without changing the controller. If omitted, the registered worker Python is used. Local jobs use `python`.
