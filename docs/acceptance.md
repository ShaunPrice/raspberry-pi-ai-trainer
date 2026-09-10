# Acceptance and evidence gates

Implemented, software-tested, SDK-tested, device-tested and release-tested are separate states. A registered worker or enrolled Pi is configuration, not proof of connectivity. Profiles are proposed defaults rather than detected hardware.

## Application acceptance

- Four interfaces share projects, immutable datasets, annotations and durable jobs.
- Reject Hailo-8/8L LLM projects and incompatible target/task specifications.
- Preserve source data and hashes; keep grouped sources together and training/validation/test sets distinct.
- Import/export data and capture bounded host-camera frames; capture failure must not invent frames.
- Launch actual provider processes, preserve logs and artifacts, expose cancellation and record interruption without automatic replay.
- Show dependency errors explicitly, including missing framework, compiler, recipe, Docker worker or SSH configuration.
- Reject cross-origin web mutations, archive traversal, symlinks, hash corruption and unsafe host arguments.
- Native controls remain responsive while jobs execute. CLI/MCP must not depend on an open window.

The original preparation baseline passed 20 tests. The expanded integrated suite currently runs 80 tests with three opt-in/dependency skips; later full-provider results belong in the validation record. Native Aqua windows and a real browser-submitted training job have passed. Do not add overlapping suite counts together.

## Host training and compilation

1. Train the implemented classifier against an immutable dataset and export real checkpoint/ONNX artifacts.
2. Train a compatible local causal model using LoRA; record held-out loss and checkpoint metadata without labelling them Hailo results.
3. Rank bounded optimisation candidates using validation quality and artifact size; do not select a winner from failed or ineligible trials.
4. Compile vision ONNX with representative calibration data using the installed target-compatible DFC.
5. For 10H LLMs, run the installed vendor-qualified recipe against its supported model family; require real target/artifact metadata and a HEF.
6. Record actual SDK/compiler versions and successful artifacts. Mocked process/contract tests do not satisfy SDK acceptance.

## 8L/8 physical end-to-end acceptance — pending

1. Record authorized SSH alias, Pi model/OS, accelerator identity, driver and runtime.
2. Run an official known-good model and retain actual inference/timing output.
3. Compile and load a target-specific custom HEF for each accelerator.
4. Reject incompatible target/runtime packages before selecting them.
5. Compare host and real Pi held-out quality; retain preprocessing/postprocessing configuration.
6. Exercise upload corruption/interruption, cancellation, activation failure and rollback.
7. Run sustained benchmarks with available thermal/memory measurements; identify metrics that are unavailable.
8. Customise an existing OS image, flash a separately selected medium and verify boot and inference.

Tested on Raspberry Pi 5 (8GB) with Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L) and Raspberry Pi AI HAT+ 26 TOPS (Hailo-8). Both completed the isolated synthetic load/inference check. The full deployment-helper activation/rollback workflow and other acceptance gates above remain separate.

## Existing image acceptance

- Preserve the input `.img` or `.img.xz`; never address a block device.
- Validate MBR/GPT boundaries and select a unique Linux filesystem partition.
- Validate deployment archive members and hashes before writing the payload.
- Inject model/scripts into the extracted filesystem; read back hashes and run consistency checks before assembling the output copy.
- Remove incomplete output images on failure; fail on insufficient filesystem space.
- Exercise both native Linux tools and the Mac-to-Docker path with real synthetic ext4 images.
- Verify the Imager catalogue binds the image hash and URI, and preparation never launches a writer; explicit Imager opening still selects no disk and writes no media.
- Distinguish filesystem verification from real Raspberry Pi OS boot. Service insertion stays disabled until account/runtime configuration is complete.

Synthetic ext4 acceptance has passed. Physical boot, actual runtime provisioning and Windows host execution remain separate tests.

## 10H physical acceptance — deferred until hardware arrives

Verify identity/runtime compatibility; load custom compiled vision and LLM models; compare task quality; measure time to first token, decode rate, context limits and sustained thermal behaviour; test release rollback and cold boot. Host execution and mocked protocol fixtures cannot substitute for these results.

## Distribution acceptance

The Mac application and companion CLI artifacts are built locally at `dist/native/Pi Trainer.app` and `dist/native/pi-trainer/`. They bundle the controller, with selected external training/SDK interpreters. Windows/Linux CI package definitions are present but have not run. Run the delivered application artifact on each supported Mac, Windows and Linux release. Verify native launch, browser/CLI/MCP operation, writable data placement, dependency error reporting and provider selection. Record signing/install status separately. A source package, CI matrix or successful packaging command alone does not establish installer usability.
