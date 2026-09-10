# Pi Trainer user guide

**Mac, Windows and Linux · September 2026**

This guide takes you from a new project to a trained model, a Hailo build, a deployment package and an existing Raspberry Pi OS image containing your model and scripts. It also explains network testing and how to repeat the same work through the command line or an MCP client.

Pi Trainer does training on your workstation or a registered Linux worker. The Raspberry Pi runs the finished application and inference tests. You can use the native application, browser, CLI and MCP together: they share projects when you select the same data directory.

The built-in vision trainer is an image **classifier**. It learns one class for each image. Detection, segmentation and other architectures can be trained elsewhere and imported as ONNX for a compatible Hailo compilation recipe. The language-model trainer produces LoRA adapters using Transformers/PEFT. A deployable Hailo-10H LLM additionally needs a supported vendor base model, its matching pre-optimised HAR and a qualified compilation recipe.

## 1. Choose your target and first exercise

| Your accelerator | Project target | Built-in training route |
|---|---|---|
| Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L) | `hailo8l` | Image classification |
| Raspberry Pi AI HAT+ 26 TOPS (Hailo-8) | `hailo8` | Image classification |
| Raspberry Pi AI HAT+ 2 (Hailo-10H) | `hailo10h` | Image classification or LLM LoRA |

For your first session, use the included `examples/vision-smoke` dataset with a vision project. The colour images make it easy to check the training workflow without downloading model weights. Treat the resulting accuracy as a demonstration of the pipeline, not a benchmark for real photographs.

The verification record includes successful synthetic vision compilation and native/quantised SDK emulation for all three targets. Physical checks were performed on **Raspberry Pi 5 (8GB)** with **Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L)** and **Raspberry Pi AI HAT+ 26 TOPS (Hailo-8)**; both completed 64 synthetic inference frames. The complete deployment-helper activation/rollback sequence is a separate test. Hailo-10H hardware testing remains pending. Consult [the verification record](../validation/README.md) for the exact evidence.

<!-- SCREENSHOT: 01-workbench-overview.png | Browser overview with a selected demonstration project. -->

## 2. Install the controller and training environment

### Get the application

Download the source repository or clone it:

```sh
git clone https://github.com/ShaunPrice/raspberry-pi-ai-trainer.git
cd raspberry-pi-ai-trainer
```

Use Python 3.11 or newer for the controller. The commands below use `python`; replace that with your selected interpreter if necessary. On Windows, `py -3.11` can create the environment; on Mac or Linux, `python3.11` is commonly available. You do not need to install the Hailo compiler merely to open the workbench or train the sample classifier.

Create a controller environment:

```sh
python -m venv .venv
```

On Mac/Linux:

```sh
.venv/bin/python -m pip install .
.venv/bin/python -m pi_trainer --data-dir .trainer doctor
```

On Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\python.exe -m pi_trainer --data-dir .trainer doctor
```

Keep the source folder in place when following this guide. Select an absolute data-directory path for normal use, particularly when launching from different applications. Relative `.trainer` means a directory beneath the current working directory; launching from another folder would select a different store.

### Add training libraries separately

Create a second environment so training dependencies remain separate from the controller:

```sh
python -m venv .venv-training
```

On Mac/Linux:

```sh
.venv-training/bin/python -m pip install torch numpy pillow onnx
.venv-training/bin/python -m pip install transformers peft safetensors
.venv-training/bin/python -m pip install opencv-python-headless
```

On Windows, replace `.venv-training/bin/python` with `.venv-training\Scripts\python.exe`. The first installation supports vision training. Transformers/PEFT are for LLM training; OpenCV is for explicit camera/video capture jobs. Install only what you need.

Record the **absolute path** to the training Python. It goes in the workflow's `python` field. The controller's environment check may suggest a nearby training environment, but inspect the generated specification before running it.

CPU training is the default. Use `"device":"mps"` for a supported Apple Silicon setup or `"device":"cuda"` for a correctly installed NVIDIA PyTorch environment. CUDA installations depend on your driver and PyTorch build. Start with the small CPU exercise before attempting a larger model.

### Native application setup

The native application needs Tk. If `python -m tkinter` cannot open its test window, use a Tk-enabled Python distribution or install your operating system's corresponding Tk package. The browser and CLI can still run without Tk.

To build a local native bundle:

```sh
python -m pip install pyinstaller==6.22.2
python packaging/build.py
```

On Mac, the application is produced at `dist/native/Pi Trainer.app`. Native bundles contain the controller; they do not embed the training frameworks or proprietary SDKs. Signing/notarisation and platform packaging status are described in the release documentation. Source-mode launching is the consistent fallback across all three operating systems.

## 3. Open the interface you prefer

### Browser

From the repository, run the controller interpreter with:

```sh
python -m pi_trainer --data-dir /absolute/path/to/trainer-data web
```

Open the printed loopback address, normally `http://127.0.0.1:8765`. If another application uses that port, add `--port 8876`. Keep the server local; this release does not provide remote authenticated web hosting.

The sidebar holds your projects. The main page includes **Create a project**, **Import training data**, build readiness, LLM preparation tools, the dataset workspace, workflow controls and the job list. Open the expandable sections for device/worker enrollment, Raspberry Pi Imager and project scripts.

### Native desktop

```sh
python -m pi_trainer --data-dir /absolute/path/to/trainer-data desktop
```

The main window provides **Create project**, **Import folder**, **Import file**, **Inspect plan**, **Export bundle**, **LLM tools**, **Training & deployment**, **Refresh** and **Raspberry Pi Imager**.

Select **Training & deployment** to open its three tabs: **Workflows**, **Devices and workers**, and **Dependencies**. In Workflows, select a project and workflow, click **Load template**, edit the JSON, and use **Choose file** or **Choose folder** to fill a selected path field. **Run job** submits the work. The lower area shows **Recent jobs**, **Progress and result**, and **Logs**.

<!-- SCREENSHOT: 02-native-workflows.png | Native Workflows tab with a reviewed training specification. -->

### Understand job status

A submitted job has an ID and persistent state. `queued` means it is waiting to start; `running` means the worker is active. Terminal states include `succeeded`, `failed`, `cancelled` and `interrupted`.

The browser's **Workflow jobs** section lets you select a job, **Read selected logs**, **Refresh jobs** or **Cancel selected job**. The native dialog has equivalent refresh and cancellation controls. Closing an interface does not normally cancel a detached worker. A worker stopped unexpectedly is marked interrupted; external operations are not silently replayed.

A successful result includes artifact paths and measured information. Read the result before proceeding. **Inspect build plan** describes readiness; it does not execute training or compilation. **Export preparation bundle** packages project preparation information; it is different from the deployable model package created later.

## 4. Create a vision project and import data

In the browser, enter a name such as **Colour classifier practice**, select your accelerator, choose **Computer vision**, and click **Create project**. In the native application, enter the equivalent name, target and `vision` task, then create the project.

Import the absolute path to `examples/vision-smoke`. In the browser use **Local dataset path** followed by **Import dataset**. In the desktop use **Import folder**. Select the project before importing.

The import saves an immutable snapshot, hashes its files and assigns training, validation and test splits. Save its dataset ID from the operation result. A later edit to your original files does not alter that snapshot. Import again when the source changes.

For your own classification data, use a class-folder layout:

```text
inspection-images/
  acceptable/
    item-001.png
    item-002.png
  damaged/
    item-003.png
    item-004.png
```

Choose simple class names containing letters, numbers, hyphens or underscores. Collect enough examples to put every class into all three splits. A large number of nearly identical frames is less useful than varied examples from different sessions and conditions.

The workbench assigns splits deterministically using content hashes, approximately 80% training, 10% validation and 10% test over a sufficiently large dataset. Small datasets may be uneven. Review the snapshot's actual counts rather than assuming those percentages. Training fails clearly when split coverage is incomplete.

The browser also provides **Upload images or JSONL from your browser** and **Upload and snapshot**. Its per-file limit is 12 MiB. Local folder import is preferable for classification because it preserves class folders. Individual browser uploads may need labels assigned afterwards.

<!-- SCREENSHOT: 03-dataset-snapshot.png | Imported dataset result showing IDs and split counts. -->

## 5. Label examples and keep related data together

Use source/session groups when images share a subject, production batch, video sequence or acquisition session. Related records should stay in one split; otherwise a model can appear accurate because its test images closely resemble its training images.

Prepare an annotation JSON file mapping image-relative paths or record hashes to a label and/or group:

```json
{
  "acceptable/item-001.png": {"label":"acceptable","group":"session-A"},
  "acceptable/item-002.png": {"label":"acceptable","group":"session-A"},
  "damaged/item-003.png": {"label":"damaged","group":"session-B"}
}
```

In **Collect, label and version data**, enter the dataset ID and the annotation file's absolute path, then click **Save annotation version**. A new dataset version is created. Groups are deterministically assigned to a split, so check the new split counts and class coverage afterwards. Add more independent groups if a split is missing a class; do not separate related samples just to make the counts look balanced.

For JSONL, use record hashes to identify individual rows when one file contains multiple records. A filename annotation can apply to multiple rows from that file.

A queued training job pins its dataset version. To select a particular snapshot explicitly, add `"dataset_id":"YOUR_DATASET_ID"` to the workflow inputs. **Export dataset snapshots** writes a materialised dataset archive for inspection or reuse.

### Collect video frames

Choose **Capture camera/video data**. For a local video, use:

```json
{
  "python":"/absolute/path/to/training-python",
  "video_path":"/absolute/path/to/inspection-video.mp4",
  "seconds":5,
  "fps":1
}
```

Run the workflow and inspect `capture_dir` and the frame count. Import that capture folder, label its examples and assign a shared group for related frames. Capture does not automatically produce reviewed class labels.

To deliberately capture an attached host camera, omit `video_path` and set `camera_index`, usually `0`. The operating system may require camera permission. Collection is bounded to 300 seconds and 1,000 requested frames. A video sampling interval includes its duration endpoint, so one second at two fps can produce frames at 0, 0.5 and 1 seconds.

## 6. Run your first vision training job

In **Train, build and deploy**, choose **Train a model** and **This computer**. Click **Load example inputs**, then replace the interpreter path:

```json
{
  "engine":"pytorch",
  "python":"/absolute/path/to/training-python",
  "epochs":2,
  "image_size":32,
  "batch_size":8,
  "device":"cpu"
}
```

These are small demonstration settings. Click **Run workflow**, select the new job and follow its logs. On the desktop, use the `train` workflow and **Run job** with the same specification.

When the job succeeds, inspect:

- `model_path`: the exported `model.onnx` for compilation.
- `calibration_path`: training-only calibration data matching preprocessing.
- `preprocessing_path`: input layout, size, RGB conversion and scaling.
- Validation accuracy/loss, test accuracy/loss, split counts and the confusion matrix.
- The hashed artifact list, including checkpoint, labels and split records.

The default architecture is `tiny-cnn`. It starts from random weights and requires no pretrained model download. Its ONNX input is named `images`, with fixed NCHW shape `[1,3,height,width]`; output `logits` contains class scores. The saved labels file defines their order. Calibration is float32 NHWC with the same 0–1 scaling.

Keep preprocessing, class order and model together. A correctly loaded model can still produce incorrect results if its application feeds BGR instead of RGB, uses another scale, or interprets score positions incorrectly.

<!-- SCREENSHOT: 04-training-result.png | Successful real training job, metrics and artifact paths. -->

## 7. Compare model size and validation quality

Choose **Optimise with bounded trials** and review a small search:

```json
{
  "engine":"pytorch",
  "python":"/absolute/path/to/training-python",
  "search":{"image_size":[32,64],"epochs":[1,3]},
  "max_trials":4,
  "min_quality":0.7
}
```

This runs four training trials. Start with fewer trials when using unfamiliar data or limited memory. The maximum allowed trial count is 12. Search fields include image size, epochs, learning rate, rank and batch size where applicable.

For vision, `min_quality` is the minimum validation accuracy. The result lists eligible trials and a Pareto frontier: trials for which another eligible trial is not both smaller and at least as accurate. Artifact bytes are host file sizes, not a measurement of Hailo memory allocation or latency. Select a trial's actual model and calibration paths when moving to compilation.

For an LLM, quality is inverse validation perplexity, `exp(-validation_loss)`. It is not an accuracy percentage. Compare it only across runs with the same data, tokenizer and loss definition. Use task-specific saved-output evaluation before selecting a deployment model. Final-test metrics are reported separately and are not the sweep's ranking criterion.

## 8. Compile the vision model for Hailo

Install the licensed Hailo SDK separately on a supported Linux x86-64 environment. Use the 8-family compiler for `hailo8l`/`hailo8` and the 10-family compiler for `hailo10h`. The verified environments used DFC 3.34.0 for 8L/8 and DFC 5.4.0 for 10H. They are separate environments; one compiler family does not accept every target.

Choose **Compile for Hailo**. **Use previous result** fills the latest successful training model and calibration paths for the selected project. Review the remaining inputs:

```json
{
  "model_path":"/absolute/path/to/model.onnx",
  "calibration_path":"/absolute/path/to/calibration.npy",
  "python":"/absolute/path/to/hailo-sdk-python"
}
```

On a Mac or Windows controller, select a registered Linux worker with an appropriate SDK environment; see section 12. Docker Desktop running a Linux x86-64 vendor container is also a suitable location for a separately operated SDK workflow. The ordinary image-customisation container is not a Hailo compiler.

Run the job and inspect its log. The provider parses ONNX, saves a parsed HAR, optimises with calibration data, saves an optimised HAR and compiles HEF bytes. Unsupported operators, unsuitable calibration and missing SDKs fail the job. No substitute HEF is generated.

A successful result exposes `model_path` pointing to the HEF and its checksum. Compile separately for each target; do not rename an 8L file and treat it as an 8 or 10H model. Calibration must follow the model's preprocessing. For externally trained architectures, use the vendor's model-specific parser and optimisation settings, including `model_script_path` where needed.

[SDK setup](hailo-sdk-setup.md) documents the board-free verification harness and target-family separation. SDK numerical emulation can examine vision outputs before hardware is available. It measures neither real board throughput nor application accuracy by itself.

## 9. Prepare an LLM project for Hailo-10H

Create a new `hailo10h` project with the language-model task. Keep its dataset and artifacts separate from vision projects.

The executable fine-tuning provider accepts Alpaca-style JSONL:

```jsonl
{"instruction":"Classify the inspection note.","input":"A cracked cover is visible.","output":"damaged"}
{"instruction":"Classify the inspection note.","input":"The cover is intact.","output":"acceptable"}
```

Use enough independent, reviewed records for training, validation and test. The importer also recognises other text/message structures, but successful import does not mean those structures can be used directly by this Alpaca training provider. The small files in `examples/llm` are demonstration fixtures, not a useful corpus.

Choose **Train a model** with:

```json
{
  "engine":"transformers-peft",
  "python":"/absolute/path/to/llm-training-python",
  "model":"/absolute/path/to/local-hugging-face-model",
  "epochs":1,
  "rank":8,
  "batch_size":1,
  "max_length":256,
  "device":"cpu",
  "allow_download":false
}
```

Start with a base model already present locally. Downloads are disabled by default, and remote model code is not trusted. A successful run saves an adapter directory with safetensors weights and tokenizer files, along with validation loss and held-out perplexity. The loss includes prompt and response tokens. It is not a measure of factual accuracy or instruction-following success.

### Evaluate with the reused LLM-Optimise tools

**Prepare and evaluate an LLM** offers **Prepare host recipe** for Soup MLX, QLoRA or experimental streaming configurations. Preparing a recipe does not execute it. The runnable built-in LLM workflow above uses Transformers/PEFT.

**Score saved outputs** compares an outputs JSON file against evaluation tasks. The included fixture can be tried with `examples/llm/tasks.jsonl` and `examples/llm/outputs.json`. Missing answers fail. Supply outputs produced by the model you intend to evaluate; the scorer does not generate answers.

**Estimate dense-transformer inference memory** accepts architecture parameters and helps identify implausible configurations early. It estimates generic weights and attention-cache payloads. It does not determine Hailo compiler placement or guarantee a model fits the accelerator.

### Turn an adapter into a deployable GenAI model

The DFC 5.4.0 guide's LoRA flow starts from a **Hailo-provided pre-optimised GenAI HAR**. It requires matching adapter weights, model scripts, calibration and compatible resources. The workbench's generic LoRA defaults are not automatically valid for that vendor model.

For custom GenAI compilation, configure the `recipe_executable` required by the compile workflow. This is a trusted, installed launcher for your qualified vendor recipe. It receives the structured job configuration and must return actual compiled artifacts. There is no generic GGUF-to-HEF conversion, and arbitrary custom base models are not qualified by this interface.

GenAI has substantial host requirements; the supplied guide calls for at least 64 GB RAM, 200 GB disk and a 24 GB NVIDIA GPU, with some optimisation paths reaching 160 GB RAM. Check the exact model recipe before reserving a worker. All DFC numerical-emulation modes exclude GenAI. Actual LLM execution therefore awaits appropriate 10H hardware, even when host adapter training has passed.

<!-- SCREENSHOT: 05-llm-workflow.png | LLM workflow inputs with a local model and the separate recipe/evaluation tools. -->

## 10. Package the model, scripts and assets

Choose **Package model and scripts**. **Use previous result** copies the preceding successful compile job's HEF path. Supply the **exact HailoRT version installed on the destination Pi**, rather than assuming the example version is suitable:

```json
{
  "model_path":"/absolute/path/to/model.hef",
  "runtime_version":"REPLACE_WITH_INSTALLED_VERSION",
  "scripts_dir":"/absolute/path/to/application-scripts",
  "assets_dir":"/absolute/path/to/application-assets",
  "entrypoint":"run_model.py"
}
```

Omit optional scripts, assets and entrypoint when you only need the model packaged. Include labels and preprocessing metadata with your application assets. Entrypoints can be Python or shell scripts; their required packages must already exist on the Pi.

The result's `bundle_path` identifies the deployment ZIP. It contains a release manifest, the HEF and optional application files. Every payload file has a size and SHA-256 hash. Keep the package intact between creation and deployment; validation rejects changed or unexpected files.

In the browser's **Project scripts and configuration** section, enter a filename and script content, then **Save project script**. Saving does not execute it. The result provides its local path; use its containing directory as `scripts_dir` when packaging reviewed application scripts.

A script run through the helper has the payload directory as its working directory. Locate packaged models under `model/` and assets under `assets/`. A systemd service uses the image-installed release location. Test these paths explicitly before enabling unattended startup.

For a qualified 10H GenAI HEF, the direct runtime route requires the bundle field `"runtime_provider":"hailort-genai-llm-v1"`. Do not apply that field to a vision HEF. It selects a runtime route; the actual vendor runtime still has to accept the compiled model.

## 11. Add the package to an existing OS image

Choose **Add bundle to an existing OS image**. The operation copies a raw `.img`, or decompresses `.img.xz`, and adds the deployment files to the copied Linux root filesystem. It preserves the source image. It does not build a distribution, resize partitions, install runtime packages or write removable media.

Prepare a base image with the required Hailo runtime and adequate free root-filesystem space. An image's eventual first-boot expansion cannot provide free space during this offline insertion step. Allow enough workstation storage for the expanded image, extracted root partition and deployment files.

Mac and Windows use the Linux Docker image worker. Build it once from the repository:

```sh
docker build -t pi-trainer-image-worker:local -f docker/image-worker.Dockerfile docker
```

Keep Docker running; on Windows select Linux-container mode. Native Linux can use installed `debugfs` and `e2fsck` from e2fsprogs.

Example image inputs:

```json
{
  "base_image":"/absolute/path/to/raspberry-pi-os.img.xz",
  "bundle_path":"/absolute/path/to/deployment.zip",
  "install_service":false
}
```

Click **Run workflow**. If the image has several candidate Linux partitions, provide its intended root `partition` number after inspecting the image. The job checks the filesystem, inserts the bundle, reads the inserted files back to verify hashes and returns an `image_path` and checksum.

The release is placed under `/opt/pi-trainer/releases/<release_id>/`. Setting `install_service` to `true` requires an entrypoint and adds a **disabled** service. Before enabling it on the Pi, create its `pi-trainer` account, arrange device-group access and install application dependencies. File insertion alone does not perform those steps.

### Write and boot the image

Open **Write a customised image with Raspberry Pi Imager**, enter the successful `image_path`, and use **Prepare Imager catalogue** or **Open in Raspberry Pi Imager**. The native application also has a **Raspberry Pi Imager** button.

The workbench opens the image/catalogue workflow. Select and verify the removable storage **inside Imager**, then explicitly confirm its write operation. Take a backup of any card you need to preserve before that confirmation.

After writing, boot the Pi and check its OS, network connection, Hailo driver and runtime. Confirm the inserted release files and run the application. The image workflow's integrity checks do not replace a real boot test.

<!-- SCREENSHOT: 06-package-image.png | Reviewed image inputs and successful output image path/checksum. -->

## 12. Configure trusted network connections

The controller uses OpenSSH `ssh` and `scp`. Configure SSH outside Pi Trainer before enrolling a device or worker. Use a neutral alias such as `bench-pi`:

```sshconfig
Host bench-pi
    HostName YOUR_PI_HOSTNAME
    User YOUR_PI_USER
    IdentityFile ~/.ssh/id_ed25519
```

Place it in your user's SSH config: normally `~/.ssh/config`, or `%USERPROFILE%\.ssh\config` on Windows. Verify the Pi's host-key fingerprint through a trusted channel before accepting it during the initial manual connection. Test key-based, noninteractive access:

```sh
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes bench-pi python3 --version
```

Pi Trainer always uses strict host-key checking and batch mode. It does not accept unknown keys or prompt for a password. When a Pi is deliberately reinstalled, verify its new fingerprint before correcting the matching known-hosts entry; do not disable checking to hide the mismatch.

In the browser, expand **Enroll Pis and Linux workers**, enter the device name and trusted alias, then **Enroll Pi**. Enrollment records configuration. Select the enrolled device in **Pi device** for a subsequent probe or deployment. The desktop offers corresponding enrollment under **Devices and workers**.

### Register a Linux compute worker

Configure a separate trusted SSH alias for your Linux worker. It needs a Python 3.11+ controller interpreter and the dependencies for the requested work. Enter its alias, display name and **Worker Python executable**, then **Register worker**.

Choose that worker under **Compute worker** for training, optimisation, compilation or image customisation. Required inputs are transferred, the operation runs remotely and validated output files are downloaded. Do not select a Pi as the training worker.

If the SDK needs an older Python, retain Python 3.11+ for the worker controller and add a worker-local provider interpreter:

```json
{
  "model_path":"/absolute/local/path/to/model.onnx",
  "calibration_path":"/absolute/local/path/to/calibration.npy",
  "provider_python":"/opt/hailo-sdk/bin/python"
}
```

The provider path belongs to the Linux worker, not the controller computer. Local jobs use `python`. Vendor recipe executables similarly have to be installed in their execution environment. A registered worker is configuration, not evidence that SSH or the SDK works; inspect the first job's result and logs.

## 13. Deploy, test and roll back on the Pi

First select **Test the Pi**, select the enrolled device and use:

```json
{"action":"probe"}
```

Read the reported hardware target and runtime version. Use that runtime version in your package. A matching name in a project is insufficient when the connected board or runtime differs.

Choose **Upload and activate on Pi**, select the device, and review:

```json
{"bundle_path":"/absolute/path/to/deployment.zip","activate":true}
```

The job uploads a standalone helper and package, verifies hashes and checks the installed target/runtime before selecting a release. Releases reside under the SSH user's `~/.local/share/pi-trainer/releases/`; atomic state records the active and previous releases. This route does not install system packages or automatically start an application service.

Inspect selection with **Test the Pi** and `{"action":"status"}`. For vision, run `{"action":"benchmark","mode":"vision"}`. This invokes HailoRT's benchmark on the active HEF and reports synthetic runtime performance. For your application's own test, use `{"action":"benchmark","mode":"script"}` to execute the declared entrypoint. Design that script to report meaningful input/output checks; process exit success alone does not establish model accuracy.

For a qualified direct 10H LLM release, select `mode:"llm_direct"`, a prompt and bounded `max_tokens`. The helper uses the HailoRT GenAI runtime. Alternatively, `mode:"llm"` queries an **already registered** model in the Pi's local Hailo-Ollama service. Copying a custom HEF does not register it with Hailo-Ollama; custom upload/adapter registration is unsupported by that route.

To restore the previous selected release, choose **Roll back Pi model selection**, select the device and submit `{}`. The helper verifies the previous package and current device before exchanging the active/previous pointers. It does not undo side effects from your scripts, reinstall the OS or stop independently running services.

If cancellation or an SSH timeout reports unknown remote state, inspect device status and logs before retrying. Closing the local SSH process does not prove remote work stopped.

<!-- SCREENSHOT: 07-device-status.png | Enrolled test device selection and probe/status result with private addresses removed. -->

## 14. Repeat operations from the CLI

Use the same absolute data directory as the GUI. These commands are valid controller commands; replace the IDs with values returned by your own store:

```sh
python -m pi_trainer --data-dir /path/trainer-data projects
python -m pi_trainer --data-dir /path/trainer-data create --name "Inspection" --target hailo8l --task vision
python -m pi_trainer --data-dir /path/trainer-data import PROJECT_ID /path/class-folders
python -m pi_trainer --data-dir /path/trainer-data datasets PROJECT_ID
```

Save a reviewed workflow JSON as `train.json`, then submit it:

```sh
python -m pi_trainer --data-dir /path/trainer-data run PROJECT_ID train --spec train.json --wait
python -m pi_trainer --data-dir /path/trainer-data workflow-jobs
python -m pi_trainer --data-dir /path/trainer-data job JOB_ID
python -m pi_trainer --data-dir /path/trainer-data logs JOB_ID
python -m pi_trainer --data-dir /path/trainer-data cancel JOB_ID
```

Omit `--wait` to return after submission while the detached worker continues. Other workflow kinds are `optimise`, `compile`, `package`, `image`, `deploy`, `benchmark`, `rollback`, `capture` and `export_dataset`. The older `jobs` command lists preparation-operation history; use `workflow-jobs` for executable jobs.

Shared operations without dedicated CLI subcommands use `invoke`. For device enrollment, save `device.json` containing `{"name":"Bench Pi","host":"bench-pi"}`, then run:

```sh
python -m pi_trainer --data-dir /path/trainer-data invoke device_enroll --arguments device.json
```

For annotations, the arguments object needs `project_id`, `dataset_id` and `annotations_path`. For worker enrollment, it needs `name`, `host` and `python`. Shared operations reject missing or unexpected argument names, helping catch accidental specifications.

## 15. Connect an MCP client

MCP uses local standard input/output. Configure your client to start the controller with absolute paths:

```json
{
  "mcpServers": {
    "pi-trainer": {
      "command":"/absolute/path/to/controller-python",
      "args":["-m","pi_trainer","--data-dir","/absolute/path/to/trainer-data","mcp"],
      "env":{"PYTHONPATH":"/absolute/path/to/raspberry-pi-ai-trainer"}
    }
  }
}
```

On Windows, use an absolute `.exe` path and JSON-escaped backslashes, or forward-slash paths accepted by your client. Installing the package into that interpreter removes reliance on `PYTHONPATH`. Restart/reconnect the MCP client after changing its configuration.

Begin by asking the client to call `capabilities`, `doctor` and `list_projects`. The shared tool names include `create_project`, `import_dataset`, `datasets`, `annotate`, `run_job`, `workflow_jobs`, `job_get`, `job_logs`, `job_cancel`, `device_enroll`, `worker_enroll`, `write_script`, `prepare_imager` and `open_imager`.

A training submission uses the following `run_job` arguments:

```json
{
  "project_id":"YOUR_PROJECT_ID",
  "kind":"train",
  "spec":{
    "engine":"pytorch",
    "python":"/absolute/path/to/training-python",
    "epochs":2,
    "image_size":32
  }
}
```

Ask the client to show the actual returned job ID, inspect `job_get`, then read `job_logs` if needed. A conversational statement that training finished should be grounded in the succeeded job and its artifacts. MCP provides the same explicit operations as the other interfaces; it is not an unrestricted terminal or a hosted HTTP/OAuth endpoint.

<!-- SCREENSHOT: 08-cli-mcp.png | CLI result or MCP tool result showing a real matching job ID. -->

## 16. Back up, restore and troubleshoot

### Back up the workstation store

There is no one-click backup/restore tool in this release. Back up the complete data directory, including `state.sqlite3`, content blobs, exports, scripts and run artifacts. Also preserve external datasets, base models, SDK configuration and OS images referenced outside that directory.

For a straightforward consistent copy, wait for jobs to finish, close desktop/web/MCP clients and verify no detached worker is writing to the store. Copy the entire directory to a versioned backup location. Do not copy only the SQLite file and expect its referenced blobs/models to appear. Keep training/SDK environments reproducible through their package versions; a copied virtual environment is not generally portable between operating systems.

To check a restore, copy the backup into a **new** directory and launch the controller with that directory as `--data-dir`. List projects, inspect dataset manifests, verify important artifact hashes and run a dataset export before using it for new work. Some recorded artifact paths are absolute; restoring to a different location can require selecting the restored paths in new workflow specifications. Keep the original backup until you have confirmed the replacement.

### Preserve the Pi before changes

Back up any valuable SD-card contents and application configuration before flashing a new image. To preserve helper deployments, copy its release directory and state only while no activation/rollback is running. An example read-only download to a new workstation folder is:

```sh
scp -r bench-pi:.local/share/pi-trainer /absolute/path/to/new-pi-backup
```

Restore only onto a compatible OS/runtime and verify the files and active release before executing them. The app's rollback selects a previous deployment; it does not replace a full card backup or recover system configuration.

### Resolve common problems

| Symptom | Next action |
|---|---|
| Native application reports missing Tk | Use a Tk-enabled Python or open the browser interface. |
| Training says `torch`, ONNX or PEFT is missing | Install the dependency in the interpreter named by the job's `python`, not just the controller environment. |
| Windows JSON paths fail to parse | Escape backslashes as `\\` or use forward slashes; save valid JSON without comments. |
| Dataset has missing split/class coverage | Inspect actual snapshot counts and group assignments; add independent examples. |
| Duplicate image/record content is rejected | Remove duplicates and re-import; do not disguise duplicates by renaming files. |
| Compilation reports missing SDK or wrong architecture | Select the correct target-family SDK and worker-local interpreter. |
| Compiler rejects an ONNX operator | Use an architecture/export recipe supported by that SDK; inspect its parser log. |
| LLM model cannot load offline | Supply an existing complete local model directory and compatible tokenizer/configuration. |
| SSH fails before a job starts | Verify the alias, key authentication and host fingerprint manually with strict checking. |
| Deployment rejects target/runtime | Probe the Pi; compile/package for its actual target and exact runtime version. |
| Image injection fails for lack of space | Prepare a base image whose existing root filesystem has enough free space; automatic resizing is not included. |
| Image job cannot find Docker | Start Docker, enable Linux containers and build the image worker. |
| Job is interrupted or remote state is unknown | Inspect logs and device/worker state before submitting another operation. |
| Previous-result chaining picks an unsuitable artifact | Select the intended successful job's explicit artifact paths; inspect target and runtime again. |

For a support report, retain the application version, operating system, selected interpreter and package versions, project target/task, redacted workflow JSON, job ID and relevant log excerpt. Include the first useful error, not only the final failed status. Remove credentials, personal data and private network addresses before sharing.

Keep [training details](training.md), [deployment details](deployment.md), [image details](images.md) and [the verification record](../validation/README.md) alongside this guide. They explain the provider-specific requirements when you move beyond the first exercise.
