# Host training and Hailo compilation

Training runs in a cancellable subprocess on the Mac, Windows or Linux host. The core has no deep-learning dependency. Select a separate environment's absolute `python` path. The implementation refuses Raspberry Pi device-tree hosts. No pretrained weights are downloaded unless `allow_download` is explicitly true.

## Install a training runtime

Create a Python 3.11/3.12 virtual environment using the platform's normal Python installer:

```sh
python -m venv .venv-training
# POSIX: .venv-training/bin/python; Windows: .venv-training\Scripts\python.exe
.venv-training/bin/python -m pip install torch numpy pillow onnx
# Add only for language-model LoRA:
.venv-training/bin/python -m pip install transformers peft safetensors
# Add for explicit video/camera frame-capture jobs (no GUI dependency):
.venv-training/bin/python -m pip install opencv-python-headless
```

CUDA installation depends on the workstation and the PyTorch distribution. CPU is the default, and `device` can explicitly select `cuda` or `mps`. The core does not silently select GPUs or spend money on cloud workers.

## Vision training

`run_training(spec, workdir, emit, cancelled)` accepts `task: vision`, `target: hailo8l|hailo8|hailo10h`, `engine: pytorch`, `model: tiny-cnn`, `dataset_dir`, and optional `python`, `epochs`, `batch_size`, `image_size`, `learning_rate`, `threads`, `seed`, `device`, `timeout`.

The built-in model is a small, untrained RGB convolutional classifier. It demonstrates a complete training/export pipeline without needing pretrained downloads. It is not an object detector. Other architectures can be trained externally and their ONNX models compiled using the same compilation interface.

Accepted dataset layouts:

- `dataset/class-name/image.png`: minimum five images per class; at least two classes. A deterministic, stratified 20% validation and 20% test split is created.
- `dataset/train/class-name/image.png`, `dataset/test/class-name/image.png`, optional `dataset/validation/class-name/image.png`: explicit splits are retained; validation is evaluated separately for model selection, never used for training or test reporting. Every class needs training and test examples.

Duplicate bytes are rejected across the dataset. Related frames, subjects and acquisition sessions must be grouped by the user to avoid semantic leakage. Only training images supply calibration. Outputs include `model.pt`, checked `model.onnx` (opset 13, fixed NCHW input), float32 NHWC `calibration.npy`, `labels.json`, `preprocessing.json`, split hashes, training loss and one final held-out accuracy/loss/confusion matrix. No test-driven checkpoint selection is performed. The saved ONNX structural check is not proof of quantized model quality.

## LLM training

Use `task: llm`, `target: hailo10h`, `engine: transformers-peft`, and `model` pointing to an existing Hugging Face causal language-model directory. Model code is not trusted/executed through `trust_remote_code`. `dataset_dir` can contain `train.jsonl` and `test.jsonl`; alternatively select one Alpaca JSONL file for a deterministic 60/20/20 train/validation/test split. Rows require `instruction` and `output` strings; `input` is optional. At least ten training rows are required for explicit splits (ten total for automatic splitting).

This provider trains real LoRA adapters through PEFT, saves safetensors and the tokenizer, and reports token-weighted held-out causal loss and perplexity. Prompt and response tokens both contribute to loss. It does not claim task correctness from perplexity. Use the reused LLM-Optimise `score` and `estimate` tools for supplied-answer evaluation and approximate memory planning. Existing `llm_recipe` prepares Soup MLX/QLoRA/streaming configuration; those engines are intentionally separate from this verified execution contract and can be executed in their own installed Soup environment.

Default context is 256 tokens, rank 8 and one epoch. `max_length`, `rank`, `target_modules`, `batch_size`, `learning_rate`, `device` and `epochs` are configurable. A compatible base-model architecture and enough host memory remain prerequisites. A host LoRA checkpoint does not establish compatibility with the Hailo GenAI compiler.

## Vision compilation

`run_compile` accepts `task: vision`, target, `model_path` (ONNX), `calibration_path` (float32 NHWC `.npy` matching model preprocessing), and the installed SDK's `python`. Optional SDK parser arguments are `start_node_names`, `end_node_names`, `net_input_shapes`; an explicit `model_script_path` can supply SDK optimization commands.

The worker uses the installed `hailo_sdk_client.ClientRunner(hw_arch=target)`, parses ONNX, saves parsed HAR, optimizes with the supplied calibration data, saves optimized HAR and calls `compile()` to write real HEF bytes. Missing SDK, unsupported architectures or operators fail the job. Hailo software licenses and the target-compatible SDK must be installed separately on a supported Linux x86_64 compilation host; the app does not redistribute the compiler. A Mac/Windows user needs a supported Linux compilation environment. Merely running this host provider on Mac does not make the vendor compiler available.

[Official Raspberry Pi DFC introduction](https://www.raspberrypi.com/news/raspberry-pi-ai-kit-update-dataflow-compiler-now-available/) and [Hailo's Model Zoo compiler source](https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/main_driver.py) establish the parse, optimize and compile workflow. SDK/version/operator compatibility must be verified with the installed release.

## Hailo-10H GenAI recipe contract

[Raspberry Pi documents adapter compilation with Hailo's Dataflow Compiler](https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/). This does not establish generic GGUF or arbitrary LoRA-to-HEF conversion.

For `task: llm`, configure `recipe_executable` as an absolute path to an installed, trusted, vendor-qualified recipe launcher. It receives exactly `--pi-trainer-config /absolute/job/compile-config.json`; the JSON contains the requested target, model path, output directory and explicit recipe options. It must compile the qualified base/adapter combination for `hailo10h`, exit successfully and write:

```json
{"target":"hailo10h","provider":"your-qualified-sdk-recipe","artifact_paths":["model.hef","tokenizer.json"]}
```

Write this as `compile-result.json` in the output directory. All declared outputs must be nonempty regular files inside that job directory, and at least one must be a `.hef`. Include any required tokenizer, adapter and metadata artifacts. The parent validates paths, target and hashes; existence is **compiler evidence only**, not a HEF semantic check. The recipe is executable operator configuration and must not be accepted from untrusted uploaded model metadata. SDK absence or a missing qualified recipe is a visible failure, never a fabricated model.

Hailo-8L, Hailo-8 and Hailo-10H physical loading, throughput, memory fit and task accuracy are separate device tests. 10H hardware validation is deferred until the user's board arrives.

## Local validation, 10 September 2026

An isolated `.venv-training` environment on this Mac ran nine provider tests, including real tiny CPU image training and ONNX execution. A separate 20-image synthetic run used 12 training, four validation and four test images; ONNX reload predictions matched the saved PyTorch model with maximum absolute error 0 on the sampled input. These easy synthetic images do not establish real-world classification quality.

A locally generated random GPT-2 model and tokenizer also completed real PEFT LoRA training with 12 training, four validation and four test records. The saved adapter reloaded into a fresh base-model instance. No pretrained model was downloaded. This verifies the host adapter pipeline, not useful language-model quality or Hailo compatibility. Subsequent vision checks passed real DFC compilation and native/quantised emulation for 8L, 8 and 10H, plus physical 8L and 8 inference of the synthetic fixture. LLM compilation and 10H hardware inference remain unverified. See the [verification record](../validation/README.md).

A synthetic local MJPG video was also decoded through the capture provider using OpenCV 5.0.0. A one-second request at two fps produced three valid 64x48 PNG frames at the inclusive sample times 0, 0.5 and 1 seconds; frame counts and SHA-256 hashes are recorded in `validation/capture-smoke.json`. No camera or network stream was accessed.
