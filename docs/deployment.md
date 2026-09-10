# Model deployment and Pi helper

The deployment backend creates real ZIP releases, uploads them over SSH, validates their bytes and the installed Hailo target/runtime, and atomically selects a release. It runs synthetic HailoRT benchmarks for Hailo-8/8L/10H vision, selected Python/shell entrypoints, or an existing Hailo-Ollama model on Hailo-10H. No training runs on the Pi.

The host needs Python 3.11+ and OpenSSH (`ssh` and `scp`). The Pi needs Python 3.9+, a working Hailo driver and `hailortcli`, and any dependencies used by supplied scripts. Configure an SSH alias with its user, key and port, and enroll the Pi's host key outside the application. All connections use `StrictHostKeyChecking=yes` and `BatchMode=yes`; this application never accepts an unknown host key. Device probes run before release activation.

## Bundle contract

`create_bundle(spec, workdir, emit, cancelled)` accepts:

```json
{
  "model_path": "/path/to/compiled/model.hef",
  "target": "hailo8l",
  "runtime_version": "4.23.0",
  "scripts_dir": "/path/to/scripts",
  "assets_dir": "/path/to/assets",
  "entrypoint": "run.py"
}
```

`runtime_version` is an exact installed HailoRT version, not a recommended example version. Supported target identifiers are `hailo8l`, `hailo8`, and `hailo10h`. Scripts, assets, and entrypoint are optional. Model metadata records the supplied target; device staging also checks the real connected device. Compiling and running the HEF remain the tests of actual hardware compatibility.

The ZIP contains `manifest.json` and `payload/<path>` files. Manifest schema is `pi-trainer/deployment/v1`, with `release_id`, `target`, `runtime_version`, `model_path`, nullable `entrypoint`, and `files` records (`path`, `sha256`, `bytes`). Paths in the manifest are relative to `payload`. Model files live under `model/`, scripts under `scripts/`, and assets under `assets/`.

Files are hashed and copied in 1 MiB chunks. Validation rejects traversal, absolute paths, backslashes, colon paths, symlinks, nonregular files, duplicate/unlisted files, CRC/hash/size corruption, files exceeding 16 GiB, aggregate contents exceeding 32 GiB, and manifests exceeding 1 MiB. ZIPs use stored files to avoid excessive compression CPU for large HEFs.

## Deployment and rollback

`deploy({"host":"pi-lab","bundle_path":"/path/release.zip","activate":true}, ...)` sends a standalone standard-library helper over SSH stdin. It uploads the ZIP through OpenSSH scp to `~/.local/share/pi-trainer/incoming`, verifies the whole ZIP SHA-256 and every payload file, and stages the immutable release under `releases/<release_id>`. Temporary upload files are removed after staging, including stage failures. Interrupted transfers may leave an incoming ZIP; they never select it.

`state.json` is the authoritative atomic active/previous pointer. Application scripts and all required dependencies remain unprivileged; deployment does not install system packages or start services. `rollback({"host":"pi-lab"}, ...)` verifies the previous release and current device again, then exchanges active and previous. Rollback cannot undo external side effects from user scripts or independently running services.

## Test modes

`test_device` accepts `host` plus `action` (`probe`, `status`, or `benchmark`, the default). For benchmark actions:

- Default or `mode:"vision"` calls `hailortcli benchmark <active-hef>` for Hailo-8/8L/10H vision bundles without a GenAI runtime provider. This measures synthetic runtime performance, not labelled-data accuracy.
- `mode:"script"` runs the active release's declared `.py` or `.sh` entrypoint with its payload directory as the working directory. Only successful process completion is reported. Scripts can locate `model/` and `assets/` relative to their working directory. Scripts are user-provided code and run with the SSH user's permissions.
- `mode:"llm_direct"` loads the selected Hailo-10H release through `hailo_platform.genai.LLM(VDevice, hef_path)` and calls `generate_all`. The bundle must explicitly declare `runtime_provider:"hailort-genai-llm-v1"`; the Pi must have the matching GenAI Python runtime. The compiled HEF must be a qualified GenAI artifact accepted by that runtime. This API needs no separate tokenizer argument; GGUF/checkpoint conversion is not performed. `max_tokens` is bounded to 1–2048 (default 200). This path is API-contract tested with a mock vendor runtime and awaits real 10H validation.
- `mode:"llm"` requires a detected Hailo-10H and an explicit already registered `model` name. It POSTs a bounded `prompt` to the Pi's loopback `http://127.0.0.1:8000/api/generate`, checks that generation returned a response, and reports elapsed time and server results. No public REST port is required because the request runs inside the SSH session.

Custom HEF registration in Hailo-Ollama is deliberately unsupported. Hailo's published API does not support user-uploaded models or LoRA adapters. A copied HEF is not claimed to be a loaded LLM. Hailo-10H vision HEFs use the normal benchmark path. Bundles declaring a GenAI provider must use `llm_direct` or an appropriate supplied application entrypoint via `mode:"script"`.

## Validation and boundaries

Automated tests cover bundle tampering/traversal/symlinks/duplicates, cancellation, wrong hardware target/runtime, upload integrity, release activation/rollback and a real local standalone-helper subprocess. Hardware checks are mocked in tests and are clearly fixtures. No Raspberry Pi host was supplied for a live transfer, benchmark, script test, or rollback; Hailo-10H hardware testing is deferred until the device is available.

Hailo primary sources reviewed 10 September 2026: [Hailo model benchmark commands](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/BENCHMARKS.rst) and [Hailo GenAI usage and API limits](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/USAGE.rst).

Direct GenAI provider follows Hailo’s [Python LLM example](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/gen_ai_apps/simple_llm_chat/simple_llm_chat.py), reviewed 10 September 2026. The constructor is the compatibility gate; declaring a provider in a bundle is not a compatibility certificate.

The HailoRT [master-branch README](https://github.com/hailo-ai/hailort/blob/master/README.md) identifies that branch as supporting Hailo-10/15, and its [benchmark implementation](https://github.com/hailo-ai/hailort/blob/master/hailort/hailortcli/benchmark_command.cpp) exposes `hailortcli benchmark <hef>` through `run2_benchmark`. This supports enabling the 10H vision command path; execution on future 10H hardware remains unverified.

Host cancellation terminates the local SSH/SCP process group (or Windows process tree) and records a cancelled job. Cancellation and transport timeout explicitly report that remote operation state may be unknown: ending SSH is not evidence that an already-started remote operation stopped. Inspect the Pi’s status and logs before retrying.
