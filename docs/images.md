# Add a deployment to an existing Raspberry Pi OS image

The image workflow **copies an existing `.img` or decompresses `.img.xz`**, adds a verified deployment bundle to the Linux root filesystem, and produces a new `.img` with a SHA-256 checksum. It does not build a distribution or write SD cards, USB disks, or host block devices. The input image is preserved.

The result contains `/opt/pi-trainer/releases/<release_id>/` with the compiled model, scripts, and deployment manifest. A release already present in the image is rejected. Root partition size is unchanged: ensure the base image has enough free space before starting. Raspberry Pi OS images often need filesystem expansion before substantial models fit; this workflow fails if space is unavailable and does not resize partitions automatically.

## Backends

Linux uses installed `debugfs` and `e2fsck` from e2fsprogs, operating on an extracted regular partition file. On Debian/Ubuntu, install `e2fsprogs` using your normal package manager.

Mac and Windows use Docker Desktop running Linux containers. Build the worker once from the repository root:

```sh
docker build -t pi-trainer-image-worker:local -f docker/image-worker.Dockerfile docker
```

Docker must be running and permit access to the workspace. The worker runs without network access, Linux capabilities, privileged mode, or device mounts. Only the isolated job folder is writable inside it. Windows requires Docker's Linux-container mode. This Docker path has the same ext4 implementation as native Linux; physical Windows execution still requires validation on Windows.

## Python API

```python
from pathlib import Path
from pi_trainer.images import customise_image

result = customise_image({
    "base_image": "/path/to/raspios.img.xz",
    "bundle_path": "/path/to/deployment.zip",
    # "partition": 2,          # required if more than one Linux ext4 candidate
    # "install_service": True,
}, Path("./image-jobs"), emit=print, cancelled=lambda: False)
print(result["image_path"], result["sha256"])
```

`max_image_bytes` defaults to 128 GiB and limits XZ expansion. Allow free host disk space for the expanded image, a copy of its root partition, and extracted bundle files. Output appears in a unique `image-<id>` job folder. Failed operations remove the incomplete output `.img`; diagnostic working files remain for inspection. Existing source images and bundles are never modified.

The bundle schema is `pi-trainer/deployment/v1` with `manifest.json` and `payload/` files. Declared sizes and SHA-256 hashes must match; undeclared files, duplicate members, path traversal, symlinks, excessive expansion, and unsafe filenames are rejected before extraction.

MBR and GPT tables use 512-byte sectors. GPT CRCs, partition boundaries, and overlaps are validated. The selected partition must have Linux type and an ext filesystem signature, and `/etc/os-release` must exist. Extended MBR partitions are rejected. Other filesystem types and encrypted images are unsupported. The root filesystem must pass read-only `e2fsck` before and after injection, and each injected file is read back and hashed before the extracted partition is copied into the output image.

`install_service=True` adds a **disabled** systemd unit and requires an entrypoint. It runs as the `pi-trainer` account; create that account, configure device-group permissions, install the correct Hailo runtime/Python dependencies, and enable the service on the Pi after reviewing the scripts. The image workflow does not claim a bootable or hardware-tested runtime merely because model files were copied successfully. Existing units are not overwritten. File-only injection works with a null entrypoint.

Output `.img` files are ordinary raw partitioned images suitable for flashing with Raspberry Pi Imager's custom-image option. Flashing remains a separate explicit action. Hardware boot/inference validation requires the target Raspberry Pi and matching HAT; Hailo-10H testing remains deferred until hardware is available.
