"""Copy the model weights from S3-compatible storage (Cloudflare R2) into the models dir.

A vast.ai pod has no network volume: its disk starts empty, so
docker/entrypoint.sh runs this before ComfyUI starts whenever
COMFY_MODELS_S3_BUCKET is set:

    COMFY_MODELS_S3_ENDPOINT  e.g. https://<account>.r2.cloudflarestorage.com
    COMFY_MODELS_S3_BUCKET    the bucket
    COMFY_MODELS_S3_PREFIX    the key prefix that maps onto the models dir, default
                              models/: models/vae/x.safetensors -> <models dir>/vae/x.safetensors
    COMFY_MODELS_S3_INCLUDE   comma-separated globs matched against the path under
                              the prefix (`*` also matches `/`), e.g.
                              "unet/*int8*,text_encoders/*,vae/*"; default: everything
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   the credentials (never printed);
                              AWS_REGION defaults to auto, which is what R2 wants

s5cmd moves the bytes. A file already on disk with the object's size is kept,
so a container that is stopped and started again downloads nothing twice. Each
download goes to <name>.s3-partial and is renamed into place only once it has
the object's size, so a sync killed halfway never leaves a truncated model
under a name ComfyUI would load; the next run deletes the leftovers. Failed
files are retried; if some still fail the exit status is 1, the pod never
becomes ready and the backend recycles it.

    python docker/s3_models_sync.py <models dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

PARTIAL_SUFFIX = ".s3-partial"
ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10
# Checkpoints are few and big (up to ~20 GB): a few files at once, each one
# fetched as many parallel ranged GETs.
FILES_AT_ONCE = 4
PARTS_PER_FILE = 16
PART_SIZE_MIB = 64
SECRET_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


@dataclass(frozen=True)
class RemoteFile:
    path: str  # under the prefix, which is also the path under the models dir
    url: str
    size: int


def say(message: str) -> None:
    """Print for a CLI script (the repo's ruff config bans print())."""
    sys.stderr.write(f"s3-sync: {message}\n")
    sys.stderr.flush()


def gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip().lstrip("/")
    return prefix if not prefix or prefix.endswith("/") else prefix + "/"


def parse_include(text: str) -> list[str]:
    return [pattern.strip() for pattern in text.split(",") if pattern.strip()]


def parse_listing(output: str, bucket: str, prefix: str) -> list[RemoteFile]:
    """The files in `s5cmd --json ls s3://<bucket>/<prefix>*` output, sorted by path."""
    base = f"s3://{bucket}/{prefix}"
    files = []
    for line in output.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        url = entry["key"]
        if entry.get("type") == "directory" or url.endswith("/"):
            continue
        if not url.startswith(base):
            raise ValueError(f"listed object {url} is not under {base}")
        path = url[len(base):]
        if any(part in ("", ".", "..") for part in path.split("/")):
            raise ValueError(f"refusing {url}: its key does not map to a path inside the models dir")
        files.append(RemoteFile(path, url, int(entry.get("size", 0))))  # s5cmd omits a zero size
    return sorted(files, key=lambda f: f.path)


def select(files: list[RemoteFile], include: list[str]) -> list[RemoteFile]:
    if not include:
        return files
    return [f for f in files if any(fnmatch.fnmatchcase(f.path, pattern) for pattern in include)]


def local_path(models_dir: str, f: RemoteFile) -> str:
    return os.path.join(models_dir, *f.path.split("/"))


def plan(files: list[RemoteFile], models_dir: str) -> tuple[list[RemoteFile], list[RemoteFile]]:
    """(to download, already present). A local file with the object's size is present."""
    missing, present = [], []
    for f in files:
        path = local_path(models_dir, f)
        same = os.path.isfile(path) and os.path.getsize(path) == f.size
        (present if same else missing).append(f)
    return missing, present


def remove_partials(models_dir: str) -> None:
    """Delete what a killed sync left: .s3-partial files and s5cmd's own temp files beside them."""
    for root, _dirs, names in os.walk(models_dir):
        for name in names:
            if PARTIAL_SUFFIX in name:
                os.remove(os.path.join(root, name))
                say(f"removed the leftover {os.path.join(root, name)}")


def redact(text: str, environ) -> str:
    for name in SECRET_ENV:
        if environ.get(name):
            text = text.replace(environ[name], "***")
    return text


def output_tail(result: subprocess.CompletedProcess, environ) -> str:
    lines = [line for line in (result.stderr + result.stdout).splitlines() if line.strip()]
    return redact(" | ".join(lines[-5:]) or f"exit status {result.returncode}", environ)


def download(s5cmd: list[str], f: RemoteFile, models_dir: str, environ) -> str | None:
    """Fetch one object under its .s3-partial name, then rename it into place. Returns an error, or None."""
    path = local_path(models_dir, f)
    partial = path + PARTIAL_SUFFIX
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.monotonic()
    result = subprocess.run([*s5cmd, "--log", "error", "cp", "--raw", "--concurrency", str(PARTS_PER_FILE),
                             "--part-size", str(PART_SIZE_MIB), f.url, partial],
                            env=environ, capture_output=True, text=True)
    size = os.path.getsize(partial) if os.path.isfile(partial) else None
    if result.returncode != 0 or size != f.size:
        if size is not None:
            os.remove(partial)
        if result.returncode != 0:
            return output_tail(result, environ)
        return f"got {size or 0} bytes, the object has {f.size}"
    os.replace(partial, path)
    say(f"ok       {f.path} ({gb(f.size)} in {time.monotonic() - t0:.0f}s)")
    return None


def list_objects(s5cmd: list[str], source: str, environ) -> str | None:
    """`s5cmd ls` JSON lines for source, retried; "" when nothing matches, None when listing kept failing."""
    for attempt in range(1, ATTEMPTS + 1):
        result = subprocess.run([*s5cmd, "--json", "ls", source], env=environ, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        if "no object found" in result.stderr:
            return ""
        say(f"listing {source} failed (attempt {attempt}/{ATTEMPTS}): {output_tail(result, environ)}")
        if attempt < ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS * attempt)
    return None


def main(argv: list[str], environ=os.environ) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models_dir", help="the ComfyUI models dir to fill")
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    opts = parser.parse_args(argv)
    models_dir = os.path.realpath(opts.models_dir)

    bucket = environ.get("COMFY_MODELS_S3_BUCKET", "").strip()
    prefix = normalize_prefix(environ.get("COMFY_MODELS_S3_PREFIX") or "models/")
    include = parse_include(environ.get("COMFY_MODELS_S3_INCLUDE", ""))
    endpoint = environ.get("COMFY_MODELS_S3_ENDPOINT", "").strip()
    if not bucket:
        say("ERROR: COMFY_MODELS_S3_BUCKET is not set")
        return 1
    if not (environ.get("AWS_ACCESS_KEY_ID") and environ.get("AWS_SECRET_ACCESS_KEY")):
        say(f"ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to read s3://{bucket}")
        return 1
    binary = shutil.which("s5cmd", path=environ.get("PATH"))
    if not binary:
        say("ERROR: s5cmd is not installed (it ships in the full image, shivanshtalwar0/comfyui:latest)")
        return 1
    environ = {**environ, "AWS_REGION": environ.get("AWS_REGION") or "auto"}
    s5cmd = [binary, *(["--endpoint-url", endpoint] if endpoint else [])]

    t0 = time.monotonic()
    source = f"s3://{bucket}/{prefix}*"
    listing = list_objects(s5cmd, source, environ)
    if listing is None:
        say(f"ERROR: could not list {source} after {ATTEMPTS} attempts")
        return 1
    try:
        files = select(parse_listing(listing, bucket, prefix), include)
    except ValueError as error:
        say(f"ERROR: {error}")
        return 1
    if not files:
        matching = f" matching COMFY_MODELS_S3_INCLUDE={','.join(include)}" if include else ""
        say(f"ERROR: no objects under s3://{bucket}/{prefix}{matching}")
        return 1

    os.makedirs(models_dir, exist_ok=True)
    remove_partials(models_dir)
    missing, present = plan(files, models_dir)
    need = sum(f.size for f in missing)
    say(f"s3://{bucket}/{prefix} -> {models_dir}: {len(files)} file(s), "
        f"{len(present)} already present ({gb(sum(f.size for f in present))}), {len(missing)} to download ({gb(need)})")
    if opts.dry_run:
        for f in missing:
            say(f"missing  {f.path} ({gb(f.size)})")
        return 0
    free = shutil.disk_usage(models_dir).free
    if need > free:
        say(f"ERROR: {gb(need)} to download but only {gb(free)} free under {models_dir}")
        return 1

    # Biggest first, so the longest download is not the one left running alone at the end.
    pending = sorted(missing, key=lambda f: -f.size)
    for attempt in range(1, ATTEMPTS + 1):
        if not pending:
            break
        for f in pending:
            say(f"download {f.path} ({gb(f.size)})")
        with ThreadPoolExecutor(FILES_AT_ONCE) as pool:
            errors = list(pool.map(lambda f: download(s5cmd, f, models_dir, environ), pending))
        failed = [(f, error) for f, error in zip(pending, errors) if error]
        for f, error in failed:
            say(f"FAILED   {f.path} (attempt {attempt}/{ATTEMPTS}): {error}")
        pending = [f for f, _ in failed]
        if pending and attempt < ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    seconds = time.monotonic() - t0
    fetched = [f for f in missing if f not in pending]
    size = sum(f.size for f in fetched)
    say(f"{len(fetched)} file(s), {gb(size)} downloaded in {seconds:.0f}s ({size / 1e6 / max(seconds, 1e-3):.0f} MB/s); "
        f"{len(present)} already present")
    if pending:
        say(f"ERROR: {len(pending)} of {len(missing)} file(s) still failed after {ATTEMPTS} attempts: "
            + ", ".join(f.path for f in pending))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
