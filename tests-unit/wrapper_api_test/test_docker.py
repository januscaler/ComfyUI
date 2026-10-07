"""Tests for the CUDA docker setup: compose + Dockerfile + .dockerignore + CI.

These are structural checks (docker itself is not required): the compose YAML
must parse, request an NVIDIA GPU, build the staged Dockerfile, persist models
as volumes, and the Dockerfile must order dependency installation before the
source copy so Docker's cache keeps the base stage until requirements change.

The `runpod` target is the slim RunPod variant: no torch or requirements in
the image; the entrypoint builds them once under a lock, archives the env to
the network volume, and every later pod unpacks it. docker/runpod-bootstrap.sh
runs the same entrypoint on RunPod's own cached image, with this code fetched
onto the volume.
"""

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def dockerfile_stages(text):
    """Stage name -> (FROM image, instruction text without comments), in file order."""
    stages = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", line, re.IGNORECASE)
        if match:
            current = match.group(2) or match.group(1)
            stages[current] = [match.group(1), []]
        elif current and not line.lstrip().startswith("#"):
            stages[current][1].append(line)
    return {name: (image, "\n".join(lines)) for name, (image, lines) in stages.items()}


class TestDockerCompose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "docker-compose.yml")) as f:
            cls.compose = yaml.safe_load(f)

    def test_services_present(self):
        self.assertIn("comfyui", self.compose["services"])
        self.assertIn("developer-api", self.compose["services"])

    def test_gpu_reservation(self):
        devices = self.compose["services"]["comfyui"]["deploy"]["resources"]["reservations"]["devices"]
        self.assertEqual(devices[0]["driver"], "nvidia")
        self.assertIn("gpu", devices[0]["capabilities"][0])

    def test_build_target_and_entrypoint_env(self):
        # The server flags are built by docker/entrypoint.sh (the same script a
        # RunPod pod runs), so compose sets env and has no `command` of its own.
        cf = self.compose["services"]["comfyui"]
        self.assertEqual(cf["build"]["target"], "comfyui")
        self.assertNotIn("command", cf)
        self.assertTrue(cf["image"].startswith("${COMFYUI_IMAGE:-shivanshtalwar0/comfyui"))
        env = cf["environment"]
        self.assertTrue(any(e.startswith("HF_TOKEN") for e in env))
        for entry in ("VRAM_HEADROOM_GB=${VRAM_HEADROOM_GB:-2}",
                      "ASYNC_OFFLOAD_STREAMS=${ASYNC_OFFLOAD_STREAMS:-0}",
                      "AUTO_DOWNLOAD_MODELS=${AUTO_DOWNLOAD_MODELS:-1}",
                      "COMFYUI_ARGS=${COMFYUI_ARGS:-}",
                      "WRAPPER_AUTH_TOKEN=${WRAPPER_AUTH_TOKEN:-}"):
            self.assertIn(entry, env)
        self.assertEqual(cf["ports"], ["${COMFYUI_HOST_PORT:-8188}:8188"])

    def test_entrypoint_builds_the_server_flags(self):
        with open(os.path.join(ROOT, "docker", "entrypoint.sh")) as f:
            script = f.read()
        self.assertIn("--auto-download-models", script)
        self.assertIn('--vram-headroom "${VRAM_HEADROOM_GB:-2}"', script)
        self.assertIn("--disable-async-offload", script)  # async offload default-off
        self.assertIn("--fast-disk", script)
        self.assertIn("COMFYUI_ARGS", script)
        self.assertIn("/workspace", script)  # RunPod network volume
        self.assertIn('"prefetch"', script)
        # One copy per checkpoint on a per-GB billed volume (not HF cache + copy).
        self.assertIn('COMFY_HF_DOWNLOAD_MODE="${COMFY_HF_DOWNLOAD_MODE:-direct}"', script)
        self.assertTrue(os.access(os.path.join(ROOT, "docker", "entrypoint.sh"), os.X_OK),
                        "docker/entrypoint.sh must be executable")

    def test_model_and_state_volumes(self):
        volumes = self.compose["services"]["comfyui"]["volumes"]
        for host in ("./input", "./output", "./temp", "./user", "./api_server/workflows", "./.triton"):
            self.assertTrue(any(v.startswith(host) for v in volumes), f"volume {host} missing")
        self.assertTrue(any("/opt/ComfyUI/models" in v for v in volumes), "models mount missing")

    def test_models_dir_is_configurable(self):
        volumes = self.compose["services"]["comfyui"]["volumes"]
        models_mount = next(v for v in volumes if "/opt/ComfyUI/models" in v)
        self.assertTrue(models_mount.startswith("${MODELS_DIR:-./models}"),
                        "models mount must default to ./models and be overridable via MODELS_DIR")

    def test_healthcheck_and_ordering(self):
        cf = self.compose["services"]["comfyui"]
        self.assertIn("system_stats", cf["healthcheck"]["test"][-1])
        dep = self.compose["services"]["developer-api"]["depends_on"]["comfyui"]
        self.assertEqual(dep["condition"], "service_healthy")


class TestDockerfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "Dockerfile")) as f:
            cls.dockerfile = f.read()

    def test_stages_and_cache_order(self):
        self.assertIn("AS base", self.dockerfile)
        self.assertIn("AS comfyui", self.dockerfile)
        requirements_copy = self.dockerfile.index("COPY requirements.txt manager_requirements.txt ./")
        source_copy = self.dockerfile.index("COPY . .")
        self.assertLess(requirements_copy, source_copy,
                        "dependency install must come before the source copy to keep the base cached")

    def test_c_compiler_installed(self):
        # Triton JIT-compiles kernels at runtime (e.g. the flux2 text encoder's
        # RoPE path) and needs a C compiler inside the container.
        apt = self.dockerfile[self.dockerfile.index("apt-get install"):self.dockerfile.index("rm -rf /var/lib/apt/lists")]
        for tool in ("gcc", "g++", "make"):
            self.assertIn(tool, apt, f"{tool} must be installed for Triton JIT")

    def test_cuda_only_and_entrypoint(self):
        # cu130: comfy/quant_ops.py turns comfy-kitchen's CUDA kernels off below CUDA 13.
        self.assertIn("ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130", self.dockerfile)
        self.assertRegex(self.dockerfile, r"(?m)^ARG TORCH_VERSION=\d+\.\d+\.\d+$", "pin torch, so a rebuild cannot move it")
        self.assertIn('--index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" torchvision torchaudio', self.dockerfile)
        self.assertIn("assert torch.version.cuda", self.dockerfile)
        torch_install = self.dockerfile.index('--index-url "${TORCH_INDEX_URL}"')
        requirements_install = self.dockerfile.index("pip install -r requirements.txt")
        self.assertLess(torch_install, requirements_install,
                        "CUDA torch must be installed before requirements.txt")
        self.assertIn('ENTRYPOINT ["tini", "--", "/opt/ComfyUI/docker/entrypoint.sh"]', self.dockerfile)
        self.assertIn("HEALTHCHECK", self.dockerfile)

    def test_copy_sources_exist(self):
        for path in ("requirements.txt", "manager_requirements.txt"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, path)),
                            f"{path} referenced by Dockerfile but missing")

    def test_full_target_stays_the_default(self):
        # `docker build .` (no --target) builds the last stage: keep that the
        # full image the local rig and compose have always used.
        stages = dockerfile_stages(self.dockerfile)
        self.assertEqual(list(stages)[-1], "comfyui")
        self.assertEqual(stages["comfyui"][0], "base")
        self.assertEqual(stages["base"][0], "system")
        self.assertIn("COPY . .", stages["comfyui"][1])


class TestServerlessImages(unittest.TestCase):
    """docker/serverless/Dockerfile: a worker in its own venv on top of the full image."""

    @classmethod
    def setUpClass(cls):
        cls.stages = dockerfile_stages(read("docker", "serverless", "Dockerfile"))

    def test_each_target_runs_its_worker_from_the_worker_venv(self):
        for target, script, sdk in (("runpod", "runpod_handler.py", "runpod=="), ("vast", "vast_worker.py", "vastai==")):
            image, text = self.stages[target]
            self.assertEqual(image, "${BASE_IMAGE}")
            self.assertIn(f"{sdk}${{", text)
            self.assertIn("uv venv", text)
            self.assertIn(f"COPY jobs.py {script} /opt/ComfyUI/docker/serverless/", text)
            self.assertIn(f'ENTRYPOINT ["tini", "--", "/opt/worker-env/bin/python", "/opt/ComfyUI/docker/serverless/{script}"]', text)
            # The base healthcheck needs a WRAPPER_AUTH_TOKEN the worker no longer exports.
            self.assertIn("HEALTHCHECK NONE", text)
            self.assertTrue(os.path.isfile(os.path.join(ROOT, "docker", "serverless", script)))
        self.assertIn("EXPOSE 3000", self.stages["vast"][1])

    def test_ci_builds_both_on_the_pushed_full_image(self):
        job = yaml.safe_load(read(".github", "workflows", "docker-publish.yml"))["jobs"]["serverless"]
        self.assertEqual(job["needs"], ["mode", "build"])
        self.assertEqual(job["strategy"]["matrix"]["target"], ["runpod", "vast"])
        steps = {step.get("name"): step for step in job["steps"]}
        self.assertIn(':sha-${GITHUB_SHA::7}', steps["Base image"]["run"])
        self.assertTrue(steps["Build"]["with"]["load"])
        self.assertTrue(steps["Push to Docker Hub"]["with"]["push"])
        meta = steps["Image tags"]["with"]
        self.assertIn("latest=false", meta["flavor"])
        self.assertIn("type=raw,value=serverless-${{ matrix.target }},enable={{is_default_branch}}", meta["tags"])
        runs = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn("--test_input", runs)
        self.assertIn("Lifecycle startup failed", runs)
        self.assertIn("ComfyUI stopped on this worker", runs)
        self.assertEqual(runs.count("Unknown workflow"), 2)


class TestRunpodTarget(unittest.TestCase):
    """The slim RunPod image: system packages + uv + sources, no Python deps."""

    @classmethod
    def setUpClass(cls):
        cls.dockerfile = read("Dockerfile")
        cls.stages = dockerfile_stages(cls.dockerfile)
        cls.runpod = cls.stages["runpod"][1]

    def test_builds_from_the_shared_system_stage(self):
        # Not from `base`: that would drag torch and requirements along.
        self.assertEqual(self.stages["runpod"][0], "system")
        system = self.stages["system"][1]
        for tool in ("ffmpeg", "gcc", "g++", "make", "git", "libglib2.0-0", "libgl1", "tini"):
            self.assertIn(tool, system, f"{tool} must be in the shared apt layer")

    def test_no_torch_or_requirements_install(self):
        for forbidden in ("pip install", "--index-url", "COPY --from=base", "site-packages"):
            self.assertNotIn(forbidden, self.runpod, f"runpod target must not contain {forbidden!r}")

    def test_uv_binary_from_a_pinned_image(self):
        self.assertIn("COPY --from=uv /uv /usr/local/bin/uv", self.runpod)
        self.assertEqual(self.stages["uv"][0], "${UV_IMAGE}")
        match = re.search(r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:(\S+)$", self.dockerfile, re.MULTILINE)
        self.assertIsNotNone(match, "UV_IMAGE must default to the official uv image")
        self.assertRegex(match.group(1), r"^\d+\.\d+\.\d+$", "pin a uv version, not a floating tag")

    def test_env_key_is_baked(self):
        self.assertIn("ARG TORCH_INDEX_URL", self.runpod)
        self.assertIn("ARG TORCH_VERSION", self.runpod)
        # One script prints the hashed inputs, for the image and for the boot-time key alike.
        self.assertIn('TORCH_INDEX_URL="${TORCH_INDEX_URL}" TORCH_VERSION="${TORCH_VERSION}" docker/env-inputs.sh > docker/env-inputs', self.runpod)
        self.assertIn("sha256sum docker/env-inputs | cut -c1-16 > docker/env-key", self.runpod)
        self.assertIn("COMFY_IMAGE_VARIANT=runpod", self.runpod)
        self.assertNotIn("ENV_RECIPE", self.dockerfile, "the recipe lives in docker/env-inputs.sh")
        inputs = read("docker", "env-inputs.sh")
        # Everything the env's contents depend on goes into the hashed inputs.
        for part in ('echo "recipe=$ENV_RECIPE"', "sys.version_info[:2]", 'echo "python_path=$python"', "platform=$(uname -m)",
                     "sha256sum requirements.txt"):
            self.assertIn(part, inputs)
        # The script's default index is the Dockerfile's, so a boot-time key matches the baked one.
        default = re.search(r"^ARG TORCH_INDEX_URL=(\S+)$", self.dockerfile, re.MULTILINE).group(1)
        self.assertIn(f"torch_index=${{TORCH_INDEX_URL:-{default}}}", inputs)
        version = re.search(r"^ARG TORCH_VERSION=(\S+)$", self.dockerfile, re.MULTILINE).group(1)
        self.assertIn(f"torch_version=${{TORCH_VERSION:-{version}}}", inputs)

    def test_entrypoint_and_healthcheck(self):
        self.assertIn('ENTRYPOINT ["tini", "--", "/opt/ComfyUI/docker/entrypoint.sh"]', self.runpod)
        self.assertIn("HEALTHCHECK", self.runpod)
        self.assertIn("chmod +x docker/entrypoint.sh", self.runpod)

    def test_cpu_torch_escape_hatch_is_never_baked(self):
        # COMFY_ALLOW_CPU_TORCH exists for local tests only; the published
        # images must stay CUDA only, so nothing may turn it on by default.
        self.assertNotIn("COMFY_ALLOW_CPU_TORCH", self.dockerfile)
        self.assertNotIn("COMFY_ALLOW_CPU_TORCH", read(".github", "workflows", "docker-publish.yml"))


class TestRunpodEntrypoint(unittest.TestCase):
    """docker/entrypoint.sh builds and reuses the Python env on the volume."""

    @classmethod
    def setUpClass(cls):
        cls.script = read("docker", "entrypoint.sh")

    def test_variant_and_env_location(self):
        self.assertIn('"${COMFY_IMAGE_VARIANT:-}" == "runpod"', self.script)
        self.assertIn('ENV_KEY=$(cat "$COMFY_ROOT/docker/env-key")', self.script)
        self.assertIn('ENVS_DIR="${COMFY_ENVS_DIR:-$DATA_DIR/envs}"', self.script)
        # One archive per key on the volume; the env itself on the container disk, at a fixed path.
        self.assertIn('ENV_ARCHIVE="$ENVS_DIR/$ENV_KEY.tar"', self.script)
        self.assertIn('ENV_DIR="${COMFY_ENV_LOCAL_DIR:-/opt/comfy-env}/$ENV_KEY"', self.script)
        # The torch index and version come from the hashed inputs, so they always match the key.
        self.assertIn("s/^torch_index=//p", self.script)
        self.assertIn("s/^torch_version=//p", self.script)
        self.assertIn('--index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" torchvision torchaudio', self.script)

    def test_key_computed_at_boot_for_fetched_code(self):
        start = self.script.index('if [[ ! -s "$COMFY_ROOT/docker/env-key" ]]; then')
        block = self.script[start:self.script.index("\tfi\n", start)]
        self.assertIn('bash "$COMFY_ROOT/docker/env-inputs.sh" >"$COMFY_ROOT/docker/env-inputs"', block)
        self.assertIn('sha256sum "$COMFY_ROOT/docker/env-inputs" | cut -c1-16 >"$COMFY_ROOT/docker/env-key"', block)
        self.assertLess(start, self.script.index('ENV_KEY=$(cat "$COMFY_ROOT/docker/env-key")'), "computed before it is read")

    def test_archive_is_published_whole(self):
        archive = self.script[self.script.index("archive_env() {"):self.script.index("unpack_env() {")]
        write = archive.index('tar -cf "$tmp" -C "$ENV_DIR" .')
        owner = archive.index('"$(lock_owner)" != "$ENV_TOKEN"')
        publish = archive.index('mv -f "$tmp" "$ENV_ARCHIVE"')
        self.assertLess(write, owner)
        self.assertLess(owner, publish, "only the lock owner publishes, and only a finished archive")
        build = self.script[self.script.index("build_env() {"):self.script.index("archive_env() {")]
        self.assertLess(build.index('mv -f "$ENV_DIR/$ENV_MARKER_NAME.tmp" "$ENV_DIR/$ENV_MARKER_NAME"'), build.index("\tarchive_env\n"),
                        "the archive carries the completion marker")

    def test_unpack_renames_a_complete_env_into_place(self):
        unpack = self.script[self.script.index("unpack_env() {"):self.script.index("ensure_env() {")]
        self.assertIn('partial="$ENV_DIR.partial"', unpack)
        extract = unpack.index('tar -xf "$ENV_ARCHIVE" -C "$partial"')
        checked = unpack.index('! -f "$partial/$ENV_MARKER_NAME"')
        renamed = unpack.index('mv "$partial" "$ENV_DIR"')
        self.assertLess(extract, checked)
        self.assertLess(checked, renamed)

    def test_archive_before_build(self):
        ensure = self.script[self.script.index("ensure_env() {"):self.script.index('if [[ "${COMFY_IMAGE_VARIANT:-}" == "runpod" ]]; then')]
        # A restarted container reuses its unpacked env; otherwise the archive; only then build.
        self.assertLess(ensure.index("if env_ready; then"), ensure.index("if archive_ready; then"))
        self.assertLess(ensure.index("if archive_ready; then"), ensure.index('mkdir "$ENV_LOCK" 2>/dev/null'))
        self.assertIn("if archive_ready; then # published by another pod", ensure, "the lock holder re-checks first")

    def test_fails_fast_without_a_volume(self):
        block = self.script[self.script.index('if [[ "${COMFY_IMAGE_VARIANT:-}" == "runpod" ]]; then'):]
        no_volume = block[:block.index("fi")]
        self.assertIn('-z "$DATA_DIR"', no_volume)
        self.assertIn("no volume is mounted", no_volume)
        self.assertIn("exit 64", no_volume)

    def test_network_filesystem_lock(self):
        # mkdir is atomic on a network filesystem where flock may not be.
        self.assertIn('ENV_LOCK="$ENVS_DIR/.lock-$ENV_KEY"', self.script)
        self.assertIn('mkdir "$ENV_LOCK" 2>/dev/null', self.script)
        code = "\n".join(line for line in self.script.splitlines() if not line.lstrip().startswith("#"))
        self.assertNotIn("flock", code)
        self.assertIn('>"$ENV_LOCK/owner"', self.script)
        self.assertIn('>"$ENV_LOCK/heartbeat"', self.script)
        self.assertIn('ENV_LOCK_STALE_SECONDS="${COMFY_ENV_LOCK_STALE_SECONDS:-120}"', self.script)
        self.assertIn('ENV_WAIT_SECONDS="${COMFY_ENV_WAIT_SECONDS:-1800}"', self.script)
        # Taking over a stale lock is itself exclusive, and re-checks the state.
        self.assertIn('mkdir "$ENV_LOCK.break"', self.script)
        self.assertIn('"$(lock_state)" == "$stale"', self.script)
        # The lock is released on failure too.
        self.assertIn("trap release_env_lock EXIT", self.script)

    def test_env_build_recipe(self):
        build = self.script[self.script.index("build_env() {"):self.script.index("ensure_env() {")]
        self.assertIn('uv venv --python "$base_python" "$ENV_DIR"', build)
        torch_install = build.index('uv pip install --python "$ENV_DIR/bin/python" --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" torchvision torchaudio')
        requirements_install = build.index('uv pip install --python "$ENV_DIR/bin/python" -r "$COMFY_ROOT/requirements.txt"')
        cuda_assert = build.index("cuda = torch.version.cuda")
        marker = build.index('mv -f "$ENV_DIR/$ENV_MARKER_NAME.tmp" "$ENV_DIR/$ENV_MARKER_NAME"')
        self.assertLess(torch_install, requirements_install, "CUDA torch must be installed before requirements.txt")
        self.assertLess(requirements_install, cuda_assert)
        self.assertLess(cuda_assert, marker, "the completion marker is written last")
        # The CPU escape hatch is explicit and off by default.
        self.assertIn('os.environ.get("COMFY_ALLOW_CPU_TORCH") != "1"', build)
        # A partial env (no marker) is removed before rebuilding.
        self.assertIn('rm -rf "$ENV_DIR"', build)
        # Only a builder that still owns the lock may mark the env complete.
        self.assertLess(build.index('"$(lock_owner)" != "$ENV_TOKEN"'), marker)
        # Nothing cached twice on the per-GB billed volume.
        self.assertIn("default_cache=/tmp/uv-cache", build)
        self.assertIn("UV_LINK_MODE=copy", build)
        self.assertIn("UV_PYTHON_DOWNLOADS=never", build)

    def test_completion_marker(self):
        self.assertIn("ENV_MARKER_NAME=.comfy-env-complete", self.script)
        self.assertIn('env_ready() { [[ -f "$ENV_DIR/$ENV_MARKER_NAME" && -x "$ENV_DIR/bin/python" ]]; }', self.script)

    def test_everything_runs_inside_the_env(self):
        activate = self.script.index('export PATH="$ENV_DIR/bin:$PATH"')
        self.assertLess(self.script.index("\tensure_env\n"), activate)
        self.assertLess(activate, self.script.index('if [[ "${1:-}" == "prefetch" ]]; then'))
        self.assertLess(activate, self.script.index('exec "$@"'))
        self.assertLess(activate, self.script.index("exec python main.py"))

    def test_prefetch_env_only(self):
        self.assertIn('"$arg" == "--env-only"', self.script)
        self.assertIn('exec python "$COMFY_ROOT/docker/prefetch_models.py" "${prefetch_args[@]}"', self.script)

    def test_logs_found_vs_built_timings(self):
        self.assertIn("reusing it (checked in $((SECONDS - t0))s)", self.script)
        self.assertIn("unpacked from $ENV_ARCHIVE in ${UNPACK_SECONDS}s, reusing it", self.script)
        self.assertIn("built in $((SECONDS - t0))s", self.script)


@unittest.skipIf(sys.platform == "win32", "bash scripts")
class TestRunpodBootstrap(unittest.TestCase):
    """docker/runpod-bootstrap.sh: code onto the volume, then the entrypoint, on RunPod's cached image."""

    @classmethod
    def setUpClass(cls):
        cls.script = read("docker", "runpod-bootstrap.sh")

    @staticmethod
    def run_script(name, env):
        return subprocess.run(["bash", os.path.join(ROOT, "docker", name)], env={"PATH": os.environ["PATH"], **env},
                              capture_output=True, text=True, timeout=60)

    def test_refuses_anything_but_a_commit(self):
        with tempfile.TemporaryDirectory() as data:
            for ref in ("", "master", "v1.2.3", "../etc", "ABCDEF1"):
                result = self.run_script("runpod-bootstrap.sh", {"COMFY_CODE_REF": ref, "COMFYUI_DATA_DIR": data})
                self.assertEqual(result.returncode, 64, f"{ref!r} must be refused")
                self.assertIn("must be a commit sha", result.stderr)
            self.assertEqual(os.listdir(data), [], "nothing is written for a refused ref")

    def test_refuses_to_run_without_a_volume(self):
        result = self.run_script("runpod-bootstrap.sh", {"COMFY_CODE_REF": "a" * 40, "COMFYUI_DATA_DIR": "/nonexistent-volume"})
        self.assertEqual(result.returncode, 64)
        self.assertIn("no writable network volume", result.stderr)

    def test_code_cached_per_commit_and_published_whole(self):
        self.assertIn('archive="$store/ComfyUI-$ref.tar.gz"', self.script)
        self.assertIn('if [[ -s "$archive" ]]; then', self.script)
        download = self.script.index('python3 - "$url" "$tmp"')
        verified = self.script.index('gzip -t "$tmp"')
        published = self.script.index('mv -f "$tmp" "$archive"')
        self.assertLess(download, verified)
        self.assertLess(verified, published)
        self.assertIn("https://codeload.github.com/${COMFY_CODE_REPO:-januscaler/ComfyUI}/tar.gz/$ref", self.script)

    def test_hands_over_to_the_entrypoint_as_the_runpod_variant(self):
        unpacked = self.script.index('tar -xzf "$archive" -C "$root" --strip-components=1')
        exported = self.script.index("export COMFY_IMAGE_VARIANT=runpod")
        handed = self.script.index('exec bash "$root/docker/entrypoint.sh" "$@"')
        self.assertLess(unpacked, exported)
        self.assertLess(exported, handed)
        self.assertIn("root=/opt/ComfyUI", self.script)

    def test_env_inputs_script_prints_every_input(self):
        result = self.run_script("env-inputs.sh", {})
        self.assertEqual(result.returncode, 0, result.stderr)
        keys = [line.split("=", 1)[0] for line in result.stdout.splitlines()]
        self.assertEqual(keys, ["recipe", "python", "python_path", "platform", "os", "torch_index", "torch_version", "requirements_sha256"])
        self.assertEqual(result.stdout, self.run_script("env-inputs.sh", {}).stdout, "the key must be deterministic")
        other = self.run_script("env-inputs.sh", {"TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128"})
        self.assertNotEqual(result.stdout, other.stdout, "another torch index is another env")
        other = self.run_script("env-inputs.sh", {"TORCH_VERSION": "2.12.0"})
        self.assertNotEqual(result.stdout, other.stdout, "another torch version is another env")


class TestVastTools(unittest.TestCase):
    """cloudflared and s5cmd: pinned release binaries, checksummed, in the full image only."""

    @classmethod
    def setUpClass(cls):
        cls.stages = dockerfile_stages(read("Dockerfile"))
        cls.tools = cls.stages["tools"][1]

    def test_pinned_and_checksummed(self):
        self.assertEqual(self.stages["tools"][0], "${BASE_IMAGE}")
        for name, version in (("CLOUDFLARED", r"\d{4}\.\d+\.\d+"), ("S5CMD", r"\d+\.\d+\.\d+")):
            self.assertRegex(self.tools, rf"(?m)^ARG {name}_VERSION={version}$")
            self.assertRegex(self.tools, rf"(?m)^ARG {name}_SHA256=[0-9a-f]{{64}}$")
        self.assertIn("https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64", self.tools)
        self.assertIn("https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz", self.tools)
        cloudflared_check = self.tools.index('echo "${CLOUDFLARED_SHA256}  /usr/local/bin/cloudflared" | sha256sum -c -')
        s5cmd_check = self.tools.index('echo "${S5CMD_SHA256}  /tmp/s5cmd.tar.gz" | sha256sum -c -')
        self.assertLess(s5cmd_check, self.tools.index("tar -xzf /tmp/s5cmd.tar.gz"), "checked before it is unpacked")
        self.assertLess(cloudflared_check, self.tools.index("chmod 0755"))

    def test_in_the_full_image_only(self):
        copy = "COPY --from=tools /usr/local/bin/cloudflared /usr/local/bin/s5cmd /usr/local/bin/"
        full = self.stages["comfyui"][1]
        self.assertLess(full.index(copy), full.index("COPY . ."), "before the sources, so a code change keeps the layer")
        self.assertNotIn("--from=tools", self.stages["runpod"][1], "the runpod image stays slim")


class TestVastEntrypoint(unittest.TestCase):
    """docker/entrypoint.sh on a vast.ai pod: the tunnel, then the weights, then ComfyUI."""

    @classmethod
    def setUpClass(cls):
        cls.script = read("docker", "entrypoint.sh")

    def test_server_mode_only_and_before_comfyui(self):
        passthrough = self.script.index('exec "$@"')
        server = self.script.index("exec python main.py")
        steps = ("\tstart_tunnel\n", 'if ! python "$COMFY_ROOT/docker/s3_models_sync.py" "$COMFY_ROOT/models"; then',
                 'if ! python "$COMFY_ROOT/docker/prefetch_models.py"; then')
        for step in steps:
            at = self.script.index(step)
            self.assertLess(passthrough, at, f"{step!r} must not run for prefetch or passthrough commands")
            self.assertLess(at, server, f"{step!r} must run before ComfyUI starts")
        self.assertLess(self.script.index(steps[0]), self.script.index(steps[1]), "the tunnel connects while the weights copy")

    def test_nothing_runs_without_the_env(self):
        self.assertIn('if [[ -n "${CF_TUNNEL_TOKEN:-}" ]]; then', self.script)
        sync = self.script.index('if [[ -n "${COMFY_MODELS_S3_BUCKET:-}" ]]; then')
        prefetch = self.script.index('elif flag_on "${COMFY_MODELS_PREFETCH:-}"; then')
        self.assertLess(sync, prefetch, "the Hugging Face prefetch is only the fallback without a bucket")

    @unittest.skipIf(sys.platform == "win32", "bash scripts")
    def test_prefetch_flag_values(self):
        flag_on = re.search(r"^flag_on\(\) \{\n.*?^\}$", self.script, re.MULTILINE | re.DOTALL).group(0)
        for value, on in (("", False), ("0", False), ("false", False), ("False", False), ("FALSE", False), ("no", False),
                          ("NO", False), ("off", False), ("1", True), ("true", True), ("yes", True)):
            result = subprocess.run(["bash", "-c", f'{flag_on}\nflag_on "$1"', "_", value], capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, on, f"COMFY_MODELS_PREFETCH={value!r}")

    def test_secrets_are_unset_before_comfyui_starts(self):
        names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "CF_TUNNEL_TOKEN", "COMFY_MODELS_S3_ENDPOINT")
        unset = re.search(r"^unset (.*)$", self.script, re.MULTILINE)
        self.assertEqual(tuple(unset.group(1).split()), names)
        self.assertLess(self.script.index("\tstart_tunnel\n"), unset.start())
        self.assertLess(self.script.index('if ! python "$COMFY_ROOT/docker/s3_models_sync.py"'), unset.start())
        after = self.script[unset.end():]
        self.assertIn("exec python main.py", after)
        for name in names:
            self.assertNotIn(name, after, f"{name} is read after it is unset")

    def test_a_failed_sync_or_prefetch_stops_the_container(self):
        block = self.script[self.script.index('if [[ -n "${COMFY_MODELS_S3_BUCKET:-}" ]]; then'):self.script.index("# --- ComfyUI server")]
        self.assertIn('if ! python "$COMFY_ROOT/docker/s3_models_sync.py"', block)
        self.assertIn('if ! python "$COMFY_ROOT/docker/prefetch_models.py"', block)
        self.assertEqual(block.count("exit 1"), 2)
        self.assertEqual(block.count("not starting ComfyUI"), 2)

    def test_the_token_stays_off_argv_and_out_of_the_log(self):
        self.assertIn('TUNNEL_TOKEN=$token cloudflared tunnel --no-autoupdate --metrics "$TUNNEL_METRICS" run', self.script)
        self.assertIn("TUNNEL_METRICS=127.0.0.1:20241", self.script)
        self.assertNotIn("--token", self.script)
        uses = [line.strip() for line in self.script.splitlines() if "$CF_TUNNEL_TOKEN" in line or "${CF_TUNNEL_TOKEN" in line]
        self.assertEqual(uses, ["local token=$CF_TUNNEL_TOKEN", 'if [[ -n "${CF_TUNNEL_TOKEN:-}" ]]; then'])

    def test_no_cloudflared_is_a_clear_error(self):
        block = self.script[self.script.index('if [[ -n "${CF_TUNNEL_TOKEN:-}" ]]; then'):self.script.index("\tstart_tunnel\n")]
        self.assertIn("if ! command -v cloudflared >/dev/null; then", block)
        self.assertIn("exit 1", block)


class TestHostDriverCheck(unittest.TestCase):
    """docker/entrypoint.sh stops on a host driver too old for the image's CUDA build."""

    @classmethod
    def setUpClass(cls):
        cls.script = read("docker", "entrypoint.sh")
        cls.function = re.search(r"^check_host_driver\(\) \{\n.*?^\}$", cls.script, re.MULTILINE | re.DOTALL).group(0)

    def test_runs_in_server_mode_before_the_weights(self):
        call = self.script.index('check_host_driver "$@"\n')
        self.assertLess(self.script.index('exec "$@"'), call, "not for prefetch or passthrough commands")
        for later in ('if [[ -n "${COMFY_MODELS_S3_BUCKET:-}" ]]; then', "exec python main.py"):
            self.assertLess(call, self.script.index(later), f"before {later!r}")

    def check(self, torch_version, driver, args=(), comfyui_args=""):
        """Runs the function with a stand-in python (prints the torch version) and nvidia-smi (prints the driver)."""
        with tempfile.TemporaryDirectory() as bin_dir:
            with open(os.path.join(bin_dir, "python"), "w") as f:
                f.write(f"#!/bin/sh\necho '{torch_version}'\n")
            if driver is not None:
                with open(os.path.join(bin_dir, "nvidia-smi"), "w") as f:
                    f.write(f"#!/bin/sh\necho '{driver}'\n")
            for name in os.listdir(bin_dir):
                os.chmod(os.path.join(bin_dir, name), 0o755)
            harness = f'set -euo pipefail\nlog() {{ echo "entrypoint: $*" >&2; }}\n{self.function}\ncheck_host_driver "$@"\necho started'
            env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "COMFYUI_ARGS": comfyui_args}
            return subprocess.run(["bash", "-c", harness, "_", *args], env=env, capture_output=True, text=True, timeout=30)

    @unittest.skipIf(sys.platform == "win32", "bash scripts")
    def test_old_driver_stops_a_cuda13_torch(self):
        result = self.check("2.11.0+cu130", "575.57.08")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("started", result.stdout)
        self.assertIn("needs an NVIDIA driver >= 580", result.stderr)
        self.assertIn("575.57.08", result.stderr)

    @unittest.skipIf(sys.platform == "win32", "bash scripts")
    def test_new_enough_drivers_start(self):
        for driver in ("580.173.02", "595.84"):
            result = self.check("2.11.0+cu130", driver)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("started", result.stdout)

    @unittest.skipIf(sys.platform == "win32", "bash scripts")
    def test_skipped_where_it_cannot_or_need_not_tell(self):
        cases = (
            ("2.11.0+cu128", "570.10", (), ""),           # CUDA 12 build: no CUDA 13 floor
            ("2.11.0+cu130", None, (), ""),               # no nvidia-smi (CI, CPU box)
            ("2.11.0+cu130", "575.57.08", ("--cpu",), ""),
            ("2.11.0+cu130", "575.57.08", (), "--fast --cpu"),
            ("2.11.0", "575.57.08", (), ""),              # not a CUDA wheel
            ("2.11.0+cu130", "not a version", (), ""),    # unreadable driver: warn, go on
        )
        for torch_version, driver, args, comfyui_args in cases:
            result = self.check(torch_version, driver, args, comfyui_args)
            self.assertEqual(result.returncode, 0, (torch_version, driver, args, comfyui_args, result.stderr))
            self.assertIn("started", result.stdout)


@unittest.skipIf(sys.platform == "win32", "bash scripts")
class TestSequenceParallelEntrypoint(unittest.TestCase):
    """docker/entrypoint.sh starts a follower per extra GPU and hands ComfyUI the first one."""

    @classmethod
    def setUpClass(cls):
        cls.script = read("docker", "entrypoint.sh")
        start = cls.script.index("SP_GROUP_SIZES=(")
        cls.block = cls.script[start:cls.script.index('start_sequence_parallel "$@"\n', start)]

    def test_runs_after_the_secrets_are_unset_and_before_comfyui(self):
        call = self.script.index('start_sequence_parallel "$@"\n')
        self.assertLess(re.search(r"^unset AWS_ACCESS_KEY_ID", self.script, re.MULTILINE).start(), call,
                        "followers must not inherit the bucket or tunnel credentials")
        self.assertLess(call, self.script.index("exec python main.py"))

    def run_block(self, gpus=4, setting=None, extra_env=None, rank1_exits=None, comfyui_args="--verbose INFO", probe="pass"):
        """The block in a bash harness with stand-in nvidia-smi and python; returns (exit code, log, launches).

        The stand-in docker/sp_probe.py passes (`probe="pass"`), passes only with
        NCCL_P2P_DISABLE=1 ("p2p") or only without it ("nccl"), or fails ("fail"). Its calls land in
        self.probes as "<NCCL_P2P_DISABLE> <GPUs>", and each follower's
        NCCL_P2P_DISABLE in self.follower_p2p as "<rank> <value>".
        """
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.mkdir(bin_dir)
            launched = os.path.join(tmp, "launched")
            probes = os.path.join(tmp, "probes")
            follower_p2p = os.path.join(tmp, "follower_p2p")
            if gpus is not None:
                with open(os.path.join(bin_dir, "nvidia-smi"), "w") as f:
                    f.write("#!/bin/sh\n" + "".join(f"echo {i}\n" for i in range(gpus)))
            with open(os.path.join(bin_dir, "python"), "w") as f:
                f.write("#!/bin/bash\n"
                        'if [[ "$1" == */docker/sp_probe.py ]]; then\n'
                        f'  echo "${{NCCL_P2P_DISABLE:-default}} ${{*:2}}" >> {probes}\n'
                        '  case "$FAKE_PROBE" in pass) exit 0 ;; p2p) [ "${NCCL_P2P_DISABLE:-}" = 1 ] && exit 0; exit 1 ;;\n'
                        '    nccl) [ -z "${NCCL_P2P_DISABLE+set}" ] && exit 0; exit 1 ;; *) exit 1 ;; esac\n'
                        'fi\n'
                        f'echo "$COMFY_SP_RANK ${{NCCL_P2P_DISABLE:-default}}" >> {follower_p2p}\n'
                        f'echo "$COMFY_SP_RANK $CUDA_VISIBLE_DEVICES $COMFY_SP_WORLD ${{*:2}}" >> {launched}\n'
                        + (f'[ "$COMFY_SP_RANK" = 1 ] && exit {rank1_exits}\n' if rank1_exits is not None else "")
                        + "sleep 20\n")
            for name in os.listdir(bin_dir):
                os.chmod(os.path.join(bin_dir, name), 0o755)
            # A non-empty `extra`: bash 3.2 (macOS) calls an empty array unbound under set -u.
            harness = ("set -euo pipefail\nlog() { echo \"entrypoint: $*\" >&2; }\nCOMFY_ROOT=/opt/ComfyUI\n"
                       "args=(--listen 0.0.0.0 --fast-disk)\nread -r -a extra <<<\"$COMFYUI_ARGS\"\n"
                       + self.block + 'start_sequence_parallel "$@"\n'
                       'echo "rank0 ${COMFY_SP_WORLD:-1} ${CUDA_VISIBLE_DEVICES:-unset}"\n'
                       'echo "rank0 p2p ${NCCL_P2P_DISABLE:-default}"\n'
                       + ("sleep 5\necho STILL_RUNNING\n" if rank1_exits is not None else ""))
            env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "COMFYUI_ARGS": comfyui_args, "FAKE_PROBE": probe, **(extra_env or {})}
            if setting is not None:
                env["COMFY_SEQUENCE_PARALLEL_GPUS"] = setting
            # A file, not a pipe: the backgrounded followers would hold a pipe open until they exit.
            log_path = os.path.join(tmp, "log")
            with open(log_path, "w") as log:
                result = subprocess.run(["bash", "-c", harness], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            output = open(log_path).read()
            world = re.search(r"^rank0 (\d+) ", output, re.MULTILINE)
            expected = int(world.group(1)) - 1 if world else 0

            def launches():
                return open(launched).read().splitlines() if os.path.exists(launched) else []

            # A new executable's first run can take a while on macOS (it is scanned).
            deadline = time.monotonic() + 15
            while len(launches()) < expected and time.monotonic() < deadline:
                time.sleep(0.1)
            subprocess.run(["pkill", "-f", bin_dir], capture_output=True)
            self.probes = open(probes).read().splitlines() if os.path.exists(probes) else []
            self.follower_p2p = sorted(open(follower_p2p).read().splitlines()) if os.path.exists(follower_p2p) else []
            return result.returncode, output, sorted(launches())

    def test_group_size_follows_the_gpus_and_the_setting(self):
        for gpus, setting, world in ((1, None, 1), (2, None, 2), (3, None, 2), (4, None, 4), (6, None, 4), (7, None, 7),
                                     (8, None, 8), (16, None, 8), (8, "4", 4), (8, "1", 1), (8, "auto", 8)):
            _, log, launched = self.run_block(gpus=gpus, setting=setting)
            self.assertIn(f"rank0 {world} ", log, (gpus, setting))
            self.assertEqual(len(launched), world - 1, (gpus, setting))
        for setting in ("3", "16", "abc"):
            _, log, launched = self.run_block(gpus=8, setting=setting)
            self.assertIn("rank0 1 unset", log)
            self.assertIn("WARNING: COMFY_SEQUENCE_PARALLEL_GPUS", log)
            self.assertEqual(launched, [])

    def test_each_follower_gets_its_own_gpu_and_comfyuis_flags(self):
        _, log, launched = self.run_block(gpus=4)
        self.assertIn("rank0 4 0", log)
        self.assertEqual(launched, [f"{rank} {rank} 4 --listen 0.0.0.0 --fast-disk --verbose INFO" for rank in (1, 2, 3)])
        _, log, launched = self.run_block(gpus=None, extra_env={"CUDA_VISIBLE_DEVICES": "2,5"})
        self.assertEqual(launched, [], "no nvidia-smi: one GPU")
        _, log, launched = self.run_block(gpus=8, extra_env={"CUDA_VISIBLE_DEVICES": "2,5"})
        self.assertIn("rank0 2 2", log)
        self.assertEqual(launched, ["1 5 2 --listen 0.0.0.0 --fast-disk --verbose INFO"])

    def test_off_with_cpu(self):
        _, log, launched = self.run_block(gpus=4, comfyui_args="--cpu")
        self.assertIn("rank0 1 unset", log)
        self.assertEqual(launched, [])

    def test_a_follower_that_fails_stops_comfyui(self):
        code, log, _ = self.run_block(gpus=2, rank1_exits=3)
        self.assertEqual(code, -15, log)  # SIGTERM
        self.assertIn("sequence-parallel follower 1 failed (status 3); stopping ComfyUI", log)
        self.assertNotIn("STILL_RUNNING", log)

    def test_a_follower_that_exits_cleanly_leaves_comfyui_running(self):
        # ComfyUI is gone, or the group could not start and ComfyUI renders on one GPU.
        code, log, _ = self.run_block(gpus=2, rank1_exits=0)
        self.assertEqual(code, 0, log)
        self.assertIn("STILL_RUNNING", log)
        self.assertNotIn("stopping ComfyUI", log)

    def test_the_group_is_tried_on_its_own_gpus_before_it_starts(self):
        _, log, launched = self.run_block(gpus=6)
        self.assertEqual(self.probes, ["1 0 1 2 3"], "the 4 GPUs of the group, not all 6, peer-to-peer off")
        self.assertEqual(len(launched), 3)
        self.assertIn("rank0 p2p 1", log)
        self.assertEqual(self.follower_p2p, ["1 1", "2 1", "3 1"])
        self.run_block(gpus=8, extra_env={"CUDA_VISIBLE_DEVICES": "2,5"})
        self.assertEqual(self.probes, ["1 2 5"])
        for gpus, args in ((1, "--verbose INFO"), (4, "--cpu")):
            self.run_block(gpus=gpus, comfyui_args=args)
            self.assertEqual(self.probes, [], "no group, nothing to try")

    def test_nccls_own_path_when_host_memory_does_not_work(self):
        _, log, launched = self.run_block(gpus=2, probe="nccl")
        self.assertEqual(self.probes, ["1 0 1", "default 0 1"])
        self.assertIn("NCCL_P2P_DISABLE=1 did not work here; ComfyUI and the followers use NCCL's own path", log)
        self.assertIn("rank0 2 0", log)
        self.assertIn("rank0 p2p default", log)
        self.assertEqual(launched, ["1 1 2 --listen 0.0.0.0 --fast-disk --verbose INFO"])
        self.assertEqual(self.follower_p2p, ["1 default"])

    def test_one_gpu_when_the_gpus_cannot_exchange_tensors(self):
        code, log, launched = self.run_block(gpus=4, probe="fail")
        self.assertEqual(code, 0, log)
        self.assertEqual(self.probes, ["1 0 1 2 3", "default 0 1 2 3"])
        self.assertIn("WARNING: sequence parallelism is off: GPUs 0 1 2 3 could not exchange tensors; ComfyUI renders on one GPU", log)
        self.assertIn("rank0 1 unset", log)
        self.assertIn("rank0 p2p default", log, "failed tries leave NCCL as it was")
        self.assertEqual(launched, [])

    def test_a_peer_to_peer_setting_from_the_environment_is_kept(self):
        _, log, launched = self.run_block(gpus=2, probe="fail", extra_env={"NCCL_P2P_DISABLE": "0"})
        self.assertEqual(self.probes, ["0 0 1"], "no retry with another setting")
        self.assertIn("rank0 1 unset", log)
        self.assertEqual(launched, [])
        _, log, launched = self.run_block(gpus=2, probe="nccl", extra_env={"NCCL_P2P_DISABLE": "0"})
        self.assertEqual(self.probes, ["0 0 1"])
        self.assertIn("rank0 1 unset", log, "an explicit 0 is not NCCL's own choice either")
        _, log, launched = self.run_block(gpus=2, probe="pass", extra_env={"NCCL_P2P_DISABLE": "0"})
        self.assertEqual(self.probes, ["0 0 1"])
        self.assertNotIn("did not work here", log)
        self.assertEqual(self.follower_p2p, ["1 0"])

    def test_the_probe_can_be_skipped(self):
        _, log, launched = self.run_block(gpus=2, probe="fail", extra_env={"COMFY_SP_PROBE": "0"})
        self.assertEqual(self.probes, [])
        self.assertIn("rank0 2 0", log)
        self.assertEqual(len(launched), 1)


@unittest.skipIf(sys.platform == "win32", "bash scripts")
class TestIdleGateEntrypoint(unittest.TestCase):
    """docker/entrypoint.sh puts docker/idle_gate.py in front of ComfyUI when COMFYUI_IDLE_STOP_SECONDS is set."""

    @classmethod
    def setUpClass(cls):
        cls.script = read("docker", "entrypoint.sh")
        cls.function = re.search(r"^start_idle_gate\(\) \{\n.*?^\}$", cls.script, re.MULTILINE | re.DOTALL).group(0)

    def run_function(self, env):
        """The function with a stand-in python that records its argv; returns (argv or None, stdout, stderr)."""
        with tempfile.TemporaryDirectory() as tmp:
            record = os.path.join(tmp, "argv")
            with open(os.path.join(tmp, "python"), "w") as f:
                f.write(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > {record}\n')
            os.chmod(os.path.join(tmp, "python"), 0o755)
            harness = ('set -euo pipefail\nlog() { echo "entrypoint: $*" >&2; }\nis_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }\n'
                       "COMFY_ROOT=/opt/ComfyUI\nargs=(--listen 0.0.0.0 --port 8188 --fast-disk)\nextra=(--verbose INFO)\n"
                       f'{self.function}\nstart_idle_gate "$@"\necho "comfyui directly"')
            result = subprocess.run(["bash", "-c", harness, "_"], env={"PATH": f"{tmp}:/usr/bin:/bin", **env},
                                    capture_output=True, text=True, timeout=30)
            argv = open(record).read().splitlines() if os.path.exists(record) else None
            return argv, result.stdout, result.stderr

    def test_off_by_default(self):
        for env in ({}, {"COMFYUI_IDLE_STOP_SECONDS": "0"}, {"COMFYUI_IDLE_STOP_SECONDS": "soon"}):
            argv, out, _ = self.run_function(env)
            self.assertIsNone(argv, env)
            self.assertIn("comfyui directly", out)

    def test_the_gate_owns_the_port_and_comfyui_listens_behind_it(self):
        argv, out, err = self.run_function({"COMFYUI_IDLE_STOP_SECONDS": "90"})
        self.assertNotIn("comfyui directly", out, "the gate replaces the entrypoint (exec)")
        self.assertIn("stops after 90s without use", err)
        self.assertEqual(argv[0], "/opt/ComfyUI/docker/idle_gate.py")
        gate, comfyui = argv[1:argv.index("--")], argv[argv.index("--") + 1:]
        self.assertEqual(gate[gate.index("--listen") + 1], "0.0.0.0:8188")
        self.assertEqual(gate[gate.index("--upstream") + 1], "127.0.0.1:8189")
        self.assertEqual(gate[gate.index("--idle-seconds") + 1], "90")
        self.assertEqual(comfyui[:2], ["python", "main.py"])
        self.assertIn("--fast-disk", comfyui)
        self.assertIn("--verbose", comfyui)
        self.assertEqual(comfyui[-4:], ["--listen", "127.0.0.1", "--port", "8189"], "last, so they win over COMFYUI_ARGS")

    def test_the_gate_accepts_the_entrypoint_flags(self):
        argv, _, _ = self.run_function({"COMFYUI_IDLE_STOP_SECONDS": "90", "COMFYUI_PORT": "9000"})
        sys.path.insert(0, os.path.join(ROOT, "docker"))
        try:
            import idle_gate
        finally:
            sys.path.pop(0)
        args = idle_gate.parse_args(argv[1:])
        self.assertEqual(args.upstream, ("127.0.0.1", 9001))
        stubs = dict(args.stub)
        self.assertEqual(json.loads(stubs[("GET", "/queue")][1]), {"queue_running": [], "queue_pending": []})
        self.assertEqual(stubs[("GET", "/ws")][0], 503)
        self.assertIn(("GET", "/system_stats"), stubs, "the healthcheck must not start ComfyUI")
        self.assertIn(("GET", "/api/wrapper/workflows"), args.cache, "voxmin-backend's 10 s probe must not start it")
        self.assertEqual(args.busy, (("GET", "/queue"), ["queue_running", "queue_pending"]))
        self.assertEqual(args.bearer_env, "WRAPPER_AUTH_TOKEN")

    def test_not_with_sequence_parallelism(self):
        argv, out, err = self.run_function({"COMFYUI_IDLE_STOP_SECONDS": "90", "COMFY_SP_WORLD": "2"})
        self.assertIsNone(argv)
        self.assertIn("comfyui directly", out)
        self.assertIn("ignored with sequence parallelism", err)

    def test_runs_after_sequence_parallelism_and_before_comfyui(self):
        call = self.script.index('start_idle_gate "$@"\n')
        self.assertLess(self.script.index('start_sequence_parallel "$@"\n'), call)
        self.assertLess(call, self.script.index("exec python main.py"))

    def test_compose_stops_an_idle_comfyui(self):
        with open(os.path.join(ROOT, "docker-compose.yml")) as f:
            env = yaml.safe_load(f)["services"]["comfyui"]["environment"]
        self.assertIn("COMFYUI_IDLE_STOP_SECONDS=${COMFYUI_IDLE_STOP_SECONDS:-120}", env)


@unittest.skipIf(sys.platform == "win32", "bash scripts")
class TestTunnelSupervisor(unittest.TestCase):
    """start_tunnel from docker/entrypoint.sh, run against a fake cloudflared."""

    TOKEN = "eyJhIjoiZmFrZS10dW5uZWwtdG9rZW4ifQ=="
    FAKE = r'''
import http.server, os, sys, threading, time
args = sys.argv[1:]
print("args: " + " ".join(args), flush=True)
print("INF token from env: " + os.environ.get("TUNNEL_TOKEN", ""), flush=True)
if os.environ["FAKE_CLOUDFLARED"] == "exit":
    sys.exit(3)
host, port = args[args.index("--metrics") + 1].rsplit(":", 1)


class Ready(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/ready" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


threading.Thread(target=http.server.HTTPServer((host, int(port)), Ready).serve_forever, daemon=True).start()
time.sleep(60)
'''

    def run_supervisor(self, mode, seconds):
        script = read("docker", "entrypoint.sh")
        log = re.search(r"^log\(\) \{.*\}$", script, re.MULTILINE).group(0)
        start = script.index("TUNNEL_METRICS=127.0.0.1:20241")
        block = script[start:script.index("\n}\n", start) + 3]
        # As in the entrypoint: the token comes from the container's env, and is
        # unset (here at once) before ComfyUI starts, while cloudflared keeps running.
        unset = re.search(r"^unset .*CF_TUNNEL_TOKEN.*$", script, re.MULTILINE).group(0)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as bin_dir:
            fake = os.path.join(bin_dir, "cloudflared")
            with open(fake, "w") as f:
                f.write(f"#!{sys.executable}\n{self.FAKE}")
            os.chmod(fake, 0o755)
            os.symlink(sys.executable, os.path.join(bin_dir, "python"))
            # kill 0 ends the whole process group: the harness, the supervisor and the fake.
            harness = "\n".join(["set -euo pipefail", log, block, f"TUNNEL_METRICS=127.0.0.1:{self.port}",
                                 "start_tunnel", unset, f"sleep {seconds}", "kill 0"])
            env = {"PATH": bin_dir + os.pathsep + os.environ["PATH"], "FAKE_CLOUDFLARED": mode, "CF_TUNNEL_TOKEN": self.TOKEN}
            result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True, timeout=60, start_new_session=True)
        return result.stdout + result.stderr

    def test_restarts_with_backoff_and_masks_the_token(self):
        out = self.run_supervisor("exit", 3.5)
        self.assertIn(f"cloudflared: args: tunnel --no-autoupdate --metrics 127.0.0.1:{self.port} run", out)
        self.assertIn("cloudflared: INF token from env: ***", out, "the token reaches cloudflared through TUNNEL_TOKEN")
        self.assertNotIn(self.TOKEN, out)
        self.assertIn("entrypoint: cloudflared exited (status 3); restarting in 1s", out)
        self.assertIn("entrypoint: cloudflared exited (status 3); restarting in 2s", out)
        self.assertNotIn("tunnel ready", out)

    def test_restarts_keep_the_token_after_it_is_unset(self):
        out = self.run_supervisor("exit", 3.5)
        runs = [line for line in out.splitlines() if line.startswith("cloudflared: INF token from env:")]
        self.assertGreaterEqual(len(runs), 2, out)
        self.assertEqual(set(runs), {"cloudflared: INF token from env: ***"}, "every restart still gets the token")

    def test_logs_when_the_tunnel_is_ready(self):
        # The readiness probe starts a Python per try, a second apart: slow CI runners (macOS) need a wide window.
        out = self.run_supervisor("serve", 12)
        self.assertIn("entrypoint: tunnel ready: cloudflared registered an edge connection", out)
        self.assertNotIn("cloudflared exited", out)
        self.assertNotIn(self.TOKEN, out)


class TestPublishWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(read(".github", "workflows", "docker-publish.yml"))
        cls.jobs = cls.workflow["jobs"]

    @staticmethod
    def steps_using(job, action):
        return [s for s in job["steps"] if s.get("uses", "").startswith(action)]

    def test_both_targets_built_and_pushed(self):
        for job, target in (("build", "comfyui"), ("runpod", "runpod")):
            builds = self.steps_using(self.jobs[job], "docker/build-push-action")
            self.assertEqual({b["with"]["target"] for b in builds}, {target})
            self.assertTrue(any(b["with"].get("push") is True for b in builds), f"{job} never pushes")
            self.assertTrue(any(b["with"].get("load") is True for b in builds), f"{job} is not smoke-tested")

    def test_full_tags_unchanged(self):
        tags = self.steps_using(self.jobs["build"], "docker/metadata-action")[0]["with"]["tags"]
        self.assertIn("type=raw,value=latest,enable={{is_default_branch}}", tags)
        self.assertIn("type=sha,prefix=sha-,format=short", tags)

    def test_runpod_tags(self):
        meta = self.steps_using(self.jobs["runpod"], "docker/metadata-action")[0]["with"]
        self.assertIn("type=raw,value=runpod,enable={{is_default_branch}}", meta["tags"])
        self.assertIn("type=sha,prefix=runpod-sha-,format=short", meta["tags"])
        self.assertIn("type=semver,pattern={{version}},prefix=runpod-", meta["tags"])
        # :latest is the full image; the runpod job must never move it.
        self.assertIn("latest=false", meta["flavor"])
        self.assertEqual(meta["images"], "docker.io/${{ env.IMAGE }}")

    def test_separate_build_caches(self):
        def cache_refs(job):
            refs = set()
            for build in self.steps_using(self.jobs[job], "docker/build-push-action"):
                for key in ("cache-from", "cache-to"):
                    refs.update(re.findall(r":(buildcache[\w-]*)", build["with"].get(key, "")))
            return refs
        self.assertEqual(cache_refs("build"), {"buildcache"})
        self.assertEqual(cache_refs("runpod"), {"buildcache-runpod"})

    def test_runpod_smoke_test_builds_then_reuses_the_env(self):
        runs = "\n".join(s.get("run", "") for s in self.jobs["runpod"]["steps"])
        self.assertIn("-v runpod-smoke-ws:/workspace", runs)
        self.assertIn("COMFYUI_ARGS=--cpu", runs)
        self.assertIn("WRAPPER_AUTH_TOKEN=smoke-token", runs)
        self.assertIn('"401"', runs)
        self.assertIn("building it", runs)
        self.assertIn("reusing it", runs)
        self.assertIn("boot second", runs)
        self.assertIn("prefetch --dry-run", runs)

    def test_runpod_smoke_test_boots_through_the_bootstrap(self):
        # The production path: code tarball onto the volume, env from the archive, no rebuild.
        runs = "\n".join(s.get("run", "") for s in self.jobs["runpod"]["steps"])
        self.assertIn('git archive --format=tar.gz --prefix="ComfyUI-$sha/"', runs)
        self.assertIn('--entrypoint bash "$RUNPOD_SMOKE_TAG" /bootstrap.sh', runs)
        self.assertIn('-e COMFY_CODE_REF="$sha"', runs)
        self.assertIn("env key computed at boot", runs)
        self.assertIn("unpacked from /workspace/envs/", runs)
        self.assertIn('kill "$server"', runs)
        self.assertIn('code $sha found on the volume', runs)

    def test_full_smoke_tests_kept(self):
        runs = "\n".join(s.get("run", "") for s in self.jobs["build"]["steps"])
        self.assertIn("assert torch.version.cuda", runs)
        self.assertIn("prefetch --dry-run", runs)
        self.assertIn('"401"', runs)

    def test_full_smoke_tests_the_vast_tools(self):
        runs = "\n".join(s.get("run", "") for s in self.jobs["build"]["steps"])
        self.assertIn('--entrypoint cloudflared "$SMOKE_TAG" --version', runs)
        self.assertIn('--entrypoint s5cmd "$SMOKE_TAG" version', runs)
        self.assertIn('timeout 120 docker run --rm --network none -e COMFY_MODELS_S3_BUCKET=smoke "$SMOKE_TAG"', runs)
        self.assertIn("not starting ComfyUI", runs)


class TestReadme(unittest.TestCase):
    def test_documents_the_runpod_variant(self):
        readme = read("README.md")
        for text in ("docker.io/shivanshtalwar0/comfyui:runpod", "prefetch --env-only", "/workspace/envs/<key>.tar",
                     "docker/runpod-bootstrap.sh", "COMFY_CODE_REF", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"):
            self.assertTrue(text in readme, f"README must document {text!r}")

    def test_documents_vast_pods(self):
        readme = read("README.md")
        for text in ("`CF_TUNNEL_TOKEN`", "`COMFY_MODELS_S3_BUCKET`", "`COMFY_MODELS_S3_ENDPOINT`", "`COMFY_MODELS_S3_PREFIX`",
                     "`COMFY_MODELS_S3_INCLUDE`", "`AWS_ACCESS_KEY_ID`", "`COMFY_MODELS_PREFETCH`", "docker/s3_models_sync.py",
                     "#### vast.ai: no volume, no proxy"):
            self.assertTrue(text in readme, f"README must document {text!r}")


class TestDockerignore(unittest.TestCase):
    def test_excludes_weights_and_state(self):
        with open(os.path.join(ROOT, ".dockerignore")) as f:
            ignore = f.read()
        for excluded in (".git", "models/*", "output/*", "*.safetensors"):
            self.assertIn(excluded, ignore)

    def test_excludes_local_environments_and_secrets(self):
        # Every top-level dot-entry: .venv (~1.5 GB), tool caches, .claude
        # worktrees and .env secrets must never reach the image.
        with open(os.path.join(ROOT, ".dockerignore")) as f:
            lines = [line.strip() for line in f]
        self.assertIn("/.*", lines)


if __name__ == "__main__":
    unittest.main()
