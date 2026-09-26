"""docker/s3_models_sync.py: a vast.ai pod copies its model weights from R2 before ComfyUI starts.

The end-to-end tests drive main() against a fake s5cmd on PATH that serves a
local directory as the bucket, so the skip, retry and temp-rename behaviour is
exercised without any network.
"""

import contextlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load():
    spec = importlib.util.spec_from_file_location("s3_models_sync", os.path.join(ROOT, "docker", "s3_models_sync.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


sync = _load()

KEY_ID = "AKIDTESTKEY0123"
SECRET = "s3cr3t-Value-xyz"

# `ls` lists <FAKE_S3_ROOT>/<bucket>/<key> the way s5cmd 2.3 prints --json;
# `cp` copies one object. FAKE_S3_FAIL ({key: times, -1 = always}),
# FAKE_S3_TRUNCATE (keys) and FAKE_S3_LS_FAIL inject failures; FAKE_S3_SLOW
# ({key: [seconds, times]}) writes half the object, then stalls that long.
FAKE_S5CMD = r'''
import json, os, sys, time

root, state = os.environ["FAKE_S3_ROOT"], os.environ["FAKE_S3_STATE"]
with open(os.path.join(state, "calls.jsonl"), "a") as calls:
    calls.write(json.dumps({"argv": sys.argv[1:], "region": os.environ.get("AWS_REGION")}) + "\n")
args = sys.argv[1:]
while args[0].startswith("--"):
    args = args[2:] if args[0] in ("--endpoint-url", "--log") else args[1:]
command, args = args[0], args[1:]


def split(url):
    bucket, _, key = url[len("s3://"):].partition("/")
    return bucket, key


if command == "ls":
    if os.environ.get("FAKE_S3_LS_FAIL"):
        sys.exit("ERROR \"ls\": RequestError: send request failed")
    bucket, pattern = split(args[0])
    prefix, base, found = pattern.rstrip("*"), os.path.join(root, bucket), False
    for dirpath, _dirs, names in sorted(os.walk(base)):
        for name in sorted(names):
            key = os.path.relpath(os.path.join(dirpath, name), base).replace(os.sep, "/")
            if key.startswith(prefix):
                found = True
                entry = {"key": "s3://%s/%s" % (bucket, key), "etag": "x", "type": "file"}
                size = os.path.getsize(os.path.join(dirpath, name))
                if size:
                    entry["size"] = size
                print(json.dumps(entry))
    if not found:
        sys.exit("ERROR \"ls %s\": no object found" % args[0])
elif command == "cp":
    src, dst = args[-2], args[-1]
    bucket, key = split(src)
    counter = os.path.join(state, "count-" + key.replace("/", "_"))
    tries = int(open(counter).read()) if os.path.exists(counter) else 0
    open(counter, "w").write(str(tries + 1))
    fail = json.loads(os.environ.get("FAKE_S3_FAIL", "{}"))
    if key in fail and (fail[key] < 0 or tries < fail[key]):
        sys.exit("ERROR \"cp %s\": AccessDenied for %s / %s" % (src, os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]))
    data = open(os.path.join(root, bucket, key), "rb").read()
    if key in os.environ.get("FAKE_S3_TRUNCATE", "").split(","):
        data = data[:len(data) // 2]
    slow = json.loads(os.environ.get("FAKE_S3_SLOW", "{}")).get(key)
    with open(dst, "wb") as out:
        if slow and (slow[1] < 0 or tries < slow[1]):
            out.write(data[:len(data) // 2])
            out.flush()
            time.sleep(slow[0])
            data = data[len(data) // 2:]
        out.write(data)
else:
    sys.exit("fake s5cmd: unsupported " + command)
'''


class TestPlanning(unittest.TestCase):
    def test_prefix_is_normalized(self):
        self.assertEqual(sync.normalize_prefix("models/"), "models/")
        self.assertEqual(sync.normalize_prefix("/models"), "models/")
        self.assertEqual(sync.normalize_prefix(" weights/v2 "), "weights/v2/")
        self.assertEqual(sync.normalize_prefix("/"), "")

    def test_include_patterns(self):
        self.assertEqual(sync.parse_include(""), [])
        self.assertEqual(sync.parse_include("unet/*int8*, vae/* ,"), ["unet/*int8*", "vae/*"])
        files = [sync.RemoteFile(p, "s3://b/models/" + p, 1) for p in ("unet/h3_int8.safetensors", "unet/h3_bf16.safetensors", "vae/v.safetensors")]
        self.assertEqual([f.path for f in sync.select(files, ["unet/*int8*", "vae/*"])], ["unet/h3_int8.safetensors", "vae/v.safetensors"])
        self.assertEqual(sync.select(files, []), files)
        self.assertEqual([f.path for f in sync.select(files, ["*int8*"])], ["unet/h3_int8.safetensors"], "* also matches /")

    def test_listing(self):
        output = "\n".join(json.dumps(e) for e in (
            {"key": "s3://b/models/vae/v.safetensors", "type": "file", "size": 7},
            {"key": "s3://b/models/unet/", "type": "directory"},
            {"key": "s3://b/models/empty.txt", "type": "file"},  # s5cmd omits a zero size
            {"key": "s3://b/models/unet/u.safetensors", "type": "file", "size": 9},
        )) + "\n"
        files = sync.parse_listing(output, "b", "models/")
        self.assertEqual([(f.path, f.url, f.size) for f in files], [
            ("empty.txt", "s3://b/models/empty.txt", 0),
            ("unet/u.safetensors", "s3://b/models/unet/u.safetensors", 9),
            ("vae/v.safetensors", "s3://b/models/vae/v.safetensors", 7),
        ])

    def test_listing_refuses_keys_that_escape_the_models_dir(self):
        for key in ("s3://b/models/../etc/passwd", "s3://b/models//abs", "s3://b/models/a/./b"):
            with self.assertRaises(ValueError):
                sync.parse_listing(json.dumps({"key": key, "size": 1}), "b", "models/")

    def test_plan_keeps_files_of_the_same_size(self):
        with tempfile.TemporaryDirectory() as models:
            os.makedirs(os.path.join(models, "vae"))
            with open(os.path.join(models, "vae", "same.safetensors"), "wb") as f:
                f.write(b"x" * 5)
            with open(os.path.join(models, "vae", "other.safetensors"), "wb") as f:
                f.write(b"x" * 4)
            files = [sync.RemoteFile(p, "s3://b/" + p, 5) for p in ("vae/same.safetensors", "vae/other.safetensors", "vae/new.safetensors")]
            missing, present = sync.plan(files, models)
            self.assertEqual([f.path for f in present], ["vae/same.safetensors"])
            self.assertEqual([f.path for f in missing], ["vae/other.safetensors", "vae/new.safetensors"])


@unittest.skipIf(sys.platform == "win32", "fake s5cmd is a script with a shebang")
class TestSync(unittest.TestCase):
    OBJECTS = {
        "models/diffusion_models/h3_int8.safetensors": 3000,
        "models/text_encoders/enc_nvfp4.safetensors": 2000,
        "models/vae/vae.safetensors": 1000,
        "other/not-a-model.bin": 10,
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = self.tmp.name
        self.bucket_root, self.state, self.models = (os.path.join(base, d) for d in ("s3", "state", "models"))
        bin_dir = os.path.join(base, "bin")
        for d in (self.state, self.models, bin_dir):
            os.makedirs(d)
        for key, size in self.OBJECTS.items():
            self.put(key, size)
        fake = os.path.join(bin_dir, "s5cmd")
        with open(fake, "w") as f:
            f.write(f"#!{sys.executable}\n{FAKE_S5CMD}")
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
        self.env = {
            "PATH": bin_dir + os.pathsep + os.environ.get("PATH", ""),
            "FAKE_S3_ROOT": self.bucket_root,
            "FAKE_S3_STATE": self.state,
            "COMFY_MODELS_S3_BUCKET": "weights",
            "COMFY_MODELS_S3_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
            "AWS_ACCESS_KEY_ID": KEY_ID,
            "AWS_SECRET_ACCESS_KEY": SECRET,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, key, size):
        path = os.path.join(self.bucket_root, "weights", *key.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(bytes(i % 251 for i in range(size)))

    def run_sync(self, *args, free=10**15, **env):
        out = io.StringIO()
        with contextlib.redirect_stderr(out), mock.patch.object(sync, "RETRY_DELAY_SECONDS", 0), \
                mock.patch.object(sync.shutil, "disk_usage", return_value=mock.Mock(free=free)):
            code = sync.main([self.models, *args], {**self.env, **env})
        return code, out.getvalue()

    def calls(self, command=None):
        path = os.path.join(self.state, "calls.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            calls = [json.loads(line) for line in f]
        return [c for c in calls if command is None or command in c["argv"]]

    def model(self, path):
        return os.path.join(self.models, *path.split("/"))

    def leftovers(self):
        return [name for _root, _dirs, names in os.walk(self.models) for name in names if sync.PARTIAL_SUFFIX in name]

    def assert_same_bytes(self, path, key):
        with open(self.model(path), "rb") as local, open(os.path.join(self.bucket_root, "weights", *key.split("/")), "rb") as remote:
            self.assertEqual(local.read(), remote.read())

    def test_downloads_the_prefix_into_the_models_dir(self):
        with mock.patch.object(sync, "FILES_AT_ONCE", 1):  # one at a time, to see the order
            code, out = self.run_sync()
        self.assertEqual(code, 0, out)
        for key in self.OBJECTS:
            if key.startswith("models/"):
                self.assert_same_bytes(key[len("models/"):], key)
        self.assertFalse(os.path.exists(self.model("other/not-a-model.bin")))
        self.assertFalse(os.path.exists(os.path.join(self.models, "models")))
        self.assertEqual(self.leftovers(), [])
        self.assertIn("3 file(s), 0.0 GB downloaded in", out)
        self.assertIn("MB/s", out)
        # s5cmd is pointed at R2 with region auto; every copy goes to a partial name first.
        (ls,) = self.calls("ls")
        self.assertEqual(ls["argv"][:2], ["--endpoint-url", "https://acct.r2.cloudflarestorage.com"])
        self.assertEqual(ls["argv"][-1], "s3://weights/models/*")
        self.assertEqual(ls["region"], "auto")
        cps = self.calls("cp")
        self.assertEqual(len(cps), 3)
        for call in cps:
            argv = call["argv"]
            self.assertIn("--raw", argv)
            self.assertEqual(argv[argv.index("--concurrency") + 1], str(sync.PARTS_PER_FILE))
            self.assertTrue(argv[-1].endswith(".s3-partial"), argv)
            self.assertEqual(argv[-1], os.path.join(os.path.realpath(self.models), *argv[-2][len("s3://weights/models/"):].split("/")) + ".s3-partial")
        self.assertEqual(cps[0]["argv"][-2], "s3://weights/models/diffusion_models/h3_int8.safetensors", "biggest first")
        self.assertNotIn(SECRET, out)
        self.assertNotIn(KEY_ID, out)

    def test_a_restarted_container_downloads_nothing(self):
        self.assertEqual(self.run_sync()[0], 0)
        code, out = self.run_sync()
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.calls("cp")), 3, "the second run copies nothing")
        self.assertIn("3 already present", out)
        self.assertIn("0 file(s), 0.0 GB downloaded", out)

    def test_a_file_of_another_size_is_replaced(self):
        os.makedirs(os.path.dirname(self.model("vae/vae.safetensors")))
        with open(self.model("vae/vae.safetensors"), "wb") as f:
            f.write(b"truncated")
        code, out = self.run_sync("--dry-run")
        self.assertEqual(code, 0, out)
        self.assertIn("missing  vae/vae.safetensors", out)
        self.assertEqual(self.calls("cp"), [], "a dry run copies nothing")
        code, out = self.run_sync()
        self.assertEqual(code, 0, out)
        self.assert_same_bytes("vae/vae.safetensors", "models/vae/vae.safetensors")

    def test_include_and_prefix(self):
        self.put("weights/v2/unet/u_int8.safetensors", 50)
        self.put("weights/v2/unet/u_bf16.safetensors", 60)
        self.put("weights/v2/vae/v.safetensors", 70)
        code, out = self.run_sync(COMFY_MODELS_S3_PREFIX="weights/v2", COMFY_MODELS_S3_INCLUDE="unet/*int8*, vae/*")
        self.assertEqual(code, 0, out)
        self.assert_same_bytes("unet/u_int8.safetensors", "weights/v2/unet/u_int8.safetensors")
        self.assert_same_bytes("vae/v.safetensors", "weights/v2/vae/v.safetensors")
        self.assertFalse(os.path.exists(self.model("unet/u_bf16.safetensors")))

    def test_a_transient_failure_is_retried(self):
        code, out = self.run_sync(FAKE_S3_FAIL=json.dumps({"models/vae/vae.safetensors": 1}))
        self.assertEqual(code, 0, out)
        self.assertIn("FAILED   vae/vae.safetensors (attempt 1/3)", out)
        self.assert_same_bytes("vae/vae.safetensors", "models/vae/vae.safetensors")
        self.assertEqual(len(self.calls("cp")), 4)

    def test_a_persistent_failure_exits_non_zero_without_a_model_or_the_credentials(self):
        code, out = self.run_sync(FAKE_S3_FAIL=json.dumps({"models/vae/vae.safetensors": -1}))
        self.assertEqual(code, 1, out)
        self.assertIn("1 of 3 file(s) still failed after 3 attempts: vae/vae.safetensors", out)
        self.assertFalse(os.path.exists(self.model("vae/vae.safetensors")))
        self.assertEqual(self.leftovers(), [])
        self.assertTrue(os.path.exists(self.model("diffusion_models/h3_int8.safetensors")), "the other files still land")
        self.assertIn("AccessDenied for *** / ***", out)
        self.assertNotIn(SECRET, out)
        self.assertNotIn(KEY_ID, out)

    def test_a_short_download_never_lands_under_the_models_name(self):
        code, out = self.run_sync(FAKE_S3_TRUNCATE="models/vae/vae.safetensors")
        self.assertEqual(code, 1, out)
        self.assertIn("got 500 bytes, the object has 1000", out)
        self.assertFalse(os.path.exists(self.model("vae/vae.safetensors")))
        self.assertEqual(self.leftovers(), [])

    def test_leftovers_of_a_killed_sync_are_removed(self):
        os.makedirs(os.path.dirname(self.model("vae/x")))
        os.makedirs(os.path.dirname(self.model("loras/x")))
        for name in ("vae/vae.safetensors.s3-partial", "vae/vae.safetensors.s3-partial123456", "loras/gone.safetensors.s3-partial"):
            with open(self.model(name), "wb") as f:
                f.write(b"half")
        with open(self.model("loras/mine.safetensors"), "wb") as f:
            f.write(b"not in the bucket")
        code, out = self.run_sync()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.leftovers(), [])
        self.assertTrue(os.path.exists(self.model("loras/mine.safetensors")), "files not in the bucket are left alone")

    def test_missing_credentials_fail_before_s5cmd_runs(self):
        env = dict(self.env)
        del env["AWS_SECRET_ACCESS_KEY"]
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            code = sync.main([self.models], env)
        self.assertEqual(code, 1)
        self.assertIn("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required", out.getvalue())
        self.assertEqual(self.calls(), [])

    def test_nothing_to_sync_is_an_error(self):
        code, out = self.run_sync(COMFY_MODELS_S3_PREFIX="missing/")
        self.assertEqual(code, 1)
        self.assertIn("no objects under s3://weights/missing/", out)
        self.assertEqual(len(self.calls("ls")), 1, "an empty prefix is not retried")
        code, out = self.run_sync(COMFY_MODELS_S3_INCLUDE="loras/*")
        self.assertEqual(code, 1)
        self.assertIn("matching COMFY_MODELS_S3_INCLUDE=loras/*", out)

    def test_a_listing_that_keeps_failing_is_an_error(self):
        code, out = self.run_sync(FAKE_S3_LS_FAIL="1")
        self.assertEqual(code, 1)
        self.assertEqual(len(self.calls("ls")), sync.ATTEMPTS)
        self.assertIn("could not list s3://weights/models/*", out)

    def test_a_listing_that_hangs_is_a_failed_attempt(self):
        with mock.patch.object(sync.subprocess, "run", side_effect=sync.subprocess.TimeoutExpired("s5cmd", 300)) as run, \
                mock.patch.object(sync, "RETRY_DELAY_SECONDS", 0), contextlib.redirect_stderr(io.StringIO()) as out:
            self.assertIsNone(sync.list_objects(["s5cmd"], "s3://weights/models/*", self.env))
        self.assertEqual(run.call_count, sync.ATTEMPTS)
        self.assertIn("failed (attempt 3/3): timed out after 300s", out.getvalue())

    def test_the_disk_must_hold_the_download_and_a_margin(self):
        need = 6000
        code, out = self.run_sync(free=need + sync.DISK_MARGIN_BYTES - 1)
        self.assertEqual(code, 1)
        self.assertIn("0.0 GB to download plus a 5.0 GB margin, but only 5.0 GB free under", out)
        self.assertEqual(self.calls("cp"), [])
        code, out = self.run_sync(free=need + sync.DISK_MARGIN_BYTES)
        self.assertEqual(code, 0, out)

    def test_a_stuck_download_is_killed_and_retried(self):
        with mock.patch.object(sync, "DOWNLOAD_TIMEOUT_FLOOR_SECONDS", 1), mock.patch.object(sync, "SLOWEST_BYTES_PER_SECOND", 1e12):
            code, out = self.run_sync(FAKE_S3_SLOW=json.dumps({"models/vae/vae.safetensors": [30, 1]}))
        self.assertEqual(code, 0, out)
        self.assertIn("FAILED   vae/vae.safetensors (attempt 1/3): timed out after 1s", out)
        self.assert_same_bytes("vae/vae.safetensors", "models/vae/vae.safetensors")

    def test_a_download_that_keeps_stalling_fails_without_leftovers(self):
        with mock.patch.object(sync, "DOWNLOAD_TIMEOUT_FLOOR_SECONDS", 1), mock.patch.object(sync, "SLOWEST_BYTES_PER_SECOND", 1e12):
            code, out = self.run_sync(FAKE_S3_SLOW=json.dumps({"models/vae/vae.safetensors": [30, -1]}))
        self.assertEqual(code, 1, out)
        self.assertEqual(out.count("timed out after 1s"), sync.ATTEMPTS)
        self.assertIn("still failed after 3 attempts: vae/vae.safetensors", out)
        self.assertFalse(os.path.exists(self.model("vae/vae.safetensors")))
        self.assertEqual(self.leftovers(), [], "the half-written partial of a killed download is removed")

    def test_the_timeout_scales_with_the_size(self):
        seen = []
        real_run = sync.subprocess.run

        def run(argv, **kwargs):
            seen.append((argv[argv.index("--json") + 1] if "--json" in argv else "cp", kwargs["timeout"]))
            return real_run(argv, **kwargs)

        with mock.patch.object(sync.subprocess, "run", side_effect=run):
            self.run_sync()
        self.assertEqual(seen, [("ls", sync.LIST_TIMEOUT_SECONDS), ("cp", 600), ("cp", 600), ("cp", 600)], "the floor, for small files")
        f = sync.RemoteFile("unet/big.safetensors", "s3://weights/models/unet/big.safetensors", 40 * 10**9)
        with mock.patch.object(sync.subprocess, "run", side_effect=sync.subprocess.TimeoutExpired("s5cmd", 2000)) as run:
            error = sync.download(["s5cmd"], f, self.models, self.env)
        self.assertEqual(run.call_args.kwargs["timeout"], 2000, "40 GB at 20 MB/s")
        self.assertEqual(error, "timed out after 2000s (slower than 20 MB/s)")

    def test_a_filesystem_error_is_a_failed_attempt_not_a_traceback(self):
        with open(os.path.join(self.models, "vae"), "w") as f:
            f.write("a file where the vae folder should be")
        code, out = self.run_sync()
        self.assertEqual(code, 1, out)
        self.assertIn("FAILED   vae/vae.safetensors (attempt 3/3): [Errno 17] File exists", out)
        self.assertNotIn("Traceback", out)
        self.assertTrue(os.path.exists(self.model("diffusion_models/h3_int8.safetensors")), "the other files still land")

    def test_progress_lines_while_downloading(self):
        with mock.patch.object(sync, "PROGRESS_SECONDS", 0.2):
            code, out = self.run_sync(FAKE_S3_SLOW=json.dumps({"models/vae/vae.safetensors": [1.5, 1]}))
        self.assertEqual(code, 0, out)
        lines = [line for line in out.splitlines() if line.startswith("s3-sync: progress ")]
        self.assertGreaterEqual(len(lines), 3, out)
        for line in lines:
            self.assertRegex(line, r"^s3-sync: progress [0-3]/3 file\(s\), 0\.0 GB of 0\.0 GB in \d+s \(\d+ MB/s\)$")
        self.assertIn("progress 2/3 file(s)", lines[-1], "the two fast files are done while the slow one runs")

    def test_partial_bytes_counts_what_s5cmd_has_written(self):
        f = sync.RemoteFile("vae/vae.safetensors", "s3://weights/models/vae/vae.safetensors", 1000)
        self.assertEqual(sync.partial_bytes(self.models, f), 0, "no folder yet")
        os.makedirs(self.model("vae"))
        with open(self.model("vae/vae.safetensors.s3-partial8123"), "wb") as out:
            out.write(b"x" * 300)
        with open(self.model("vae/other.safetensors.s3-partial"), "wb") as out:
            out.write(b"x" * 999)
        self.assertEqual(sync.partial_bytes(self.models, f), 300)
        with open(self.model("vae/vae.safetensors.s3-partial"), "wb") as out:
            out.write(b"x" * 900)
        self.assertEqual(sync.partial_bytes(self.models, f), 1000, "capped at the object's size")

    def test_empty_prefix_means_models_and_slash_means_the_bucket_root(self):
        code, out = self.run_sync(COMFY_MODELS_S3_PREFIX="")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.calls("ls")[-1]["argv"][-1], "s3://weights/models/*")
        code, out = self.run_sync(COMFY_MODELS_S3_PREFIX="/")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.calls("ls")[-1]["argv"][-1], "s3://weights/*")
        self.assert_same_bytes("other/not-a-model.bin", "other/not-a-model.bin")
        self.assert_same_bytes("models/vae/vae.safetensors", "models/vae/vae.safetensors")


if __name__ == "__main__":
    unittest.main()
