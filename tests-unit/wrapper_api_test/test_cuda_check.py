"""Tests for docker/cuda_check.py: the idle gate's check that the driver still sees a GPU.

The CUDA driver is replaced with a stand-in, so these run on any machine.
"""

import ctypes
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "docker"))
try:
    import cuda_check
finally:
    sys.path.pop(0)


class FakeDriver:
    """libcuda as ctypes sees it: cuInit, cuDeviceGetCount and cuGetErrorString."""

    def __init__(self, init=0, count=1, count_status=0, error_names=True):
        self.init, self.count, self.count_status = init, count, count_status
        self.calls = []
        if not error_names:
            self.cuGetErrorString = None
            del self.cuGetErrorString

    def cuInit(self, flags):
        self.calls.append(("cuInit", flags))
        return self.init

    def cuDeviceGetCount(self, ref):
        self.calls.append(("cuDeviceGetCount",))
        ctypes.cast(ref, ctypes.POINTER(ctypes.c_int))[0] = self.count
        return self.count_status

    def cuGetErrorString(self, status, ref):
        names = {100: b"no CUDA-capable device is detected", 999: b"unknown error"}
        if status not in names:
            return 1
        ctypes.cast(ref, ctypes.POINTER(ctypes.c_char_p))[0] = names[status]
        return 0


def loader(driver, missing=()):
    def load(name):
        if name in missing or driver is None:
            raise OSError(f"{name}: cannot open shared object file")
        return driver
    return load


class CudaCheckTest(unittest.TestCase):
    def test_a_driver_that_sees_a_gpu_passes(self):
        driver = FakeDriver(count=1)
        self.assertIsNone(cuda_check.check(loader(driver)))
        self.assertEqual(driver.calls, [("cuInit", 0), ("cuDeviceGetCount",)])

    def test_a_driver_that_lost_the_gpu_fails_with_the_cuda_error(self):
        # What a GPU hung in the driver gives (nvidia-smi: ERR!); torch said "No CUDA GPUs are available".
        self.assertEqual(cuda_check.check(loader(FakeDriver(init=100))),
                         "cuInit failed: CUDA error 100 (no CUDA-capable device is detected)")
        self.assertEqual(cuda_check.check(loader(FakeDriver(init=999))), "cuInit failed: CUDA error 999 (unknown error)")

    def test_an_error_without_a_name_still_says_its_number(self):
        self.assertEqual(cuda_check.check(loader(FakeDriver(init=3))), "cuInit failed: CUDA error 3")
        self.assertEqual(cuda_check.check(loader(FakeDriver(init=3, error_names=False))), "cuInit failed: CUDA error 3")

    def test_no_gpu_or_a_failed_count_fails(self):
        self.assertEqual(cuda_check.check(loader(FakeDriver(count=0))), "the CUDA driver sees no GPU")
        self.assertEqual(cuda_check.check(loader(FakeDriver(count_status=999))),
                         "cuDeviceGetCount failed: CUDA error 999 (unknown error)")

    def test_the_unversioned_library_is_the_fallback(self):
        self.assertIsNone(cuda_check.check(loader(FakeDriver(), missing=("libcuda.so.1",))))

    def test_no_driver_fails(self):
        self.assertEqual(cuda_check.check(loader(None)), "the CUDA driver (libcuda.so.1) could not be loaded")

    @unittest.skipIf(sys.platform == "win32", "the image is Linux")
    def test_the_script_exits_non_zero_with_one_line_the_gate_reports(self):
        # The gate answers with the last line a failed check printed. A machine with
        # a working CUDA driver passes instead, which is fine too.
        result = subprocess.run([sys.executable, os.path.join(ROOT, "docker", "cuda_check.py")],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            self.assertEqual(result.stdout, "")
        else:
            self.assertEqual(result.returncode, 1)
            self.assertRegex(result.stdout, r"^GPU check: .+\n$")


if __name__ == "__main__":
    unittest.main()
