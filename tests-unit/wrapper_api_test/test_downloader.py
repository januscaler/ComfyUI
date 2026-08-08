"""Tests for the model downloader's truncation protection.

An interrupted download used to leave a silently truncated model file (the
safetensors loader then crashes with 'shape [...] is invalid for input of
size ...'). The resolver now verifies byte sizes against the server metadata
before and after copying to the models dir.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import huggingface_hub

from comfy import model_downloader

URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"


def _make_file(path, size):
    with open(path, "wb") as f:
        f.write(b"x" * size)
    return path


class _Meta:
    def __init__(self, size):
        self.size = size


class TestResolveUrlDownloadVerification(unittest.TestCase):
    def test_truncated_cache_blob_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 100)  # truncated cache
            dest = os.path.join(tmp, "model.safetensors")
            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest), "truncated file must not be copied to the models dir")

    def test_truncated_copy_is_rejected_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 1000)
            dest = os.path.join(tmp, "model.safetensors")

            def truncated_copy(src, dst):
                _make_file(dst, 100)  # simulate a disk-full copy

            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)), \
                 mock.patch.object(model_downloader, "_copy_to_dest", side_effect=truncated_copy):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest), "truncated copy must be removed")

    def test_full_download_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 1000)
            dest = os.path.join(tmp, "model.safetensors")
            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(huggingface_hub, "get_hf_file_metadata", return_value=_Meta(1000)):
                result = model_downloader._download_huggingface_resolve_url(URL, dest)
            self.assertTrue(result)
            self.assertEqual(os.path.getsize(dest), 1000)


class TestAtFormatDownloadVerification(unittest.TestCase):
    def test_truncated_copy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = _make_file(os.path.join(tmp, "blob"), 500)
            dest = os.path.join(tmp, "model.safetensors")

            def truncated_copy(src, dst):
                _make_file(dst, 10)

            with mock.patch.object(huggingface_hub, "hf_hub_download", return_value=cache_file), \
                 mock.patch.object(model_downloader, "_copy_to_dest", side_effect=truncated_copy):
                result = model_downloader._download_huggingface_file(
                    "Comfy-Org/MiniMax-H3@text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest))


if __name__ == "__main__":
    unittest.main()
