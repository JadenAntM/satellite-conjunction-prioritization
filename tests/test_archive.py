import hashlib

import pytest
import requests

from conjunctions import archive


def test_checksum_failure_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "dataset.zip"
    path.write_bytes(b"incorrect archive")
    monkeypatch.setattr(archive, "SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        archive.verify_archive(path)


def test_valid_existing_download_does_not_use_network(tmp_path, monkeypatch):
    path = tmp_path / archive.ARCHIVE_NAME
    path.write_bytes(b"test fixture")
    monkeypatch.setattr(archive, "SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Unnecessary download"))
    assert archive.download_archive(tmp_path) == path


def test_failed_download_leaves_no_partial_archive(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("test connection failure")
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(requests.ConnectionError):
        archive.download_archive(tmp_path)
    assert list(tmp_path.iterdir()) == []
