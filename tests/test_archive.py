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


def make_zip(tmp_path, members):
    import gzip
    import zipfile
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for name, frame in members.items():
            bundle.writestr(name, gzip.compress(frame.to_csv(index=False).encode()))
    return path


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate member"])
def test_unexpected_release_layout_is_rejected(tmp_path, monkeypatch, messages, change):
    frame, members = messages.copy(), {}
    if change == "missing":
        frame = frame.drop(columns="AP")
    elif change == "extra":
        frame["surprise"] = 1.
    members["data/raw_data_2015-2019.gz"] = frame
    if change == "duplicate member":
        members["copy/raw_data_2015-2019.gz"] = frame
    path = make_zip(tmp_path, members)
    monkeypatch.setattr(archive, "SOURCE_SHA256", archive.sha256(path))
    with pytest.raises(ValueError, match="schema|exactly one"):
        archive.read_messages(path)


def test_release_is_read_in_schema_order(tmp_path, monkeypatch, messages):
    path = make_zip(tmp_path, {"data/raw_data_2015-2019.gz": messages[messages.columns[::-1]]})
    monkeypatch.setattr(archive, "SOURCE_SHA256", archive.sha256(path))
    assert list(archive.read_messages(path).columns) == list(messages.columns)
