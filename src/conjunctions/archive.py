"""Download and read the checksum-verified official ESA release."""

import gzip
import hashlib
from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import requests

from .schema import ARCHIVE_NAME, EXPECTED_COLUMNS, SOURCE_SHA256, SOURCE_URL


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path):
    actual = sha256(path)
    if actual != SOURCE_SHA256:
        raise ValueError(f"Archive SHA-256 mismatch: {actual}")
    return actual


def download_archive(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / ARCHIVE_NAME
    if destination.exists():
        verify_archive(destination)
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            with requests.get(SOURCE_URL, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        verify_archive(temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def read_messages(path):
    verify_archive(path)
    # Read in place: no extraction of untrusted archive member paths.
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.endswith("/raw_data_2015-2019.gz")]
        if len(matches) != 1:
            raise ValueError("Expected exactly one official raw message table")
        with archive.open(matches[0]) as compressed, gzip.GzipFile(fileobj=compressed) as stream:
            messages = pd.read_csv(stream)
    missing = set(EXPECTED_COLUMNS) - set(messages.columns)
    unexpected = set(messages.columns) - set(EXPECTED_COLUMNS)
    if missing or unexpected:
        raise ValueError(f"Unexpected release schema: missing={sorted(missing)}, extra={sorted(unexpected)}")
    return messages.loc[:, list(EXPECTED_COLUMNS)]
