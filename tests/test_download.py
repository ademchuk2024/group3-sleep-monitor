"""Tests for scripts/download_data.py. They never touch the internet.

Instead of a website we "download" from a file on this computer, using a
file:// address, which Python's downloader treats the same way.
"""

import hashlib
import importlib.util
import os
import zipfile

import pytest

# The script is not part of the sleepmon package, so load it by its path.
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "download_data.py")
spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """The script pauses between retries. Skip the pauses so tests stay fast."""
    monkeypatch.setattr(download_data.time, "sleep", lambda seconds: None)


def make_source(tmp_path, content=b"hello sleep data"):
    source = tmp_path / "source.txt"
    source.write_bytes(content)
    return source.as_uri(), hashlib.sha256(content).hexdigest()


def test_checksum_matches_pythons_own():
    # Compare against a value anyone can verify: the SHA-256 of nothing at all.
    assert hashlib.sha256(b"").hexdigest() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_file_checksum(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert download_data.file_checksum(str(path), "sha256") == hashlib.sha256(b"abc").hexdigest()
    assert download_data.file_checksum(str(path), "md5") == hashlib.md5(b"abc").hexdigest()


def test_a_good_download_is_kept(tmp_path):
    url, checksum = make_source(tmp_path)
    destination = tmp_path / "out" / "nested" / "file.txt"
    assert download_data.download_file(url, str(destination), checksum) == "downloaded"
    assert destination.read_bytes() == b"hello sleep data"
    assert not os.path.exists(str(destination) + ".part")


def test_a_file_we_already_have_is_not_downloaded_again(tmp_path):
    url, checksum = make_source(tmp_path)
    destination = tmp_path / "file.txt"
    download_data.download_file(url, str(destination), checksum)
    assert download_data.download_file("file:///does/not/exist", str(destination), checksum) == "skipped"


def test_a_download_with_the_wrong_checksum_is_thrown_away(tmp_path):
    url, _ = make_source(tmp_path)
    destination = tmp_path / "file.txt"
    with pytest.raises(RuntimeError, match="checksum did not match"):
        download_data.download_file(url, str(destination), "0" * 64, attempts=2)
    assert not destination.exists()                      # no bad file left behind
    assert not os.path.exists(str(destination) + ".part")


def test_a_damaged_copy_on_disk_is_replaced(tmp_path):
    url, checksum = make_source(tmp_path)
    destination = tmp_path / "file.txt"
    destination.write_bytes(b"half a downlo")            # a broken earlier download
    assert download_data.download_file(url, str(destination), checksum) == "downloaded"
    assert destination.read_bytes() == b"hello sleep data"


def test_a_missing_source_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="Could not download"):
        download_data.download_file((tmp_path / "nope.txt").as_uri(), str(tmp_path / "out.txt"), "0" * 64, attempts=2)


def test_parse_checksum_list():
    text = "aaa LICENSE.txt\nbbb motion/1_acceleration.txt\n\nccc acc/MECSLEEP01_left wrist_x.bin\n"
    assert download_data.parse_checksum_list(text) == {
        "LICENSE.txt": "aaa",
        "motion/1_acceleration.txt": "bbb",
        "acc/MECSLEEP01_left wrist_x.bin": "ccc",        # file names may contain spaces
    }


@pytest.mark.parametrize("path, safe", [
    ("motion/1.txt", True),
    ("Bidslab00/1/hr.csv", True),
    ("../outside.txt", False),
    ("a/../../outside.txt", False),
    ("/etc/passwd", False),
    ("a\\..\\b", False),
])
def test_paths_that_would_escape_the_data_folder_are_refused(path, safe):
    assert download_data.is_safe_relative_path(path) is safe


def test_certificate_checking_is_on():
    import ssl
    assert download_data.SSL_CONTEXT.verify_mode == ssl.CERT_REQUIRED
    assert download_data.SSL_CONTEXT.check_hostname is True


def test_dreamt_is_not_downloadable():
    # DREAMT is restricted access. The script must not know how to fetch it.
    source = open(SCRIPT).read()
    assert "dreamt" not in {name.lower() for name in download_data.PHYSIONET}
    assert "physionet.org/files/dreamt" not in source.lower()


def test_newcastle_unzips_only_inside_its_folder(tmp_path, monkeypatch):
    # Build a small zip, point the script at it, and check it lands in data/raw/newcastle.
    archive = tmp_path / "fake.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("dataset/psg/a.txt", "hello")
    monkeypatch.setattr(download_data, "RAW_FOLDER", str(tmp_path / "raw"))
    monkeypatch.setattr(download_data, "NEWCASTLE_URL", archive.as_uri())
    monkeypatch.setattr(download_data, "NEWCASTLE_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    download_data.download_newcastle()
    assert (tmp_path / "raw" / "newcastle" / "dataset" / "psg" / "a.txt").read_text() == "hello"


def test_a_zip_that_tries_to_write_outside_is_refused(tmp_path, monkeypatch):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped.txt", "bad")
    monkeypatch.setattr(download_data, "RAW_FOLDER", str(tmp_path / "raw"))
    monkeypatch.setattr(download_data, "NEWCASTLE_URL", archive.as_uri())
    monkeypatch.setattr(download_data, "NEWCASTLE_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="suspicious"):
        download_data.download_newcastle()
    assert not (tmp_path / "raw" / "escaped.txt").exists()
