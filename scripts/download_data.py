"""
download_data.py - Downloads the open sleep datasets into data/raw/.

Run it from the project folder, naming what you want:

    python scripts/download_data.py walch            (about 2.2 GB on disk)
    python scripts/download_data.py newcastle        (about 1 GB download)
    python scripts/download_data.py bidsleep-sample  (3 nights, for testing the loader)
    python scripts/download_data.py bidsleep         (ALL 253 nights, about 28 GB - meant for Colab)

Why this script exists:
  - The datasets are too big, and too private, to keep in git. Anyone who
    clones the repo runs this once to get their own copy in data/raw/,
    which git ignores.
  - Every file is checked against the checksum the dataset authors
    published. A checksum is a fingerprint of a file's contents: if even one
    byte is wrong (a broken download), the fingerprint does not match and we
    throw the file away instead of training on damaged data.
  - It only uses Python's built-in tools, so it needs nothing installed and
    works the same on a laptop and on Google Colab.

It is safe to run again: files that are already downloaded and correct are skipped.

Dataset pages (read these for the license and how to cite):
  Walch:     https://doi.org/10.13026/hmhs-py35      (ODC Attribution License 1.0)
  BIDSleep:  https://doi.org/10.13026/rees-1092      (ODC Attribution License 1.0)
  Newcastle: https://zenodo.org/records/1160410      (CC BY 4.0)

DREAMT is NOT here on purpose. It is restricted access and must not be used
without approval.
"""

import argparse
import hashlib
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")

# PhysioNet serves every file of a dataset under one web folder, next to a
# SHA256SUMS.txt file that lists the checksum of each file.
PHYSIONET = {
    "walch": "https://physionet.org/files/sleep-accel/1.0.0/",
    "bidsleep": "https://physionet.org/files/bidsleep-dataset/1.0.1/",
}

# Zenodo serves Newcastle as one zip file, with an MD5 checksum.
NEWCASTLE_URL = "https://zenodo.org/api/records/1160410/files/dataset_psgnewcastle2015_v1.0.zip/content"
NEWCASTLE_ZIP = "dataset_psgnewcastle2015_v1.0.zip"
NEWCASTLE_MD5 = "8ebadbc55cb0e76230f293fc6d53f504"

# The few BIDSleep nights used to write and test the loader on a laptop:
# two nights from one person (to test "several nights, one person") and one
# night from a second person (to test "several people").
BIDSLEEP_SAMPLE_NIGHTS = ["Bidslab00/1/", "Bidslab00/2/", "Bidslab01/1/"]

# Some servers refuse requests that do not say what program is asking.
# This name is generic on purpose: it contains nothing personal.
HEADERS = {"User-Agent": "sleep-dataset-downloader/1.0"}

CHUNK_BYTES = 1024 * 1024  # read and write 1 MB at a time, so memory use stays small


def make_ssl_context():
    """Build the settings Python uses to check a website's security certificate.

    "https" only protects a download if Python can check that the website is
    really who it says it is. For that it needs a list of trusted certificate
    issuers. Python installed from python.org on a Mac often has no such list,
    and every https download fails with CERTIFICATE_VERIFY_FAILED.

    The fix is NOT to switch the check off. Instead we use the trusted list
    from the "certifi" package, which is already installed because Streamlit
    needs it. If certifi is missing (for example on Colab, where Python
    already has a list), we use Python's normal settings.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = make_ssl_context()


def open_url(url):
    """Open a web address for reading, with certificate checking on."""
    request = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(request, timeout=120, context=SSL_CONTEXT)


def file_checksum(path, kind):
    """Return the checksum of a file. kind is "sha256" or "md5"."""
    digest = hashlib.new(kind)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, destination, expected_checksum, kind="sha256", attempts=3):
    """Download one file and check its checksum.

    Returns "skipped" if a correct copy was already there, else "downloaded".
    Raises RuntimeError if it still fails after a few tries.
    """
    # Already have a good copy? Then do nothing.
    if os.path.exists(destination) and file_checksum(destination, kind) == expected_checksum:
        return "skipped"

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Download to a temporary ".part" name first. Only a file that passed the
    # checksum gets the real name, so a half-finished download is never
    # mistaken for a complete one.
    temporary = destination + ".part"
    last_problem = None
    for attempt in range(1, attempts + 1):
        try:
            with open_url(url) as response, open(temporary, "wb") as out:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
            actual = file_checksum(temporary, kind)
            if actual == expected_checksum:
                os.replace(temporary, destination)
                return "downloaded"
            last_problem = f"checksum did not match (expected {expected_checksum}, got {actual})"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as problem:
            last_problem = str(problem)
        # Wait a little longer after each failure before trying again.
        time.sleep(2 * attempt)

    if os.path.exists(temporary):
        os.remove(temporary)
    raise RuntimeError(f"Could not download {url}: {last_problem}")


def parse_checksum_list(text):
    """Turn the text of a SHA256SUMS.txt file into {file path: checksum}.

    Each line looks like:  <checksum> <path>
    """
    checksums = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        checksum, path = line.split(maxsplit=1)
        checksums[path.strip()] = checksum
    return checksums


def is_safe_relative_path(path):
    """True if the path stays inside the folder we are downloading into.

    A path like "../../somewhere/else" or "/etc/..." could write outside the
    data folder, so we refuse those.
    """
    if os.path.isabs(path):
        return False
    return ".." not in path.replace("\\", "/").split("/")


def download_physionet(name, only_prefixes=None):
    """Download a PhysioNet dataset file by file.

    only_prefixes: if given, only download files whose path starts with one of
    these (used to fetch just a few BIDSleep nights).
    """
    base_url = PHYSIONET[name]
    target = os.path.join(RAW_FOLDER, name)
    os.makedirs(target, exist_ok=True)

    print(f"Getting the file list for {name} ...")
    try:
        with open_url(base_url + "SHA256SUMS.txt") as response:
            checksum_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as problem:
        raise RuntimeError(f"Could not get the file list for {name}: {problem}")
    # Keep the list next to the data so the loaders' tests can re-check files later.
    with open(os.path.join(target, "SHA256SUMS.txt"), "w") as f:
        f.write(checksum_text)

    checksums = parse_checksum_list(checksum_text)
    wanted = sorted(checksums)
    if only_prefixes:
        # Always keep the small top-level files (license, readme).
        wanted = [p for p in wanted if "/" not in p or p.startswith(tuple(only_prefixes))]

    downloaded = skipped = 0
    for number, path in enumerate(wanted, start=1):
        if not is_safe_relative_path(path):
            raise RuntimeError(f"Refusing a suspicious file path in the file list: {path}")
        # Web addresses cannot contain spaces etc., so encode the path.
        url = base_url + urllib.request.quote(path)
        result = download_file(url, os.path.join(target, path), checksums[path])
        if result == "downloaded":
            downloaded += 1
        else:
            skipped += 1
        print(f"  [{number}/{len(wanted)}] {result}: {path}")

    print(f"Done with {name}: {downloaded} downloaded, {skipped} already there. Folder: {target}")


def download_newcastle():
    """Download the Newcastle zip from Zenodo, check it, and unzip it."""
    target = os.path.join(RAW_FOLDER, "newcastle")
    zip_path = os.path.join(target, NEWCASTLE_ZIP)

    print("Getting Newcastle (one zip file, about 1 GB) ...")
    result = download_file(NEWCASTLE_URL, zip_path, NEWCASTLE_MD5, kind="md5")
    print(f"  {result}: {NEWCASTLE_ZIP}")

    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not is_safe_relative_path(member):
                raise RuntimeError(f"Refusing a suspicious file path inside the zip: {member}")
        archive.extractall(target)
    print(f"Done with newcastle. Folder: {target}")


def main():
    parser = argparse.ArgumentParser(description="Download the open sleep datasets into data/raw/.")
    parser.add_argument(
        "dataset",
        choices=["walch", "newcastle", "bidsleep-sample", "bidsleep"],
        help="which dataset to download",
    )
    args = parser.parse_args()

    if args.dataset == "walch":
        download_physionet("walch")
    elif args.dataset == "newcastle":
        download_newcastle()
    elif args.dataset == "bidsleep-sample":
        download_physionet("bidsleep", only_prefixes=BIDSLEEP_SAMPLE_NIGHTS)
    elif args.dataset == "bidsleep":
        download_physionet("bidsleep")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as problem:
        print(f"\nPROBLEM: {problem}", file=sys.stderr)
        sys.exit(1)
