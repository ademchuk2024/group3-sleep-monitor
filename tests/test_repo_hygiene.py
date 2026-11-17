"""
Tests that guard the rule: dataset files and recordings never go into git.

Sleep data is health data, and the dataset licenses do not let us hand out
copies from our own repository. These tests fail loudly if someone (or some
tool) ever stages such a file.
"""

import os
import shutil
import subprocess

import pytest

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not os.path.isdir(os.path.join(PROJECT_FOLDER, ".git")),
    reason="not a git checkout",
)


def git(*arguments):
    result = subprocess.run(["git", *arguments], cwd=PROJECT_FOLDER, capture_output=True, text=True)
    return result.returncode, result.stdout


def tracked_files():
    """Every file git knows about: already committed, or staged for the next commit."""
    _, out = git("ls-files", "--cached")
    return [line for line in out.splitlines() if line]


def test_nothing_inside_the_data_folder_is_tracked():
    inside = [path for path in tracked_files() if path.startswith("data/")]
    assert inside == [], f"dataset files must never be committed: {inside[:5]}"


def test_the_data_folder_is_ignored_by_git():
    for path in ("data/raw/walch/motion/1_acceleration.txt", "data/raw/newcastle/x.bin", "data/features/a.npz"):
        code, _ = git("check-ignore", "-q", path)
        assert code == 0, f"{path} is not ignored. Check .gitignore."


def test_no_dataset_style_files_are_tracked_anywhere():
    # Raw recordings and arrays built from them, whatever folder they sit in.
    forbidden = (".bin", ".mat", ".zip", ".npz", ".npy", ".h5", ".hdf5", ".parquet", ".pkl", ".edf", ".cwa")
    found = [path for path in tracked_files() if path.lower().endswith(forbidden)]
    assert found == [], f"these look like data files: {found}"


def test_no_csv_is_tracked():
    # Recordings are never committed. The sample night is generated in memory.
    csvs = [path for path in tracked_files() if path.lower().endswith(".csv")]
    assert csvs == []


def test_no_tracked_file_is_suspiciously_large():
    limit = 10 * 1024 * 1024    # 10 MB. The largest tracked file is the 0.5 MB model.
    large = [path for path in tracked_files()
             if os.path.exists(os.path.join(PROJECT_FOLDER, path))
             and os.path.getsize(os.path.join(PROJECT_FOLDER, path)) > limit]
    assert large == [], f"large files in git: {large}"


def test_the_split_file_holds_ids_only():
    # splits/people_split.json IS committed, on purpose. It may contain the
    # datasets' own person codes and nothing else about anyone.
    with open(os.path.join(PROJECT_FOLDER, "splits", "people_split.json")) as f:
        text = f.read().lower()
    for word in ("diagnosis", "insomnia", "apnoea", "narcolepsy", "female", "birth", '"age"', '"sex"'):
        assert word not in text
