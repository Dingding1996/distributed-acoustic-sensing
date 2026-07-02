"""
Valencia-IslaLink DAS Dataset (PubDAS) - Download via Globus
===============================================================
Downloads Distributed Acoustic Sensing data from the PubDAS Globus guest
collection (https://app.globus.org/file-manager?origin_id=706e304c-5def-11ec-9b5c-f9dfb1abb183&origin_path=%2FValencia%2F).

Can be used as a library or run directly as a script.

As a library:
    from utils.download_dataset import ensure_data
    data_dir = ensure_data(max_hours=4)

As a script:
    python utils/download_dataset.py                       # first 4 hours (default cap)
    python utils/download_dataset.py --max_hours 1          # just the first hour
    python utils/download_dataset.py --output_dir /path/to/save

Requirements:
    - `globus-cli` and `globus-sdk` installed in this Python environment
      (already the case in ds-py311 — `pip install globus-cli` otherwise).
    - A one-time interactive `globus login` (run from a terminal) to
      authenticate; this script shells out to the already-authenticated CLI
      rather than re-implementing the OAuth flow, since the login session is
      a one-time setup step the user does directly, not something to
      automate per-run.
    - Globus Connect Personal installed and running locally as the transfer
      destination (https://www.globus.org/globus-connect-personal), with its
      collection granted filesystem access (via the GCP tray app ->
      Preferences -> Access) to wherever `output_dir` resolves to — Globus
      Connect Personal restricts transfers to an explicit allow-list of local
      paths, it does not default to full-disk access.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


# PubDAS Globus guest collection (public; hosts multiple DAS test-site datasets)
PUBDAS_COLLECTION_ID = "706e304c-5def-11ec-9b5c-f9dfb1abb183"
VALENCIA_REMOTE_DIR = "/Valencia/Data"

# Default data directory relative to this file (matches the manually-placed
# geometry/readme/license files already on disk — see pubDAS_data/readme.txt)
DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'pubDAS_data'

# Verified directly against a live Globus listing (see pubDAS_data/readme.txt
# for the acquisition story): each hour of recording was split into six
# 10-minute, 250 Hz, 2977-channel HDF5 files.
FILES_PER_HOUR = 6
BYTES_PER_FILE = 1_789_226_584  # ~1.78 GB

# Disk-space cap, not a dataset limit — the full Valencia release spans nearly
# a week (Sept 1-7, 2020) of continuous recording. Don't raise this default
# without checking available disk space first (~10.7 GB per additional hour).
DEFAULT_MAX_HOURS = 4


def _local_collection_id() -> str:
    """Resolve the local Globus Connect Personal collection ID.

    Checks the GLOBUS_LOCAL_COLLECTION_ID env var first, then falls back to
    the standard GCP-for-Windows install location. There's no safe default
    to assume across machines, so this raises rather than guessing.
    """
    env_id = os.environ.get("GLOBUS_LOCAL_COLLECTION_ID")
    if env_id:
        return env_id

    client_id_file = Path(os.environ.get("LOCALAPPDATA", "")) / "Globus Connect" / "client-id.txt"
    if client_id_file.exists():
        return client_id_file.read_text().strip()

    raise RuntimeError(
        "Could not find a local Globus Connect Personal collection ID. "
        "Install and run Globus Connect Personal "
        "(https://www.globus.org/globus-connect-personal), or set the "
        "GLOBUS_LOCAL_COLLECTION_ID environment variable to your collection's UUID."
    )


def _to_gcp_windows_path(local_path: Path) -> str:
    """Convert a local Windows path to the `/<Drive>/...` form Globus Connect
    Personal for Windows exposes (verified directly: GCP lists its root as
    `/C/`, `/D/`, etc., not a bare drive letter with a colon).

    e.g. C:\\8. DSPandML\\foo -> /C/8. DSPandML/foo
    """
    local_path = local_path.resolve()
    drive = local_path.drive.rstrip(':')
    rest = local_path.as_posix().split(':', 1)[1]
    return f"/{drive}{rest}"


def _run_globus_cli(args: List[str], input_text: Optional[str] = None) -> str:
    """Run a `globus` CLI subcommand via this interpreter's globus_cli module.

    Shells out to the CLI rather than re-implementing OAuth with globus_sdk
    directly: `globus login` is a one-time interactive step the user runs
    themselves, and reusing that already-authenticated session is far simpler
    and more robust than re-deriving the same tokens from this script.

    Args:
        args: CLI arguments after `globus`, e.g. ['ls', '-F', 'json', '...'].
        input_text: Optional stdin content (used for `transfer --batch -`).

    Returns:
        Captured stdout.

    Raises:
        RuntimeError: If the CLI exits non-zero.
    """
    cmd = [sys.executable, "-m", "globus_cli"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, input=input_text)
    if result.returncode != 0:
        raise RuntimeError(
            f"globus CLI failed ({' '.join(args)}):\n{result.stderr or result.stdout}"
        )
    return result.stdout


def list_hour_folders(max_hours: Optional[int] = None) -> List[str]:
    """List available hour-folders under the remote Valencia Data/ directory.

    Folder names are the recording start time (e.g.
    'SR_Valencia_2020-09-01_13-21-28_UTC') and sort chronologically as-is.

    Args:
        max_hours: If given, return only the first N (earliest) folders.

    Returns:
        Hour-folder names, chronologically sorted.
    """
    out = _run_globus_cli(["ls", "-F", "json", f"{PUBDAS_COLLECTION_ID}:{VALENCIA_REMOTE_DIR}/"])
    entries = json.loads(out)["DATA"]
    folders = sorted(e["name"] for e in entries if e["type"] == "dir")
    return folders[:max_hours] if max_hours else folders


def list_files_in_hour(hour_folder: str) -> List[Dict]:
    """List the 10-minute .h5 files inside one remote hour-folder, with sizes."""
    out = _run_globus_cli(
        ["ls", "-F", "json", f"{PUBDAS_COLLECTION_ID}:{VALENCIA_REMOTE_DIR}/{hour_folder}/"]
    )
    entries = json.loads(out)["DATA"]
    return [e for e in entries if e["type"] == "file" and e["name"].endswith(".h5")]


def ensure_data(
    max_hours: int = DEFAULT_MAX_HOURS,
    output_dir: Path = DEFAULT_DATA_DIR,
    local_collection_id: Optional[str] = None,
) -> Path:
    """Download the earliest `max_hours` of Valencia DAS data, skipping files
    already on disk (checked by name + exact byte size).

    This is the main library entry point. Call this at the top of any script
    that needs the dataset — it's a no-op for hour-folders already complete
    on disk. Safe to re-run after an interrupted download: only missing or
    size-mismatched files are re-requested, and the underlying Globus
    transfer uses checksum verification.

    Args:
        max_hours: Number of earliest hour-folders to download (6 files each,
            ~10.7 GB/hour) — bounds disk usage, see DEFAULT_MAX_HOURS.
        output_dir: Local root directory; data lands in `output_dir/Data/`,
            mirroring the remote layout.
        local_collection_id: Destination Globus collection ID. Defaults to
            this machine's Globus Connect Personal collection.

    Returns:
        Path to the Data/ directory containing the downloaded hour-folders.
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / 'Data'
    data_dir.mkdir(parents=True, exist_ok=True)
    local_collection_id = local_collection_id or _local_collection_id()

    hour_folders = list_hour_folders(max_hours)

    print(f"\n{'='*60}")
    print("Valencia-IslaLink DAS Dataset (PubDAS) - Globus Downloader")
    print(f"{'='*60}")
    print(f"Hours requested : {len(hour_folders)} "
          f"(~{len(hour_folders) * FILES_PER_HOUR * BYTES_PER_FILE / 1e9:.1f} GB)")

    batch_lines = []
    for folder in hour_folders:
        local_folder = data_dir / folder
        remote_files = list_files_in_hour(folder)
        missing = [
            rf for rf in remote_files
            if not (local_folder / rf["name"]).exists()
            or (local_folder / rf["name"]).stat().st_size != rf["size"]
        ]
        if not missing:
            print(f"  {folder}: already complete ({len(remote_files)} files)")
            continue
        local_folder.mkdir(parents=True, exist_ok=True)
        for rf in missing:
            batch_lines.append(f"{folder}/{rf['name']} {folder}/{rf['name']}")

    if not batch_lines:
        print(f"\nDataset ready: all {len(hour_folders)} hour(s) found in {data_dir}")
        return data_dir

    print(f"To download     : {len(batch_lines)} file(s)")
    print(f"Destination     : {data_dir.absolute()}")
    print(f"{'='*60}\n")

    dest_path = _to_gcp_windows_path(data_dir) if os.name == 'nt' else str(data_dir)
    out = _run_globus_cli(
        [
            "transfer",
            f"{PUBDAS_COLLECTION_ID}:{VALENCIA_REMOTE_DIR}/",
            f"{local_collection_id}:{dest_path}/",
            "--batch", "-",
            "--sync-level", "checksum",
            "--label", f"valencia-das-{len(hour_folders)}h",
            "-F", "json",
        ],
        input_text="\n".join(batch_lines),
    )
    task_id = json.loads(out)["task_id"]
    print(f"Submitted Globus transfer task {task_id} ({len(batch_lines)} files).")
    print("Waiting for completion (this can take a while for multiple hours)...")

    # No fixed timeout: large multi-hour transfers can legitimately take a long
    # time, and silently giving up would leave a partially-downloaded hour on
    # disk. If this is interrupted, re-running ensure_data() will pick up
    # exactly where it left off.
    wait_result = subprocess.run(
        [sys.executable, "-m", "globus_cli", "task", "wait", task_id,
         "--polling-interval", "15", "-H"],
    )
    if wait_result.returncode != 0:
        print(f"\nWarning: task {task_id} did not report success. "
              f"Check status with: globus task show {task_id}")
        print("Re-running ensure_data() will retry only the missing/incomplete files.")
    else:
        print(f"\nAll {len(batch_lines)} file(s) downloaded successfully.")

    return data_dir


def main():
    parser = argparse.ArgumentParser(
        description='Download the Valencia-IslaLink DAS dataset (PubDAS) via Globus'
    )
    parser.add_argument('--output_dir', default=str(DEFAULT_DATA_DIR),
                        help=f'Directory to save data (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--max_hours', type=int, default=DEFAULT_MAX_HOURS,
                        help=f'Number of earliest hours to download (default: {DEFAULT_MAX_HOURS})')
    parser.add_argument('--local_collection_id', default=None,
                        help='Destination Globus collection ID (default: auto-detect GCP)')
    args = parser.parse_args()

    data_dir = ensure_data(
        max_hours=args.max_hours,
        output_dir=args.output_dir,
        local_collection_id=args.local_collection_id,
    )

    h5_files = list(data_dir.rglob('*.h5'))
    print(f"\nTotal .h5 files : {len(h5_files)}")
    print(f"Data path       : {data_dir.absolute()}")


if __name__ == '__main__':
    main()
