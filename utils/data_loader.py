"""
Valencia-IslaLink DAS Dataset (PubDAS) - Data Loader
=======================================================
Loads HDF5 strain-rate recordings (Febus A1-R interrogator) and provides
structured access to per-event, per-channel data.

HDF5 layout (verified directly with h5py against a real downloaded file,
not just the bundled Matlab_read_data_example/read_das_data.m — MATLAB's
high-level h5read() reports dimensions in reverse order relative to the raw
HDF5 storage, which h5py reads in natively, so the two are not directly
comparable without checking):

    /<device>/Source1                       (attrs: SamplingRate, GaugeLength,
                                               FiberLength, PulseRateFreq, ...)
    /<device>/Source1/time                  shape (1, n_blocks), float64,
                                               UNIX timestamp per block
    /<device>/Source1/ZoneN                 (attrs: Spacing, Origin, Extent)
    /<device>/Source1/ZoneN/SR_Valencia     shape (n_blocks, block_len, n_channels),
                                               float32, strain rate (nanostrain/s)

`<device>` (e.g. 'fa1-20050027') is the interrogator unit's hostname and is
discovered at read time, never hardcoded. Only Zone1 has been observed in the
files inspected so far (the full 50 km fiber in one zone) — `zone` is exposed
as a parameter in case a future file splits the fiber into multiple zones.

Blocks concatenate end-to-end with no overlap (verified against the `time`
array: consecutive block timestamps are spaced ~1.0 s apart, matching
block_len=250 samples at SamplingRate=250 Hz) — a straight reshape from
(n_blocks, block_len, n_channels) to (n_blocks * block_len, n_channels) is
the correct way to flatten the time axis. The bundled Matlab example's
quarter-block trimming only applies when reading a sub-range of blocks for
plotting an x-axis; it does not trim samples when reading in full, which is
the only mode this loader implements so far.
"""

import h5py
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union


@dataclass
class DASEvent:
    """Container for one DAS recording. May span a single 10-minute file or
    a full concatenated hour, depending on which loader function built it."""
    event_id: str                  # hour-folder name, e.g. 'SR_Valencia_2020-09-01_13-21-28_UTC'
    files: List[str]               # source filename(s), in time order
    strain_rate: np.ndarray        # (n_samples, n_channels) or (n_samples,) if a single channel was selected, float32, nanostrain/s
    block_time: np.ndarray         # (n_blocks,) float64 UNIX timestamp, one per 1s block, concatenated across files
    fs: float                      # sampling rate, Hz
    channel_spacing: float         # m, distance between adjacent channels
    gauge_length: float            # m
    fiber_length: float            # m, total interrogated fiber length
    n_channels: int                # channel count in `strain_rate` (1 if a single channel was loaded)


def _device_group(f: h5py.File) -> str:
    """Return the top-level device group name (e.g. 'fa1-20050027').

    This is the interrogator unit's hostname, not a fixed dataset constant —
    discover it instead of hardcoding it.
    """
    return list(f.keys())[0]


def load_h5_file(
    filepath: Union[str, Path],
    zone: int = 1,
    channels: Optional[Union[int, Sequence[int], slice]] = None,
) -> DASEvent:
    """Load one Valencia DAS 10-minute .h5 file.

    Args:
        filepath: Path to a `*_10mins.h5` file.
        zone: Which Zone group to read. Only Zone1 has been observed so far.
        channels: If given, only read this channel index / list of indices /
            slice — sliced directly on the HDF5 dataset, so this keeps
            memory use proportional to the requested channels rather than
            loading all ~2977 channels first (a full file at full channel
            count is ~1.8 GB in memory; a single channel is a few MB).

    Returns:
        DASEvent for this one file (event_id is its parent hour-folder name).
    """
    filepath = Path(filepath)

    with h5py.File(filepath, 'r') as f:
        dev = _device_group(f)
        src = f[f'{dev}/Source1']
        zone_grp = src[f'Zone{zone}']
        ds = zone_grp['SR_Valencia']

        n_blocks, block_len, n_channels_total = ds.shape

        if channels is None:
            strain_rate = ds[:, :, :].reshape(n_blocks * block_len, n_channels_total)
            n_channels = n_channels_total
        else:
            strain_rate = ds[:, :, channels]
            n_channels = 1 if isinstance(channels, int) else strain_rate.shape[-1]
            strain_rate = strain_rate.reshape(n_blocks * block_len, -1)
            if n_channels == 1:
                strain_rate = strain_rate[:, 0]

        block_time = src['time'][0].astype(np.float64)
        fs = float(src.attrs['SamplingRate'][0])
        channel_spacing = float(zone_grp.attrs['Spacing'][0])
        gauge_length = float(src.attrs['GaugeLength'][0])
        fiber_length = float(src.attrs['FiberLength'][0])

    return DASEvent(
        event_id=filepath.parent.name,
        files=[filepath.name],
        strain_rate=strain_rate,
        block_time=block_time,
        fs=fs,
        channel_spacing=channel_spacing,
        gauge_length=gauge_length,
        fiber_length=fiber_length,
        n_channels=n_channels,
    )


def discover_events(data_dir: Union[str, Path], files_per_event: int = 6) -> List[str]:
    """List local hour-folders ("events") with a complete set of .h5 files.

    Args:
        data_dir: The Data/ directory produced by download_dataset.ensure_data().
        files_per_event: Expected file count per complete hour-folder.

    Returns:
        Event (hour-folder) names, chronologically sorted, that have at
        least `files_per_event` .h5 files on disk. Incomplete/in-progress
        folders are silently skipped.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    events = []
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and len(list(folder.glob('*.h5'))) >= files_per_event:
            events.append(folder.name)
    return events


def list_event_files(data_dir: Union[str, Path], event_id: str) -> List[Path]:
    """Return the .h5 files for one event, sorted in time order."""
    return sorted((Path(data_dir) / event_id).glob('*.h5'))


def load_event(
    data_dir: Union[str, Path],
    event_id: str,
    zone: int = 1,
    channels: Optional[Union[int, Sequence[int], slice]] = None,
) -> DASEvent:
    """Load and concatenate all 10-minute files within one hour-folder.

    Loading every channel for a full hour is memory-heavy (~10.7 GB at full
    channel count) — pass `channels` to load only what's needed (e.g. a
    single representative channel for feature extraction is a few MB/hour).

    Args:
        data_dir: The Data/ directory produced by download_dataset.ensure_data().
        event_id: Hour-folder name, e.g. 'SR_Valencia_2020-09-01_13-21-28_UTC'.
        zone: Which Zone group to read.
        channels: Optional channel subset — see load_h5_file().

    Returns:
        DASEvent with strain_rate/block_time concatenated across all files
        in the event, in time order.

    Raises:
        FileNotFoundError: If no .h5 files are found for this event.
    """
    files = list_event_files(data_dir, event_id)
    if not files:
        raise FileNotFoundError(f"No .h5 files found for event {event_id!r} under {data_dir}")

    parts = [load_h5_file(f, zone=zone, channels=channels) for f in files]

    return DASEvent(
        event_id=event_id,
        files=[p.files[0] for p in parts],
        strain_rate=np.concatenate([p.strain_rate for p in parts], axis=0),
        block_time=np.concatenate([p.block_time for p in parts], axis=0),
        fs=parts[0].fs,
        channel_spacing=parts[0].channel_spacing,
        gauge_length=parts[0].gauge_length,
        fiber_length=parts[0].fiber_length,
        n_channels=parts[0].n_channels,
    )


def load_dataset(
    data_dir: Union[str, Path],
    max_events: Optional[int] = None,
    channels: Optional[Union[int, Sequence[int], slice]] = None,
) -> List[DASEvent]:
    """Load every available event under data_dir, optionally capped.

    Args:
        data_dir: The Data/ directory produced by download_dataset.ensure_data().
        max_events: If given, only load the first N (earliest) events.
        channels: Optional channel subset — see load_h5_file(). Strongly
            recommended when loading multiple events at full channel count,
            since each full-channel hour is ~10.7 GB in memory.

    Returns:
        List of DASEvent, one per event, chronologically ordered.
    """
    event_ids = discover_events(data_dir)
    if max_events:
        event_ids = event_ids[:max_events]

    events = []
    for event_id in event_ids:
        try:
            events.append(load_event(data_dir, event_id, channels=channels))
            print(f"  Loaded {event_id} ({len(list_event_files(data_dir, event_id))} files)")
        except Exception as e:
            print(f"  Error loading {event_id}: {e}")

    print(f"\nTotal loaded: {len(events)} events")
    return events


# ============================================================
# Quick test
# ============================================================
if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent / 'pubDAS_data' / 'Data'
    event_ids = discover_events(data_dir)

    if event_ids:
        # Load just one channel for the first event — fast, low-memory smoke test
        event = load_event(data_dir, event_ids[0], channels=1500)
        print(f"Event: {event.event_id}")
        print(f"Files: {event.files}")
        print(f"Strain rate shape: {event.strain_rate.shape}")
        print(f"Sampling rate: {event.fs} Hz")
        print(f"Channel spacing: {event.channel_spacing} m")
        print(f"Duration: {len(event.strain_rate) / event.fs:.1f} s")
    else:
        print(f"No complete events found in {data_dir}. Run utils/download_dataset.py first.")
