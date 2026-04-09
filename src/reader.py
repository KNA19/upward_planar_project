from __future__ import annotations

from pathlib import Path
import os
from typing import Iterator

import numpy as np
from numpy.typing import DTypeLike


# ---------------------------
# Core helpers
# ---------------------------

def _infer_bits(n: int, path: str | None = None, bits: int | None = None) -> int:
    """
    Decide whether to use 8-bit or 16-bit.

    Priority:
    1) explicit bits
    2) filename suffix (.b08 / .b16)
    3) fallback rule: n <= 8 -> 8-bit, else 16-bit
    """
    if bits in (8, 16):
        return bits

    if path is not None:
        suffix = str(path).lower()
        if suffix.endswith(".b08"):
            return 8
        if suffix.endswith(".b16"):
            return 16

    return 8 if n <= 8 else 16


def _default_path(n: int, bits: int, root_dir: str | Path = ".") -> str:
    """
    Construct the default canonical filename:
    otypesNN.b08 or otypesNN.b16 inside root_dir.
    """
    root = Path(root_dir)
    return str(root / f"otypes{n:02d}.b{bits:02d}")


def _dtype_for_bits(bits: int) -> np.dtype:
    """
    Return the numpy dtype matching the file bit-width.
    """
    if bits == 8:
        return np.dtype(np.uint8)
    if bits == 16:
        return np.dtype("<u2")  # little-endian uint16
    raise ValueError("bits must be 8 or 16.")


def _load_raw_vector(path: str, bits: int, mmap: bool) -> np.ndarray:
    """
    Load the file as a flat 1D array (or memmap).
    """
    dt = _dtype_for_bits(bits)

    if mmap:
        n_elems = os.path.getsize(path) // dt.itemsize
        return np.memmap(path, dtype=dt, mode="r", shape=(n_elems,))
    return np.fromfile(path, dtype=dt)


def _validate_stride_and_counts(arr: np.ndarray, n: int) -> tuple[int, int]:
    """
    Validate that the file length matches complete point sets of shape (n, 2).
    Each set uses 2*n scalar entries.
    """
    stride = 2 * n
    total = int(arr.size)

    if total % stride != 0:
        raise ValueError(
            f"File length ({total} elements) is not divisible by stride=2n={stride}. "
            f"Wrong n, bits, or file?"
        )

    num_sets = total // stride
    return stride, num_sets


def _optional_range_check(arr: np.ndarray, bits: int) -> None:
    """
    Quick sanity check that values fit the declared bit-width.
    """
    if arr.size == 0:
        return

    sample_end = min(int(arr.size), 1_000_000)
    vmax = int(arr[:sample_end].max())

    if bits == 8 and vmax > 255:
        raise ValueError(f"Detected value {vmax} exceeds 8-bit range in a .b08 file.")
    if bits == 16 and vmax > 65535:
        raise ValueError(f"Detected value {vmax} exceeds 16-bit range in a .b16 file.")


def as_int64_for_geometry(points: np.ndarray) -> np.ndarray:
    """
    Cast a point array to int64 before geometry operations
    to avoid unsigned underflow during subtraction/orientation tests.
    """
    if points.dtype == np.int64:
        return points
    return points.astype(np.int64, copy=False)


# --------------------------------------------
# Public API: read all sets at once
# --------------------------------------------

def read_otypes_all(
    n: int,
    path: str | None = None,
    *,
    bits: int | None = None,
    root_dir: str | Path = ".",
    mmap: bool = False,
    validate_range: bool = True,
) -> np.ndarray:
    """
    Load all order-type point sets for a given n.

    Returns:
        ndarray of shape (num_sets, n, 2)

    Notes:
    - If mmap=True, the returned array is memmap-backed.
    - If path is None, the canonical filename is used.
    """
    actual_bits = _infer_bits(n, path, bits)

    if path is None:
        path = _default_path(n, actual_bits, root_dir)

    raw = _load_raw_vector(path, actual_bits, mmap=mmap)

    if validate_range:
        _optional_range_check(raw, actual_bits)

    _validate_stride_and_counts(raw, n)

    return raw.reshape(-1, n, 2)


# --------------------------------------------
# Public API: iterate in batches
# --------------------------------------------

def iter_otypes_sets(
    n: int,
    path: str | None = None,
    *,
    bits: int | None = None,
    root_dir: str | Path = ".",
    mmap: bool = True,
    start_set: int = 0,
    stop_set: int | None = None,
    batch_size_sets: int = 10_000,
    out_dtype: DTypeLike | None = None,
    validate_range: bool = True,
) -> Iterator[np.ndarray]:
    """
    Yield batches of point sets with shape (k, n, 2).

    Parameters:
        n: number of points per set
        path: optional file path
        bits: optional explicit bit-width
        root_dir: base directory if path is omitted
        mmap: use memory mapping
        start_set: first set index to include
        stop_set: stop before this set index
        batch_size_sets: number of sets per yielded block
        out_dtype: optional dtype conversion for each block
        validate_range: run quick range check

    Yields:
        blocks of shape (k, n, 2)
    """
    actual_bits = _infer_bits(n, path, bits)

    if path is None:
        path = _default_path(n, actual_bits, root_dir)

    raw = _load_raw_vector(path, actual_bits, mmap=mmap)

    if validate_range:
        _optional_range_check(raw, actual_bits)

    stride, num_sets = _validate_stride_and_counts(raw, n)

    start_set_int = max(0, int(start_set))

    if stop_set is None:
        stop_set_int = num_sets
    else:
        stop_set_int = min(int(stop_set), num_sets)

    if start_set_int >= stop_set_int:
        return

    for base in range(start_set_int, stop_set_int, batch_size_sets):
        end = min(base + batch_size_sets, stop_set_int)

        lo = base * stride
        hi = end * stride

        block = raw[lo:hi].reshape(end - base, n, 2)

        if out_dtype is not None:
            block = block.astype(out_dtype, copy=False)

        yield block


# --------------------------------------------
# Public API: iterate one set at a time
# --------------------------------------------

def iter_otypes_one_by_one(
    n: int,
    path: str | None = None,
    *,
    bits: int | None = None,
    root_dir: str | Path = ".",
    mmap: bool = True,
    start_set: int = 0,
    stop_set: int | None = None,
    cast_int64_for_geometry: bool = False,
) -> Iterator[np.ndarray]:
    """
    Yield one point set at a time with shape (n, 2).

    Parameters:
        cast_int64_for_geometry:
            If True, each set is cast to int64 before yielding.
    """
    out_dtype: DTypeLike | None = np.dtype(np.int64) if cast_int64_for_geometry else None

    for block in iter_otypes_sets(
        n=n,
        path=path,
        bits=bits,
        root_dir=root_dir,
        mmap=mmap,
        start_set=start_set,
        stop_set=stop_set,
        batch_size_sets=4096,
        out_dtype=out_dtype,
        validate_range=True,
    ):
        for pts in block:
            yield pts