from typing import Callable, Optional
import torch
import torch.nn as nn
import logging
import numpy.typing as npt
from reader import DiskVolume

def chunk_starts(axis_len: int, C: int, K: int, S: int) -> list[int]:
    """
    Return stride-aligned, non-negative chunk start indices.
    Each start is chosen so that the slice
        [start : start + C + max(0, K-S)]
    lives entirely inside the volume.
    """
    H = max(0, K - S)  # halo (never negative)
    step = max(1, C - H)  # distance between consecutive starts

    last = axis_len - (C + H)

    # ─── Patch: if the volume is smaller than one chunk+halo, use a single chunk at 0
    if last <= 0:
        return [0]

    starts = list(range(0, last + 1, step))
    if starts[-1] != last:  # cover the tail
        starts.append(last)

    # snap to stride grid and remove duplicates
    starts = sorted({s - (s % S) for s in starts})

    return starts


def make_plan(axis_len: int, C: int, K: int, S: int):
    """
    Returns three parallel lists (starts, out_starts, keeps) that
    collectively cover the full convolution output on one spatial axis.
    """
    H = max(0, K - S)  # halo
    max_in = C + H  # chunk payload + halo
    full_out = max(0, (axis_len - K) // S + 1)

    starts: list[int] = []
    out_starts: list[int] = []
    keeps: list[int] = []
    out_idx = 0  # first *uncovered* output plane

    safety = axis_len

    while out_idx < full_out:
        safety -= 1
        if safety < 0:
            raise RuntimeError("This loop will run indefinitely")
        start = out_idx * S

        # Actual slice length and number of outputs this chunk can emit
        in_len = min(max_in, axis_len - start)
        if in_len < K:  # kernel doesn’t even fit once
            break
        prod = (in_len - K) // S + 1
        logging.debug(f"prod: {prod}; full_out: {full_out}; out_idx: {out_idx}")

        starts.append(start)
        out_starts.append(out_idx)
        keeps.append(min(prod, full_out - out_idx))

        out_idx += prod  # advance to next uncovered plane

    return starts, out_starts, keeps


def chunked_convolution(
    disk_volume: DiskVolume[npt.ArrayLike],
    chunk_size: tuple[int, int, int],
    conv_filter: nn.Conv3d,
    out_volume: Optional[DiskVolume[npt.ArrayLike]] = None,
    cast_fn: Optional[Callable[[npt.ArrayLike], npt.ArrayLike]] = None,
) -> torch.Tensor | DiskVolume[npt.ArrayLike]:
    Cx, Cy, Cz = chunk_size
    Kx, Ky, Kz = conv_filter.kernel_size
    Sx, Sy, Sz = conv_filter.stride

    # ----- output volume size, identical to torch's conv formula -------------

    plan_x = make_plan(disk_volume.shape[0], Cx, Kx, Sx)  # (starts_x, out_x, keep_x)
    plan_y = make_plan(disk_volume.shape[1], Cy, Ky, Sy)
    plan_z = make_plan(disk_volume.shape[2], Cz, Kz, Sz)

    if out_volume is None:
        # output volume will reside in RAM, not on disk
        out_shape = tuple(
            (n - k) // s + 1 for n, k, s in zip(disk_volume.shape, conv_filter.kernel_size, conv_filter.stride)
        )
        out_volume: torch.Tensor = torch.empty(out_shape, dtype=disk_volume.torch_dtype, device="cpu")  # disk_volume.device)

    # ----------------------------- main loop ---------------------------------
    for si_x, so_x, kx in zip(*plan_x):
        if kx <= 0:  # nothing new along x
            continue
        for si_y, so_y, ky in zip(*plan_y):
            if ky <= 0:
                continue
            for si_z, so_z, kz in zip(*plan_z):
                logging.debug(
                    f"si_x: {si_x}; si_y: {si_y}; si_x: {si_z};\nso_x: {so_x}; so_y: {so_y}; so_z: {so_z};\nkx: {kx}; ky: {ky}; kz: {kz};\n\n"
                )
                if kz <= 0:
                    continue

                chunk = disk_volume[
                    si_x : si_x + Cx + Kx - Sx,  # == payload + halo
                    si_y : si_y + Cy + Ky - Sy,
                    si_z : si_z + Cz + Kz - Sz,
                ]

                chunk = torch.as_tensor(chunk)
                if cast_fn is not None:
                    chunk = cast_fn(chunk)

                conv_chunk = conv_filter(chunk.unsqueeze(0)).squeeze(0)

                c = conv_chunk[
                    :kx,
                    :ky,
                    :kz,
                ]

                logging.debug(f"c.shape: {c.shape}")
                logging.debug(f"[{kx}, {ky}, {kz}]")
                logging.debug(f"[{so_x} - {so_x + kx}, {so_y} - {so_y + ky}, {so_z} - {so_z + kz}]")
                out_volume[so_x : so_x + kx, so_y : so_y + ky, so_z : so_z + kz] = c

    return out_volume


class ChunkedConv3d:
    """
    A class that wraps the chunked_convolution function for easier use.
    """

    def __init__(self, chunk_size: tuple[int, int, int], conv_filter: nn.Conv3d):
        self.chunk_size = chunk_size
        assert conv_filter.padding == (0, 0, 0), "ChunkedConv3d only supports filters without padding. (for now)"
        self.conv_filter = conv_filter

    def __call__(self, 
                 disk_volume: DiskVolume[npt.ArrayLike], 
                 create_out_volume: Optional[Callable[[tuple[int, ...]], DiskVolume[npt.ArrayLike]]] = None, 
                 cast_fn: Optional[Callable[[npt.ArrayLike], npt.ArrayLike]] = None
                 ) -> torch.Tensor | DiskVolume[npt.ArrayLike]:
        out_shape = tuple(
            (n - k) // s + 1
            for n, k, s in zip(disk_volume.shape, self.conv_filter.kernel_size, self.conv_filter.stride)
        )
        if create_out_volume is not None:
            out_volume = create_out_volume(out_shape)
        else:
            out_volume = None
        if self.conv_filter.weight.data.dtype != disk_volume.torch_dtype and cast_fn is None:
            self.conv_filter.weight.data = self.conv_filter.weight.data.to(disk_volume.torch_dtype)

        return chunked_convolution(disk_volume, self.chunk_size, self.conv_filter, out_volume=out_volume, cast_fn=cast_fn)


import unittest
from unittest import TestCase


class TestConv(TestCase):

    def __init__(self):
        super().__init__()

        self.fuzz_size: int = 100
        self.max_volume_size: tuple[int, int, int] = (256, 256, 256)
        self.max_filter_size: tuple[int, int, int] = (32, 32, 32)
        self.max_chunk_ratio: tuple[int, int, int] = (6, 6, 6)

    def test_fuzz(
        self,
    ):

        size = self.fuzz_size
        max_filter_size = self.max_filter_size
        max_volume_size = self.max_volume_size
        max_chunk_ratio = self.max_chunk_ratio

        with torch.no_grad():
            filter_sizes = torch.max(
                torch.ones(size, 3, dtype=torch.int32),
                torch.tensor(torch.rand(size, 3) * torch.tensor(max_filter_size), dtype=torch.int32),
            )
            strides = torch.max(
                torch.ones(size, 3, dtype=torch.int32),
                torch.tensor(torch.rand(size, 3) * filter_sizes, dtype=torch.int32),
            )
            chunk_sizes = (
                torch.max(
                    torch.ones(size, 3, dtype=torch.int32) * 2,
                    torch.tensor(torch.rand(size, 3) * torch.tensor(max_chunk_ratio), dtype=torch.int32),
                )
                * filter_sizes
                + 1
            )
            volume_sizes = torch.max(
                chunk_sizes, torch.tensor(torch.rand(size, 3) * torch.tensor(max_volume_size), dtype=torch.int32)
            )

            for v_size, f_size, f_stride, c_size in zip(volume_sizes, filter_sizes, strides, chunk_sizes):
                test_volume = torch.randn(*v_size.tolist())
                test_volume /= (
                    test_volume.sum()
                )  # normalize volume to not overcorrect for floating point arithmetic errors

                rand_chunk_size: tuple[int, int, int] = tuple(d for d in c_size.tolist())  # get our random chunk size
                conv_filter = nn.Conv3d(1, 1, f_size.tolist(), f_stride.tolist())
                kernel_weights = torch.randn(*f_size).unsqueeze(0).unsqueeze(0)
                kernel_weights /= kernel_weights.sum()  # normalize kernel to make it a distribution/preserve energy
                conv_filter.weight.data = kernel_weights

                ground_truth = conv_filter(test_volume.unsqueeze(0)).squeeze()
                chunked = chunked_convolution(test_volume, rand_chunk_size, conv_filter)
                assert all(
                    (s_c - s_g) == 0 for s_c, s_g in zip(chunked.shape, ground_truth.shape)
                ), f"Sizes: {ground_truth.shape} and {chunked.shape} did not match"
                assert torch.allclose(
                    ground_truth,
                    chunked,
                    rtol=1e-2,
                    atol=1e-7,  # relax tolerances a little because volumes can be large and accumulate a lot of error in total.
                ), f"Result for input size: {v_size}; filter size: {f_size}; stride: {f_stride}; chunk size: {rand_chunk_size}; was not close... diff: {(ground_truth - chunked)} total: {(ground_truth - chunked).abs().sum()} max: {(ground_truth - chunked).abs().max()}"
                logging.info(
                    f"✅ Passed: input size: {v_size}; filter size: {f_size}; stride: {f_stride}; chunk size: {rand_chunk_size}"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=[""], verbosity=2, exit=False)
    # Run the unittest main to ensure all tests are executed
