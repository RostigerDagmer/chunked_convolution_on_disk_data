from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
import h5py
import os
from multiprocessing.pool import ThreadPool
from queue import Queue
from threading import Thread
from tqdm import tqdm
from PIL import Image


def load_slice(args: Any) -> NDArray:
    """Load a single image slice."""
    data_dir, fname = args
    img_path = os.path.join(data_dir, fname)
    img = Image.open(img_path)
    return np.array(img)


def process_slice_to_chunks(
    slice_data: NDArray,
    chunk_size_xy: tuple[int, int],
    xy_chunk_coords: list[tuple[int, int]],
    slice_idx: int,
    z_chunk_size: int,
):
    """Split a slice into x-y chunks and determine their z-index."""
    height, width = slice_data.shape
    z_idx = slice_idx // z_chunk_size  # Which z-chunk this slice belongs to
    z_offset = slice_idx % z_chunk_size  # Position within that z-chunk
    chunks = []
    for cy, cx in xy_chunk_coords:
        x_start = cx * chunk_size_xy
        y_start = cy * chunk_size_xy
        x_end = min(x_start + chunk_size_xy, width)
        y_end = min(y_start + chunk_size_xy, height)
        chunk = slice_data[y_start:y_end, x_start:x_end]
        chunks.append(((cy, cx, z_idx), chunk, z_offset))
    return chunks


def hdf5_writer(
    queue: Queue[list[tuple[tuple[int, int, int], NDArray, int]]],
    output_file: str,
    chunk_shapes: dict[tuple[int, int, int], tuple[int, int, int]],
    total_slices: int,
):
    """Dedicated thread to write chunks to HDF5 file."""
    with h5py.File(output_file, "w") as f:
        # Initialize datasets for each chunk in the 3D grid
        datasets = {}
        for (cy, cx, cz), (h, w, d) in chunk_shapes.items():
            ds_name = f"chunk_{cy}_{cx}_{cz}"
            datasets[(cy, cx, cz)] = f.create_dataset(
                ds_name,
                shape=(h, w, d),
                chunks=(h, w, 1),  # Small z-chunk for appending efficiency
                dtype="uint16",  # Adjust dtype based on your data
            )

        # Process queue until all slices are written
        slice_count = 0
        with tqdm(total=total_slices, desc="Writing to HDF5") as pbar:
            while slice_count < total_slices:
                chunk_list = queue.get()
                for (cy, cx, cz), chunk, z_offset in chunk_list:
                    ds = datasets[(cy, cx, cz)]
                    ds[:, :, z_offset] = chunk
                slice_count += 1
                pbar.update(1)
                queue.task_done()


def parallel_chunk_and_write(
    sample_paths: list[str], data_dir: str, output_file: str, chunk_size: int = 128, num_workers: Optional[int] = None
):
    """Process slices one at a time, chunk them in 3D, and write to HDF5."""
    if num_workers is None:
        num_workers = os.cpu_count()

    # Precompute 3D chunk grid (x, y, z)
    first_slice = load_slice((data_dir, sample_paths[0]))
    height, width = first_slice.shape
    depth = len(sample_paths)
    n_chunks_x = (width + chunk_size - 1) // chunk_size
    n_chunks_y = (height + chunk_size - 1) // chunk_size
    n_chunks_z = (depth + chunk_size - 1) // chunk_size

    # Define chunk coordinates and shapes
    chunk_shapes = {}
    xy_chunk_coords = [(cy, cx) for cy in range(n_chunks_y) for cx in range(n_chunks_x)]
    for cy in range(n_chunks_y):
        for cx in range(n_chunks_x):
            x_start = cx * chunk_size
            y_start = cy * chunk_size
            x_end = min(x_start + chunk_size, width)
            y_end = min(y_start + chunk_size, height)
            for cz in range(n_chunks_z):
                z_start = cz * chunk_size
                z_end = min(z_start + chunk_size, depth)
                chunk_shapes[(cy, cx, cz)] = (y_end - y_start, x_end - x_start, z_end - z_start)

    # Start writer thread with queue
    queue = Queue()
    writer_thread = Thread(
        target=hdf5_writer, args=(queue, output_file, chunk_shapes, len(sample_paths), chunk_size), daemon=True
    )
    writer_thread.start()

    # Process each slice
    with ThreadPool(processes=num_workers) as pool:
        for slice_idx, fname in enumerate(tqdm(sample_paths, desc="Processing slices")):
            # Load one slice
            slice_data = load_slice((data_dir, fname))

            # Split into chunks in parallel, with z-indexing
            chunk_list = process_slice_to_chunks(slice_data, chunk_size, xy_chunk_coords, slice_idx, chunk_size)

            # Send to writer queue
            queue.put(chunk_list)

    # Wait for all writes to complete
    queue.join()
    writer_thread.join()


if __name__ == "__main__":

    """
    Example call:
    python preprocess.py --data_dir="D:\data\micro_CT\D25-E10_raw_AP\D25-E10_raw_AP\D25-E10 Skull_01" --output_file="optimized.h5" --chunk_size=256 --dtype=".tif"

    """

    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw image slices into chunked HDF5 format.")
    parser.add_argument("--data_dir", type=str, help="Directory containing raw image slices.")
    parser.add_argument("--output_file", type=str, help="Output HDF5 file.")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size in pixels.")
    parser.add_argument("--dtype", type=str, default=".tif", help="File extension of raw image slices.")
    args = parser.parse_args()

    data_dir = args.data_dir.replace("\\", "/")
    output_file = args.output_file
    dtype = args.dtype

    sample_paths: list[str] = [f for f in os.listdir(data_dir) if f.endswith(dtype)]
    parallel_chunk_and_write(sample_paths, data_dir, output_file, chunk_size=128)
