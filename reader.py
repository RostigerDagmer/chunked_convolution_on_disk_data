from typing import Literal, Optional, TypeVar, Generic
import numpy as np
import h5py
import abc
import numpy.typing as npt
import torch

T = TypeVar("T")

def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)
    
    return dtype

class DiskVolume(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def __getitem__(self, key: int | tuple[slice | Literal[None], slice | Literal[None], slice | Literal[None]] | slice ) -> T:
        """Get a slice or subvolume from the disk."""
        pass

    @abc.abstractmethod
    def __setitem__(self, key: int | tuple[slice | Literal[None], slice | Literal[None], slice | Literal[None]] | slice, value: T):
        """Set a slice or subvolume on the disk."""
        pass

    @abc.abstractmethod
    def close(self):
        """Close the volume."""
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the volume."""
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> str:
        """Get the data type of the volume."""
        pass

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the PyTorch data type of the volume."""
        return get_torch_dtype(self.dtype)


class DummyVolume(DiskVolume[npt.ArrayLike]):
    def __init__(self, shape: tuple[int, int, int], dtype: str = "uint16"):
        """Initialize a dummy volume with a given shape and data type."""
        self._shape = shape
        self._dtype = dtype
        self._data = np.zeros(shape, dtype=dtype)

    def __getitem__(self, key: int | tuple[slice | Literal[None], slice | Literal[None], slice | Literal[None]] | slice) -> npt.ArrayLike:
        return self._data[key]

    def __setitem__(self, key: int | tuple[slice | Literal[None], slice | Literal[None], slice | Literal[None]] | slice, value: npt.ArrayLike):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=self._dtype)
        self._data[key] = value

    def close(self):
        """Dummy close method, nothing to do."""
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

# New HDF5Volume class
class HDF5Volume(DiskVolume[npt.ArrayLike]):
    def __init__(self, hdf5_file: str, write_location: Optional[str] = None, dtype: str = "uint16"):
        """Initialize with an HDF5 file and optional write location."""
        self.read_file = h5py.File(hdf5_file, "r" if write_location else "r+")  # Source data
        self.chunk_size = None
        self._shape = None
        self.chunk_shapes = {}
        self.read_chunks = {}  # For reading
        self.write_chunks = {}  # For writing
        self._dtype = dtype

        # Infer volume properties from source file
        for ds_name in self.read_file.keys():
            if ds_name.startswith("chunk_"):
                cy, cx, cz = map(int, ds_name.split("_")[1:])
                chunk_shape = self.read_file[ds_name].shape
                self.chunk_shapes[(cy, cx, cz)] = chunk_shape
                self.read_chunks[(cy, cx, cz)] = self.read_file[ds_name]
                if self.chunk_size is None:
                    self.chunk_size = chunk_shape[0]

        # Calculate full volume shape
        max_y = max(cy for cy, _, _ in self.chunk_shapes.keys()) + 1
        max_x = max(cx for _, cx, _ in self.chunk_shapes.keys()) + 1
        max_z = max(cz for _, _, cz in self.chunk_shapes.keys()) + 1
        self._shape = (
            max_y * self.chunk_size - (self.chunk_size - self.chunk_shapes[(max_y - 1, 0, 0)][0]),
            max_x * self.chunk_size - (self.chunk_size - self.chunk_shapes[(0, max_x - 1, 0)][1]),
            max_z * self.chunk_size - (self.chunk_size - self.chunk_shapes[(0, 0, max_z - 1)][2]),
        )

        # Set up write file
        self.write_location = write_location
        if write_location is None:
            self.write_chunks = self.read_chunks
        else:
            # New file mode: create a new HDF5 file with same structure
            self.write_file = h5py.File(write_location, "w")
            for (cy, cx, cz), shape in self.chunk_shapes.items():
                ds_name = f"chunk_{cy}_{cx}_{cz}"
                self.write_chunks[(cy, cx, cz)] = self.write_file.create_dataset(
                    ds_name, shape=shape, chunks=(shape[0], shape[1], 1), dtype=self._dtype
                )
                # Copy original data to new file
                self.write_chunks[(cy, cx, cz)][:] = self.read_chunks[(cy, cx, cz)][:]

    def __getitem__(self, key: int | tuple[slice | Literal[None], slice | Literal[None], slice | Literal[None]] | slice) -> npt.ArrayLike:
        if not isinstance(key, tuple):
            key = (key,)
        slices = []
        for i, s in enumerate(key[:3]):
            if isinstance(s, int):
                start = s if s >= 0 else self.shape[i] + s
                slices.append(slice(start, start + 1, 1))
            elif isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else self.shape[i]
                step = s.step if s.step is not None else 1
                start = start if start >= 0 else self.shape[i] + start
                stop = stop if stop >= 0 else self.shape[i] + stop
                slices.append(slice(start, stop, step))
            else:
                raise ValueError("Invalid index type")
        while len(slices) < 3:
            slices.append(slice(0, self.shape[len(slices)], 1))

        out_shape = [(s.stop - s.start + s.step - 1) // s.step for s in slices]
        y_chunk_start = slices[0].start // self.chunk_size
        y_chunk_stop = (slices[0].stop - 1) // self.chunk_size + 1
        x_chunk_start = slices[1].start // self.chunk_size
        x_chunk_stop = (slices[1].stop - 1) // self.chunk_size + 1
        z_chunk_start = slices[2].start // self.chunk_size
        z_chunk_stop = (slices[2].stop - 1) // self.chunk_size + 1

        result = np.zeros(out_shape, dtype=self._dtype)
        for cy in range(y_chunk_start, y_chunk_stop):
            for cx in range(x_chunk_start, x_chunk_stop):
                for cz in range(z_chunk_start, z_chunk_stop):
                    if (cy, cx, cz) not in self.read_chunks:
                        continue
                    chunk = self.read_chunks[(cy, cx, cz)]
                    chunk_y_start = cy * self.chunk_size
                    chunk_x_start = cx * self.chunk_size
                    chunk_z_start = cz * self.chunk_size
                    chunk_y_end = chunk_y_start + chunk.shape[0]
                    chunk_x_end = chunk_x_start + chunk.shape[1]
                    chunk_z_end = chunk_z_start + chunk.shape[2]

                    y_start = max(slices[0].start, chunk_y_start) - chunk_y_start
                    y_end = min(slices[0].stop, chunk_y_end) - chunk_y_start
                    x_start = max(slices[1].start, chunk_x_start) - chunk_x_start
                    x_end = min(slices[1].stop, chunk_x_end) - chunk_x_start
                    z_start = max(slices[2].start, chunk_z_start) - chunk_z_start
                    z_end = min(slices[2].stop, chunk_z_end) - chunk_z_start

                    if y_start >= y_end or x_start >= x_end or z_start >= z_end:
                        continue

                    res_y_start = max(slices[0].start, chunk_y_start) - slices[0].start
                    res_y_end = min(slices[0].stop, chunk_y_end) - slices[0].start
                    res_x_start = max(slices[1].start, chunk_x_start) - slices[1].start
                    res_x_end = min(slices[1].stop, chunk_x_end) - slices[1].start
                    res_z_start = max(slices[2].start, chunk_z_start) - slices[2].start
                    res_z_end = min(slices[2].stop, chunk_z_end) - slices[2].start

                    chunk_data = chunk[
                        y_start : y_end : slices[0].step,
                        x_start : x_end : slices[1].step,
                        z_start : z_end : slices[2].step,
                    ]
                    result[
                        res_y_start : res_y_end : slices[0].step,
                        res_x_start : res_x_end : slices[1].step,
                        res_z_start : res_z_end : slices[2].step,
                    ] = chunk_data

        if len(key) == 1 and isinstance(key[0], int):
            return result[0, 0, 0]
        elif len(key) == 2 and isinstance(key[1], int):
            return result[:, 0, :]
        return result

    def __setitem__(
        self, key: int | tuple[slice | Literal[None], slice | Literal[None], slice | Literal[None]] | slice, value: npt.ArrayLike
    ):
        """Write data to the volume."""
        if not isinstance(key, tuple):
            key = (key,)
        slices = []
        for i, s in enumerate(key[:3]):
            if isinstance(s, int):
                start = s if s >= 0 else self.shape[i] + s
                slices.append(slice(start, start + 1, 1))
            elif isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else self.shape[i]
                step = s.step if s.step is not None else 1
                start = start if start >= 0 else self.shape[i] + start
                stop = stop if stop >= 0 else self.shape[i] + stop
                slices.append(slice(start, stop, step))
            else:
                raise ValueError("Invalid index type")
        while len(slices) < 3:
            slices.append(slice(0, self.shape[len(slices)], 1))

        # Check value shape matches slice
        expected_shape = [(s.stop - s.start + s.step - 1) // s.step for s in slices]
        if np.asarray(value).shape != tuple(expected_shape):
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {value.shape}")

        y_chunk_start = slices[0].start // self.chunk_size
        y_chunk_stop = (slices[0].stop - 1) // self.chunk_size + 1
        x_chunk_start = slices[1].start // self.chunk_size
        x_chunk_stop = (slices[1].stop - 1) // self.chunk_size + 1
        z_chunk_start = slices[2].start // self.chunk_size
        z_chunk_stop = (slices[2].stop - 1) // self.chunk_size + 1

        value = np.asarray(value)
        for cy in range(y_chunk_start, y_chunk_stop):
            for cx in range(x_chunk_start, x_chunk_stop):
                for cz in range(z_chunk_start, z_chunk_stop):
                    if (cy, cx, cz) not in self.write_chunks:
                        continue  # Skip if chunk doesn’t exist
                    chunk = self.write_chunks[(cy, cx, cz)]
                    chunk_y_start = cy * self.chunk_size
                    chunk_x_start = cx * self.chunk_size
                    chunk_z_start = cz * self.chunk_size
                    chunk_y_end = chunk_y_start + chunk.shape[0]
                    chunk_x_end = chunk_x_start + chunk.shape[1]
                    chunk_z_end = chunk_z_start + chunk.shape[2]

                    y_start = max(slices[0].start, chunk_y_start) - chunk_y_start
                    y_end = min(slices[0].stop, chunk_y_end) - chunk_y_start
                    x_start = max(slices[1].start, chunk_x_start) - chunk_x_start
                    x_end = min(slices[1].stop, chunk_x_end) - chunk_x_start
                    z_start = max(slices[2].start, chunk_z_start) - chunk_z_start
                    z_end = min(slices[2].stop, chunk_z_end) - chunk_z_start

                    if y_start >= y_end or x_start >= x_end or z_start >= z_end:
                        continue

                    res_y_start = max(slices[0].start, chunk_y_start) - slices[0].start
                    res_y_end = min(slices[0].stop, chunk_y_end) - slices[0].start
                    res_x_start = max(slices[1].start, chunk_x_start) - slices[1].start
                    res_x_end = min(slices[1].stop, chunk_x_end) - slices[1].start
                    res_z_start = max(slices[2].start, chunk_z_start) - slices[2].start
                    res_z_end = min(slices[2].stop, chunk_z_end) - slices[2].start

                    chunk[
                        y_start : y_end : slices[0].step,
                        x_start : x_end : slices[1].step,
                        z_start : z_end : slices[2].step,
                    ] = value[
                        res_y_start : res_y_end : slices[0].step,
                        res_x_start : res_x_end : slices[1].step,
                        res_z_start : res_z_end : slices[2].step,
                    ]

    def __array__(self):
        return self[:, :, :]

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def close(self):
        """Close both read and write files."""
        self.read_file.close()
        self.write_file.close()


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hdf5_path = "optimized.h5"

    volume = HDF5Volume(hdf5_path, write_location=None)  # <- specify write_location to create a new file to write to

    print(f"Volume shape: {volume.shape}")

    # Get a single slice
    slice_50 = volume[:, :, 50]  # <- in memory slice
    plt.imshow(slice_50, cmap="gray")
    plt.title("Slice 50")
    plt.show()

    # Get a 3D region
    subvolume = volume[1000:, 1000:, 40:60]  # <- in memory slice
    # modifications to subvolume will not be written to disk automatically
    subvolume = np.log(subvolume + 1)  # <- not written to disk
    volume[1000:, 1000:, 40:60] = subvolume  # <- write to disk

    print(f"Subvolume shape: {subvolume.shape}")

    plt.imshow(subvolume[:, :, 0], cmap="gray")
    plt.title("Subvolume slice 0")
    plt.show()
