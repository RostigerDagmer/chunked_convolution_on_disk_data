# Utilities for working with large volumetric data

This repo holds several scripts and examples for working with volumetric data that can not fit into DRAM.

**These utilities are not HIGHLY optimized** however they provide reasonable performance on volumes exceeding volatile memory sizes.

#### preprocess.py

Transposes standard formats like .tif stacks (in parallel) into sequences of volume chunks for fast access.

```bash
python preprocess.py --data_dir="D:\data\micro_CT\D25-E10_raw_AP\D25-E10_raw_AP\D25-E10 Skull_01" --output_file="optimized.h5" --chunk_size=256 --dtype=".tif"
```

#### reader.py
Contains an HDF5Volume utility class that allows to arbitrarily slice into volume data on disk with a numpy ArrayLike interface.

```python
hdf5_path = "optimized.h5"

volume = HDF5Volume(hdf5_path, write_location=None)  # <- specify write_location to create a new file to write to

print(f"Volume shape: {volume.shape}")

# Get a single slice
slice_50 = volume[:, :, 50]  # <- in memory slice

# Get a 3D region
subvolume = volume[1000:, 1000:, 40:60]  # <- in memory slice
# modifications to subvolume will not be written to disk automatically
subvolume = np.log(subvolume + 1)  # <- not written to disk
volume[1000:, 1000:, 40:60] = subvolume  # <- write to disk
```

### chunked_convolution.py

Provides a ChunkedConv3d utility class that can be constructed from a torch.Conv3d module.

```python

chunk_size = (1024, ) * 3
conv_filter = nn.Conv3d(1, 1, (15, 15, 15), (7, 7, 7))
kernel_weights = torch.randn(15, 15, 15).unsqueeze(0).unsqueeze(0)
kernel_weights /= kernel_weights.sum()

chunked_conv = ChunkedConv3d(chunk_size, conv_filter)
```

It works directly using any object that inherits from DiskVolume (like HDF5Volume).
If called with default parameters the result of the convolution will end up in a buffer in DRAM (torch.Tensor).

TODO:
Support for writing filtered results straight back to disk is only stubbed out at the moment!