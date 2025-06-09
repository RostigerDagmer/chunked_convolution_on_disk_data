import torch
from torch import nn
from chunked_convolution import ChunkedConv3d
from reader import DummyVolume, HDF5Volume
import numpy as np
import matplotlib.pyplot as plt

def make_gaussian_weights(kernel_size: tuple[int, int, int], sigma: float) -> torch.Tensor:
    """
    Create a Gaussian kernel with the specified size and standard deviation.
    """
    kx, ky, kz = kernel_size
    x = torch.arange(0, kx, dtype=torch.float32) - (kx - 1) / 2
    y = torch.arange(0, ky, dtype=torch.float32) - (ky - 1) / 2
    z = torch.arange(0, kz, dtype=torch.float32) - (kz - 1) / 2

    x = torch.exp(-0.5 * (x / sigma)**2)
    y = torch.exp(-0.5 * (y / sigma)**2)
    z = torch.exp(-0.5 * (z / sigma)**2)

    weights = x[:, None, None] * y[None, :, None] * z[None, None, :]
    return weights / weights.sum()


def dummy_volume():
    gaussian_weights = make_gaussian_weights((5, 5, 5), sigma=1.0)
    conv_filter = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(9, 9, 9),
        stride=2,
        bias=False
    )

    disk_volume = DummyVolume(
        shape=(128, 128, 128),
        dtype="float32"
    )

    cast_fn = lambda x: x.float()  # Ensure the input is float32
    
    disk_volume[:, :, :] = torch.rand(disk_volume.shape, dtype=torch.float32)
    print(f"Disk volume: {disk_volume[:]}")

    conv_filter.weight.data = gaussian_weights.unsqueeze(0).unsqueeze(0)
    chunked_conv = ChunkedConv3d(chunk_size=(32, 32, 32), conv_filter=conv_filter)
    output_volume = chunked_conv(disk_volume, cast_fn=cast_fn)
    assert output_volume.shape == (60, 60, 60), "Output shape does not match expected dimensions."

    plt.imshow(output_volume[31, :, :].detach().cpu().numpy(), cmap='gray')
    plt.show()

    print(f"Output volume: {output_volume}")


def disk_volume():
    gaussian_weights = make_gaussian_weights((5, 5, 5), sigma=1.0)
    conv_filter = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(15, 15, 15),
        stride=7,
        bias=False
    )

    disk_volume = HDF5Volume(
        hdf5_file="test_volume.h5",
        dtype="float16",
    )

    # this operates entirely in float16 precision
    print(f"Disk volume size: {disk_volume.shape}")

    conv_filter.weight.data = gaussian_weights.unsqueeze(0).unsqueeze(0)
    chunked_conv = ChunkedConv3d(chunk_size=(128, 128, 128), conv_filter=conv_filter)
    output_volume = chunked_conv(disk_volume)

    plt.imshow(output_volume[31, :, :].detach().cpu().numpy(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    # dummy_volume()
    disk_volume()
    
    