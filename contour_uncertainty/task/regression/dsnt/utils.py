from functools import reduce
from operator import mul

import torch


def dsnt(heatmaps, compute_skew=False):
    linespace = normalized_linspace(heatmaps.shape[-1], dtype=heatmaps.dtype, device=heatmaps.device)[None]
    X = linespace.repeat(heatmaps.shape[-1], 1)
    Y = X.t()

    X = X[None, None]
    Y = Y[None, None]

    x = torch.inner(heatmaps.flatten(-2), X.flatten(-2))
    y = torch.inner(heatmaps.flatten(-2), Y.flatten(-2))

    coords = torch.cat([x.squeeze(2), y.squeeze(2)], dim=-1)

    X_temp = X - x
    X_temp_square = X_temp * X_temp
    Y_temp = Y - y
    Y_temp_square = Y_temp * Y_temp
    xy_temp = X_temp * Y_temp

    # print(heatmaps.flatten(-2).shape, X_temp_square.flatten(-2).shape)
    var_x = (heatmaps.flatten(-2) * X_temp_square.flatten(-2)).sum(-1)
    var_y = (heatmaps.flatten(-2) * Y_temp_square.flatten(-2)).sum(-1)
    covar = (heatmaps.flatten(-2) * xy_temp.flatten(-2)).sum(-1)
    var = torch.cat([var_x[..., None], var_y[..., None]], dim=-1)

    if compute_skew:
        print(x)
        print(var_x)
        X_temp2 = (X - x / torch.sqrt(var_x))
        X_temp2 = X_temp2 * X_temp2 * X_temp2

        Y_temp2 = (Y - y / torch.sqrt(var_y))
        Y_temp2 = Y_temp2 * Y_temp2 * Y_temp2

        skew_x = (heatmaps.flatten(-2) * X_temp2.flatten(-2)).sum(-1)
        skew_y = (heatmaps.flatten(-2) * Y_temp2.flatten(-2)).sum(-1)

        skew = torch.cat([skew_x[..., None], skew_y[..., None]], dim=-1)
        return coords, var, covar, skew

    return coords, var, covar


def normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.
    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:
    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```
    Args:
        length: The length of the vector
    Returns:
        The generated vector
    """
    if isinstance(length, torch.Tensor):
        length = length.to(device, dtype)
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype, device=device) * (2.0 / length) + first


def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""

    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)


def euclidean_losses(actual, target):
    """Calculate the Euclidean losses for multi-point samples.
    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).
    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    Returns:
        Tensor: Losses (B x L)
    """
    assert actual.size() == target.size(), 'input tensors must have the same size'
    return torch.norm(actual - target, p=2, dim=-1, keepdim=False)


def normalized_to_pixel_coordinates(coords, size):
    """Convert from normalized coordinates to pixel coordinates.
    Args:
        coords: Coordinate tensor, where elements in the last dimension are ordered as (x, y, ...).
        size: Number of pixels in each spatial dimension, ordered as (..., height, width).
    Returns:
        `coords` in pixel coordinates.
    """
    if torch.is_tensor(coords):
        size = coords.new_tensor(size).flip(-1)
    return 0.5 * ((coords + 1) * size - 1)


def pixel_to_normalized_coordinates(coords, size):
    """Convert from pixel coordinates to normalized coordinates.
    Args:
        coords: Coordinate tensor, where elements in the last dimension are ordered as (x, y, ...).
        size: Number of pixels in each spatial dimension, ordered as (..., height, width).
    Returns:
        `coords` in normalized coordinates.
    """
    if torch.is_tensor(coords):
        size = coords.new_tensor(size).flip(-1)
    return ((2 * coords + 1) / size) - 1