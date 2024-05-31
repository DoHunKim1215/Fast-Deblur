import random

import torch
from torch import Tensor

from utils.converter import rgb2lab


def get_color_hint_randomly(color_image: Tensor, patch_size: int) -> Tensor:
    """
    make hint tensor using label color image.
    pick colors randomly at each image patch.
    """

    B, C, H, W = color_image.shape

    lab_image = rgb2lab(color_image)
    hint_tensor = torch.zeros(B, 3, H, W).to('cuda')

    h_patch_num = H // patch_size
    w_patch_num = W // patch_size

    for i in range(h_patch_num + 1):
        for j in range(w_patch_num + 1):
            x = random.randrange(0, patch_size)
            y = random.randrange(0, patch_size)
            x = j * patch_size + x
            y = i * patch_size + y
            if 0 <= x < W and 0 <= y < H:
                hint_tensor[:, 1:3, y, x] = lab_image[:, 1:3, y, x]
                hint_tensor[:, 0, y, x] = 1

    return hint_tensor


def get_color_hint_evenly(lab_image: Tensor, patch_size: int) -> Tensor:
    """
    make hint tensor using label color image.
    pick colors randomly at each image patch.
    """

    B, C, H, W = lab_image.shape

    hint_tensor = torch.zeros(B, 3, H, W).to('cuda')

    h_patch_num = H // patch_size
    w_patch_num = W // patch_size

    for i in range(h_patch_num):
        for j in range(w_patch_num):
            x = j * patch_size + 7
            y = i * patch_size + 7
            hint_tensor[:, 1:3, y, x] = lab_image[:, 1:3, y, x]
            hint_tensor[:, 0, y, x] = 1

    return hint_tensor


def get_color_hint_evenly_faster(lab_image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Make hint tensor using label color image.
    Pick colors randomly at each image patch.
    """

    B, C, H, W = lab_image.shape

    hint_tensor = torch.zeros(B, 3, H, W, device='cuda')

    h_offsets = torch.arange(patch_size // 2, H, patch_size, device='cuda')
    w_offsets = torch.arange(patch_size // 2, W, patch_size, device='cuda')

    h_patch_indices = h_offsets.view(-1, 1)
    w_patch_indices = w_offsets.view(1, -1)

    hint_tensor[:, 0, h_patch_indices, w_patch_indices] = 1.
    hint_tensor[:, 1:3, h_patch_indices, w_patch_indices] = lab_image[:, 1:3, h_patch_indices, w_patch_indices]

    return hint_tensor


def get_color_hint_at_similar(color_image: Tensor, gray_image: Tensor, patch_size: int) -> Tensor:
    """
    make hint tensor using label color image.
    pick colors at a pixel having most similar luminance with deblur gray image.
    """

    B, C, H, W = color_image.shape

    lab_image = rgb2lab(color_image)
    hint_tensor = torch.zeros(B, 3, H, W)

    h_patch_num = H // patch_size
    w_patch_num = W // patch_size

    for i in range(h_patch_num):
        for j in range(w_patch_num):
            blur_l = lab_image[:, 0:1, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
            sharp_l = gray_image[:, 0:1, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
            diff = abs(blur_l - sharp_l)

            for b in range(B):
                min_val = 100000
                min_w = 0
                min_h = 0
                for h in range(patch_size):
                    for w in range(patch_size):
                        if diff[b, 0, h, w] < min_val:
                            min_val = diff[b, 0, h, w]
                            min_w = w
                            min_h = h

                hint_tensor[b, 1:3, i * patch_size + min_h, j * patch_size + min_w] \
                    = lab_image[b, 1:3, i * patch_size + min_h, j * patch_size + min_w]
                hint_tensor[b, 0, i * patch_size + min_h, j * patch_size + min_w] = 1

    return hint_tensor


def get_laplacian(img: Tensor) -> Tensor:
    B, C, H, W = img.shape

    laplacian = torch.zeros(B, 1, H, W)

    for b in range(B):
        for i in range(H):
            for j in range(W):
                left = img[b, :, i, j - 2 < 0 if 0 else j - 2]
                right = img[b, :, i, j + 2 >= W if W - 1 else j + 2]
                top = img[b, :, i - 2 < 0 if 0 else i - 2, j]
                bottom = img[b, :, i + 2 >= H if H - 1 else i + 2, j]

                pixel_lap = ((right - left) * 0.5).pow(2.0) + ((top - bottom) * 0.5).pow(2.0)
                laplacian[b, 0:1, i, j] = pixel_lap.sum(3).sqrt()

    return laplacian


def get_color_hint_at_smooth(color_image: Tensor, patch_size: int) -> Tensor:
    """
    make hint tensor using label color image.
    pick colors at a smooth area in blur gray image using Laplacian operator.
    """

    B, C, H, W = color_image.shape

    lab_image = rgb2lab(color_image)
    laplacian = get_laplacian(color_image)
    hint_tensor = torch.zeros(B, 3, H, W)

    h_patch_num = H // patch_size
    w_patch_num = W // patch_size

    for i in range(h_patch_num):
        for j in range(w_patch_num):
            for b in range(B):
                min_val = 100000
                min_w = 0
                min_h = 0
                for h in range(patch_size):
                    for w in range(patch_size):
                        if laplacian[b, 0, i * patch_size + h, j * patch_size + w] < min_val:
                            min_val = laplacian[b, 0, i * patch_size + h, j * patch_size + w]
                            min_w = j * patch_size + w
                            min_h = i * patch_size + h

                hint_tensor[b, 1:3, min_h, min_w] = lab_image[b, 1:3, min_h, min_w]
                hint_tensor[b, 0, min_h, min_w] = 1

    return hint_tensor
