import cv2
import numpy as np
import torch
import math
import torch.nn as nn
from cv2 import waitKey
from matplotlib import pyplot as plt


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

# img = torch.randn([1,3,64,64]).cuda()
# print(img)
img = cv2.imread("1.png")
cv2.imshow("Image1",img)
# cv2.waitKey(0)
# h , w , c
blur_layer = get_gaussian_kernel().cuda()
img_t = torch.from_numpy(img).unsqueeze(0)
img_t = torch.transpose(img_t, 1 , 2)
img_t = torch.transpose(img_t, 1 , 3).type(torch.FloatTensor).cuda()

blured_img = blur_layer(img_t)
blured_img = torch.squeeze(blured_img)
blured_img = torch.transpose(blured_img , 0 , 2)
blured_img = torch.transpose(blured_img , 0 , 1)
blured_img = blured_img.cpu().numpy()

cv2.imshow("Image2",blured_img.astype(np.uint8))
cv2.waitKey(0)


# print(blured_img.shape)
# print(blured_img)