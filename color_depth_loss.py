# utils/loss_utils.py
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    loss = torch.abs((network_output - gt))
    loss = torch.where(gt!=0, loss, 0.)
    return loss, loss.mean()

def l2_loss(network_output, gt):
    loss = ((network_output - gt) ** 2)
    loss = torch.where(gt!=0, loss, 0.)
    return loss.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img, gt, window_size=11, size_average=True):
    img = torch.where(gt!=0, img, 0.)
    channel = img.size(-3)
    window = create_window(window_size, channel)

    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    return _ssim(img, gt, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map, ssim_map.mean()
    else:
        return ssim_map, ssim_map.mean(1).mean(1).mean(1)
    

def gradient_l1(image, gt_image):
    mask = (gt_image != 0).float()
    grad_map = torch.sign(image - gt_image) * mask
    grad = grad_map / image.numel()
    return grad

def gradient_ssim(image, gt_image):
    pass


def main():
    d_max        = 10
    lambda_dssim = 0.2
    image = torch.tensor([[
        [[1.0, 2.0], [0.0, 4.0]],
        [[5.0, 6.0], [7.0, 0.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ]], requires_grad=True)

    gt_image = torch.tensor([[
        [[1.0, 1.0], [1.0, 1.0]],
        [[2.0, 2.0], [2.0, 2.0]],
        [[3.0, 3.0], [3.0, 3.0]]
    ]])

    depth_image    = torch.tensor([[[[1.0, 0.0], [2.0, 3.0]]]], requires_grad=True)
    gt_depth_image = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])


    Ll1_map, Ll1 = l1_loss(image, gt_image)
    L_ssim_map, L_ssim = ssim(image, gt_image)
    loss_rgb = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - L_ssim)

    Ll1_d_map, Ll1_d = l1_loss(depth_image/d_max, gt_depth_image/d_max)
    loss_d = Ll1_d
    
    loss = loss_rgb + 0.1*loss_d

    loss.backward()
    print("Loss:", loss.item())
    print("Gradient w.r.t. image:", image.grad)
    print("Gradient w.r.t. depth:", depth_image.grad)

    depth_image_grad = 0.1 * gradient_l1(depth_image, gt_depth_image)
    print("Gradient w.r.t. depth image(Self):", depth_image_grad)

if __name__ == "__main__":
    main()
