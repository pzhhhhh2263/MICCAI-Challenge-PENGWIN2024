"""
计算两幅图的梯度图的Bce_loss
"""
import numpy as np
from torch.nn import functional as F
from torch import nn
import torch
import matplotlib.pyplot as plt


def show_tensor(image):
    gt = image["gt"]
    gt_sobel = image["gt_sobel"]
    pred_sobel = image["pred_sobel"]
    plt.subplot(1, 3, 1).imshow(gt[0, 0, :, :].detach().cpu().numpy())
    plt.subplot(1, 3, 2).imshow(gt_sobel[0, 0, :, :].detach().cpu().numpy())
    plt.subplot(1, 3, 3).imshow(pred_sobel[0, 0, :, :].detach().cpu().numpy())
    plt.show()


def compute_loss_and_metrics(images):
    """
    This part compute loss and metrics for the generator
    """
    grad_loss = F.l1_loss(images['gt_sobel'], images['pred_sobel'])
    # 取总体差的平方，适用于两组图不对应的情况
    # grad_loss = (torch.mean(images['gt_sobel']) - torch.mean(images['pred_sobel'])) ** 2
    return grad_loss


class SobelOperator(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = torch.tensor(x_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_x.weight.requires_grad = False

        y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight.data = torch.tensor(y_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)
        x = x.view(b, c, h, w)

        return x


class SobelComputer:
    def __init__(self):
        self.sobel = SobelOperator(1e-4)

    def compute_edges(self, images):
        images['gt_sobel'] = self.sobel(images['gt'])
        images['pred_sobel'] = self.sobel(images['pred'])
