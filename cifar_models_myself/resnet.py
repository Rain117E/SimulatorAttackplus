'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL.Image import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def visualize_feature_map(feature_batch):
    '''
    创建特征子图，创建叠加后的特征图
    :param feature_batch: 一个卷积层所有特征图
    :return:
    '''
    shapesize = feature_batch.shape
    b = shapesize[0]
    c = shapesize[1]
    h = shapesize[2]
    w = shapesize[3]

    feature_map = torch.squeeze(feature_batch, 0)
    feature_map_combination = []
    plt.figure(figsize=(8, 7))
    # 取出 featurn map 的数量，因为特征图数量很多，这里直接手动指定了。
    #num_pic = feature_map.shape[2]
    count = 0
    # 将 每一层卷积的特征图，拼接层 5 × 5
    feature_batch_sum = torch.sum(feature_batch, dim=1)
    # feature_map_split = feature_batch[:, 0, :, :]
    feature_map_split = torch.div(feature_batch_sum, 64)
    for i in range(0, 64):
        feature_map_split = torch.squeeze(feature_batch[:, i, :, :], 0)
        for k in range(0, b):
            feature_map_split_np_img = feature_map_split[k, :, :].cpu().detach().numpy()
            plt.subplot(1, b, k + 1)
            plt.imshow(feature_map_split_np_img)
            plt.axis('off')
        plt.show()


def visualize_orginal_imgs(x):
    shapesize = x.shape
    b = shapesize[0]
    c = shapesize[1]
    h = shapesize[2]
    w = shapesize[3]

    plt.figure(figsize=(8, 7))
    count = 0
    for k in range(0, b):
        orginal_np_imgs = x[k, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(1, b, k + 1)
        plt.imshow(orginal_np_imgs)
        plt.axis('off')
    plt.show()
    # path = "/home/djy/Desktop/pics/original/" + str(count) + ".jpg"
    # orginal_np_imgs = cv2.cvtColor(orginal_np_imgs * 255, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(path, orginal_np_imgs)

def feature_map_attention(feature_batch): # 0.75
    shapesize = feature_batch.shape
    b = shapesize[0]
    c = shapesize[1]
    h = shapesize[2]
    w = shapesize[3]
    attention_zeros = torch.zeros([b,h,w]).cuda()
    attention_ones = torch.ones([b,h,w]).cuda()
    attention_res = torch.zeros([b,h,w]).cuda()
# for i in range(0, 25):
    feature_batch_sum = torch.sum(feature_batch ,dim = 1)
    # feature_map_split = feature_batch[:, 0, :, :]
    feature_map_split = torch.div(feature_batch_sum, 64)
    for k in range(0, b):
        feature_map_split_k_np = feature_map_split[k, :, :].cpu().detach().numpy()
        attention_res = torch.where(feature_map_split[k, :, :] > 2, attention_ones, attention_zeros)
        attention_res_np = attention_res.cpu().detach().numpy()
    return attention_res


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x):
    #     # visualize_orginal_imgs(x)
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     # visualize_feature_map(out)
    #     # ===================================================================
    #     # ===================================================================
    #     out = self.layer1(out)
    #     atten_res = feature_map_attention(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     # ===================================================================
    #     return out, atten_res
    #     # ===================================================================
    #     # return out
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # visualize_feature_map(out)
        # ===================================================================
        out = self.layer1(out)
        atten_res = feature_map_attention(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        last_feature = out
        out = self.linear(out)
        # ===================================================================
        return out, atten_res , last_feature


def ResNet18(in_channels, num_classes):
    return ResNet(in_channels, BasicBlock, [2,2,2,2], num_classes)

def ResNet34(in_channels, num_classes):
    return ResNet(in_channels, BasicBlock, [3,4,6,3], num_classes)

def ResNet50(in_channels, num_classes):
    return ResNet(in_channels, Bottleneck, [3,4,6,3], num_classes)

def ResNet101(in_channels, num_classes):
    return ResNet(in_channels, Bottleneck, [3,4,23,3], num_classes)

def ResNet152(in_channels, num_classes):
    return ResNet(in_channels, Bottleneck, [3,8,36,3], num_classes)
