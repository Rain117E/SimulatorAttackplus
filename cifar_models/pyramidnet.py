import cv2
import torch
import torch.nn as nn
import math

__all__ = ['pyramidnet272']

import torchvision
from matplotlib import pyplot as plt


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def calc_prob(curr_layer, total_layers, p_l):
    """Calculates drop prob depending on the current layer."""
    return 1 - (float(curr_layer) / total_layers) * p_l

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
    # num_pic = feature_map.shape[2]

    # 将 每一层卷积的特征图，拼接层 5 × 5
    count = 0
    # feature_batch_sum = torch.sum(feature_batch, dim=1)
    # # feature_map_split = feature_batch[:, 0, :, :]
    # feature_map_split = torch.div(feature_batch_sum, 16)
    for i in range(0, 16):
        feature_map_split = torch.squeeze(feature_map[:, i, :, :], 0)
        for k in range(0, 9):
            feature_map_split_np_img = feature_map_split[k, :, :].cpu().detach().numpy()
            plt.subplot(1, 9, k + 1)
            plt.imshow(feature_map_split_np_img)
            plt.axis('off')
        plt.show()
    # path = "/home/djy/Desktop/pics/black-box/" + str(count) + ".jpg"
    # cv2.imwrite(path, feature_map_split_np_img)

def visualize_orginal_imgs(x):
    shapesize = x.shape
    b = shapesize[0]
    c = shapesize[1]
    h = shapesize[2]
    w = shapesize[3]

    plt.figure(figsize=(8, 7))

    for k in range(0, b):
        orginal_np_imgs = x[k, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(1, b, k + 1)
        plt.imshow(orginal_np_imgs)
        plt.axis('off')
        torchvision.utils.save_image(orginal_np_imgs, "/home/djy/Desktop/pics/original/1.png")
    plt.show()

def feature_map_attention(feature_batch): # 0.75
    shapesize = feature_batch.shape
    b = shapesize[0]
    c = shapesize[1]
    h = shapesize[2]
    w = shapesize[3]
    plt.figure(figsize=(8, 7))
    attention_zeros = torch.zeros([b,h,w]).cuda()
    attention_ones = torch.ones([b,h,w]).cuda()
    attention_res = torch.zeros([b,h,w]).cuda()
    feature_map = torch.squeeze(feature_batch, 0)
# for i in range(0, 25):
    # feature_batch_sum = torch.sum(feature_batch ,dim = 1)
    feature_map_split = feature_batch[:, 12, :, :]
    # feature_map_split = torch.div(feature_batch_sum, 64)
    for k in range(0, 9):
        feature_map_split_k_np = feature_map_split[k, :, :].cpu().detach().numpy()
        attention_res = torch.where(feature_map_split[k, :, :] > 3, attention_ones, attention_zeros)
        plt.subplot(1, 9, k + 1)
        plt.imshow(attention_res[k, :, :].cpu().detach().numpy())
        plt.axis('off')
    plt.show()
    return attention_res

class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, prob=1.):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.conv2 = nn.Conv2d(planes, (planes * 1), kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            self.conv2 = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)),
                                       nn.Conv2d(planes, (planes * 1), kernel_size=3, stride=stride,
                                                 padding=0, bias=False))
        self.bn3 = nn.BatchNorm2d((planes * 1))
        self.conv3 = nn.Conv2d((planes * 1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        # shake drop inference
        # we may support shake drop training in a future version
        out = out * self.prob

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        if residual_channel != shortcut_channel:
            padding_tensor = torch.zeros(batch_size, residual_channel - shortcut_channel,
                                           featuremap_size[0], featuremap_size[1]).type(shortcut.dtype).to(shortcut.device)
            padding_tensor.requires_grad = True
            out += torch.cat((shortcut, padding_tensor), 1)
        else:
            out += shortcut

        return out


class PyramidNet(nn.Module):

    def __init__(self, in_channels, depth, alpha, num_classes):
        super(PyramidNet, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 9)
        block = Bottleneck

        self.addrate = alpha / (3 * n * 1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(in_channels, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim

        self.p_l = 0.5
        self.layer_num = 1
        self.total_layers = n * 3

        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        prob = calc_prob(self.layer_num, self.total_layers, self.p_l)
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample, prob))
        self.layer_num += 1
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            prob = calc_prob(self.layer_num, self.total_layers, self.p_l)
            layers.append(
                block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1,
                      prob=prob))
            self.layer_num += 1
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # visualize_feature_map(x)
        x = self.bn1(x)
        # attenres = feature_map_attention(x)
        # visualize_feature_map(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def pyramidnet272(in_channels, num_classes):
    return PyramidNet(in_channels=in_channels, depth=272, alpha=200, num_classes=num_classes)
