import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
import math

'''
EGA模块原理（Edge-GaussianAggregationModule)
1.构建空间不确定性先验：EGA模块首先引入一个固定参数的高斯滤波器，用于模拟图像中局部区域的平
滑变化特性。通过深度可分离卷积方式，该高斯核在每个通道上独立作用，有效保留模糊区域的边缘轮廓并
抑制高频噪声，从而建立一种具有不确定性建模能力的特征表达基础。
2.边缘响应增强与激活归一化：在完成高斯滤波后，EGA模块通过标准化与非线性激活进一步强化边缘响
应特征。这一步骤可提高图像中结构性信息的辨识度，特别是在存在噪声干扰或纹理模糊的场景下，使得边
缘区域在特征图中的表现更加清晰、稳定。
3.特征重构与残差融合：为提升模型对语义特征的表达能力，EGA模块引入可选的特征增强分支。该分支
通过多层卷积对边缘增强后的特征图进行进一步建模，并与原始输入进行残差连接。这种融合策略一方面保
留了底层结构信息，另一方面提升了模型对关键区域的关注能力，从而增强整体的特征判别力和抗干扰能力。
'''

class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x)
        return out

class EdgeGaussianAggregation(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()
    
# 测试代码
if __name__ == "__main__":
    batch_size = 1
    channels = 3
    height, width = 256, 256
    size = 3  # Gaussian kernel size
    sigma = 1.0  # Standard deviation for Gaussian
    norm_layer = dict(type='BN', requires_grad=True)  # Normalization type
    act_layer = nn.ReLU  # Activation function
    
    # 创建输入张量
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # 初始化 Gaussian 模块
    gaussian = EdgeGaussianAggregation(dim=channels, size=size, sigma=sigma, norm_layer=norm_layer, act_layer=act_layer, feature_extra=True)
    print(gaussian)
    print("\n哔哩哔哩: CV缝合救星!\n")

    # 前向传播测试
    output = gaussian(input_tensor)
    
    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")