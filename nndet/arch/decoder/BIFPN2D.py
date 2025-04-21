import math
import torch
from torch import nn
from torch.nn import functional as F

# Swish Activation Functions
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Convolution and Pooling with Static Same Padding for 2D
class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            bias=bias, groups=groups, dilation=dilation
        )
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Calculate padding for height and width
        pad_h = max((math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0], 0)
        pad_w = max((math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1], 0)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad expects padding in the order of (W_left, W_right, H_top, H_bottom)
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        x = self.conv(x)
        return x

class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Calculate padding for height and width
        pad_h = max((math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0], 0)
        pad_w = max((math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1], 0)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad expects padding in the order of (W_left, W_right, H_top, H_bottom)
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        x = self.pool(x)
        return x

# Separable Convolution Block for 2D
class SeparableConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock2D, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False
        )
        self.pointwise_conv = Conv2dStaticSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1
        )

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x) 
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

# BiFPN Module for 2D
class BiFPN2D(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN2D, self).__init__()
        self.epsilon = epsilon
        self.attention = attention
        self.first_time = first_time

        # 定义对齐层，直接使用显式的对齐层而非循环生成
        self.align_p3 = nn.Conv2d(conv_channels[0], num_channels//2, kernel_size=1)
        self.align_p4 = nn.Conv2d(conv_channels[1], num_channels, kernel_size=1)
        self.align_p5 = nn.Conv2d(conv_channels[2], num_channels, kernel_size=1)
        self.align_p6 = nn.Conv2d(conv_channels[3], num_channels, kernel_size=1)
        self.align_p7 = nn.Conv2d(conv_channels[4], num_channels, kernel_size=1)

        # 下面是与 BiFPN 计算流程相关的层/模块
        self.conv6_up = SeparableConvBlock2D(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock2D(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock2D(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock2D(num_channels//2, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock2D(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock2D(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock2D(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock2D(num_channels, onnx_export=onnx_export)

        # 上采样和下采样操作
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.p5_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.p6_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.p7_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # 如果依然需要 first_time=True 支持，就保留原先对 (p3,p4,p5) 的逻辑
        if self.first_time:
            # 这里可以添加first_time=True的相关代码
            pass

        # Attention 相关权重
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.align_layers = nn.ModuleList()
        for in_ch in conv_channels:
            align_conv = nn.Conv2d(in_channels=in_ch, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
            self.align_layers.append(align_conv)

        # 添加通道调整层
        self.p4_td_channel_reduce = nn.Conv2d(num_channels, num_channels//2, kernel_size=1)
        self.p3_out_channel_expand = nn.Conv2d(num_channels//2, num_channels, kernel_size=1)

    def forward(self, inputs):
        """
        如果是 5 层输入，则先用 align_p* 把 p3,p4,p5,p6,p7 都变到统一通道数 (num_channels)。
        然后再执行 BiFPN 的多尺度融合。
        """
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # 应用对齐层
        p3_in = self.align_p3(p3_in)
        p4_in = self.align_p4(p4_in)
        p5_in = self.align_p5(p5_in)
        p6_in = self.align_p6(p6_in)
        p7_in = self.align_p7(p7_in)

        # 根据attention开关，进入fast_attention或普通_forward
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention((p3_in, p4_in, p5_in, p6_in, p7_in))
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward((p3_in, p4_in, p5_in, p6_in, p7_in))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if not self.first_time:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0, keepdim=True) + self.epsilon)
            p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0, keepdim=True) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0, keepdim=True) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            p4_td_reduced = self.p4_td_channel_reduce(p4_td)  # 将p4_td的通道数减半
            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td_reduced)))
            
            # 将p3_out的通道数扩展到128，然后再进行下采样
            p3_out_expanded = self.p3_out_channel_expand(p3_out)  # 扩展到128通道

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0, keepdim=True) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out_expanded))
            )

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0, keepdim=True) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out))
            )

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0, keepdim=True) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out))
            )

            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0, keepdim=True) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            # first_time的处理逻辑，如果需要可以实现
            pass
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
            p5_td = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_td)))
            p4_td = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_td)))
            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td + self.p4_downsample(p3_out))
            )
            p5_out = self.conv5_down(
                self.swish(p5_in + p5_td + self.p5_downsample(p4_out))
            )
            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + self.p6_downsample(p5_out))
            )
            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

# Example Usage
if __name__ == '__main__':
    import torch
    import torch.nn as nn

    # 假设 BiFPN2D 已正确修改
    bifpn = BiFPN2D(
        num_channels=128,
        conv_channels=[64, 128, 128, 128, 128],
        first_time=False,
        attention=True
    )

    # 创建五层特征图，匹配编码器输出
    p3_in = torch.rand(1, 64, 112, 112)
    p4_in = torch.rand(1, 128, 56, 56)
    p5_in = torch.rand(1, 128, 28, 28)
    p6_in = torch.rand(1, 128, 14, 14)
    p7_in = torch.rand(1, 128, 7, 7)

    features = (p3_in, p4_in, p5_in, p6_in, p7_in)
    outputs = bifpn(features)

    p3_out, p4_out, p5_out, p6_out, p7_out = outputs

    print("=== BiFPN 输出特征 ===")
    print("p3_out.shape:", p3_out.shape)
    print("p4_out.shape:", p4_out.shape)
    print("p5_out.shape:", p5_out.shape)
    print("p6_out.shape:", p6_out.shape)
    print("p7_out.shape:", p7_out.shape)