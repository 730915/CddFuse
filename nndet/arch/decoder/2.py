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

# Convolution and Pooling with Static Same Padding for 3D
class Conv3dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            bias=bias, groups=groups, dilation=dilation
        )
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 3
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 3

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 3

    def forward(self, x):
        d, h, w = x.shape[-3:]
        
        # Calculate padding for depth, height, and width
        pad_d = max((math.ceil(d / self.stride[0]) - 1) * self.stride[0] - d + self.kernel_size[0], 0)
        pad_h = max((math.ceil(h / self.stride[1]) - 1) * self.stride[1] - h + self.kernel_size[1], 0)
        pad_w = max((math.ceil(w / self.stride[2]) - 1) * self.stride[2] - w + self.kernel_size[2], 0)
        
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad expects padding in the order of (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back])

        x = self.conv(x)
        return x

class MaxPool3dStaticSamePadding(nn.Module):
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride=stride, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 3
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 3

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 3

    def forward(self, x):
        d, h, w = x.shape[-3:]
        
        # Calculate padding for depth, height, and width
        pad_d = max((math.ceil(d / self.stride[0]) - 1) * self.stride[0] - d + self.kernel_size[0], 0)
        pad_h = max((math.ceil(h / self.stride[1]) - 1) * self.stride[1] - h + self.kernel_size[1], 0)
        pad_w = max((math.ceil(w / self.stride[2]) - 1) * self.stride[2] - w + self.kernel_size[2], 0)
        
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad expects padding in the order of (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back])

        x = self.pool(x)
        return x

# Separable Convolution Block for 3D
class SeparableConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock3D, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv3dStaticSamePadding(
            in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False
        )
        self.pointwise_conv = Conv3dStaticSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1
        )

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm3d(num_features=out_channels, momentum=0.01, eps=1e-3)

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

# BiFPN Module for 3D
class BiFPN3D(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN3D, self).__init__()
        self.epsilon = epsilon
        self.attention = attention
        self.first_time = first_time

        # 定义对齐层，直接使用显式的对齐层而非循环生成
        self.align_p3 = nn.Conv3d(conv_channels[0], num_channels, kernel_size=1)
        self.align_p4 = nn.Conv3d(conv_channels[1], num_channels, kernel_size=1)
        self.align_p5 = nn.Conv3d(conv_channels[2], num_channels, kernel_size=1)
        self.align_p6 = nn.Conv3d(conv_channels[3], num_channels, kernel_size=1)
        self.align_p7 = nn.Conv3d(conv_channels[4], num_channels, kernel_size=1)

        # 下面是与 BiFPN 计算流程相关的层/模块，保持原样即可
        self.conv6_up = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock3D(num_channels, onnx_export=onnx_export)

        # 注意：以下 Upsample(2,2,2) 或 (1,2,2) 等需手动与实际 Encoder 匹配
        self.p6_upsample = nn.Upsample(scale_factor=(1, 1, 2), mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')

        self.p4_downsample = MaxPool3dStaticSamePadding(kernel_size=3, stride=(1, 2, 2))
        self.p5_downsample = MaxPool3dStaticSamePadding(kernel_size=3, stride=(1, 2, 2))
        self.p6_downsample = MaxPool3dStaticSamePadding(kernel_size=3, stride=(1, 2, 2))
        self.p7_downsample = MaxPool3dStaticSamePadding(kernel_size=3, stride=(1, 1, 2))

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # 如果依然需要 first_time=True 支持，就保留原先对 (p3,p4,p5) 的逻辑
        # 这里只示例让输入 5 层时 first_time=False
        if self.first_time:
            # ...
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
            align_conv = nn.Conv3d(in_channels=in_ch, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
            self.align_layers.append(align_conv)

    def forward(self, inputs):
        """
        如果是 5 层输入，则先用 align_p* 把 p3,p4,p5,p6,p7 都变到统一通道数 (num_channels)。
        然后再执行 BiFPN 的多尺度融合。
        """
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        print("对齐前的形状：")
        print(f"p3_in shape: {p3_in.shape}")
        print(f"p4_in shape: {p4_in.shape}")
        print(f"p5_in shape: {p5_in.shape}")
        print(f"p6_in shape: {p6_in.shape}")
        print(f"p7_in shape: {p7_in.shape}")

        # 应用对齐层
        p3_in = self.align_p3(p3_in)
        p4_in = self.align_p4(p4_in)
        p5_in = self.align_p5(p5_in)
        p6_in = self.align_p6(p6_in)
        p7_in = self.align_p7(p7_in)

        print("对齐后的形状：")
        print(f"p3_in shape: {p3_in.shape}")
        print(f"p4_in shape: {p4_in.shape}")
        print(f"p5_in shape: {p5_in.shape}")
        print(f"p6_in shape: {p6_in.shape}")
        print(f"p7_in shape: {p7_in.shape}")

        # 根据你的 attention 开关，进入 fast_attention 或普通 _forward
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention((p3_in, p4_in, p5_in, p6_in, p7_in))
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward((p3_in, p4_in, p5_in, p6_in, p7_in))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs
            p3_in = self.p3_down_channel(p3)

            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            # Attention mechanisms
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0, keepdim=True) + self.epsilon)
            p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0, keepdim=True) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_td)))

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0, keepdim=True) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))

            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0, keepdim=True) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0, keepdim=True) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out))
            )

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0, keepdim=True) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out))
            )

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0, keepdim=True) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out))
            )

            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0, keepdim=True) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        else:
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

            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0, keepdim=True) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out))
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
            p3, p4, p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
            p5_td = self.conv5_up(self.swish(p5_in_1 + self.p5_upsample(p6_td)))
            p4_td = self.conv4_up(self.swish(p4_in_1 + self.p4_upsample(p5_td)))
            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in_2 + p4_td + self.p4_downsample(p3_out))
            )
            p5_out = self.conv5_down(
                self.swish(p5_in_2 + p5_td + self.p5_downsample(p4_out))
            )
            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + self.p6_downsample(p5_out))
            )
            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))
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

    # 假设 BiFPN3D 已正确修改
    bifpn = BiFPN3D(
        num_channels=128,
        conv_channels=[64, 128, 128, 128, 128],
        first_time=False,
        attention=True
    )

    # 创建五层特征图，匹配编码器输出
    p3_in = torch.rand(1, 64, 5, 48, 64)
    p4_in = torch.rand(1, 128, 5, 24, 32)
    p5_in = torch.rand(1, 128, 5, 12, 16)
    p6_in = torch.rand(1, 128, 5, 6, 8)
    p7_in = torch.rand(1, 128, 5, 6, 4)

    features = (p3_in, p4_in, p5_in, p6_in, p7_in)
    outputs = bifpn(features)

    p3_out, p4_out, p5_out, p6_out, p7_out = outputs

    print("=== BiFPN 输出特征 ===")
    print("p3_out.shape:", p3_out.shape)
    print("p4_out.shape:", p4_out.shape)
    print("p5_out.shape:", p5_out.shape)
    print("p6_out.shape:", p6_out.shape)
    print("p7_out.shape:", p7_out.shape)
