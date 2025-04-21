import torch
from nndet.arch.decoder.base import BiFPN, decoder_cls

# 假设卷积模块和其他必要参数已经定义
def dummy_conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)

# 定义参数
conv = dummy_conv
strides = [2, 2, 2]
in_channels = [40, 112, 320]
conv_kernels = 3
decoder_levels = None
fixed_out_channels = 88

# 实例化 BiFPN
bifpn_decoder = decoder_cls(conv=conv,
                             strides=strides,
                             in_channels=in_channels,
                             conv_kernels=conv_kernels,
                             decoder_levels=decoder_levels,
                             fixed_out_channels=fixed_out_channels,
                             )

# 创建示例输入
p3 = torch.rand(8, 40, 80, 80)
p4 = torch.rand(8, 112, 40, 40)
p5 = torch.rand(8, 320, 20, 20)

# 前向传播
features = (p3, p4, p5)
outputs = bifpn_decoder(features)
o3, o4, o5 = outputs[:3]

print("o3.shape:", o3.shape)
print("o4.shape:", o4.shape)
print("o5.shape:", o5.shape) 