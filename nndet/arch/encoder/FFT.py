import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        通道注意力模块

        :param in_channels: 输入通道数
        :param reduction: 通道缩减比例
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        :param x: 输入张量，形状 (B, C, D, H, W)
        :return: 加权后的张量，形状 (B, C, D, H, W)
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class FFT3D(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        初始化3D FFT模块，并添加通道注意力

        :param in_channels: 输入通道数
        :param reduction: 通道注意力中的缩减比例
        """
        super(FFT3D, self).__init__()
        self.channel_attention_real = ChannelAttention(in_channels=in_channels, reduction=reduction)
        self.channel_attention_imag = ChannelAttention(in_channels=in_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行3D FFT，应用通道注意力，逆FFT，并返回重建的空间域张量

        :param x: 输入张量，形状为 (B, C, D, H, W)
        :return: 重建后的张量，形状为 (B, C, D, H, W)
        """
        # 将张量转换为 float32
        x = x.float()
        
        # 断言张量类型为 float32
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        
        with torch.cuda.amp.autocast(enabled=False):
            # 执行3D FFT
            x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
            
            # 分离实部和虚部
            real = x_fft.real
            imag = x_fft.imag

            # 应用通道注意力
            real_weighted = self.channel_attention_real(real)  # (B, C, D, H, W)
            imag_weighted = self.channel_attention_imag(imag)  # (B, C, D, H, W)

            # 重新组合并进行逆 FFT
            x_fft_weighted = real_weighted + 1j * imag_weighted  # (B, C, D, H, W)
            x_reconstructed = torch.fft.ifftn(x_fft_weighted, dim=(-3, -2, -1)).real  # (B, C, D, H, W)

        return x_reconstructed

class BAM(nn.Module):
    def __init__(self, in_channels, external_attention_map=None):
        super(BAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 如果有外部病变注意力图，初始化它
        self.external_attention_map = external_attention_map

    def forward(self, x):
        # 计算通道注意力
        ca_map = self.channel_attention(x)
        # 计算空间注意力
        sa_map = self.spatial_attention(x)
        
        # 合并通道和空间注意力图
        attention_map = ca_map + sa_map
        
        # 如果外部病变注意力图存在，进行尺寸调整
        if self.external_attention_map is not None:
            # 确保 external_attention_map 的维度为 (C, H, W) 或 (1, C, H, W)
            if len(self.external_attention_map.shape) == 3:  # 如果它是 (C, H, W)，加一个 batch 维度
                self.external_attention_map = self.external_attention_map.unsqueeze(0)  # (1, C, H, W)
            elif len(self.external_attention_map.shape) == 4:  # 如果它已经是 (1, C, D, H, W)
                pass
            else:
                raise ValueError("external_attention_map 应具有3D或4D维度，当前维度: {}".format(self.external_attention_map.shape))
            
            # 确保病变注意力图和输入特征图具有相同的空间维度
            external_attention_map_resized = F.interpolate(
                self.external_attention_map, size=attention_map.shape[2:], mode='bilinear', align_corners=False
            )
            attention_map = attention_map * external_attention_map_resized  # 用外部注意力图加权

        attention_map = torch.sigmoid(attention_map)  # 归一化到[0,1]
        
        # 自适应加权特征图
        out = x * attention_map
        return out