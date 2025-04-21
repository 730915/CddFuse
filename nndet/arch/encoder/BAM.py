import torch
import torch.nn as nn
import torch.nn.functional as F

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
            if len(self.external_attention_map.shape) == 3:
                self.external_attention_map = self.external_attention_map.unsqueeze(0)
            elif len(self.external_attention_map.shape) == 5:
                pass
            else:
                raise ValueError("external_attention_map 应具有3D或5D维度，当前维度: {}".format(self.external_attention_map.shape))
            
            external_attention_map_resized = F.interpolate(
                self.external_attention_map, size=attention_map.shape[2:], mode='trilinear', align_corners=False
            )
            attention_map = attention_map * external_attention_map_resized

        attention_map = torch.sigmoid(attention_map)
        out = x * attention_map
        return out

