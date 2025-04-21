import torch
import torch.nn as nn

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, D, H, W = x.size()

        # 计算query、key和value
        query = self.query(x).view(batch_size, -1, D * H * W)  # [B, C', N]
        key = self.key(x).view(batch_size, -1, D * H * W)      # [B, C', N]
        value = self.value(x).view(batch_size, -1, D * H * W)  # [B, C, N]

        # 计算注意力
        attention = torch.bmm(query.permute(0, 2, 1), key)      # [B, N, N]
        attention = self.softmax(attention)                    # 对最后一个维度进行softmax

        # 计算输出
        out = torch.bmm(value, attention.permute(0, 2, 1))     # [B, C, N]
        out = out.view(batch_size, C, D, H, W)                # 恢复形状

        # 用gamma权重加上输入x
        out = self.gamma * out + x
        return out





