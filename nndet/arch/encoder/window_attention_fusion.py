import torch
import torch.nn as nn
from nndet.arch.encoder.swimTransformer import WindowAttention3D, window_partition3D, window_reverse3D

class WindowAttentionFusion(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads

        self.attention = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 如果输入尺寸不能被窗口大小整除，需要进行padding
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        # 更新尺寸
        D_padded = D + pad_d
        H_padded = H + pad_h
        W_padded = W + pad_w

        # 将通道维度移到最后
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)

        # 划分窗口
        x_windows, D_windows, H_windows, W_windows = window_partition_fusion(x, self.window_size)

        # 计算窗口内的自注意力
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attention(x_windows)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        x = window_reverse_fusion(attn_windows, self.window_size, B, D_windows, H_windows, W_windows, D_padded, H_padded, W_padded)

        # 恢复通道维度
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # 去除填充
        x = x[:, :, :D, :H, :W]

        return x

def window_partition_fusion(x, window_size):
    # x: (B, D, H, W, C)
    B, D, H, W, C = x.shape

    D_windows = D // window_size[0]
    H_windows = H // window_size[1]
    W_windows = W // window_size[2]

    x = x.view(B,
               D_windows, window_size[0],
               H_windows, window_size[1],
               W_windows, window_size[2],
               C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # (B, D_windows, H_windows, W_windows, wd, wh, ww, C)
    windows = x.view(-1, window_size[0], window_size[1], window_size[2], C)  # (num_windows*B, wd, wh, ww, C)
    return windows, D_windows, H_windows, W_windows

def window_reverse_fusion(windows, window_size, B, D_windows, H_windows, W_windows, D_padded, H_padded, W_padded):
    # windows: (num_windows*B, wd, wh, ww, C)
    x = windows.view(B,
                     D_windows, H_windows, W_windows,
                     window_size[0], window_size[1], window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()  # (B, D_padded, H_padded, W_padded, C)
    x = x.view(B, D_padded, H_padded, W_padded, -1)
    return x
