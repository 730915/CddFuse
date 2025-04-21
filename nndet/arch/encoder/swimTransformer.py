import torch
import torch.nn as nn
import logging
import math

# 配置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatchPartition3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4)):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.unfold(2, self.patch_size[0], self.patch_size[0]) \
              .unfold(3, self.patch_size[1], self.patch_size[1]) \
              .unfold(4, self.patch_size[2], self.patch_size[2])  # (B, C, D_p, H_p, W_p, p_D, p_H, p_W)
        x = x.contiguous().view(B, C, -1, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        x = x.permute(0, 2, 1, 3, 4, 5)  # (B, N_patches, C, p_D, p_H, p_W)
        x = x.contiguous().view(B, -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * C)
        return x

class PatchMerging3D(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (D, H, W)
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """
        x: (B, D*H*W, C)
        """
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, f"input feature has wrong size. Expected {D*H*W}, got {L}"

        x = x.view(B, D, H, W, C)

        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            pad_d = (2 - D % 2) % 2
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        D = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C

        x = x.view(B, -1, 8 * C)  # B D/2*H/2*W/2 8*C
        x = self.norm(x)
        x = self.reduction(x)  # B D/2*H/2*W/2 2*C

        return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1), num_heads)
        )

        # 计算相对位置索引
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        # x: (B_*num_windows, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, num_heads, N, C // num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, C // num_heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1).to(torch.long)
        ].view(
            N, N, -1
        )  # N, N, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition3D(x, window_size):
    # x: (B, D, H, W, C)
    B, D, H, W, C = x.shape
    pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
    pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
    pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
    x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

    D_padded = D + pad_d
    H_padded = H + pad_h
    W_padded = W + pad_w

    D_windows = D_padded // window_size[0]
    H_windows = H_padded // window_size[1]
    W_windows = W_padded // window_size[2]

    x = x.view(B, D_windows, window_size[0],
               H_windows, window_size[1],
               W_windows, window_size[2], C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = x.view(-1, window_size[0] * window_size[1] * window_size[2], C)
    return windows, D_padded, H_padded, W_padded, D_windows, H_windows, W_windows

def window_reverse3D(windows, window_size, B, D_windows, H_windows, W_windows, D_padded, H_padded, W_padded, D, H, W):
    x = windows.view(B, D_windows, H_windows, W_windows, window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D_padded, H_padded, W_padded, -1)
    x = x[:, :D, :H, :W, :].contiguous()
    return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0), mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, input_resolution):
        # x: (B, L, C)
        B, L, C = x.shape
        D, H, W = input_resolution
        assert L == D * H * W, f"input feature has wrong size. Expected {D*H*W}, got {L}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # 循环移位
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3)
            )
        else:
            shifted_x = x

        # 将特征划分为窗口
        x_windows, D_padded, H_padded, W_padded, D_windows, H_windows, W_windows = window_partition3D(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows)

        # 将窗口特征合并
        x = window_reverse3D(attn_windows, self.window_size, 
                             B, D_windows, H_windows, W_windows, 
                             D_padded, H_padded, W_padded, 
                             D, H, W)  # (B, D, H, W, C)

        # 逆移位
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3)
            )

        x = x.view(B, D * H * W, C)

        # 残差连接和 MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinTransformer3D(nn.Module):
    def __init__(self,
                 in_chans=1,
                 embed_dim=32,
                 depths=[2, 2, 2],
                 num_heads=[4, 8, 16],
                 window_size=(2, 7, 7),
                 patch_size=(2, 4, 4),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchPartition3D(patch_size=patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.num_heads = num_heads
        self.depths = depths
        assert len(self.depths) == self.num_layers, "Depths length mismatch"
        assert len(self.num_heads) == self.num_layers, "Num_heads length mismatch"

        # 打印检查
        print(f"SwinTransformer3D - num_layers: {self.num_layers}")
        print(f"Depths: {self.depths}")
        print(f"Num heads: {self.num_heads}")

        # 计算补丁嵌入的维度
        self.patch_dim = (
            self.patch_embed.patch_size[0] *
            self.patch_embed.patch_size[1] *
            self.patch_embed.patch_size[2] *
            in_chans
        )
        # 添加线性投影层
        self.linear = nn.Linear(self.patch_dim, self.embed_dim)

        # 构建每一层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            current_dim = int(self.embed_dim * 2 ** i_layer)
            current_num_heads = self.num_heads[i_layer]
            assert current_dim % current_num_heads == 0, f"dim {current_dim} not divisible by num_heads {current_num_heads}"
            logger.info(f"Layer {i_layer}: dim={current_dim}, num_heads={current_num_heads}")

            layer = nn.ModuleList()
            for j in range(self.depths[i_layer]):
                block = SwinTransformerBlock3D(
                    dim=current_dim,
                    num_heads=current_num_heads,
                    window_size=self.window_size,
                    shift_size=(0, 0, 0) if (j % 2 == 0) else (
                        self.window_size[0] // 2,
                        self.window_size[1] // 2,
                        self.window_size[2] // 2),
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=self.drop_path_rate,
                    norm_layer=self.norm_layer,
                )
                layer.append(block)
            self.layers.append(layer)
            if i_layer < self.num_layers -1:
                # 添加 Patch Merging 层
                merge_layer = PatchMerging3D(None, dim=current_dim, norm_layer=self.norm_layer)
                self.layers.append(merge_layer)

        self.norm = self.norm_layer(int(self.embed_dim * 2 ** (self.num_layers - 1)))

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.patch_embed(x)  # (B, N_patches, patch_dim)
        x = self.linear(x)       # (B, N_patches, embed_dim)
        x = self.pos_drop(x)

        # 计算初始的输入分辨率
        D_p = D // self.patch_embed.patch_size[0]
        H_p = H // self.patch_embed.patch_size[1]
        W_p = W // self.patch_embed.patch_size[2]
        input_resolution = (D_p, H_p, W_p)

        features = []
        layer_index = 0
        while layer_index < len(self.layers):
            layer = self.layers[layer_index]
            # 如果是 SwinTransformerBlock3D 的列表
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    x = block(x, input_resolution)
                # 收集当前阶段的特征
                D_p, H_p, W_p = input_resolution
                x_feature = x.view(B, D_p, H_p, W_p, -1).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D_p, H_p, W_p)
                features.append(x_feature)
                layer_index += 1
            # 如果是 PatchMerging3D 层
            elif isinstance(layer, PatchMerging3D):
                layer.input_resolution = input_resolution
                x = layer(x)
                # 更新输入分辨率
                input_resolution = (
                    (input_resolution[0] + 1) // 2,
                    (input_resolution[1] + 1) // 2,
                    (input_resolution[2] + 1) // 2,
                )
                layer_index += 1
            else:
                raise TypeError(f"Unknown layer type at index {layer_index}")

        x = self.norm(x)
        # 重塑 x 的形状
        D_p, H_p, W_p = input_resolution
        x = x.view(B, D_p, H_p, W_p, -1).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D_p, H_p, W_p)
        features[-1] = x  # 更新最后一个特征

        return features  # 返回每个阶段的特征列表