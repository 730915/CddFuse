import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from nndet.arch.decoder.BiFPN import BiFPN3D, MemoryEfficientSwish, Swish


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detailFeature = DetailFeatureExtraction()
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


# 为2D特征图创建的Conv2dStaticSamePadding类
class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            bias=bias, groups=groups, dilation=dilation, padding=0
        )
        self.stride = self.conv.stride if isinstance(self.conv.stride, tuple) else (self.conv.stride, self.conv.stride)
        self.kernel_size = self.conv.kernel_size if isinstance(self.conv.kernel_size, tuple) else (self.conv.kernel_size, self.conv.kernel_size)
        self.dilation = self.conv.dilation if isinstance(self.conv.dilation, tuple) else (self.conv.dilation, self.conv.dilation)

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # 计算padding
        pad_h = max((math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0], 0)
        pad_w = max((math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1], 0)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 应用padding
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
        x = self.conv(x)
        return x

# 为2D特征图创建的MaxPool2dStaticSamePadding类
class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, **kwargs)
        self.stride = self.pool.stride if isinstance(self.pool.stride, tuple) else (self.pool.stride, self.pool.stride)
        self.kernel_size = self.pool.kernel_size if isinstance(self.pool.kernel_size, tuple) else (self.pool.kernel_size, self.pool.kernel_size)

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # 计算padding
        pad_h = max((math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0], 0)
        pad_w = max((math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1], 0)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 应用padding
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
        x = self.pool(x)
        return x

# 为2D特征图创建的SeparableConvBlock2D类
class SeparableConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
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
            self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = self.depthwise_conv(x) 
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

# 为2D特征图创建的BiFPN2D类
class BiFPN2D(nn.Module):
    def __init__(self, num_channels, conv_channels, epsilon=1e-4, attention=True):
        super(BiFPN2D, self).__init__()
        self.epsilon = epsilon
        self.attention = attention

        # 定义对齐层
        self.align_p3 = nn.Conv2d(conv_channels[0], num_channels//2, kernel_size=1)
        self.align_p4 = nn.Conv2d(conv_channels[1], num_channels, kernel_size=1)
        self.align_p5 = nn.Conv2d(conv_channels[2], num_channels, kernel_size=1)
        self.align_p6 = nn.Conv2d(conv_channels[3], num_channels, kernel_size=1)
        self.align_p7 = nn.Conv2d(conv_channels[4], num_channels, kernel_size=1)

        # BiFPN计算流程相关的层
        self.conv6_up = SeparableConvBlock2D(num_channels)
        self.conv5_up = SeparableConvBlock2D(num_channels)
        self.conv4_up = SeparableConvBlock2D(num_channels)
        self.conv3_up = SeparableConvBlock2D(num_channels//2)

        self.conv4_down = SeparableConvBlock2D(num_channels)
        self.conv5_down = SeparableConvBlock2D(num_channels)
        self.conv6_down = SeparableConvBlock2D(num_channels)
        self.conv7_down = SeparableConvBlock2D(num_channels)

        # 修改上采样和下采样层，不再使用固定的scale_factor
        # 上采样时将使用目标特征图的尺寸
        self.swish = MemoryEfficientSwish()

        # 通道调整层
        self.p4_td_channel_reduce = nn.Conv2d(num_channels, num_channels//2, kernel_size=1)
        self.p3_out_channel_expand = nn.Conv2d(num_channels//2, num_channels, kernel_size=1)

        # Attention权重
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

    def forward(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # 应用对齐层
        p3_in = self.align_p3(p3_in)
        p4_in = self.align_p4(p4_in)
        p5_in = self.align_p5(p5_in)
        p6_in = self.align_p6(p6_in)
        p7_in = self.align_p7(p7_in)

        # 使用attention机制的前向传播
        if self.attention:
            # 自顶向下路径
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0, keepdim=True) + self.epsilon)
            # 使用size参数进行上采样，确保尺寸匹配
            p6_upsample = F.interpolate(p7_in, size=p6_in.shape[2:], mode='nearest')
            p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * p6_upsample))

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0, keepdim=True) + self.epsilon)
            # 使用size参数进行上采样，确保尺寸匹配
            p5_upsample = F.interpolate(p6_td, size=p5_in.shape[2:], mode='nearest')
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * p5_upsample))

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0, keepdim=True) + self.epsilon)
            # 使用size参数进行上采样，确保尺寸匹配
            p4_upsample = F.interpolate(p5_td, size=p4_in.shape[2:], mode='nearest')
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * p4_upsample))

            p4_td_reduced = self.p4_td_channel_reduce(p4_td)  # 将p4_td的通道数减半
            # 使用size参数进行上采样，确保尺寸匹配
            p3_upsample = F.interpolate(p4_td_reduced, size=p3_in.shape[2:], mode='nearest')
            p3_out = self.conv3_up(self.swish(p3_in + p3_upsample))
            
            # 将p3_out的通道数扩展
            p3_out_expanded = self.p3_out_channel_expand(p3_out)

            # 自底向上路径
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0, keepdim=True) + self.epsilon)
            # 使用size参数进行下采样，确保尺寸匹配
            p4_downsample = F.interpolate(p3_out_expanded, size=p4_in.shape[2:], mode='nearest')
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * p4_downsample)
            )

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0, keepdim=True) + self.epsilon)
            # 使用size参数进行下采样，确保尺寸匹配
            p5_downsample = F.interpolate(p4_out, size=p5_in.shape[2:], mode='nearest')
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * p5_downsample)
            )

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0, keepdim=True) + self.epsilon)
            # 使用size参数进行下采样，确保尺寸匹配
            p6_downsample = F.interpolate(p5_out, size=p6_in.shape[2:], mode='nearest')
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * p6_downsample)
            )

            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0, keepdim=True) + self.epsilon)
            # 使用size参数进行下采样，确保尺寸匹配
            p7_downsample = F.interpolate(p6_out, size=p7_in.shape[2:], mode='nearest')
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * p7_downsample))
        else:
            # 不使用attention的前向传播
            # 使用size参数进行上采样，确保尺寸匹配
            p6_upsample = F.interpolate(p7_in, size=p6_in.shape[2:], mode='nearest')
            p6_td = self.conv6_up(self.swish(p6_in + p6_upsample))
            
            p5_upsample = F.interpolate(p6_td, size=p5_in.shape[2:], mode='nearest')
            p5_td = self.conv5_up(self.swish(p5_in + p5_upsample))
            
            p4_upsample = F.interpolate(p5_td, size=p4_in.shape[2:], mode='nearest')
            p4_td = self.conv4_up(self.swish(p4_in + p4_upsample))
            
            p3_upsample = F.interpolate(p4_td, size=p3_in.shape[2:], mode='nearest')
            p3_out = self.conv3_up(self.swish(p3_in + p3_upsample))

            p4_downsample = F.interpolate(p3_out, size=p4_in.shape[2:], mode='nearest')
            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td + p4_downsample)
            )
            
            p5_downsample = F.interpolate(p4_out, size=p5_in.shape[2:], mode='nearest')
            p5_out = self.conv5_down(
                self.swish(p5_in + p5_td + p5_downsample)
            )
            
            p6_downsample = F.interpolate(p5_out, size=p6_in.shape[2:], mode='nearest')
            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + p6_downsample)
            )
            
            p7_downsample = F.interpolate(p6_out, size=p7_in.shape[2:], mode='nearest')
            p7_out = self.conv7_down(self.swish(p7_in + p7_downsample))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class BiFPNDecoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(BiFPNDecoder, self).__init__()
        
        # 通道减少层，将base_feature和detail_feature连接后的通道数减半
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        
        # 创建BiFPN2D模块
        self.bifpn = BiFPN2D(
            num_channels=dim,  # 主通道数
            conv_channels=[dim, dim, dim, dim, dim],  # 输入通道数列表，修正p3_in的通道数为dim而非dim//2
            attention=True
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(int(dim//2), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inp_img, base_feature, detail_feature):
        # 合并base_feature和detail_feature
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        
        # 创建多尺度特征图
        batch_size, channels, height, width = out_enc_level0.shape
        
        # 创建不同尺度的特征图，使用size参数而不是scale_factor
        p3_in = out_enc_level0  # 原始尺寸
        p4_in = F.interpolate(p3_in, size=(height//2, width//2), mode='nearest')  # 1/2尺寸
        p5_in = F.interpolate(p3_in, size=(height//4, width//4), mode='nearest')  # 1/4尺寸
        p6_in = F.interpolate(p3_in, size=(height//8, width//8), mode='nearest')  # 1/8尺寸
        p7_in = F.interpolate(p3_in, size=(height//16, width//16), mode='nearest')  # 1/16尺寸
        
        # 通过BiFPN处理特征
        p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn((p3_in, p4_in, p5_in, p6_in, p7_in))
        
        # 使用p3_out作为主要输出特征
        out_feature = p3_out  # 使用最高分辨率的特征图
        
        # 应用输出层
        if inp_img is not None:
            out_enc_level1 = self.output(out_feature) + inp_img
        else:
            out_enc_level1 = self.output(out_feature)
            
        return self.sigmoid(out_enc_level1), out_enc_level0
    
if __name__ == '__main__':
    height = 551
    width = 794
    # height = 128
    # width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = BiFPNDecoder().cuda()
    
    # 测试前向传播
    x = torch.randn(1, 1, height, width).cuda()
    base_feature, detail_feature, out_enc_level1 = modelE(x)
    output, _ = modelD(x, base_feature, detail_feature)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")