import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
# 导入必要的库和模块，包括PyTorch基础库、数学函数、特殊层和数据重排工具

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    # 随机深度方法：在训练过程中随机丢弃一些路径，增强模型的鲁棒性和泛化能力
    # 如果丢弃概率为0或不在训练模式，直接返回输入
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 创建适用于任意维度张量的形状，不仅限于2D卷积网络
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化处理
    # 缩放输出以保持期望值不变
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        # DropPath模块：随机深度的PyTorch模块实现，用于提高模型的泛化能力
        # drop_prob: 路径丢弃的概率
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 前向传播时调用drop_path函数，传入当前训练状态
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        # 基础注意力机制模块：实现多头自注意力机制，用于捕获图像特征的全局依赖关系
        # dim: 输入特征的通道数
        # num_heads: 注意力头的数量，将特征分成多个头并行处理
        # qkv_bias: 是否在qkv投影中使用偏置
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习的缩放参数
        # 两层卷积用于生成查询(q)、键(k)和值(v)
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)  # 1x1卷积投影
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)  # 3x3卷积增强特征
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)  # 输出投影

    def forward(self, x):
        # 前向传播实现自注意力机制
        # x的形状: [batch_size, channels, height, width]
        b, c, h, w = x.shape
        # 生成qkv并分割
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)  # 将特征分成查询、键和值
        
        # 重排形状以适应多头注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        
        # 归一化查询和键以提高稳定性
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # 计算注意力分数并应用缩放
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 矩阵乘法计算相似度
        attn = attn.softmax(dim=-1)  # softmax归一化得到注意力权重

        # 应用注意力权重到值向量
        out = (attn @ v)
        
        # 重排回原始形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        # 最终投影
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
        # 多层感知机模块：用于特征变换和非线性映射，类似Vision Transformer中的FFN
        # in_features: 输入特征的通道数
        # hidden_features: 隐藏层的通道数，默认为None
        # ffn_expansion_factor: 隐藏层相对于输入层的扩展倍数
        # bias: 是否使用偏置
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)  # 计算隐藏层维度

        # 输入投影：将输入特征投影到更高维度的隐藏空间
        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        # 深度可分离卷积：在空间维度上进行特征交互，同时保持计算效率
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        # 输出投影：将特征映射回原始维度
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
            
    def forward(self, x):
        # 前向传播实现特征变换
        x = self.project_in(x)  # 输入投影
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 将特征分成两部分
        x = F.gelu(x1) * x2  # 门控机制：GELU激活的特征与另一部分特征相乘
        x = self.project_out(x)  # 输出投影
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        # 基础特征提取模块：结合注意力机制和MLP，用于提取图像的基础特征
        # dim: 输入特征的通道数
        # num_heads: 注意力头的数量
        # ffn_expansion_factor: MLP中隐藏层的扩展倍数
        # qkv_bias: 是否在注意力机制中使用偏置
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')  # 第一个层归一化
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)  # 注意力模块
        self.norm2 = LayerNorm(dim, 'WithBias')  # 第二个层归一化
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)  # MLP模块
                       
    def forward(self, x):
        # 前向传播实现残差连接的Transformer结构
        x = x + self.attn(self.norm1(x))  # 第一个子层：注意力机制与残差连接
        x = x + self.mlp(self.norm2(x))   # 第二个子层：MLP与残差连接
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        # 倒置残差块：MobileNetV2中提出的高效卷积块，先扩展通道再压缩
        # inp: 输入通道数
        # oup: 输出通道数
        # expand_ratio: 通道扩展比例
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)  # 计算扩展后的通道数
        self.bottleneckBlock = nn.Sequential(
            # 点卷积(pw)：通道扩展，1x1卷积增加通道数
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),  # 批归一化（已注释）
            nn.ReLU6(inplace=True),  # ReLU6激活函数
            
            # 深度卷积(dw)：空间特征提取，每个通道单独卷积
            nn.ReflectionPad2d(1),  # 反射填充，保持边缘信息
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),  # 分组卷积
            # nn.BatchNorm2d(hidden_dim),  # 批归一化（已注释）
            nn.ReLU6(inplace=True),  # ReLU6激活函数
            
            # 点卷积(pw-linear)：通道压缩，1x1卷积减少通道数
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),  # 批归一化（已注释）
        )
        
    def forward(self, x):
        # 前向传播直接应用bottleneckBlock序列
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        # 细节节点：用于处理和增强图像的细节特征，基于可逆神经网络(INN)的思想
        super(DetailNode, self).__init__()
        # 三个变换函数，用于特征间的相互作用和信息流动
        # theta_phi: 从z1到z2的加性耦合函数
        # theta_rho: 从z2到z1的乘性耦合函数
        # theta_eta: 从z2到z1的加性耦合函数
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        # 特征混合卷积层
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
                                    
    def separateFeature(self, x):
        # 将特征分成两半，用于可逆网络的两个分支
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
        
    def forward(self, z1, z2):
        # 前向传播实现特征的耦合变换
        # 首先连接并混合特征，然后再分离
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        # 加性耦合：z2更新
        z2 = z2 + self.theta_phi(z1)
        # 仿射耦合：z1更新（乘性+加性）
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        # 细节特征提取模块：堆叠多个DetailNode，用于提取和增强图像的细节特征
        # num_layers: 堆叠的DetailNode层数
        super(DetailFeatureExtraction, self).__init__()
        # 创建多个DetailNode实例并组成序列
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        
    def forward(self, x):
        # 前向传播：将输入特征分成两半，依次通过每个DetailNode
        # 首先将特征分成两个分支
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        # 依次通过每个DetailNode层
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        # 最后将两个分支的特征连接起来
        return torch.cat((z1, z2), dim=1)

# =============================================================================
# 以下是层归一化和Transformer相关组件的实现
# =============================================================================
import numbers
##########################################################################
## Layer Norm 实现
def to_3d(x):
    # 将4D张量(B,C,H,W)转换为3D张量(B,H*W,C)，用于层归一化
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    # 将3D张量(B,H*W,C)转换回4D张量(B,C,H,W)，层归一化后恢复形状
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        # 无偏置的层归一化：只使用缩放参数，没有偏置参数
        # normalized_shape: 需要归一化的特征维度
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1  # 确保是一维形状

        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算方差并归一化
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算特征维度上的方差
        return x / torch.sqrt(sigma+1e-5) * self.weight  # 归一化并应用缩放


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        # 带偏置的层归一化：使用缩放和偏置参数
        # normalized_shape: 需要归一化的特征维度
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1  # 确保是一维形状

        # 可学习的缩放和偏置参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算均值和方差并归一化
        mu = x.mean(-1, keepdim=True)  # 计算特征维度上的均值
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算特征维度上的方差
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias  # 归一化并应用缩放和偏置

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        # 层归一化包装器：根据类型选择使用带偏置或无偏置的层归一化
        # dim: 需要归一化的特征维度
        # LayerNorm_type: 归一化类型，'BiasFree'或'WithBias'
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)  # 无偏置的层归一化
        else:
            self.body = WithBias_LayerNorm(dim)  # 带偏置的层归一化

    def forward(self, x):
        # 前向传播：先将4D张量转为3D进行归一化，再转回4D
        h, w = x.shape[-2:]  # 保存高度和宽度信息
        return to_4d(self.body(to_3d(x)), h, w)  # 转换、归一化、再转换回来

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        # 门控深度卷积前馈网络：结合门控机制和深度卷积的前馈网络
        # dim: 输入特征的通道数
        # ffn_expansion_factor: 隐藏层相对于输入层的扩展倍数
        # bias: 是否使用偏置
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)  # 计算隐藏层维度

        # 输入投影：将输入特征投影到更高维度的隐藏空间，通道数翻倍用于门控机制
        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        # 深度卷积：在空间维度上进行特征交互，同时保持计算效率
        # groups=hidden_features*2表示每个通道单独卷积（深度可分离卷积）
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        # 输出投影：将特征映射回原始维度
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 前向传播实现门控机制
        x = self.project_in(x)  # 输入投影
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 将特征分成两部分
        x = F.gelu(x1) * x2  # 门控机制：GELU激活的特征与另一部分特征相乘
        x = self.project_out(x)  # 输出投影
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        # 多头深度卷积转置自注意力：结合深度卷积和多头自注意力的高效实现
        # dim: 输入特征的通道数
        # num_heads: 注意力头的数量
        # bias: 是否使用偏置
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习的温度参数，用于缩放注意力分数

        # 生成查询、键、值的卷积层
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)  # 1x1卷积生成qkv
        # 深度卷积增强qkv的空间信息
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 前向传播实现多头自注意力机制
        b, c, h, w = x.shape  # 获取输入形状

        # 生成qkv并应用深度卷积
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  # 将特征分成查询、键和值

        # 重排形状以适应多头注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        # 归一化查询和键以提高稳定性
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力分数并应用温度缩放
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 矩阵乘法计算相似度
        attn = attn.softmax(dim=-1)  # softmax归一化得到注意力权重

        # 应用注意力权重到值向量
        out = (attn @ v)

        # 重排回原始形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        # 最终投影
        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        # Transformer块：结合自注意力和前馈网络的基本Transformer结构单元
        # dim: 输入特征的通道数
        # num_heads: 注意力头的数量
        # ffn_expansion_factor: 前馈网络中隐藏层的扩展倍数
        # bias: 是否使用偏置
        # LayerNorm_type: 层归一化的类型
        super(TransformerBlock, self).__init__()

        # 第一个子层：层归一化和多头自注意力
        self.norm1 = LayerNorm(dim, LayerNorm_type)  # 第一个层归一化
        self.attn = Attention(dim, num_heads, bias)  # 多头自注意力
        
        # 第二个子层：层归一化和前馈网络
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # 第二个层归一化
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # 前馈网络

    def forward(self, x):
        # 前向传播实现残差连接的Transformer结构
        x = x + self.attn(self.norm1(x))  # 第一个子层：注意力机制与残差连接
        x = x + self.ffn(self.norm2(x))   # 第二个子层：前馈网络与残差连接

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        # 重叠图像块嵌入：使用3x3卷积将图像转换为特征表示
        # in_c: 输入通道数，默认为3（RGB图像）
        # embed_dim: 嵌入维度，即输出通道数
        # bias: 是否使用偏置
        super(OverlapPatchEmbed, self).__init__()

        # 3x3卷积进行特征提取和维度变换，padding=1保持空间尺寸不变
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        # 前向传播：直接应用卷积投影
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
        # Restormer编码器：基于Transformer的图像编码器，提取基础特征和细节特征
        # inp_channels: 输入通道数，默认为1（灰度图像）
        # out_channels: 输出通道数，默认为1
        # dim: 特征维度
        # num_blocks: 每个级别的Transformer块数量
        # heads: 各级别的注意力头数量
        # ffn_expansion_factor: 前馈网络中隐藏层的扩展倍数
        # bias: 是否使用偏置
        # LayerNorm_type: 层归一化的类型
        super(Restormer_Encoder, self).__init__()

        # 图像块嵌入层：将输入图像转换为特征表示
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 第一级编码器：堆叠多个Transformer块
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # 基础特征提取模块：捕获图像的全局结构和主要内容
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        # 细节特征提取模块：增强图像的细节和纹理信息
        self.detailFeature = DetailFeatureExtraction()
             
    def forward(self, inp_img):
        # 前向传播：提取基础特征和细节特征
        inp_enc_level1 = self.patch_embed(inp_img)  # 图像块嵌入
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # 第一级编码
        base_feature = self.baseFeature(out_enc_level1)  # 提取基础特征
        detail_feature = self.detailFeature(out_enc_level1)  # 提取细节特征
        return base_feature, detail_feature, out_enc_level1

class Restormer_Decoder(nn.Module):
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
        # Restormer解码器：基于Transformer的图像解码器，融合基础特征和细节特征生成输出图像
        # inp_channels: 输入通道数，默认为1（灰度图像）
        # out_channels: 输出通道数，默认为1
        # dim: 特征维度
        # num_blocks: 每个级别的Transformer块数量
        # heads: 各级别的注意力头数量
        # ffn_expansion_factor: 前馈网络中隐藏层的扩展倍数
        # bias: 是否使用偏置
        # LayerNorm_type: 层归一化的类型
        super(Restormer_Decoder, self).__init__()
        # 通道减少层：将连接的特征通道数减半
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        # 第二级编码器：堆叠多个Transformer块进行特征融合
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # 输出层：逐步减少通道数并生成最终输出
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),  # 第一个卷积层减半通道数
            nn.LeakyReLU(),  # 激活函数
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)  # 第二个卷积层生成最终输出
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，将输出值限制在[0,1]范围              
    def forward(self, inp_img, base_feature, detail_feature):
        # 前向传播：融合特征并生成输出图像
        # 连接基础特征和细节特征
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        # 减少通道数
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        # 通过第二级编码器进行特征融合
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        # 生成输出，如果有输入图像则添加残差连接
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img  # 残差连接
        else:
            out_enc_level1 = self.output(out_enc_level1)
        # 应用Sigmoid激活并返回结果
        return self.sigmoid(out_enc_level1), out_enc_level0
    
if __name__ == '__main__':
    # 主函数：用于测试模型的前向传播
    height = 551
    width = 794
    # height = 128
    # width = 128
    window_size = 8  # 窗口大小参数（未使用）
    
    # 创建编码器和解码器模型实例并移至GPU
    modelE = Restormer_Encoder().cuda()  # 编码器
    modelD = Restormer_Decoder().cuda()  # 解码器
    
    # 测试前向传播
    x = torch.randn(1, 1, height, width).cuda()  # 创建随机输入张量
    # 通过编码器提取特征
    base_feature, detail_feature, out_enc_level1 = modelE(x)
    # 通过解码器生成输出
    output, _ = modelD(x, base_feature, detail_feature)
    # 打印输入和输出的形状信息
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")

