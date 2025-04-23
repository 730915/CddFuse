import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from ImprovementModule.EGA import EdgeGaussianAggregation

class EGAFeatureEnhancement(nn.Module):
    """
    EGA特征增强模块：将EdgeGaussianAggregation模块与网络集成，用于增强边缘特征
    可以作为BaseFeatureExtraction或TransformerBlock的补充，提升图像融合的边缘保留能力
    """
    def __init__(self, dim, size=3, sigma=1.0, norm_layer=dict(type='BN', requires_grad=True), 
                 act_layer=nn.ReLU, feature_extra=True, apply_mode='sequential'):
        super().__init__()
        self.ega = EdgeGaussianAggregation(dim=dim, size=size, sigma=sigma, 
                                         norm_layer=norm_layer, act_layer=act_layer, 
                                         feature_extra=feature_extra)
        self.apply_mode = apply_mode  # 'sequential'或'parallel'
        
    def forward(self, x):
        if self.apply_mode == 'sequential':
            # 顺序模式：直接将输入通过EGA模块处理
            return self.ega(x)
        else:  # parallel模式
            # 并行模式：EGA处理后与原始输入相加（残差连接）
            return x + self.ega(x)


class EGAEnhancedTransformer(nn.Module):
    """
    EGA增强的Transformer块：结合TransformerBlock和EGA模块，增强边缘特征提取能力
    """
    def __init__(self, transformer_block, dim, size=3, sigma=1.0, 
                 norm_layer=dict(type='BN', requires_grad=True), 
                 act_layer=nn.ReLU, feature_extra=True, 
                 apply_position='after'):
        super().__init__()
        self.transformer = transformer_block
        self.ega = EdgeGaussianAggregation(dim=dim, size=size, sigma=sigma, 
                                         norm_layer=norm_layer, act_layer=act_layer, 
                                         feature_extra=feature_extra)
        self.apply_position = apply_position  # 'before', 'after', 或 'both'
        
    def forward(self, x):
        if self.apply_position == 'before':
            # EGA处理后再通过Transformer
            enhanced = self.ega(x)
            return self.transformer(enhanced)
        elif self.apply_position == 'after':
            # 先通过Transformer再EGA增强
            transformed = self.transformer(x)
            return self.ega(transformed)
        else:  # 'both'
            # 前后都使用EGA增强
            enhanced = self.ega(x)
            transformed = self.transformer(enhanced)
            return self.ega(transformed)


class EGAEnhancedRestormerEncoder(nn.Module):
    """
    EGA增强的Restormer编码器：在Restormer编码器的基础上集成EGA模块，增强边缘特征提取
    """
    def __init__(self, original_encoder, dim=64, size=3, sigma=1.0, 
                 norm_layer=dict(type='BN', requires_grad=True), 
                 act_layer=nn.ReLU, feature_extra=True):
        super().__init__()
        self.encoder = original_encoder
        self.ega = EdgeGaussianAggregation(dim=dim, size=size, sigma=sigma, 
                                         norm_layer=norm_layer, act_layer=act_layer, 
                                         feature_extra=feature_extra)
        
    def forward(self, inp_img):
        # 获取原始编码器的特征
        base_feature, detail_feature, out_enc_level1 = self.encoder(inp_img)
        
        # 使用EGA增强细节特征
        enhanced_detail = self.ega(detail_feature)
        
        # 返回增强后的特征
        return base_feature, enhanced_detail, out_enc_level1