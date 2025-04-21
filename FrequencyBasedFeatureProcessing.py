import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ImprovementModule.FrequencyDecomposition import FrequencyDecomposition

class FrequencyBasedFeatureProcessing(nn.Module):
    """
    基于频率分解的特征处理模块，将输入特征分解为低频和高频部分，
    然后分别传递给BaseFeatureExtraction和DetailFeatureExtraction进行处理。
    """
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(FrequencyBasedFeatureProcessing, self).__init__()
        
        # 频率分解模块
        self.freq_decomp = FrequencyDecomposition(feature_dim=dim, cutoff=0.2, N=900)
        
        # 从net.py导入的模块
        from net import BaseFeatureExtraction, DetailFeatureExtraction, LayerNorm
        
        # 基础特征处理模块 - 处理低频部分
        self.base_feature_extraction = BaseFeatureExtraction(
            dim=dim, 
            num_heads=num_heads, 
            ffn_expansion_factor=ffn_expansion_factor, 
            qkv_bias=bias
        )
        
        # 细节特征处理模块 - 处理高频部分
        self.detail_feature_extraction = DetailFeatureExtraction(num_layers=3)
        
        # 特征融合层
        self.fusion_conv = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入特征，形状为 [B, C, H, W]
            
        返回:
            tuple: 包含处理后的基础特征和细节特征的元组
        """
        # 获取输入特征的形状
        b, c, h, w = x.shape
        
        # 将特征重塑为频率分解模块所需的形状 [B, C, H*W]
        reshaped_feature = x.view(b, c, h*w)
        
        # 进行频率分解，获取低频和高频部分
        low_freq, high_freq = self.freq_decomp(reshaped_feature)
        
        # 将分解后的特征重塑回原始形状 [B, C, H, W]
        low_freq = low_freq.view(b, c, h, w)
        high_freq = high_freq.view(b, c, h, w)
        
        # 使用BaseFeatureExtraction处理低频部分
        base_feature = self.base_feature_extraction(low_freq)
        
        # 使用DetailFeatureExtraction处理高频部分
        detail_feature = self.detail_feature_extraction(high_freq)
        
        # 融合处理后的特征
        enhanced_base = self.fusion_conv(torch.cat([base_feature, low_freq], dim=1))
        enhanced_detail = self.fusion_conv(torch.cat([detail_feature, high_freq], dim=1))
        
        return enhanced_base, enhanced_detail, x


class FrequencyBasedRestormerEncoder(nn.Module):
    """
    基于频率分解的Restormer编码器，集成了频率分解特征处理模块
    """
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):

        super(FrequencyBasedRestormerEncoder, self).__init__()
        
        # 从net.py导入必要的模块
        from net import OverlapPatchEmbed, TransformerBlock
        
        # 特征嵌入层
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # 编码器第一层
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        # 频率分解特征处理模块
        self.freq_feature_processing = FrequencyBasedFeatureProcessing(
            dim=dim, 
            num_heads=heads[2], 
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, 
            LayerNorm_type=LayerNorm_type
        )
             
    def forward(self, inp_img):
        # 特征嵌入
        inp_enc_level1 = self.patch_embed(inp_img)
        
        # 第一层编码
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        # 频率分解特征处理
        enhanced_base, enhanced_detail, _ = self.freq_feature_processing(out_enc_level1)
        
        return enhanced_base, enhanced_detail, out_enc_level1


# 测试代码
if __name__ == '__main__':
    # 设置输入尺寸
    height = 128
    width = 128
    
    # 创建模型实例
    modelE = FrequencyBasedRestormerEncoder().cuda()
    
    # 从net.py导入解码器
    from net import Restormer_Decoder
    modelD = Restormer_Decoder().cuda()
    
    # 创建测试输入
    x = torch.randn(1, 1, height, width).cuda()
    
    # 前向传播测试
    base_feature, detail_feature, out_enc_level1 = modelE(x)
    output, _ = modelD(x, base_feature, detail_feature)
    
    # 打印形状信息
    print(f"输入尺寸: {x.shape}")
    print(f"基础特征尺寸: {base_feature.shape}")
    print(f"细节特征尺寸: {detail_feature.shape}")
    print(f"输出尺寸: {output.shape}")
    
    # 验证频率分解的效果
    print("\n频率分解特征处理测试:")
    freq_processor = FrequencyBasedFeatureProcessing().cuda()
    enhanced_base, enhanced_detail, original = freq_processor(out_enc_level1)
    print(f"增强的基础特征尺寸: {enhanced_base.shape}")
    print(f"增强的细节特征尺寸: {enhanced_detail.shape}")
    
    print("\n测试完成，模型工作正常！")