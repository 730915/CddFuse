import torch 
import torch.nn as nn
from typing import Callable, Tuple, Sequence, Union, List, Optional
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as SwinTransformer
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.blocks.basic import AbstractBlock
from nndet.arch.encoder.SE import SEBlock
from nndet.arch.encoder.DropBlock3D import DropBlock3D  # 适用于3D数据
from nndet.arch.encoder.SelfAttention import SelfAttention3D  # 导入自注意力模块
from nndet.arch.encoder.swimTransformer import SwinTransformer3D
import torch.nn.functional as F  # 加入插值函数
import logging
from nndet.arch.encoder.window_attention_fusion import WindowAttentionFusion
from nndet.arch.encoder.BAM import BAM  # 导入 BAM 模块
from nndet.arch.encoder.FFT import FFT3D  # 导入 FFT3D 模块

# 配置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["Encoder"]

class SelfAttentionFusion(nn.Module):
    def __init__(self, conv_channels, transformer_channels, fused_channels, window_size, num_heads):
        super().__init__()
        # 投影卷积特征和Transformer特征到相同的维度
        self.conv_proj = nn.Conv3d(conv_channels, fused_channels, kernel_size=1)
        self.transformer_proj = nn.Conv3d(transformer_channels, fused_channels, kernel_size=1)
        
        # 使用窗口自注意力模块
        self.self_attention = WindowAttentionFusion(
            dim=fused_channels * 2,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
        )
        
        # 残差连接
        self.residual_conv = nn.Identity()
    
    def forward(self, conv_feat, transformer_feat):
        conv_feat_proj = self.conv_proj(conv_feat)
        transformer_feat_proj = self.transformer_proj(transformer_feat)
        combined = torch.cat([conv_feat_proj, transformer_feat_proj], dim=1)
        fused_feat = self.self_attention(combined)
        C = conv_feat_proj.size(1)
        conv_fused, transformer_fused = torch.split(fused_feat, C, dim=1)
        fused_feat = conv_fused + transformer_fused
        fused_feat = fused_feat + self.residual_conv(conv_feat)
        return fused_feat

class Encoder(AbstractEncoder):
    def __init__(self,
                 conv: Callable[[], nn.Module],
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                 block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                 stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                 first_block_cls: Optional[AbstractBlock] = None,
                 bam_reduction_ratio: int = 16,  # 使用 BAM 时的瓶颈缩放比例
                 drop_prob: float = 0.2,
                 block_size: int = 5):
        super().__init__()
        #self.logger = logging.getLogger(self.__class__.__name__)
        #self.logger.setLevel(logging.WARNING)
        self.num_stages = len(conv_kernels)
        self.dim = conv.dim
        if stage_kwargs is None:
            stage_kwargs = [{}] * self.num_stages
        elif isinstance(stage_kwargs, dict):
            stage_kwargs = [stage_kwargs] * self.num_stages
        assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        if first_block_cls is None:
            first_block_cls = block_cls

        stages = []
        self.out_channels = []
        self.self_attention_fusion_modules = nn.ModuleList()
        self.fft_modules = nn.ModuleList()  # 初始化 FFT3D 模块列表
        in_ch = in_channels
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides

        # 定义并行的Swin Transformer分支
        self.use_transformer = True
        if self.use_transformer:
            # 确保 depths 和 num_heads 的长度与 self.num_stages 一致
            self.depths = [2] * self.num_stages
            self.num_heads = [4 * (2 ** i) for i in range(self.num_stages)]  # 例如 [4, 8, 16, 32, ...]
            self.transformer = SwinTransformer3D(
                in_chans=in_channels,
                embed_dim=start_channels,
                window_size=(2,7,7),
                patch_size=(2,4,4),
                depths=self.depths,
                num_heads=self.num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm
            )
            # 计算每个阶段的Transformer输出通道数
            self.transformer_out_channels = [int(start_channels * 2 ** i) for i in range(self.num_stages)]

        for stage_id in range(self.num_stages):
            current_in_channels = in_ch
            if stage_id == 0:
                _block = first_block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=start_channels,
                    conv_kernel=conv_kernels[stage_id],
                    stride=None,
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            else:
                _block = block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=None,
                    conv_kernel=conv_kernels[stage_id],
                    stride=strides[stage_id - 1],
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            in_ch = _block.get_output_channels()
            self.out_channels.append(in_ch)

            # 在卷积块后添加 BAM 模块（代替 SEBlock）
            bam = BAM(in_ch)  # 只传递 in_channels
            drop_block = DropBlock3D(drop_prob=drop_prob, block_size=block_size)

            stages.append(nn.Sequential(_block, bam, drop_block))

            # 初始化 SelfAttentionFusion 模块
            if self.use_transformer:
                fused_channels = in_ch  # 假设融合后的通道数与卷积特征的输出通道数相同
                transformer_channels = self.transformer_out_channels[stage_id]
                window_size = self.transformer.window_size  # 使用与 Swin Transformer 相同的窗口大小
                num_heads = self.num_heads[stage_id]
                self_attention_fusion = SelfAttentionFusion(
                    conv_channels=in_ch,
                    transformer_channels=transformer_channels,
                    fused_channels=fused_channels,
                    window_size=window_size,
                    num_heads=num_heads
                )
                self.self_attention_fusion_modules.append(self_attention_fusion)
            
            # 初始化对应阶段的 FFT3D 模块
            fft_module = FFT3D(in_channels=in_ch, reduction=16)  # 使用默认 reduction 比例
            self.fft_modules.append(fft_module)

        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        if self.use_transformer:
            transformer_feats = self.transformer(x)
        
        for stage_id, module in enumerate(self.stages):
            x = module(x)
            x = self.fft_modules[stage_id](x)

            if self.use_transformer and stage_id < len(self.transformer_out_channels):
                transformer_feat = transformer_feats[stage_id]
                transformer_feat_resized = F.interpolate(
                    transformer_feat, size=x.shape[2:], mode='trilinear', align_corners=False)
                x = self.self_attention_fusion_modules[stage_id](x, transformer_feat_resized)

            if stage_id in self.out_stages:
                outputs.append(x)

        return outputs


    def get_channels(self) -> List[int]:
        """
        计算每个返回特征图的通道数
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        计算每个输出特征图相对于输入大小的步幅
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * s for prev_stride, s
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides