import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock3D(nn.Module):
    def __init__(self, drop_prob: float = 0.2, block_size: int = 5):
        super(DropBlock3D, self).__init__()
        self.drop_prob = drop_prob  # 丢弃的概率
        self.block_size = block_size  # DropBlock 的块大小

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
            mask = F.max_pool3d(mask.unsqueeze(1), kernel_size=(self.block_size, self.block_size, self.block_size), stride=(1, 1, 1), padding=self.block_size // 2)
            mask = 1 - mask.squeeze(1)
            x = x * mask.unsqueeze(1) * (mask.numel() / mask.sum())
            return x

    def _compute_gamma(self, x: torch.Tensor) -> float:
        return self.drop_prob / (self.block_size ** 3)

