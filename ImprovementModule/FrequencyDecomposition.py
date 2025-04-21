import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionMatching(nn.Module):
    def __init__(self, feature_dim=512, seq_len=5000):
        super(AttentionMatching, self).__init__()
        self.fc_spt = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_qry = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len // 5),
            nn.ReLU(),
            nn.Linear(seq_len // 5, 2 * seq_len),
        )
        self.sigmoid = nn.Sigmoid()

    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        """
        计算空间前景特征和查询前景特征之间的相关矩阵。

        参数:
            spt_fg_fts (torch.Tensor): 空间前景特征。
            qry_fg_fts (torch.Tensor): 查询前景特征。

        返回:
            torch.Tensor: 余弦相似度矩阵。形状: [1, 1, N]。
        """

        spt_fg_fts = F.normalize(spt_fg_fts, p=2, dim=1)  # 形状 [1, 512, N]
        qry_fg_fts = F.normalize(qry_fg_fts, p=2, dim=1)  # 形状 [1, 512, N]

        cosine_similarity = torch.sum(spt_fg_fts * qry_fg_fts, dim=1, keepdim=True)  # 形状: [1, 1, N]

        return cosine_similarity

    def forward(self, spt_fg_fts, qry_fg_fts, band):
        """
        参数:
            spt_fg_fts (torch.Tensor): 空间前景特征。
            qry_fg_fts (torch.Tensor): 查询前景特征。
            band (str): 频段类型，可以是'low'、'high'或其他。

        返回:
            torch.Tensor: 融合后的张量。
        """

        spt_proj = F.relu(self.fc_spt(spt_fg_fts))  # 形状: [1, 512, N]
        qry_proj = F.relu(self.fc_qry(qry_fg_fts))  # 形状: [1, 512, N]

        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))
        
        if band == 'low' or band == 'high':
            weighted_spt = (1 - similarity_matrix) * spt_proj
            weighted_qry = (1 - similarity_matrix) * qry_proj
        else:
            weighted_spt = similarity_matrix * spt_proj
            weighted_qry = similarity_matrix * qry_proj

        combined = torch.cat((weighted_spt, weighted_qry), dim=2)  # 形状: [1, 512, 2*N]
        fused_tensor = F.relu(self.fc_fusion(combined))  # 形状: [1, 512, 2*N]

        return fused_tensor

class FrequencyDecomposition(nn.Module):
    """
    一个将输入特征分解为低频和高频频段的模块，并使用注意力机制进行处理和融合。
    这是基于FAM模块的频率滤波和注意力处理功能的实现。
    """
    def __init__(self, feature_dim=512, cutoff=0.2, N=900):
        """
        初始化FrequencyDecomposition模块。
        
        参数:
            feature_dim (int, 可选): 特征维度。默认为512。
            cutoff (float, 可选): 频段滤波的截止值。
                                 默认为0.2，这意味着低频段将包含最低20%的频率，
                                 而高频段将包含最高20%的频率。
            N (int, 可选): 序列长度。默认为900。
        """
        # 调用父类(nn.Module)的初始化方法
        super(FrequencyDecomposition, self).__init__()
        # 存储截止频率值
        self.cutoff = cutoff
        # 检查是否有可用的CUDA设备，并相应地设置设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # 添加注意力匹配模块
        self.attention_matching = AttentionMatching(feature_dim, N)
        # 添加自适应池化层以确保输入大小一致
        self.adapt_pooling = nn.AdaptiveAvgPool1d(N)

    def reshape_to_square(self, tensor):
        """
        将张量重塑为方形形状，以便进行FFT处理。

        参数:
            tensor (torch.Tensor): 形状为(B, C, N)的输入张量，其中B是批量大小，
                C是通道数，N是元素数量。

        返回:
            tuple: 包含以下内容的元组:
                - square_tensor (torch.Tensor): 形状为(B, C, side_length, side_length)的重塑后的张量
                - side_length (int): 方形张量每边的长度
                - side_length (int): 方形张量每边的长度
                - N (int): 输入张量中原始元素的数量
        """
        # 获取输入张量的形状
        B, C, N = tensor.shape
        # 计算方形张量的边长，使用向上取整确保能容纳所有元素
        side_length = int(np.ceil(np.sqrt(N)))
        # 计算填充后的总长度
        padded_length = side_length ** 2
        
        # 创建一个填充的张量，初始值为零
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        # 将原始数据复制到填充张量的前N个位置
        padded_tensor[:, :, :N] = tensor

        # 将填充后的张量重塑为方形
        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        
        # 返回重塑后的方形张量和相关参数
        return square_tensor, side_length, side_length, N

    def filter_frequency_bands(self, tensor):
        """
        将输入张量过滤为低频和高频频段。

        参数:
            tensor (torch.Tensor): 要过滤的输入张量，形状为(B, C, N)。

        返回:
            tuple: 包含以下内容的元组:
                - low_freq_tensor (torch.Tensor): 输入张量的低频段。
                - high_freq_tensor (torch.Tensor): 输入张量的高频段。
        """
        # 确保张量是浮点类型，因为FFT操作需要浮点数
        tensor = tensor.float()
        # 将张量重塑为方形，以便进行FFT处理
        tensor, H, W, N = self.reshape_to_square(tensor)
        # 获取重塑后张量的形状
        B, C, _, _ = tensor.shape

        # 计算频域中的最大半径（从中心到角落的距离）
        max_radius = np.sqrt((H // 2)**2 + (W // 2)**2)
        # 计算低频截止值
        low_cutoff = max_radius * self.cutoff
        # 计算高频截止值
        high_cutoff = max_radius * (1 - self.cutoff)

        # 对张量应用FFT变换
        # fftshift将零频率分量移到频谱中心
        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

        # 创建频率滤波器的内部函数
        def create_filter(shape, cutoff, mode='low', device=self.device):
            # 获取形状参数
            rows, cols = shape
            # 计算中心点位置
            center_row, center_col = rows // 2, cols // 2
            
            # 创建网格坐标
            y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device))
            # 计算每个点到中心的距离
            distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
            
            # 创建初始掩码，全为零
            mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)
            
            # 根据模式设置掩码值
            if mode == 'low':
                # 低通滤波器：距离小于等于截止值的点设为1
                mask[distance <= cutoff] = 1
            elif mode == 'high':
                # 高通滤波器：距离大于等于截止值的点设为1
                mask[distance >= cutoff] = 1
            
            # 返回创建的滤波器掩码
            return mask

        # 创建低通和高通滤波器
        # [None, None, :, :] 用于扩展维度以匹配张量形状
        low_pass_filter = create_filter((H, W), low_cutoff, mode='low')[None, None, :, :]
        high_pass_filter = create_filter((H, W), high_cutoff, mode='high')[None, None, :, :]

        # 将滤波器应用于FFT张量
        # 元素级乘法实现频域滤波
        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter

        # 应用逆FFT变换回到空间域
        # ifftshift将中心的零频率分量移回原位
        # .real获取复数结果的实部
        low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

        # 将结果重塑回原始形状
        low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
        high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]

        # 返回低频和高频分量
        return low_freq_tensor, high_freq_tensor

    def forward(self, spt_fts, qry_fts=None):
        """
        FrequencyDecomposition模块的前向传播。

        参数:
            spt_fts (torch.Tensor): 形状为(B, C, N)的支持特征张量。
            qry_fts (torch.Tensor, 可选): 形状为(B, C, N)的查询特征张量。
                                        如果为None，则只执行频率分解而不进行注意力处理。

        返回:
            如果qry_fts为None:
                tuple: 包含以下内容的元组:
                    - low_freq (torch.Tensor): 输入的低频分量。
                    - high_freq (torch.Tensor): 输入的高频分量。
            否则:
                torch.Tensor: 经过注意力处理和融合后的特征。
        """
        # 如果只有支持特征，只进行频率分解
        if qry_fts is None:
            # 调用频率分解函数处理输入
            low_freq, high_freq = self.filter_frequency_bands(spt_fts)
            # 返回低频和高频分量
            return low_freq, high_freq
        
        # 如果有查询特征，进行注意力处理和融合
        # 确保输入大小一致
        spt_fts = self.adapt_pooling(spt_fts)
        qry_fts = self.adapt_pooling(qry_fts)
        
        # 分解支持特征和查询特征为低频和高频分量
        spt_low_freq, spt_high_freq = self.filter_frequency_bands(spt_fts)
        qry_low_freq, qry_high_freq = self.filter_frequency_bands(qry_fts)
        
        # 对低频和高频分量分别进行注意力处理
        fused_low_freq = self.attention_matching(spt_low_freq, qry_low_freq, 'low')
        fused_high_freq = self.attention_matching(spt_high_freq, qry_high_freq, 'high')
        
        # 融合处理后的低频和高频分量
        fused_features = torch.cat([fused_low_freq, fused_high_freq], dim=2)
        
        return fused_features


# 测试主函数
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一个形状为[1, 64, 128, 128]的随机张量
    input_tensor = torch.rand(1, 64, 128, 128, device=device)
    print("输入张量的形状:", input_tensor.shape)
    
    # 将4D张量转换为3D张量，因为FrequencyDecomposition期望输入形状为(B, C, N)
    B, C, H, W = input_tensor.shape
    input_tensor_reshaped = input_tensor.view(B, C, H * W)
    print("重塑后的输入张量形状:", input_tensor_reshaped.shape)
    
    # 创建另一个随机张量作为查询特征
    query_tensor = torch.rand(1, 64, 128, 128, device=device)
    query_tensor_reshaped = query_tensor.view(B, C, H * W)
    print("查询张量的形状:", query_tensor_reshaped.shape)
    
    # 实例化FrequencyDecomposition模块
    feature_dim = 64  # 特征维度
    N = 900  # 序列长度
    freq_decomp = FrequencyDecomposition(feature_dim=feature_dim, cutoff=0.2, N=N).to(device)
    
    print("\n测试1: 只进行频率分解")
    # 将输入张量传递给模块，只进行频率分解
    low_freq, high_freq = freq_decomp(input_tensor_reshaped)
    
    # 打印结果形状
    print("低频分量的形状:", low_freq.shape)
    print("高频分量的形状:", high_freq.shape)
    
    # 将结果重塑回原始形状
    low_freq_reshaped = low_freq.view(B, C, H, W)
    high_freq_reshaped = high_freq.view(B, C, H, W)
    
    print("重塑回原始形状后的低频分量形状:", low_freq_reshaped.shape)
    print("重塑回原始形状后的高频分量形状:", high_freq_reshaped.shape)
    
    print("\n测试2: 进行注意力处理和融合")
    # 将支持特征和查询特征传递给模块，进行注意力处理和融合
    fused_features = freq_decomp(input_tensor_reshaped, query_tensor_reshaped)
    
    # 打印融合后特征的形状
    print("融合后特征的形状:", fused_features.shape)
    
    # 检查融合后特征是否可以重塑回原始形状
    # 注意：由于注意力处理，输出形状可能与输入不同
    try:
        # 尝试将融合后的特征重塑为与输入相似的形状
        fused_features_reshaped = fused_features.view(B, C, H, W)
        print("重塑后的融合特征形状:", fused_features_reshaped.shape)
    except RuntimeError as e:
        print("无法将融合特征重塑为原始形状，因为形状已更改:", str(e))
        # 打印实际形状以便了解
        print("融合特征的实际形状:", fused_features.shape)