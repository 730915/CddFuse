# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
导入所需的包和模块
------------------------------------------------------------------------------
'''

# 导入自定义的网络模型组件
from net_fft import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

# 导入数据集处理类
from utils.dataset import H5Dataset
import os
# 设置环境变量，解决macOS上的库重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 导入自定义的损失函数
from utils.loss import Fusionloss, cc
# 导入kornia库，用于图像处理和损失计算
import kornia
# 导入TensorBoard
from torch.utils.tensorboard import SummaryWriter

# 创建TensorBoard日志记录器
writer = SummaryWriter(log_dir=f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

'''
------------------------------------------------------------------------------
配置网络参数和训练设置
------------------------------------------------------------------------------
'''

# 设置使用的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 初始化融合损失函数
criteria_fusion = Fusionloss()
# 模型名称
model_str = 'CDDFuse'

# 设置训练的超参数
num_epochs = 120 # 总训练轮数
epoch_gap = 40  # 第一阶段训练的轮数（Phase I）

# 学习率设置
lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# 损失函数的系数设置
coeff_mse_loss_VF = 1. # alpha1 - 可见光图像重建损失系数
coeff_mse_loss_IF = 1. # 红外图像重建损失系数
coeff_decomp = 2.      # alpha2和alpha4 - 分解损失系数
coeff_tv = 5. # 总变差损失系数

# 梯度裁剪值
clip_grad_norm_value = 0.01
# 学习率调整步长和衰减率
optim_step = 20
optim_gamma = 0.5


# 模型初始化
# 设置计算设备（GPU或CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 初始化编码器并设置为并行模式
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
# 初始化解码器并设置为并行模式（使用BiFPN_Decoder替换原来的Restormer_Decoder）
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
# 初始化基础特征融合层
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
# 初始化细节特征融合层
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# 优化器、调度器和损失函数设置
# 为编码器设置Adam优化器
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
# 为解码器设置Adam优化器
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
# 为基础特征融合层设置Adam优化器
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
# 为细节特征融合层设置Adam优化器
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

# 为各个优化器设置学习率调度器，使用StepLR策略
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

# 定义损失函数
MSELoss = nn.MSELoss()  # 均方误差损失
L1Loss = nn.L1Loss()    # L1损失
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')  # 结构相似性损失


# 数据加载器设置
# 创建训练数据加载器，使用H5Dataset加载数据
trainloader = DataLoader(H5Dataset(r"data/MSRS_train/train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,  # 随机打乱数据
                         num_workers=0)  # 不使用多进程加载数据

# 将数据加载器放入字典中，便于后续扩展
loader = {'train': trainloader, }
# 生成时间戳，用于保存模型
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
训练过程
------------------------------------------------------------------------------
'''

# 初始化训练步数计数器
step = 0
# 启用cudnn的自动优化，提高运行速度
torch.backends.cudnn.benchmark = True
# 记录开始时间，用于计算剩余时间
prev_time = time.time()

# 用于记录每个epoch的平均损失
epoch_losses = []

# 开始训练循环
for epoch in range(num_epochs):
    ''' 训练阶段 '''
    # 记录当前epoch的所有batch的loss
    batch_losses = []
    
    # 遍历训练数据集
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        # 将数据移至GPU
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        # 设置模型为训练模式
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        # 清除模型梯度
        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        # 清除优化器梯度
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap: # 第一阶段训练（Phase I）
            # 提取可见光图像的特征
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            # 提取红外图像的特征
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            # 重建可见光图像
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            # 重建红外图像
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            # 计算基础特征和细节特征的相关系数损失
            cc_loss_B = cc(feature_V_B, feature_I_B)  # 基础特征相关系数
            cc_loss_D = cc(feature_V_D, feature_I_D)  # 细节特征相关系数
            # 计算可见光图像重建损失（SSIM损失和MSE损失的组合）
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            # 计算红外图像重建损失（SSIM损失和MSE损失的组合）
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            # 计算梯度损失，使用空间梯度滤波器
            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            # 计算分解损失，鼓励细节特征相关性高而基础特征相关性低
            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            # 计算总损失，各部分损失按系数加权
            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            # 反向传播计算梯度
            loss.backward()
            # 对编码器参数进行梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            # 对解码器参数进行梯度裁剪
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            # 更新编码器参数
            optimizer1.step()  
            # 更新解码器参数
            optimizer2.step()
        else:  # 第二阶段训练（Phase II）
            # 提取可见光和红外图像的特征
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
            # 融合基础特征
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            # 融合细节特征
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)

            # 使用融合特征生成融合图像
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)  

            # 计算可见光图像与融合图像的损失
            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            # 计算红外图像与融合图像的损失
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

            # 计算基础特征和细节特征的相关系数
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            # 计算分解损失
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            # 计算融合损失
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            # 计算总损失
            loss = fusionloss + coeff_decomp * loss_decomp
            # 反向传播
            loss.backward()
            # 对各个网络模块进行梯度裁剪
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            # 更新各个网络模块的参数
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            
        # 记录当前batch的loss
        batch_losses.append(loss.item())
        # 记录到TensorBoard
        writer.add_scalar('Loss/train/batch_loss', loss.item(), step)
        step += 1

        # 计算剩余训练时间
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        # 打印训练进度信息
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )
    
    # 计算并记录当前epoch的平均loss
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_loss)
    # 记录到TensorBoard
    writer.add_scalar('Loss/train/epoch_loss', epoch_loss, epoch)
    
    # 打印当前epoch的平均loss
    print(f"\nEpoch {epoch}/{num_epochs} Average Loss: {epoch_loss:.6f}")

    # 调整学习率
    # 更新编码器和解码器的学习率
    scheduler1.step()  
    scheduler2.step()
    # 在第二阶段，更新融合层的学习率
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    # 确保学习率不低于设定的最小值
    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6

# 将loss历史保存到文件
import numpy as np
np.save(f'models/loss_history_{timestamp}.npy', np.array(epoch_losses))

# 关闭TensorBoard写入器
writer.close()
    
# 训练完成后保存模型
if True:
    # 创建包含所有网络模块参数的检查点字典
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
        'loss_history': epoch_losses,  # 同时保存loss历史到模型文件
    }
    # 保存模型到文件，文件名包含时间戳
    torch.save(checkpoint, os.path.join("models/CDDFuse_"+timestamp+'.pth'))