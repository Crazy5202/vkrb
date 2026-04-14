import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

# from pytorch.imp.convnext import convnextv2_atto
from pytorch.imp.convnext_ss import SS_ConvNeXt as SS_ConvNeXt_HSI



# class SS_ConvNeXt_HSI(BaseModel):
#     def __init__(self, num_classes=4, depths = [2, 2, 6, 2], dims = [32, 64, 128, 256], 
#             num_channels = 204, drop_path_rate = 0.5, 
#             layer_scale_init_value = 1e-6, head_init_scale = 1.):
#         """
#         :param num_classes: число классов (как у HybridSN_improved)
#         :param in_chans: число спектральных каналов (по умолчанию 204)
#         :param depths, dims: конфигурация ConvNeXtV2 (как в оригинале)
#         :param drop_path_rate: stochastic depth rate
#         """
#         super().__init__()
#         self.backbone = SS_ConvNeXt(
#             num_classes=num_classes,
#             depths = depths, dims = dims, 
#             num_channels = num_channels, drop_path_rate = drop_path_rate, 
#             layer_scale_init_value = layer_scale_init_value, 
#             head_init_scale = head_init_scale
#         )

#     def forward(self, x):
#         """
#         :param x: torch.Tensor формы (N, H, W, C) — как у HybridSN_improved
#         :return: логиты (N, num_classes)
#         """
#         logits = self.backbone(x)
#         return logits

# class ConvNextV2_Atto_HSI(BaseModel):
#     def __init__(self, num_classes=2, in_chans=204):
#         """
#         :param num_classes: число классов (как у HybridSN_improved)
#         :param in_chans: число спектральных каналов (по умолчанию 204)
#         :param depths, dims: конфигурация ConvNeXtV2 (как в оригинале)
#         :param drop_path_rate: stochastic depth rate
#         """
#         super().__init__()
#         self.backbone = convnextv2_atto(
#             in_chans=in_chans,
#             num_classes=num_classes,
#         )

#     def forward(self, x):
#         """
#         :param x: torch.Tensor формы (N, H, W, C) — как у HybridSN_improved
#         :return: логиты (N, num_classes)
#         """
#         logits = self.backbone(x)
#         return logits


# class HybridSN_improved(BaseModel):
#     def __init__(self, num_classes = 2, sp_channels = 204, dropout_rate=0.4):
#         super().__init__()

#         self.cs = nn.Conv2d(in_channels=sp_channels, out_channels=30, 
#                             kernel_size=(1,1), bias = False)
#         self.bn2d_cs = nn.BatchNorm2d(30)
        
#         self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=8, 
#                                 kernel_size=(8, 3, 3), stride=(4, 2, 2), bias = False)
#         self.bn3d_1 = nn.BatchNorm3d(8)
        
#         self.conv3d_2 = nn.Conv3d(in_channels=8, out_channels=16, 
#                                 kernel_size=(5, 3, 3), stride=(2, 2, 2), bias = False)
#         self.bn3d_2 = nn.BatchNorm3d(16)
        
#         self.conv2d_1 = nn.Conv2d(in_channels=32, out_channels=64, 
#                                   kernel_size=3, stride=2, bias = False)
#         self.bn2d_1 = nn.BatchNorm2d(64)
        
#         self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, 
#                                   kernel_size=3, stride=2, bias = False)
#         self.bn2d_2 = nn.BatchNorm2d(128)

#         self.fc_1 = nn.Linear(128, 256)
#         self.bn1d_1 = nn.BatchNorm2d(256)

#         self.dropout_1 = nn.Dropout(p=dropout_rate)

#         self.fc_2 = nn.Linear(256, num_classes)

#     def forward(self, x):

#         x = F.relu(self.bn2d_cs(self.cs(x)))

#         x = x.unsqueeze(1) 
        
#         x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        
#         x = F.relu(self.bn3d_2(self.conv3d_2(x)))

#         # x = x.squeeze(2) 

#         # #x = x.mean(dim=2) 
        
#         # x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        
#         # x = F.relu(self.bn2d_2(self.conv2d_2(x)))
        
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
        
#         # x = self.dropout(x)
#         # x = self.fc(x)
        
#         return x
    
    # class HybridSN(BaseModel):
#     def __init__(self, num_classes, dropout_rate=0.5):
#         super().__init__()
        
#         self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=8, 
#                                 kernel_size=(8, 3, 3), stride=(4, 2, 2))
#         self.bn3d_1 = nn.BatchNorm3d(8)
        
#         self.conv3d_2 = nn.Conv3d(in_channels=8, out_channels=16, 
#                                 kernel_size=(5, 3, 3), stride=(2, 2, 2))
#         self.bn3d_2 = nn.BatchNorm3d(16)
        
#         #self.spectral_reduce = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=1)
#         self.spectral_squeeze = nn.Conv3d(in_channels=16, out_channels=32, 
#                                           kernel_size=(23, 1, 1))
        
#         self.conv2d_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
#         self.bn2d_1 = nn.BatchNorm2d(64)
        
#         self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
#         self.bn2d_2 = nn.BatchNorm2d(128)

#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):

#         x = x.unsqueeze(1) 
        
#         x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        
#         x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        
#         x = self.spectral_squeeze(x)

#         x = x.squeeze(2) 

#         #x = x.mean(dim=2) 
        
#         x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        
#         x = F.relu(self.bn2d_2(self.conv2d_2(x)))
        
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
        
#         x = self.dropout(x)
#         x = self.fc(x)
        
#         return x

# class HybridSN_TT(BaseModel):
#     def __init__(self, num_classes, dropout_rate=0.5, tt_rank=0.15):
#         """
#         Args:
#             num_classes (int): Количество классов для классификации.
#             dropout_rate (float): Вероятность Dropout для регуляризации.
#             tt_rank (float/int/'same'): Rank of the TT decomposition. A float < 1.0 represents 
#                                         the fraction of parameters to keep (e.g., 0.5 = 50% parameters).
#         """
#         super().__init__()
        
#         self.conv3d_1 = tltorch.FactorizedConv(in_channels=1, out_channels=8, 
#                                                kernel_size=(8, 3, 3), order=3, stride=(4, 2, 2),
#                                                factorization='tt', rank=tt_rank)
#         self.bn3d_1 = nn.BatchNorm3d(8)

#         self.conv3d_2 = tltorch.FactorizedConv(in_channels=8, out_channels=16, 
#                                                kernel_size=(5, 3, 3), order=3, stride=(2, 2, 2),
#                                                factorization='tt', rank=tt_rank)
#         self.bn3d_2 = nn.BatchNorm3d(16)
        
#         #self.spectral_reduce = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=1)

#         self.spectral_squeeze = tltorch.FactorizedConv(in_channels=16, out_channels=32, 
#                                         kernel_size=(23, 1, 1), order=3,
#                                         factorization='tt', rank=tt_rank)
        
#         self.conv2d_1 = tltorch.FactorizedConv(in_channels=32, out_channels=64, 
#                                                kernel_size=3, order=2, stride=2,
#                                                factorization='tt', rank=tt_rank)
#         self.bn2d_1 = nn.BatchNorm2d(64)
        
#         self.conv2d_2 = tltorch.FactorizedConv(in_channels=64, out_channels=128, 
#                                                kernel_size=3, order=2, stride=2,
#                                                factorization='tt', rank=tt_rank)
#         self.bn2d_2 = nn.BatchNorm2d(128)

#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = x.unsqueeze(1) 
        
#         x = F.relu(self.bn3d_1(self.conv3d_1(x)))
#         x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        
#         #x = self.spectral_reduce(x) 
#         #x = x.mean(dim=2) 

#         x = self.spectral_squeeze(x)
#         x = x.squeeze(2) 
        
#         x = F.relu(self.bn2d_1(self.conv2d_1(x)))
#         x = F.relu(self.bn2d_2(self.conv2d_2(x)))
        
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
        
#         x = self.dropout(x)
#         x = self.fc(x)
        
#         return x