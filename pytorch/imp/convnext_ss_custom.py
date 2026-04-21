import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class spatial_ConvBlock(nn.Module):

    def __init__(self, dim, kernel_size, drop_rate=0.5):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding='same', groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class spectral_ConvBlock(nn.Module):

    def __init__(self, dim, drop_rate=0.5):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x

class SS_ConvNeXt_V2(nn.Module):
    def __init__(self, num_classes: int = 4, num_channels = 204, 
                depths: list = [1, 1, 2, 1],  dims: list = [64, 128, 256, 512], 
                type_mix = 'spatial_first', spa_kernel_size = (3,11),
                drop_path_rate: float = 0.1, head_init_scale: float = 1.):
        if type_mix not in ['mixed', 'spatial_first', 'spectral_first']:
            raise RuntimeError("Тип смешивания слоёв должен быть один из 'mixed', 'spatial_first', 'spectral_first'")
        if len(depths) != len(dims):
            raise RuntimeError("Несоответствие depths и dims")
        if len(depths) <1:
            raise RuntimeError("Минимум 1 слой")

        super(SS_ConvNeXt_V2, self).__init__()

        self.layer_count = len(depths)

        spm_layer = nn.Sequential(nn.Conv2d(num_channels, dims[0], kernel_size=1, stride=1, padding=0),
                        LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                        nn.GELU())
        
        self.downsample_layers = nn.ModuleList()

        self.downsample_layers.append(spm_layer)

        for i in range(1, len(dims)):
            down_layer = nn.Sequential(LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2, padding=0))
            self.downsample_layers.append(down_layer)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)] #sum(sum(depths, [])))]

        self.main_layers = nn.ModuleList()

        depth_counter = 0

        for i in range(len(depths)):
            module_list = []
            if (type_mix == 'mixed'):
                for _ in range(depths[i]):
                    module_list.append(spatial_ConvBlock(dim=dims[i], kernel_size=spa_kernel_size, drop_rate=dp_rates[depth_counter]))
                    depth_counter += 1
                    module_list.append(spectral_ConvBlock(dim=dims[i], drop_rate=dp_rates[depth_counter]))
                    depth_counter += 1
            elif (type_mix == 'spatial_first'):
                for _ in range(depths[i]):
                    module_list.append(spatial_ConvBlock(dim=dims[i], kernel_size=spa_kernel_size, drop_rate=dp_rates[depth_counter]))
                    depth_counter += 1
                for _ in range(depths[i]):
                    module_list.append(spectral_ConvBlock(dim=dims[i], drop_rate=dp_rates[depth_counter]))
                    depth_counter += 1
            elif (type_mix == 'spectral_first'):
                for _ in range(depths[i]):
                    module_list.append(spectral_ConvBlock(dim=dims[i], drop_rate=dp_rates[depth_counter]))
                    depth_counter += 1
                for _ in range(depths[i]):
                    module_list.append(spatial_ConvBlock(dim=dims[i], kernel_size=spa_kernel_size, drop_rate=dp_rates[depth_counter]))
                    depth_counter += 1
            
            spatial_spectral = nn.Sequential(*module_list)

            self.main_layers.append(spatial_spectral)

        self.ln = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.fc = nn.Linear(dims[-1], num_classes)
        self.activate = nn.GELU()   # GELU

        self.apply(self.initialize_weights)
        self.fc.weight.data.mul_(head_init_scale)
        self.fc.bias.data.mul_(head_init_scale)

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # elif isinstance(module, LayerNorm):
        #     nn.init.constant_(module.bias, 0)
        #     nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        for i in range(self.layer_count):
            x = self.downsample_layers[i](x)
            x = self.main_layers[i](x)

        x = self.activate(self.ln(x))
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x