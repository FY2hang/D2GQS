"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.   
---------------------------------------------------------------------------------   
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.  
"""
  
import copy   
from collections import OrderedDict
     
import torch
import torch.nn as nn   
import torch.nn.functional as F   
     
from .utils import get_activation   

from ..core import register  
   
__all__ = ['HybridEncoder_DRE']  

class ConvNormLayer_fuse(nn.Module):    
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()  
        padding = (kernel_size-1)//2 if padding is None else padding     
        self.conv = nn.Conv2d( 
            ch_in,     
            ch_out,    
            kernel_size,    
            stride,    
            groups=g,     
            padding=padding,    
            bias=bias)    
        self.norm = nn.BatchNorm2d(ch_out) 
        self.act = nn.Identity() if act is None else get_activation(act)    
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias
     
    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)   
        else:
            y = self.norm(self.conv(x))
        return self.act(y)  
  
    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(  
                self.ch_in,     
                self.ch_out,   
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)   
   
        kernel, bias = self.get_equivalent_kernel_bias()  
        self.conv_bn_fused.weight.data = kernel   
        self.conv_bn_fused.bias.data = bias   
        self.__delattr__('conv')
        self.__delattr__('norm') 
  
    def get_equivalent_kernel_bias(self): 
        kernel3x3, bias3x3 = self._fuse_bn_tensor() 
 
        return kernel3x3, bias3x3    

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var 
        gamma = self.norm.weight     
        beta = self.norm.bias    
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1) 
        return kernel * t, beta - running_mean * gamma / std
    

class ConvNormLayer(nn.Module): 
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):    
        super().__init__() 
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,   
            groups=g,
            padding=padding,
            bias=bias)     
        self.norm = nn.BatchNorm2d(ch_out)   
        self.act = nn.Identity() if act is None else get_activation(act)    
     
    def forward(self, x): 
        return self.act(self.norm(self.conv(x))) 
  
   
# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)    
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):   
    def __init__(self, c1, c2, k, s, act=None): 
        super().__init__()    
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1) 
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):     
        return self.cv2(self.cv1(x))  


class VGGBlock(nn.Module):    
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()  
        # 初始化输入和输出通道数
        self.ch_in = ch_in
        self.ch_out = ch_out   
        
        # 定义两个卷积层：conv1 是3x3卷积，conv2 是1x1卷积
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)  
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)     
        
        # 激活函数的选择，默认为'ReLU'，如果传入None则为Identity（即没有激活）     
        self.act = nn.Identity() if act is None else get_activation(act)  
 
    def forward(self, x):     
        # 如果有`conv`属性，则直接使用它，否则使用两个卷积层的和（残差连接） 
        if hasattr(self, 'conv'):    
            y = self.conv(x)
        else: 
            y = self.conv1(x) + self.conv2(x)
 
        # 返回激活后的结果 
        return self.act(y)   

    def convert_to_deploy(self): 
        # 将模块转换为推理时使用的部署模型
        if not hasattr(self, 'conv'):   
            # 如果没有 `conv` 属性，说明我们需要融合卷积 
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)  
 
        # 获取融合后的卷积核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()
     
        # 将卷积核和偏置赋值给 `self.conv`
        self.conv.weight.data = kernel
        self.conv.bias.data = bias    
  
        # 删除不再需要的卷积层 `conv1` 和 `conv2`
        self.__delattr__('conv1')
        self.__delattr__('conv2')
   
    def get_equivalent_kernel_bias(self):  
        # 获取两个卷积层融合后的卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)  
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)  
   
        # 将1x1卷积的kernel pad到3x3大小，并返回融合后的kernel和bias
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1  
  
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):     
        # 如果1x1卷积的kernel为空，则返回0
        if kernel1x1 is None:
            return 0  
        else:     
            # 否则将1x1卷积的kernel pad到3x3
            return F.pad(kernel1x1, [1, 1, 1, 1])  
 
    def _fuse_bn_tensor(self, branch: ConvNormLayer): 
        # 如果卷积层为空，则返回0
        if branch is None:
            return 0, 0    
        
        # 获取卷积层的权重、BN层的均值、方差、权重、偏置等
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var    
        gamma = branch.norm.weight
        beta = branch.norm.bias  
        eps = branch.norm.eps  
 
        # 计算标准差并进行归一化   
        std = (running_var + eps).sqrt() 
        t = (gamma / std).reshape(-1, 1, 1, 1)
 
        # 返回归一化后的卷积核和偏置
        return kernel * t, beta - running_mean * gamma / std
     
     
class CSPLayer(nn.Module):   
    def __init__(self,
                 in_channels,  
                 out_channels,    
                 num_blocks=3,   
                 expansion=1.0,    
                 bias=False,    
                 act="silu",    
                 bottletype=VGGBlock):     
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion) 
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)     
        ])  
        if hidden_channels != out_channels:    
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)   
        else:
            self.conv3 = nn.Identity()
  
    def forward(self, x):  
        x_2 = self.conv2(x)  
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)     
        return self.conv3(x_1 + x_2)
     
class RepNCSPELAN4(nn.Module):
    # csp-elan     
    def __init__(self, c1, c2, c3, c4, n=3,   
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))  
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)) 
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)
 
    def forward_chunk(self, x):     
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))     
    
    def forward(self, x): 
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])  
        return self.cv4(torch.cat(y, 1))   
  
   
# transformer   
class TransformerEncoderLayer(nn.Module): 
    def __init__(self,     
                 d_model,    
                 nhead, 
                 dim_feedforward=2048,
                 dropout=0.1,    
                 activation="relu",
                 normalize_before=False):  
        super().__init__()   
        self.normalize_before = normalize_before
 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)   

        self.linear1 = nn.Linear(d_model, dim_feedforward)  
        self.dropout = nn.Dropout(dropout)    
        self.linear2 = nn.Linear(dim_feedforward, d_model)  

        self.norm1 = nn.LayerNorm(d_model)     
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)
  
    @staticmethod
    def with_pos_embed(tensor, pos_embed): 
        return tensor if pos_embed is None else tensor + pos_embed   

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:     
        residual = src
        if self.normalize_before: 
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)     
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)     

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)     

        residual = src    
        if self.normalize_before:   
            src = self.norm2(src) 
        src = self.linear2(self.dropout(self.activation(self.linear1(src)))) 
        src = residual + self.dropout2(src)
        if not self.normalize_before:   
            src = self.norm2(src)
        return src

class TransformerEncoderBlock(nn.Module):  
    def __init__(self,  
                 d_model,
                 nhead,
                 dim_feedforward=2048,    
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 normalize_before=False):  
        super().__init__()
        self.normalize_before = normalize_before    
        self.pe_temperature = pe_temperature     

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)   
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
   
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)  
        self.dropout2 = nn.Dropout(dropout)    
        self.activation = get_activation(activation)
     
    @staticmethod  
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def build_2d_sincos_position_embedding(self, w, h, embed_dim=256, temperature=10000.):
        """
        生成 2D sine-cosine 位置编码
        Args:
            w (int): 特征图宽度
            h (int): 特征图高度
            embed_dim (int): 嵌入维度，必须能被 4 整除
            temperature (float): 温度参数，控制频率    
        Returns:   
            torch.Tensor: 位置编码张量，形状为 [1, w*h, embed_dim]
        """
        # 创建宽度和高度的网格   
        grid_w = torch.arange(int(w), dtype=torch.float32)  
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')  # 生成 2D 网格
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4  # 每个方向 (w, h) 的编码维度    
        # 计算频率因子
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)  

        # 计算宽度和高度的 sin 和 cos 编码 
        out_w = grid_w.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]    
        out_h = grid_h.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]
  
        # 拼接 sin 和 cos 编码，形成最终的位置编码
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]  

    def forward(self, src, src_mask=None) -> torch.Tensor:     
        b, c, h, w = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c, self.pe_temperature).to(src.device)  
    
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)  
        if not self.normalize_before:
            src = self.norm1(src) 
    
        residual = src   
        if self.normalize_before:     
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))   
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)     

        return src.permute(0, 2, 1).reshape(-1, c, h, w).contiguous()

class DynamicRegionEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, out_channels=256, dim_feedforward=1024, dropout=0.0, activation='gelu', window_size=10, stride=5):
        super().__init__()
        self.d_model = d_model
        self.out_channels = out_channels
        self.window_size = window_size
        self.stride = stride
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 模仿 DART 的局部坐标投影
        self.region_bias_proj = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, out_channels) if d_model != out_channels else nn.Identity()

    def forward(self, x, density_map=None, adaptive_mu=None):
        B, C, H, W = x.shape  # C 此时可能是 512 (如果是 Stride 4 层)
        ws, st = self.window_size, self.stride

        # 1. 物理切分 (生成所有可能的 Patch)
        x_unfold = F.unfold(x, kernel_size=ws, stride=st) 
        num_patches = x_unfold.shape[-1]
        
        # 形状变换: [B, num_patches, ws*ws, C]
        x_all_patches = x_unfold.view(B, C, ws * ws, num_patches).permute(0, 3, 2, 1)

        # 2. 物理筛选逻辑 (Dome-DETR 核心)
        if density_map is not None and adaptive_mu is not None:
            density_unfold = F.unfold(density_map, kernel_size=ws, stride=st) 
            patch_density = density_unfold.max(dim=1)[0] 
            active_mask = patch_density > adaptive_mu.view(B, 1)
        else:
            active_mask = torch.ones(B, num_patches, dtype=torch.bool, device=x.device)

        # 3. 物理抽离 (Token Indexing)
        batch_idx, patch_idx = active_mask.nonzero(as_tuple=True)
        
        # --- 🚀 专门针对 get_info.py 的测试逻辑 (正式训练请注释掉) ---
        # if not self.training:
        #     batch_idx, patch_idx = batch_idx[:len(batch_idx)//10], patch_idx[:len(patch_idx)//10]
        batch_idx, patch_idx = batch_idx[:len(batch_idx)//10], patch_idx[:len(patch_idx)//10]

        K = batch_idx.shape[0] 
        if K == 0: return x 

        x_active = x_all_patches[batch_idx, patch_idx] 

        # 4. 核心计算 (Attention + FFN)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, ws), torch.linspace(-1, 1, ws), indexing='ij')
        local_coords = torch.stack([grid_x, grid_y], dim=-1).to(x.device).view(ws * ws, 2)
        pos_embed = self.region_bias_proj(local_coords) 

        x_active = self.norm1(x_active + self.self_attn(x_active + pos_embed, x_active + pos_embed, x_active)[0])
        x_active = self.norm2(x_active + self.linear2(self.act(self.linear1(x_active))))

        # =========================================================
        # 🌟 关键修改 1：通道投影对齐 (512 -> 256)
        # =========================================================
        # 将计算完的特征通过 output_proj，确保输出通道是 self.out_channels (256)
        x_active = self.output_proj(x_active)  # 现在是 [K, L, 256]

        # 3. 🌟 修正回填容器 (不要用 zeros_like，因为维度变了)
        B, _, H, W = x.shape
        ws, st = self.window_size, self.stride
        # 最后一个维度必须是 self.out_channels (256)
        out_all_patches = torch.zeros(B, num_patches, ws * ws, self.out_channels, 
                                 device=x.device, dtype=x.dtype)
        out_all_patches[batch_idx, patch_idx] = x_active
        
        # 5. Fold 回 2D
        # 注意：C 通道现在是 self.out_channels (256)
        out_unfold = out_all_patches.permute(0, 3, 2, 1).reshape(B, self.out_channels * ws * ws, num_patches)
        x_recovered = F.fold(out_unfold, output_size=(H, W), kernel_size=ws, stride=st)

        # 5. 处理重叠均值
        ones_unfold = torch.ones(B, 1 * ws * ws, num_patches, device=x.device, dtype=x.dtype)
        divisor = F.fold(ones_unfold, output_size=(H, W), kernel_size=ws, stride=st)
        return x_recovered / (divisor + 1e-5)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):    
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
   
    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:  
        output = src    
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:   
            output = self.norm(output)

        return output


@register()   
class HybridEncoder_DRE(nn.Module): 
    # 定义共享属性，'eval_spatial_size' 可在模型实例间共享  
    __share__ = ['eval_spatial_size', ]
    
    def __init__(self,    
                 in_channels=[512, 1024, 2048],        # 输入特征图的通道数列表，例如来自骨干网络的不同层    
                 feat_strides=[8, 16, 32],             # 输入特征图的步幅列表，表示特征图相对于输入图像的缩放比例
                 hidden_dim=256,                       # 隐藏层维度，所有特征图将被投影到这个维度  
                 nhead=8,                              # Transformer 编码器中多头自注意力的头数 
                 dim_feedforward=1024,                 # Transformer 编码器中前馈网络的维度 
                 dropout=0.0,                          # Transformer 编码器中的 dropout 概率   
                 enc_act='gelu',                       # Transformer 编码器中的激活函数类型  
                 use_encoder_idx=[2],                  # 指定哪些层使用 Transformer 编码器（索引列表）   
                 num_encoder_layers=1,                 # Transformer 编码器的层数
                 pe_temperature=10000,                 # 位置编码的温度参数，用于控制频率
                 expansion=1.0,                        # FPN 和 PAN 中特征扩展因子
                 depth_mult=1.0,                       # 深度乘数，用于调整网络深度    
                 act='silu',                           # FPN 和 PAN 中使用的激活函数类型
                 eval_spatial_size=None,               # 评估时的空间尺寸 (H, W)，用于预计算位置编码
                 version='dfine',                      # 模型版本，决定使用哪些具体模块（如 'dfine' 或其他）
                 ):
        # 调用父类 nn.Module 的构造函数
        super().__init__()   
    
        # 保存传入的参数为类的成员变量
        self.in_channels = in_channels              # 输入通道数列表
        self.feat_strides = feat_strides            # 输入特征步幅列表   
        self.hidden_dim = hidden_dim                # 隐藏层维度
        self.use_encoder_idx = use_encoder_idx      # 使用 Transformer 编码器的层索引
        self.num_encoder_layers = num_encoder_layers # Transformer 编码器层数
        self.pe_temperature = pe_temperature        # 位置编码温度参数
        self.eval_spatial_size = eval_spatial_size  # 评估时的空间尺寸    
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]  # 输出通道数，统一为 hidden_dim
        self.out_strides = feat_strides             # 输出步幅，与输入相同
        

        # self.output_proj = nn.Linear(d_model, out_channels) if d_model != out_channels else nn.Identity()
        # 输入投影层：将不同通道数的输入特征图投影到统一的 hidden_dim  
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            # 每个投影层包含 1x1 卷积和批量归一化
            proj = nn.Sequential(OrderedDict([     
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),  # 1x1 卷积变换通道数
                ('norm', nn.BatchNorm2d(hidden_dim))                                    # 批量归一化     
            ]))   
            self.input_proj.append(proj)
   
        # Transformer 编码器：对指定层进行特征增强     
        # ===== 替换为我们的 ADRE =====
        self.encoder = nn.ModuleList([
            DynamicRegionEncoder(
                d_model=hidden_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation=enc_act,
                window_size=10,  # Dome-DETR 推荐的窗口大小
                stride=5         # 步长为5产生 50% 的重叠，防止目标被割裂
            )
            for _ in range(len(use_encoder_idx)) 
        ])

        # FPN（特征金字塔网络）：自顶向下融合高层特征到低层特征
        self.lateral_convs = nn.ModuleList()  # 横向连接卷积
        self.fpn_blocks = nn.ModuleList()     # FPN 融合块
        for i, idx in enumerate(range(len(in_channels) - 1, 0, -1)):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            stride = self.feat_strides[idx - 1]
            
            if stride == 4:
                self.fpn_blocks.append(
                    DynamicRegionEncoder(
                        d_model=hidden_dim * 2, # 输入是 512
                        # 🌟 新增参数：告诉 DRE 内部最后做一个线性映射变回 256
                        # 如果你的 DRE 类还没写这个参数，看下文的类修改建议
                        out_channels=hidden_dim, 
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        window_size=10,
                        stride=5
                    )
                )
            else:
                self.fpn_blocks.append(
                    # 🌟 确保第二个参数是 hidden_dim (256)
                    RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, 
                                 round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act)
                )
 
        # PAN（路径聚合网络）：自底向上融合低层特征到高层特征
        self.downsample_convs = nn.ModuleList()  # 下采样卷积   
        self.pan_blocks = nn.ModuleList()        # PAN 融合块   
        for i in range(len(in_channels) - 1):
            # 🌟 关键修正：
            # 根据调试信息，i=0 时传入的 P2 已经是 256 通道了。
            # 所以这里不需要判断，直接全部使用 hidden_dim
            input_c = hidden_dim  # <--- 修正为 256
            
            self.downsample_convs.append(
                nn.Sequential(SCDown(input_c, hidden_dim, 3, 2))
            )
            
            # 🌟 核心修正：确保所有 Pan Block 输出都是 hidden_dim (256)
            stride = self.feat_strides[i + 1]
            if stride == 4:
                self.pan_blocks.append(
                    DynamicRegionEncoder(
                        d_model=hidden_dim * 2,    # 输入 512
                        out_channels=hidden_dim,   # 🌟 必须显式让它输出 256
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        window_size=10,
                        stride=5
                    )
                )
            else:
                # 确保第二个参数是 hidden_dim
                self.pan_blocks.append(
                    RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, 
                                 round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act)
                )
     
        # 初始化参数，包括预计算位置编码   
        self._reset_parameters()
     
    def _reset_parameters(self):    
        # 如果指定了评估时的空间尺寸，则预计算位置编码
        if self.eval_spatial_size:  
            for idx in self.use_encoder_idx:    
                stride = self.feat_strides[idx]  # 当前层的步幅  
                # 根据特征图尺寸和步幅计算位置编码
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,  # 宽度
                    self.eval_spatial_size[0] // stride,  # 高度     
                    self.hidden_dim,                      # 嵌入维度
                    self.pe_temperature                   # 温度参数 
                )
                # 将位置编码存储为类的属性
                setattr(self, f'pos_embed{idx}', pos_embed)    
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod  
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):     
        """  
        生成 2D sine-cosine 位置编码     
        Args:  
            w (int): 特征图宽度
            h (int): 特征图高度
            embed_dim (int): 嵌入维度，必须能被 4 整除    
            temperature (float): 温度参数，控制频率   
        Returns:
            torch.Tensor: 位置编码张量，形状为 [1, w*h, embed_dim] 
        """    
        # 创建宽度和高度的网格    
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')  # 生成 2D 网格
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'     
        pos_dim = embed_dim // 4  # 每个方向 (w, h) 的编码维度    
        # 计算频率因子
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)  
  
        # 计算宽度和高度的 sin 和 cos 编码 
        out_w = grid_w.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]
        out_h = grid_h.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]     
  
        # 拼接 sin 和 cos 编码，形成最终的位置编码
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, density_map=None, adaptive_mu=None):     
        """
        前向传播函数
        Args:
            feats (list[torch.Tensor]): 输入特征图列表，形状为 [B, C, H, W]，长度需与 in_channels 一致
        Returns:     
            list[torch.Tensor]: 融合后的多尺度特征图列表 
        """
        # 检查输入特征图数量是否与预期一致
        assert len(feats) == len(self.in_channels)

        # 1. 输入投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 🌟 关键准备：提前生成每一层对应的物理掩码 (借鉴 Dome-DETR)
        masks = []
        if density_map is not None and adaptive_mu is not None:
            for feat in proj_feats:
                h, w = feat.shape[2:]
                # 对密度图下采样，生成当前尺度的掩码
                m = F.interpolate(density_map, size=(h, w), mode='bilinear') > adaptive_mu.view(-1, 1, 1, 1)
                masks.append(m)
        else:
            masks = [None] * len(proj_feats)

        # 2. Transformer 编码器 (你已经改好的物理裁剪版)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                current_density = None
                if density_map is not None:
                    current_density = F.interpolate(density_map, size=(h, w), mode='bilinear')
                
                # 这里内部已经是物理裁剪了
                proj_feats[enc_ind] = self.encoder[i](proj_feats[enc_ind], current_density, adaptive_mu)

        # 3. FPN 融合 (🌟 重点改造：让卷积块变动态)
        inner_outs = [proj_feats[-1]]
        for i, idx in enumerate(range(len(self.in_channels) - 1, 0, -1)):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            h, w = feat_low.shape[2:] # 获取目标尺寸
            
            feat_heigh = self.lateral_convs[i](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, size=(h, w), mode='bilinear')
            
            concat_feat = torch.concat([upsample_feat, feat_low], dim=1)
            
            # 🌟 关键：判断当前 Block 类型并执行物理裁剪
            if isinstance(self.fpn_blocks[i], DynamicRegionEncoder):
                # 如果是 Stride 4 层的动态块，送入密度图，实现真正的 Token 物理剔除
                current_density = F.interpolate(density_map, size=(h, w), mode='bilinear') if density_map is not None else None
                inner_out = self.fpn_blocks[i](concat_feat, current_density, adaptive_mu)
            else:
                inner_out = self.fpn_blocks[i](concat_feat)
            
            inner_outs.insert(0, inner_out)

        # 4. PAN 融合
        outs = [inner_outs[0]]
        for i in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[i + 1]
            h, w = feat_height.shape[2:]
            print(f"DEBUG: i={i}, feat_low shape={feat_low.shape}, weight_in_channel={self.downsample_convs[i][0].cv1.conv_bn_fused.weight.shape[1]}")
            downsample_feat = self.downsample_convs[i](feat_low)
            concat_feat = torch.concat([downsample_feat, feat_height], dim=1)
            
            # 🌟 同样逻辑：替换掉 PAN 里最重的那几层
            if isinstance(self.pan_blocks[i], DynamicRegionEncoder):
                current_density = F.interpolate(density_map, size=(h, w), mode='bilinear') if density_map is not None else None
                out = self.pan_blocks[i](concat_feat, current_density, adaptive_mu)
            else:
                out = self.pan_blocks[i](concat_feat)
                
            outs.append(out)

        return outs

    def convert_to_deploy(model): 
        """
        融合 input_proj 中的 Conv+BN 层
        """
        for i, proj in enumerate(model.input_proj):
            try:
                conv = proj.conv
                bn = proj.norm
            except:    
                continue
     
            # 计算融合参数
            inv_std = torch.rsqrt(bn.running_var + bn.eps)
            fused_weight = conv.weight * (bn.weight * inv_std).reshape(-1, 1, 1, 1)
            fused_bias = (conv.bias - bn.running_mean) * bn.weight * inv_std + bn.bias if conv.bias is not None else (-bn.running_mean) * bn.weight * inv_std + bn.bias 
            
            # 创建融合后的卷积层 
            fused_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, 
                conv.stride, conv.padding, conv.dilation, conv.groups, True
            )
            fused_conv.weight.data = fused_weight     
            fused_conv.bias.data = fused_bias    
            
            # 替换投影层
            model.input_proj[i] = nn.Sequential(OrderedDict([('conv', fused_conv)])) 
   
@register()   
class SimpleEncoder_DRE(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],        # 输入特征图的通道数列表，例如来自骨干网络的不同层
                 feat_strides=[8, 16, 32],             # 输入特征图的步幅列表，表示特征图相对于输入图像的缩放比例
                 ):  
        # 调用父类 nn.Module 的构造函数   
        super().__init__()   
    
        # 保存传入的参数为类的成员变量 
        self.in_channels = in_channels              # 输入通道数列表 
        self.feat_strides = feat_strides            # 输入特征步幅列表
        self.out_channels = in_channels  # 输出通道数，统一为 hidden_dim    
        self.out_strides = feat_strides             # 输出步幅，与输入相同 
    
    def forward(self, feats):
        """
        前向传播函数   
        Args:     
            feats (list[torch.Tensor]): 输入特征图列表，形状为 [B, C, H, W]，长度需与 in_channels 一致
        Returns:    
            list[torch.Tensor]: 融合后的多尺度特征图列表     
        """ 
        # 检查输入特征图数量是否与预期一致
        assert len(feats) == len(self.in_channels)

        return feats