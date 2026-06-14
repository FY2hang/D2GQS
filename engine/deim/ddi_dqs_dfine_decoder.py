"""
DFINE with Density-aware Query Selection
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob, visualize_density_map_only
from ..core import register

from .hybrid_encoder import ConvNormLayer_fuse
from .dfine_decoder import Integral, MLP, TransformerDecoder, TransformerDecoderLayer
# from .dfine_decoder import DynamicManifoldIntegral, Integral, MLP, TransformerDecoder, TransformerDecoderLayer
from .dq_dfine_decoder import MultiScaleFeature, CGFE

from ..logger_module import get_logger

logger = get_logger(__name__)

__all__ = ['DDIDQSDFINETransformer']

def extract_1d_density_rays(density_map, ref_bboxes, reg_max=32):
    """
    密度空间探针：沿边界框的 左、上、右、下 4条射线，采样 1D 密度分布。
    density_map: [B, 1, H, W]
    ref_bboxes: [B, L, 4] (cx, cy, w, h) 归一化坐标
    """
    if density_map is None:
        return None
        
    B, L, _ = ref_bboxes.shape
    cx, cy, w, h = ref_bboxes.unbind(-1)
    
    # 构建 0 到 1 的 33 个采样步长 [1, 1, 33]
    steps = torch.linspace(0, 1, reg_max + 1, device=ref_bboxes.device).view(1, 1, -1)
    
    # 计算 4 条射线上的采样点坐标 (从中心发散到边界)
    # 1. 左边界射线 (向左 x 减小)
    x_left = cx.unsqueeze(-1) - (w.unsqueeze(-1) / 2) * steps
    y_left = cy.unsqueeze(-1).expand_as(x_left)
    
    # 2. 上边界射线 (向上 y 减小)
    x_top = cx.unsqueeze(-1).expand_as(steps.expand(B, L, -1))
    y_top = cy.unsqueeze(-1) - (h.unsqueeze(-1) / 2) * steps
    
    # 3. 右边界射线 (向右 x 增大)
    x_right = cx.unsqueeze(-1) + (w.unsqueeze(-1) / 2) * steps
    y_right = cy.unsqueeze(-1).expand_as(x_right)
    
    # 4. 下边界射线 (向下 y 增大)
    x_bot = cx.unsqueeze(-1).expand_as(steps.expand(B, L, -1))
    y_bot = cy.unsqueeze(-1) + (h.unsqueeze(-1) / 2) * steps
    
    # 拼接所有的 (x,y) 坐标: [B, L, 4条边, 33个点, 2]
    pts = torch.stack([
        torch.stack([x_left, y_left], dim=-1),
        torch.stack([x_top, y_top], dim=-1),
        torch.stack([x_right, y_right], dim=-1),
        torch.stack([x_bot, y_bot], dim=-1)
    ], dim=2)
    
    # 转换到 grid_sample 需要的 [-1, 1] 坐标系
    grid = pts * 2.0 - 1.0
    grid = grid.view(B, L * 4, reg_max + 1, 2) # 调整形状为 [B, H_new, W_new, 2]
    
    # 在密度图上采样
    # sampled_density: [B, 1, L*4, 33]
    sampled_density = F.grid_sample(density_map, grid, align_corners=False)
    
    # Reshape 回归格式: [B*L, 4, 33]
    return sampled_density.view(B * L, 4, reg_max + 1)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class LightDMG(nn.Module):
    def __init__(self, chs, scale, kernel_sizes=(3, 5, 7, 9, 11)) -> None:
        super().__init__()

        self.scale = scale

        self.dw_conv = nn.ModuleList(nn.Conv2d(chs, chs // 4, kernel_size=k, padding=autopad(k), groups=math.gcd(chs, chs // 4)) for k in kernel_sizes)
        self.pw_conv = ConvNormLayer_fuse(chs // 4 * len(kernel_sizes), chs, 1, 1)

        self.densehead = nn.Sequential(
            ConvNormLayer_fuse(chs, chs // 4, 3, 1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(chs // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        
        x_in = x 
        
        # 1. 多尺度特征提取
        x_processed = torch.concat([layer(x) for layer in self.dw_conv], dim=1)
        # 2. 融合
        x_processed = self.pw_conv(x_processed)
        
        # 3. 生成密度图
        density_map = self.densehead(x_processed)
        
        # # [核心修复] 返回三个值，匹配主代码的解包需求
        # _, densityData, densityMap = self.LDMG(...)
        return x_in, x_processed, density_map
    
class ChannelGate(nn.Module):
    def __init__(self, in_chs, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_chs = max(8, in_chs // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_chs, mid_chs),
            nn.ReLU(inplace=True),
            nn.Linear(mid_chs, in_chs),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvNormLayer_fuse(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, bias=False, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


@register()
class DDIDQSDFINETransformer(nn.Module):
    # 定义共享参数，这些参数可能在其他地方被引用
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,              # 类别数量，默认为80（例如COCO数据集的类别数）
                 hidden_dim=256,              # Transformer隐藏层的维度
                 num_queries=300,             # 查询（query）的数量，即模型预测的最大目标数
                 feat_channels=[512, 1024, 2048],  # 输入特征图的通道数
                 feat_strides=[8, 16, 32],    # 特征图相对于输入图像的步幅
                 num_levels=3,                # 多尺度特征的层数
                 num_points=4,                # 每个查询点的数量（用于采样）
                 nhead=8,                     # Transformer中多头注意力的头数
                 num_layers=6,                # Transformer解码器层数
                 dim_feedforward=1024,        # 前馈网络的隐藏层维度
                 dropout=0.,                  # Dropout比率，防止过拟合
                 activation="relu",           # 激活函数类型
                 num_denoising=100,           # 去噪训练的查询数量
                 label_noise_ratio=0.5,       # 标签噪声比例，用于去噪训练
                 box_noise_scale=1.0,         # 边界框噪声比例，用于去噪训练
                 learn_query_content=False,   # 是否学习查询内容嵌入
                 eval_spatial_size=None,      # 评估时的空间分辨率
                 eval_idx=-1,                 # 评估时使用的解码器层索引，负数表示从最后一层计数
                 eps=1e-2,                    # 小值阈值，用于边界框的有效性检查
                 aux_loss=True,               # 是否使用辅助损失
                 cross_attn_method='default', # 交叉注意力机制类型
                 query_select_method='default', # 查询选择方法
                 reg_max=32,                  # 回归最大值，用于边界框回归
                 reg_scale=4.,                # 回归缩放因子
                 layer_scale=1,               # 层缩放因子，用于调整隐藏层维度
                 mlp_act='relu',              # MLP激活函数类型
                 using_densitymap_iter=10000,
                 densitymap_temperature=10,
                 query_factor=3,
                 min_query_num=100,
                 max_query_num=1500,
                 using_dynamic_query=False,
                 use_ldmg=False,  # 控制 LDMG
                 use_cgfe=False,  # 控制 CGFE
                 ):
        
        super().__init__()
        

        self.using_dynamic_query = using_dynamic_query
        # [新增] 保存开关状态
        self.use_ldmg = use_ldmg
        self.use_cgfe = use_cgfe

        # 参数校验，确保输入特征通道数不超过多尺度层数
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        # 如果特征步幅数量不足，自动扩展到num_levels层
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        # 初始化核心参数
        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)  # 根据层缩放调整隐藏维度
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        

       
        # 校验查询选择和交叉注意力方法的有效性
        assert query_select_method in ('default', 'one2many', 'agnostic'), '查询选择方法无效'
        assert cross_attn_method in ('default', 'discrete'), '交叉注意力方法无效'
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # 构建输入投影层，将主干网络特征投影到hidden_dim维度
        self._build_input_proj_layer(feat_channels)

        # 定义Transformer模块的参数
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)  # 上采样因子，固定为0.5
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)  # 回归缩放参数

        # 定义解码器层
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)

        # 去噪训练相关参数
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            # 为去噪训练创建类别嵌入，+1表示包括背景类
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])  # 初始化类别嵌入权重（除背景类）

        # 解码器嵌入
        self.learn_query_content = learn_query_content
        if learn_query_content:
            # 如果学习查询内容，则创建可学习的查询嵌入
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)  # 查询位置的MLP

        # 编码器输出层
        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),  # 线性投影
            ('norm', nn.LayerNorm(hidden_dim)),          # 层归一化
        ]))

        # 根据查询选择方法定义得分头
        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)  # 类无关得分
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)  # 类别相关得分

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)  # 边界框预测MLP

        # 解码器头
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx  # 计算评估层索引
        # 类别得分预测头，根据层数和缩放维度分段定义
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
          + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)  # 预边界框预测
        # 边界框回归头，输出4*(reg_max+1)表示分布回归
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)  # 积分模块，用于将分布转换为边界框坐标
        # self.integral = DynamicManifoldIntegral(self.reg_max)  # 积分模块，用于将分布转换为边界框坐标
        
        # 初始化评估时的锚点和有效掩码
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)  # 注册锚点为缓冲区
            self.register_buffer('valid_mask', valid_mask)  # 注册有效掩码
        
        
        # 1. 控制 LDMG
        if self.use_ldmg:
            self.LDMG = LightDMG(self.hidden_dim, feat_strides[0], kernel_sizes=[3, 5, 7, 9, 11])
        else:
            self.LDMG = None # 没开就不创建，参数量为 0

                # 3. 控制 CGFE
        if self.use_ldmg and self.use_cgfe:
            # self.CGFE = CGFE(gate_channels=self.hidden_dim, reduction_ratio=4, num_feature_levels=self.num_levels)
            self.CGFE = CGFE(self.hidden_dim)
        else:
            self.CGFE = None    

        self.iter = 0
        self.using_densitymap_iter = using_densitymap_iter
        self.densitymap_temperature = densitymap_temperature
        self.query_factor = query_factor
        
        
        # [逻辑检查] CGFE 依赖 LDMG 的特征，如果 LDMG 关闭，CGFE 也强制关闭
        if not self.use_ldmg:
            self.use_cgfe = False

        self.min_query_num = min_query_num
        self.max_query_num = max_query_num

        # 重置参数
        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        # 将模型转换为部署模式，仅保留评估层的预测头
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )
        self.iter = 0
        self.using_densitymap_iter = 0

    def _reset_parameters(self, feat_channels):
        # 参数初始化
        bias = bias_init_with_prob(0.01)  # 初始化偏置，假设函数返回一个偏置值
        init.constant_(self.enc_score_head.bias, bias)  # 初始化编码器得分头的偏置
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)  # 初始化边界框头的权重
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)  # 初始化边界框头的偏置

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        # 初始化解码器得分头和边界框头的偏置和权重
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)  # Xavier初始化编码器输出投影权重
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)  # 初始化查询嵌入权重
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)  # 初始化查询位置MLP权重
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)  # 初始化输入投影层的权重

    def _build_input_proj_layer(self, feat_channels):
        # 构建输入投影层，将不同通道数的特征投影到hidden_dim
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())  # 如果通道数匹配，直接使用恒等映射
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),  # 1x1卷积
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])  # 批归一化
                    )
                )

        in_channels = feat_channels[-1]
        # 为剩余的特征层添加投影层
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),  # 3x3卷积，下采样
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # 获取编码器输入，将特征图投影并展平
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # 展平特征并记录空间形状
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))  # [b, c, h, w] -> [b, h*w, c]
            spatial_shapes.append([h, w])  # 记录每层的空间分辨率

        feat_flatten = torch.concat(feat_flatten, 1)  # 拼接所有层特征
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        # 生成锚点和有效掩码
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # 生成网格坐标
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)  # 归一化到[0,1]
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)  # 根据层级缩放锚点大小
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)  # 拼接中心点和宽高
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)  # 检查锚点是否有效
        anchors = torch.log(anchors / (1 - anchors))  # 将锚点转换为logit形式(数值稳定性：避免边界值的梯度消失)
        anchors = torch.where(valid_mask, anchors, torch.inf)  # 无效锚点置为无穷大

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None,
                           densityMap=None):
        # 准备解码器输入
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)  # 为batch扩展锚点

        memory = valid_mask.to(memory.dtype) * memory  # 应用有效掩码

        output_memory: torch.Tensor = self.enc_output(memory)  # 编码器输出
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)  # 计算得分

        
        # --- [修改] 密度图处理逻辑 ---
        densityMapMemory = None
        
        if densityMap is not None:
            densityMapMemory = []
            for idx, s in enumerate(self.feat_strides):
                if idx == 0:
                    densityMapMemory.append(nn.AvgPool2d(kernel_size=s, stride=s)(densityMap))
                else:
                    densityMapMemory.append(nn.AvgPool2d(kernel_size=s // self.feat_strides[idx - 1], stride=s // self.feat_strides[idx - 1])(densityMapMemory[-1]))
            for idx in range(len(densityMapMemory)):
                densityMapMemory[idx] = densityMapMemory[idx].flatten(2).permute(0, 2, 1)
            densityMapMemory = torch.cat(densityMapMemory, dim=1).squeeze()

        # 选择top-k查询
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(output_memory, enc_outputs_logits, densityMapMemory, anchors, self.num_queries)

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors  # 预测边界框

        # 如果是训练阶段，记录编码器输出
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # 获取查询内容
        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])  # 可学习嵌入
        else:
            content = enc_topk_memory.detach()  # 使用编码器输出

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        # 如果有去噪输入，拼接去噪和正常查询
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits

    def _select_topk(self, memory, outputs_logits, densityMapMemory, outputs_anchors_unact, topk):
        # ============================================================
        # 策略：自适应保底阈值 (Adaptive Recall-Preserving Threshold)
        # 灵感来源：Dome-DETR 的 MWAS 模块 [cite: 31, 281, 284]
        # 核心：mu = min(Hard_Threshold, Kth_Largest_Density)
        # 效果：图强则强杀背景，图弱则降权保 Recall。
        # ============================================================
        
        warm_up_iter = self.using_densitymap_iter
        suppress_start_iter = self.using_densitymap_iter + 1  # 30E
        if self.query_select_method == 'default':
            # -----------------------------------------------------------
            # Stage 1: 冷启动 (Warm-up)
            # -----------------------------------------------------------
            if (self.iter < warm_up_iter) or (densityMapMemory is None):
                _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

            # -----------------------------------------------------------
            # Stage 2: 助推期 (T=10 + 全局归一化)
            # -----------------------------------------------------------
            elif self.iter < suppress_start_iter:
                if self.iter == warm_up_iter:
                    logger.info("Stage 2 START: T=10 Boost with Global Norm...")
                
                # T=10 提亮
                density_processed = densityMapMemory ** (1 / self.densitymap_temperature)
                # [Fix] 全局归一化
                density_processed = density_processed / (density_processed.max() + 1e-12)
                
                # 简单融合
                final_score = outputs_logits.max(-1).values * density_processed
                _, topk_ind = torch.topk(final_score, topk, dim=-1)

            # -----------------------------------------------------------
            # Stage 3: 自适应抑制期 (Adaptive Suppression)
            # -----------------------------------------------------------
            else:
                if self.iter == suppress_start_iter:
                    logger.info("Stage 3 START: Adaptive Thresholding (Dome-DETR Style)...")

                # [Step A] 准备数据 (保持 T=10，因为我们需要它的分布特性)
                density_processed = densityMapMemory ** (1 / self.densitymap_temperature)
                density_processed = density_processed / (density_processed.max() + 1e-12) # Global Norm

                # [Step B] 计算"第 K 大"的密度值 (Batch-wise)
                # 我们需要保证每张图至少能选出 topk 个 Query，或者接近这个数
                # 展平: [B, H, W, 1] -> [B, HW]
                density_flat = density_processed.view(density_processed.shape[0], -1)
                
                # 找到每张图中第 K 大的值 (val_k)
                # 如果 K > HW (虽不可能但为了鲁棒)，取最小值
                k_safe = min(topk, density_flat.shape[1])
                # topk_vals: [B, K] -> 取最后一个也就是第 K 大的: [B, 1]
                val_k, _ = torch.topk(density_flat, k=k_safe, dim=1)
                boundary_val = val_k[:, -1:].unsqueeze(-1).unsqueeze(-1) # [B, 1, 1, 1] 用于广播

                # [Step C] 动态计算 mu (阈值)
                # 逻辑：
                # 1. 理想阈值是 0.62 (target_mu)，我们希望能杀掉 < 0.62 的。
                # 2. 现实边界是 boundary_val。如果 boundary_val 只有 0.2，说明这张图全是弱目标。
                # 3. 妥协：mu = min(0.62, boundary_val - 0.05)
                #    减去 0.05 是为了留一点余量，让第 K 个点能过 Sigmoid 的中心点(0.5)
                
                target_mu = 0.65    # 你想要的严厉阈值
                min_floor = 0.05     # 最低底线 (防止选中纯黑背景)
                
                # 核心公式：自适应下降
                adaptive_mu = torch.clamp(boundary_val - 0.05, min=min_floor, max=target_mu)
                
                # [Step D] 动态斜率 k (可选优化)
                # 如果我们被迫降低了阈值(图很弱)，slope 应该平缓一点，给它机会
                # 如果我们维持高阈值(图很强)，slope 可以陡峭一点，杀伐果断
                # 简单线性映射: mu=0.1 -> k=5, mu=0.6 -> k=12
                adaptive_k = 4.0 + (adaptive_mu - min_floor) / (target_mu - min_floor) * (12.0 - 4.0)

                # # 打印日志观察 (仅调试时)
                # if self.iter % 2000 == 0:
                #     logger.info(f"Iter {self.iter} | Target Mu: {target_mu} | Real TopK Val: {boundary_val.mean().item():.3f} | Adapted Mu: {adaptive_mu.mean().item():.3f}")

                # [Step E] Sigmoid 门控
                # 此时的 adaptive_mu 是针对每一张图片(Batch)动态计算的！
                # 这解决了"有的图满是目标，有的图全是背景"的不平衡问题
                density_gated = torch.sigmoid(adaptive_k * (density_processed - adaptive_mu))
                
                # [Step F] 融合
                final_score = outputs_logits.max(-1).values * density_gated
                _, topk_ind = torch.topk(final_score, topk, dim=-1)
            
                self.iter = self.iter + 1

           
        
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor # [bs, topk]

        # 提取top-k对应的锚点、得分和记忆
        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    
    def forward(self, feats, targets=None):
        """
        feats: List[Tensor], 来自 Backbone 的多尺度特征 [C3, C4, C5]
        targets: 标签 (训练时用)
        """
        # =========================================================
        # 1. 准备 Encoder 输入 (语义源 E)
        # =========================================================
        # memory: [B, Total_L, 256] (语义强，但细节模糊)
        # spatial_shapes: 记录了每一层特征的 H, W
        memory, spatial_shapes = self._get_encoder_input(feats)

        # =========================================================
        # 2. 生成密度图 (位置引导 D)
        # =========================================================
        densityMap = None
        # 初始化占位符，防止后面报错
        densityData = None 
        
        if self.use_ldmg and self.LDMG is not None:
            # LDMG 通常基于第一层(最浅层)特征生成，因为细节最丰富
            shallow_spatial_shapes = spatial_shapes[0]
            # 计算第一层 token 的长度
            len_shallow = int(shallow_spatial_shapes[0] * shallow_spatial_shapes[1])
            
            # 还原第一层特征的空间结构: [B, L, C] -> [B, C, H, W]
            shallow_feature = memory[:, :len_shallow].transpose(1, 2).reshape(
                memory.size(0), memory.size(2), 
                int(shallow_spatial_shapes[0]), int(shallow_spatial_shapes[1])
            )
            
            # [修复后] LDMG 现在返回 3 个值，不会报错了
            _, densityData, densityMap = self.LDMG(shallow_feature)

        
        if self.use_cgfe and self.CGFE is not None and densityMap is not None:
            enhanced_feats = []
            idx = 0
            
            # 遍历每一层 (Spatial Shapes 对应 feat_strides)
            # 假设 self.feat_strides = [8, 16, 32] 或 [4, 8, 16, 32]
            # 我们需要获取当前层的 stride
            
            for i, (h, w) in enumerate(spatial_shapes):
                length = int(h * w)
                
                # 获取当前层的 Stride
                # 这种计算方式是自适应的: img_size / feat_size
                # 假设输入是正方形，取 h 即可
                # 或者直接从 self.feat_strides 获取 (如果对齐的话)
                current_stride = 640 // h # 简单估算，或者用 self.feat_strides[i] 如果索引对齐
                
                # 更稳健的方法：直接用 self.feat_strides (如果 i 和 feats 对应)
                # 如果你的 feats 只有 3 层 (C3-C5)，但 strides 有 4 层，需要注意偏移
                # 假设 feats 是 [C2, C3, C4, C5] 或 [C3, C4, C5]
                # 最好的判定 P2 的方法是看尺寸: 160x160 (对于640输入) 就是 P2
                
                # A. 准备 Backbone (2D)
                if i < len(self.input_proj):
                    raw_backbone_2d = self.input_proj[i](feats[i]) 
                else:
                    continue 
                
                # 判断是不是 P2 (Stride 4)
                # 只要当前特征图宽/高 >= 100 (对于640输入)，肯定就是 P2
                is_p2_stride = 4 if (h >= 100 or w >= 100) else 8 # 传给模块做判断
                
                # B. 准备 Encoder
                curr_enc_flat = memory[:, idx:idx+length, :]
                
                # C. 调用模块 (传入 stride)
                feat_new = self.CGFE(
                    backbone_feat_2d=raw_backbone_2d, 
                    encoder_feat_flat=curr_enc_flat, 
                    density_map=densityMap,
                    stride=is_p2_stride # <--- 关键参数
                )
                
                enhanced_feats.append(feat_new)
                idx += length
            
            memory = torch.cat(enhanced_feats, dim=1)

        num_queries_list = None
        # 只有在 (开启了动态查询) AND (有密度图) 的情况下才计算
        if self.using_dynamic_query and densityMap is not None:
            num_queries_list = list(map(int, (densityMap.sum(dim=[1,2,3]) * self.query_factor).cpu().detach().tolist()))
            num_queries_list = [max(min(q, self.max_query_num), self.min_query_num) for q in num_queries_list]
            self.num_queries = int(max(num_queries_list))

        # if densityMap is not None:
        #     # a. 设定二值化阈值。可以使用一个固定超参，或者结合你 DQS 里的自适应均值
        #     # 这里的目的是区分“绝对的背景”和“可能的缺陷”
        #     mra_threshold = 0.05  # 这个值可以根据你的 gt_densitymap 归一化后的分布来调
            
        #     # b. 生成硬掩码 (Hard Mask): [B, 1, H, W]
        #     # 大于阈值的保留为 1，小于阈值的（背景）变为 0
        #     binary_density_mask = (densityMap > mra_threshold).float() 
            
        #     # c. 将 2D 掩码展平以对齐 memory 的形状 [B, Total_L, 1]
        #     mra_mask_flat = []
        #     for idx, s in enumerate(self.feat_strides):
        #         if idx == 0:
        #             pooled_mask = nn.MaxPool2d(kernel_size=s, stride=s)(binary_density_mask)
        #         else:
        #             pooled_mask = nn.MaxPool2d(kernel_size=s // self.feat_strides[idx - 1], 
        #                                        stride=s // self.feat_strides[idx - 1])(mra_mask_flat[-1])
        #         mra_mask_flat.append(pooled_mask)
                
        #     for idx in range(len(mra_mask_flat)):
        #         mra_mask_flat[idx] = mra_mask_flat[idx].flatten(2).permute(0, 2, 1)
                
        #     mra_mask_flat = torch.cat(mra_mask_flat, dim=1) # [B, Total_L, 1]
            
        #     # d. 【核心操作】将 MRA 掩码直接乘到 memory 上！
        #     # 这样一来，背景区域的特征全部归零。
        #     # 当 Decoder 里的 Deformable Attention 去采样时，即使它想看背景，也只能采到 0。
        #     # 强制网络把注意力权重分配给非 0 的高密度区域。
        #     memory = memory * mra_mask_flat 
            
        # # =========================================================
            # self.num_queries = max(min(self.num_queries, self.max_query_num), self.min_query_num)
        # self.num_queries = 300
        else:
            # 如果不开 DQS，或者没有 LDMG，使用默认固定查询数
            # 这里的 self.num_queries 保持 init 里的默认值 (如 300)
            pass

        # visualize_density_map_only(densityMap.squeeze().cpu().detach().numpy(), 'result.png')
        # visualize_density_map_only(densityMap.squeeze().cpu().detach().numpy() ** (1 / self.densitymap_temperature), 'result.png')
        # visualize_density_map_only(densityMap.squeeze().cpu().detach().numpy() ** (1 / self.densitymap_temperature) / float((densityMap ** (1 / self.densitymap_temperature)).max()), 'result.png')

        # 准备去噪训练数据
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                    )
            if memory.size(0) > 1 and self.using_dynamic_query: # bs等于1的时候就不需要处理attn_mask
                if attn_mask is not None:
                    attn_mask = attn_mask.unsqueeze(0).repeat(memory.size(0) * self.nhead, 1, 1) # 每个样本不同的attn_mask，需要扩展到三维
                    for i, qn in enumerate(num_queries_list):
                        # 假设这个batch最大的query是1500，当这个batch的其中一个样本为900的时候，前900个不能看到后600个查询
                        attn_mask[i * self.nhead:(i + 1) * self.nhead, dn_meta['dn_num_split'][0] + qn:, :dn_meta['dn_num_split'][0] + qn] = True 
                else:
                    self.init_attn_mask(memory.size(0), num_queries_list, memory.device)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
            if memory.size(0) > 1 and self.using_dynamic_query: # bs等于1的时候就不需要处理attn_mask
                self.init_attn_mask(memory.size(0), num_queries_list, memory.device)

        # 获取解码器输入
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, densityMap)
        # 解码器前向传播
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta
            # densityMap=densityMap
        )

        # 如果有去噪训练，分割去噪和正常输出
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)

        # 构造输出字典
        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale, 'pred_densitymap': densityMap,
                   'num_queries_list': num_queries_list}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_densitymap': densityMap, 'num_queries_list': num_queries_list}

        # 如果是训练阶段且使用辅助损失，添加辅助输出
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                     out_corners[-1], out_logits[-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta
            
            out['enc_outputs_logits'] = enc_outputs_logits

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                     'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]

    def init_attn_mask(self, bs, num_queries_list, device):
        attn_mask = torch.full([bs * self.nhead,self.num_queries, self.num_queries], False, dtype=torch.bool, device=device)
        for i, qn in enumerate(num_queries_list):
            # 假设这个batch最大的query是1500，当这个batch的其中一个样本为900的时候，前900个不能看到后600个查询
            attn_mask[i * self.nhead:(i + 1) * self.nhead, qn:, :qn] = True 
        return attn_mask
