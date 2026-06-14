"""
DFINE with Density-aware Query Selection & OQI (Occlusion-aware Query Interaction)
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
from .dq_dfine_decoder import MultiScaleFeature

from ..logger_module import get_logger

logger = get_logger(__name__)

__all__ = ['DQSDFINETransformer']

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

# =========================================================================================
# 🌟 OQI (Occlusion-aware Query Interaction) 模块
# =========================================================================================
class OQI(nn.Module):
    """
    Occlusion-aware Query Interaction (抗遮挡查询交互模块 - 局部距离感知版)
    针对目标相互遮挡：引入空间距离掩码 (Distance Mask)，限制特征只在局部邻域内进行交互，
    防止全局注意力导致的目标粘连和平滑，强制解耦遮挡边缘特征。
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, interaction_radius=0.15):
        """
        interaction_radius: 交互半径 (基于 0~1 的归一化坐标)。
                            0.15 表示只有距离在整图宽/高 15% 范围内的 Token 才允许互相施加注意力。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.interaction_radius = interaction_radius
        self.scale = (hidden_dim // num_heads) ** -0.5
        
        # 空间位置编码投影 
        self.pos_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 手动实现 Attention 的线性投影
        # self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.qk_proj = nn.Linear(hidden_dim, hidden_dim * 2) # 负责 Q 和 K
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)     # 负责 V (原生语义)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, density_map_1d, anchors, interact_num=15000):
        B, L, C = memory.shape
        N = min(interact_num, L)
        
        if density_map_1d.dim() > 2:
            density_map_1d = density_map_1d.squeeze(-1)
            
        # ==========================================
        # 1. 动态阈值保底 + Top-N 提取 (防止弱势目标丢失)
        # ==========================================
        # 为了防止被遮挡的弱目标连 OQI 都进不去，我们不仅取 Top-N，
        # 还可以借用 DQS 里密度图归一化的思路。不过最快的方式依然是保留足够的 N。
        _, top_n_idx = torch.topk(density_map_1d, N, dim=1) # [B, N]
        
        salient_tokens = memory.gather(1, top_n_idx.unsqueeze(-1).expand(-1, -1, C)) # [B, N, C]
        
        salient_coords = anchors.gather(1, top_n_idx.unsqueeze(-1).expand(-1, -1, 4))[..., :2]
        salient_coords = salient_coords.sigmoid() # [B, N, 2] 绝对坐标
        
        # ==========================================
        # 2. 计算空间距离掩码 (Distance Mask)
        # ==========================================
        # 计算 N 个 Token 两两之间的欧氏距离，形状: [B, N, N]
        dist_matrix = torch.cdist(salient_coords, salient_coords, p=2.0)
        
        # 生成布尔掩码：距离大于交互半径 interaction_radius 的位置为 True（代表需要被屏蔽）
        # [B, 1, N, N] 增加头部的维度以便广播
        spatial_mask = (dist_matrix > self.interaction_radius).unsqueeze(1)
        
        # ==========================================
        # 3. 局部感知的自注意力重组 (核心逻辑)
        # ==========================================
        pos_embed = self.pos_proj(salient_coords) # [B, N, C]
        q_k_input = salient_tokens + pos_embed
        
        # # 生成 Q, K, V
        # qkv = self.qkv_proj(q_k_input) # [B, N, 3C]
        # qkv = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], salient_tokens.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # # q, k: [B, Heads, N, Head_dim], v: 原生特征不加位置编码
        # Q, K 包含位置信息
        qk = self.qk_proj(q_k_input).view(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        
        # V 仅包含原生纹理 (使用独立的 v_proj)
        v = self.v_proj(salient_tokens).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # 计算 Attention Score
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, Heads, N, N]
        
        # 🚨 应用空间距离掩码：将距离过远的 Token 注意力强制设为 -inf
        attn = attn.masked_fill(spatial_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        # 防止全是 -inf 导致 softmax 产出 NaN
        attn = torch.nan_to_num(attn, nan=0.0) 
        
        attn = self.dropout(attn)
        
        # 聚合特征
        out = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, N, C]
        out = self.out_proj(out)
        
        salient_tokens = self.norm(salient_tokens + self.dropout(out))
        
        # ==========================================
        # 4. 特征回写
        # ==========================================
        memory_out = memory.clone() 
        memory_out.scatter_(1, top_n_idx.unsqueeze(-1).expand(-1, -1, C), salient_tokens)
        
        return memory_out

class CGFE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x

@register()
class DQSDFINETransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 using_densitymap_iter=10000,
                 densitymap_temperature=10,
                 query_factor=3,
                 min_query_num=100,
                 max_query_num=1500,
                 using_dynamic_query=False,
                 use_ldmg=False,
                 use_cgfe=False, # yaml 中的参数，映射给 OQI
                 ):
        super().__init__()

        self.using_dynamic_query = using_dynamic_query
        self.use_ldmg = use_ldmg
        # 将 YAML 传进来的 use_cgfe 映射给 OQI
        self.use_oqi = use_cgfe

        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
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

        assert query_select_method in ('default', 'one2many', 'agnostic')
        assert cross_attn_method in ('default', 'discrete')
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead, reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
          + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)
        
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        
        # 控制 LDMG
        if self.use_ldmg:
            self.LDMG = LightDMG(self.hidden_dim, feat_strides[0], kernel_sizes=[3, 5, 7, 9, 11])
        else:
            self.LDMG = None

        # 实例化 OQI 模块
        if not self.use_ldmg:
            self.use_oqi = False
            
        if self.use_ldmg and self.use_oqi:
            self.OQI = OQI(hidden_dim=self.hidden_dim, num_heads=8)
        else:
            self.OQI = None

        self.iter = 0
        self.using_densitymap_iter = using_densitymap_iter
        self.densitymap_temperature = densitymap_temperature
        self.query_factor = query_factor
        self.min_query_num = min_query_num
        self.max_query_num = max_query_num

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )
        self.iter = 0
        self.using_densitymap_iter = 0

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])

        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self, memory: torch.Tensor, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None, densityMap=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory 

        # 优先提取和展平 densityMap
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
            # squeeze(-1) 防止因为单通道而出现额外的维度
            densityMapMemory = torch.cat(densityMapMemory, dim=1).squeeze(-1) 

        # =========================================================
        # 🌟 OQI: 在计算得分和 DQS 之前，缝合遮挡导致的断裂特征
        # =========================================================
        if self.use_oqi and getattr(self, 'OQI', None) is not None and densityMapMemory is not None:
            interact_num = min(1500, self.max_query_num * 2) 
            memory = self.OQI(memory, densityMapMemory, anchors, interact_num)

        # 此时得到的 output_memory 是已经过缝合的高质量特征
        output_memory: torch.Tensor = self.enc_output(memory) 
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory) 

        # DQS 动态挑选 Query 
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, densityMapMemory, anchors, self.num_queries)

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        # 返回时也将更新后的 memory 传出，以供 Decoder 使用
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits, memory


    def _select_topk(self, memory, outputs_logits, densityMapMemory, outputs_anchors_unact, topk):
        warm_up_iter = self.using_densitymap_iter
        suppress_start_iter = self.using_densitymap_iter + 1  
        
        if self.query_select_method == 'default':
            if (self.iter < warm_up_iter) or (densityMapMemory is None):
                _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
            elif self.iter < suppress_start_iter:
                if self.iter == warm_up_iter:
                    logger.info("Stage 2 START: T=10 Boost with Global Norm...")
                density_processed = densityMapMemory ** (1 / self.densitymap_temperature)
                density_processed = density_processed / (density_processed.max() + 1e-12)
                final_score = outputs_logits.max(-1).values * density_processed
                _, topk_ind = torch.topk(final_score, topk, dim=-1)
            else:
                if self.iter == suppress_start_iter:
                    logger.info("Stage 3 START: Fully Data-Driven Adaptive Thresholding...")

                density_processed = densityMapMemory ** (1 / self.densitymap_temperature)
                density_processed = density_processed / (density_processed.max() + 1e-12)

                density_flat = density_processed.view(density_processed.shape[0], -1)
                k_safe = min(topk, density_flat.shape[1])
                val_k, _ = torch.topk(density_flat, k=k_safe, dim=1)

                topk_mean = val_k.mean(dim=1).view(-1, 1, 1, 1) if density_processed.dim() == 4 else val_k.mean(dim=1).unsqueeze(-1)
                topk_std = val_k.std(dim=1).view(-1, 1, 1, 1) if density_processed.dim() == 4 else val_k.std(dim=1).unsqueeze(-1)
                global_mean = density_flat.mean(dim=1).view(-1, 1, 1, 1) if density_processed.dim() == 4 else density_flat.mean(dim=1).unsqueeze(-1)

                adaptive_mu = topk_mean - topk_std
                adaptive_mu = torch.max(adaptive_mu, global_mean) 

                adaptive_k = 1.0 / (topk_std + 1e-4)
                adaptive_k = torch.clamp(adaptive_k, min=2.0, max=15.0)

                density_gated = torch.sigmoid(adaptive_k * (density_processed - adaptive_mu))
                
                final_score = outputs_logits.max(-1).values * density_gated
                _, topk_ind = torch.topk(final_score, topk, dim=-1)
            
                self.iter = self.iter + 1
        
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor # [bs, topk]

        topk_anchors = outputs_anchors_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    
    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        densityMap = None
        if self.use_ldmg and self.LDMG is not None:
            shallow_spatial_shapes = spatial_shapes[0]
            len_shallow = int(shallow_spatial_shapes[0] * shallow_spatial_shapes[1])
            shallow_feature = memory[:, :len_shallow].transpose(1, 2).reshape(
                memory.size(0), memory.size(2), 
                int(shallow_spatial_shapes[0]), int(shallow_spatial_shapes[1])
            )
            _, _, densityMap = self.LDMG(shallow_feature)

        num_queries_list = None
        if self.using_dynamic_query and densityMap is not None:
            num_queries_list = list(map(int, (densityMap.sum(dim=[1,2,3]) * self.query_factor).cpu().detach().tolist()))
            num_queries_list = [max(min(q, self.max_query_num), self.min_query_num) for q in num_queries_list]
            self.num_queries = int(max(num_queries_list))

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, 
                    self.num_classes, self.num_queries, self.denoising_class_embed,
                    num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio, box_noise_scale=1.0)
            if memory.size(0) > 1 and self.using_dynamic_query: 
                if attn_mask is not None:
                    attn_mask = attn_mask.unsqueeze(0).repeat(memory.size(0) * self.nhead, 1, 1) 
                    for i, qn in enumerate(num_queries_list):
                        attn_mask[i * self.nhead:(i + 1) * self.nhead, dn_meta['dn_num_split'][0] + qn:, :dn_meta['dn_num_split'][0] + qn] = True 
                else:
                    self.init_attn_mask(memory.size(0), num_queries_list, memory.device)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
            if memory.size(0) > 1 and self.using_dynamic_query:
                self.init_attn_mask(memory.size(0), num_queries_list, memory.device)

        # 核心：_get_decoder_input 内部封装了 OQI 的交互和 DQS 的挑选，同时返回 OQI 缝合后的 memory
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits, memory = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, densityMap)

        # Decoder 接收的 memory 是经历了 OQI 缝合连通后的高质量特征
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents, init_ref_points_unact, memory, spatial_shapes,
            self.dec_bbox_head, self.dec_score_head, self.query_pos_head, self.pre_bbox_head,
            self.integral, self.up, self.reg_scale, attn_mask=attn_mask, dn_meta=dn_meta)

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)

        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale, 'pred_densitymap': densityMap,
                   'num_queries_list': num_queries_list}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_densitymap': densityMap, 'num_queries_list': num_queries_list, 'init_reference': init_ref_points_unact.sigmoid()}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1], out_corners[-1], out_logits[-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta
            out['enc_outputs_logits'] = enc_outputs_logits

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref, teacher_corners=None, teacher_logits=None):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                     'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]

    def init_attn_mask(self, bs, num_queries_list, device):
        attn_mask = torch.full([bs * self.nhead,self.num_queries, self.num_queries], False, dtype=torch.bool, device=device)
        for i, qn in enumerate(num_queries_list):
            attn_mask[i * self.nhead:(i + 1) * self.nhead, qn:, :qn] = True 
        return attn_mask