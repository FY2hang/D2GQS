# """     
# DEIM: DETR with Improved Matching for Fast Convergence   
# Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
# ---------------------------------------------------------------------------------    
# Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)   
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.    
# """   


# import os, sys
# import math     
# import json   
# import gc  
# import numpy as np
# from typing import Iterable
# from tqdm import tqdm
# from prettytable import PrettyTable
# # from pycocotools.coco import COCO
# # from pycocotools.cocoeval import COCOeval
# from tidecv import TIDE, datasets
# from skimage.metrics import structural_similarity as ssim_func
# from skimage.metrics import peak_signal_noise_ratio as psnr_func
# import torch
# import torch.amp 
# from torch.utils.tensorboard import SummaryWriter     
# from torch.cuda.amp.grad_scaler import GradScaler

# from ..optim import ModelEMA, Warmup    
# from ..data import CocoEvaluator
# from ..misc import MetricLogger, MetricLogger_progress, SmoothedValue, dist_utils, plot_sample   
# from ..logger_module import get_logger
# from ..extre_module.ops import Profile 
# from ..extre_module.utils import TQDM, RANK    
# from ..deim.utils import coco_evaluator_per_class
# from .sample_adapter import (  
#     move_samples_to_device, 
#     select_model_input_for_model,  
#     select_plot_samples_for_logging,
# )
    
# from pycocoeval.yoloeval import get_yolo_det_metrice, get_yolo_seg_metrice   
  
# CLEAR_MEMORY_STEP = 100   
# TIME_DEBUG = False
# RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
# logger = get_logger(__name__)  


# def _plot_training_modalities(samples, targets, data_loader, output_dir, epoch):
#     modality_samples = select_plot_samples_for_logging(samples, keys=("rgb", "npy"))    
#     is_multimodal_plot = len(modality_samples) > 1
 
#     for modality, plot_samples in modality_samples:
#         if modality == "npy":
#             plot_samples = normalize_tensor_minmax_per_sample(plot_samples)    
#         if modality == "rgb" and not is_multimodal_plot: 
#             suffix = ""
#         else:    
#             suffix = f"_{modality}"
#         save_path = output_dir / f"train_batch_{epoch}{suffix}.png"     
#         if data_loader.dataset.remap_mscoco_category:
#             plot_sample((plot_samples, targets), data_loader.dataset.category2name, save_path, data_loader.dataset.label2category) 
#         else:
#             plot_sample((plot_samples, targets), data_loader.dataset.category2name, save_path)  
     
# # 训练单个 epoch  
# def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module, 
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer, 
#                     device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
#     model.train()  # 设置模型为训练模式 
#     criterion.train()  # 设置损失函数为训练模式    

#     print_freq = kwargs.get('print_freq', 10)  # 日志打印频率
#     writer: SummaryWriter = kwargs.get('writer', None)  # TensorBoard 记录器
#     ema: ModelEMA = kwargs.get('ema', None)  # 指数移动平均模型
#     scaler: GradScaler = kwargs.get('scaler', None)  # 混合精度训练的梯度缩放器
#     lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)  # 预热学习率调度器
#     plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)
#     output_dir = kwargs.get('output_dir', None)    
#     epoches = kwargs.get('epoches', -1) # 总的训练次数
#     verbose_type = kwargs.get('verbose_type', 'origin') # 显示方式  
#     header = 'Epoch: {}/{}'.format(epoch, epoches)  # 训练过程的日志标题

#     cur_iters = epoch * len(data_loader)  # 计算当前 epoch 的起始迭代数
    
#     if verbose_type == 'origin':     
#         metric_logger = MetricLogger(delimiter="  ")  # 记录训练过程中的度量信息
#     else:
#         metric_logger = MetricLogger_progress(delimiter="  ")  # 记录训练过程中的度量信息  
#     metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率变化 
#     pbar = enumerate(metric_logger.log_every(data_loader, print_freq if verbose_type == 'origin' else 1, header))
   
#     dt = [
#         Profile(device=device),   
#         Profile(device=device),
#         Profile(device=device),  
#         Profile(device=device),
#         Profile(device=device) 
#     ]  

#     for i, (samples, targets) in pbar:   
#         if i % CLEAR_MEMORY_STEP == 0:
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()   
  
#         if epoch % plot_train_batch_freq == 0 and i == 0:
#             _plot_training_modalities(samples, targets, data_loader, output_dir, epoch)     
#         with dt[0]:
#             samples = move_samples_to_device(samples, device, non_blocking=True)  # 将输入数据移动到指定设备 
#             model_inputs = select_model_input_for_model(samples, model=model, key='rgb')
#             targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]  # 目标数据也移动到设备    
    
#         global_step = epoch * len(data_loader) + i  # 计算全局训练步数 
#         metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))  # 训练元数据
     
#         # 使用混合精度训练 
#         if scaler is not None: 
#             with dt[1]:
#                 with torch.autocast(device_type=str(device), cache_enabled=True):
#                     outputs = model(model_inputs, targets=targets)   
            
#             # 处理异常情况，避免 NaN 或 Inf 影响训练  
#             if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():     
#                 logger.warning(outputs['pred_boxes'])
#                 state = model.state_dict()   
#                 new_state = {}     
#                 for key, value in model.state_dict().items():
#                     new_key = key.replace('module.', '')  # 兼容多 GPU 训练的情况 
#                     state[new_key] = value    
#                 new_state['model'] = state
#                 dist_utils.save_on_master(new_state, "./NaN.pth")  # 保存异常模型状态    
            
#             with dt[2]:   
#             # 计算损失 
#                 with torch.autocast(device_type=str(device), enabled=False):
#                     loss_dict = criterion(outputs, targets, **metas)    
#                 loss = sum(loss_dict.values())  # 总损失  
            
#             with dt[3]:
#                 scaler.scale(loss).backward()  # 反向传播   
   
#                 # 进行梯度裁剪（如果 max_norm > 0）  
#                 if max_norm > 0:
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)   
     
#                 scaler.step(optimizer)  # 更新参数
#                 scaler.update()  # 更新梯度缩放因子
#                 optimizer.zero_grad()  # 清空梯度   
        
#         else:
#             with dt[1]:   
#                 outputs = model(model_inputs, targets=targets)  # 前向传播
#             with dt[2]:
#                 loss_dict = criterion(outputs, targets, **metas)  # 计算损失
#                 loss: torch.Tensor = sum(loss_dict.values())  # 总损失   
#             with dt[3]:  
#                 optimizer.zero_grad()  # 清空梯度
#                 loss.backward()  # 反向传播
    
#                 # 进行梯度裁剪
#                 if max_norm > 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#                 optimizer.step()  # 更新参数  

#         with dt[4]:
#             # 更新 EMA（指数移动平均）  
#             if ema is not None:
#                 ema.update(model)
   
#             # 更新学习率   
#             if self_lr_scheduler:    
#                 optimizer = lr_scheduler.step(cur_iters + i, optimizer) 
#             else: 
#                 if lr_warmup_scheduler is not None:   
#                     lr_warmup_scheduler.step()

#             # 计算损失并检查是否异常    
#             loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
#             loss_value = sum(loss_dict_reduced.values())
#             if not math.isfinite(loss_value):     
#                 logger.warning("Loss is {}, stopping training".format(loss_value))
#                 logger.info(loss_dict_reduced)
#                 sys.exit(1)

#             # 记录日志   
#             metric_logger.update(loss=loss_value, **loss_dict_reduced)
#             metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#             # 记录到 TensorBoard  
#             if writer and dist_utils.is_main_process() and global_step % 10 == 0:    
#                 writer.add_scalar('Loss/total', loss_value.item(), global_step)
#                 for j, pg in enumerate(optimizer.param_groups):
#                     writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
#                 for k, v in loss_dict_reduced.items():
#                     writer.add_scalar(f'Loss/{k}', v.item(), global_step)   
  
#     # 统计并打印训练结果
#     metric_logger.synchronize_between_processes()    
#     logger.info(f'Averaged stats:{metric_logger}')
#     if TIME_DEBUG:  
#         time_data = [x.t / len(data_loader) for x in dt] 
#         logger.debug(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# class DensityTopologyMetric:
#     def __init__(self):
#         """
#         针对 DPDL 峰值密度图定制的空间拓扑质量评估器
#         摒弃易受量纲漂移干扰的计数指标，从几何形态（SSIM）和噪声抑制（PSNR）两个维度量化模型质量
#         """
#         self.total_ssim = 0.0
#         self.total_psnr = 0.0
#         self.total_images = 0

#     @torch.no_grad()
#     def update(self, outputs, targets):
#         if 'pred_densitymap' not in outputs or outputs['pred_densitymap'] is None:
#             return 
            
#         from ..deim.utils import DensityMapGenerator
        
#         pred_densitymap = outputs['pred_densitymap'] # [BS, 1, H, W]
#         bs, _, h, w = pred_densitymap.shape
#         dmg = DensityMapGenerator([h, w])

#         for b in range(bs):
#             # 1. 提取预测密度图并进行自适应最值归一化 (100% 对齐 DPDL 的形态特征空间)
#             raw_pred = pred_densitymap[b, 0].detach().cpu().numpy()
#             p_max = raw_pred.max()
#             pred_norm = (raw_pred / (p_max + 1e-6)) if p_max > 0 else np.zeros_like(raw_pred)
            
#             # 2. 截断负噪声并限定在标准 [0, 1] 像素空间
#             pred_norm = np.clip(pred_norm, 0.0, 1.0)

#             # 3. 实时生成当前图像完全同构的标准真实 Ground Truth 密度图
#             boxes = targets[b].get('boxes', None)
#             gt_count = boxes.shape[0] if boxes is not None else 0
            
#             if gt_count > 0:
#                 boxes_np = boxes.cpu().detach().numpy().copy()
#                 boxes_np[:, [0, 2]] *= w  
#                 boxes_np[:, [1, 3]] *= h  
#                 gt_densitymap = dmg.generate_from_boxes(boxes_np, method='gaussian')
                
#                 # 对 GT 执行标准的 DPDL 归一化限制
#                 gt_max = gt_densitymap.max()
#                 gt_norm = (gt_densitymap / (gt_max + 1e-6)) if gt_max > 0 else np.zeros_like(gt_densitymap)
#                 gt_norm = np.clip(gt_norm, 0.0, 1.0)
#             else:
#                 # 纯背景图
#                 gt_norm = np.zeros_like(pred_norm)

#             # 4. 计算结构相似性 (SSIM)
#             cur_ssim = ssim_func(pred_norm, gt_norm, data_range=1.0)
            
#             # 5. 计算峰值信噪比 (PSNR)
#             if np.all(pred_norm == gt_norm):
#                 cur_psnr = 80.0 # 行业级最高品质保底dB
#             else:
#                 cur_psnr = psnr_func(gt_norm, pred_norm, data_range=1.0)

#             self.total_ssim += cur_ssim
#             self.total_psnr += cur_psnr
#             self.total_images += 1

#     def compute(self):
#         if self.total_images == 0:
#             return 0.0, 0.0
#         avg_ssim = self.total_ssim / self.total_images
#         avg_psnr = self.total_psnr / self.total_images
#         return avg_ssim, avg_psnr


# @torch.no_grad()
# def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, test_only=False, output_dir=None, yolo_metrice=False, other_platform_model=None):
#     if model is not None: 
#         model.eval()
#     criterion.eval()
#     coco_evaluator.cleanup()   
  
#     metric_logger = MetricLogger_progress(delimiter="  ")
#     header = 'Test:'

#     iou_types = coco_evaluator.iou_types

#     dt = [ 
#         Profile(device=device),
#         Profile(device=device)
#     ]

#     # 🌟 新增：在线实例化新一代空间拓扑质量评估器
#     topology_metric = DensityTopologyMetric()

#     # 遍历数据集进行评估
#     coco_det_pred_json, coco_seg_pred_json = [], []
#     for samples, targets in metric_logger.log_every(data_loader, 1, header):   
#         samples = move_samples_to_device(samples, device, non_blocking=True) 
#         model_inputs = select_model_input_for_model(samples, model=model, key='rgb')    
#         targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets] 

#         if model is not None:    
#             with dt[0]:
#                 outputs = model(model_inputs)  # 前向传播 

#             # 🌟 新增：在线提取预测密度分布特征，累加 SSIM 和 PSNR 分项指标
#             topology_metric.update(outputs, targets)
 
#             orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

#             with dt[1]:
#                 results = postprocessor(outputs, orig_target_sizes, for_eval=True)     
#         else:   
#             if 'onnx' in other_platform_model:    
#                 with dt[0]:   
#                     orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0) 
#                     onnx_result = other_platform_model['onnx'].run(  
#                         output_names=None,
#                         input_feed={'images': model_inputs.cpu().detach().numpy(), "orig_target_sizes": orig_target_sizes.cpu().detach().numpy()}
#                     )
    
#                     results = [] 
#                     if len(onnx_result) == 3:
#                         labels, boxes, scores = onnx_result 
#                         for lab, box, sco in zip(labels, boxes, scores): 
#                             result = dict(labels=torch.from_numpy(lab), boxes=torch.from_numpy(box), scores=torch.from_numpy(sco))
#                             results.append(result)
#                     elif len(onnx_result) == 4:
#                         labels, boxes, scores, masks = onnx_result  
#                         for lab, box, sco, mask in zip(labels, boxes, scores, masks):
#                             result = dict(labels=torch.from_numpy(lab), boxes=torch.from_numpy(box), scores=torch.from_numpy(sco), masks=torch.from_numpy(mask))
#                             results.append(result)
#             elif 'engine' in other_platform_model:
#                 with dt[0]:   
#                     orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)    
#                     output = other_platform_model['engine']({'images': model_inputs,    
#                                                              'orig_target_sizes': orig_target_sizes.to(device)})   
#                     labels, boxes, scores, masks = output['labels'], output['boxes'], output['scores'], output.get('masks', None)
     
#                     results = []
#                     if masks is None:  
#                         for lab, box, sco in zip(labels, boxes, scores):    
#                             result = dict(labels=lab, boxes=box, scores=sco)
#                             results.append(result)
#                     else:
#                         for lab, box, sco, mask in zip(labels, boxes, scores, masks):    
#                             result = dict(labels=lab, boxes=box, scores=sco, masks=mask)  
#                             results.append(result)    
       
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res) 
#             coco_det_pred_json.extend(list(coco_evaluator.coco_eval['bbox'].cocoDt.anns.values()))
#             if 'segm' in coco_evaluator.coco_eval:   
#                 coco_seg_pred_json.extend(list(coco_evaluator.coco_eval['segm'].cocoDt.anns.values()))    

#     metric_logger.synchronize_between_processes()     
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()  
        
#     # 🌟 新增核心：提取最终收敛的 SSIM 与 PSNR 指标，并执行高亮显现
#     final_ssim, final_psnr = topology_metric.compute()
#     if topology_metric.total_images > 0:
#         logger.info(ORANGE + f"------------------ DPDL Density Topology Metrice ------------------" + RESET)
#         logger.info(GREEN + f"[*] Density Structural Similarity (SSIM) : {final_ssim:.4f} (Topology Match)" + RESET)
#         logger.info(GREEN + f"[*] Density Peak Signal-Noise Ratio (PSNR): {final_psnr:.2f} dB (Background Suppression)" + RESET)
#         logger.info(ORANGE + f"-------------------------------------------------------------------" + RESET)

#     if test_only:     
#         if model is not None:
#             speed = dict(zip(['inference', 'postprocess'], (x.t / len(data_loader.dataset) * 1e3 for x in dt)))  
#             logger.info(GREEN + f'Test On BatchSize:{data_loader.batch_size}' + RESET)    
#             logger.info(GREEN + f"Speed: {speed['inference']:.4f}ms inference, {speed['postprocess']:.4f}ms postprocess per image" + RESET)
#             logger.info(GREEN + f"FPS(inference+postprocess): {1000 / (speed['inference'] + speed['postprocess']):.2f}" + RESET) 
#         else:  
#             inference_speed = dt[0].t / len(data_loader.dataset) * 1e3  
#             logger.info(GREEN + f'Test On BatchSize:{data_loader.batch_size}' + RESET)
#             logger.info(GREEN + f"Speed: {inference_speed:.4f}ms inference per image" + RESET) 
#             logger.info(GREEN + f"FPS(inference): {1000 / inference_speed:.2f}" + RESET)    
   
#     if yolo_metrice:     
#         get_yolo_det_metrice(logger, coco_evaluator, coco_det_pred_json, output_dir if test_only else None)
#         if 'segm' in coco_evaluator.coco_eval:
#             get_yolo_seg_metrice(logger, coco_evaluator, coco_seg_pred_json, save_vis=False)

#     if coco_evaluator is not None:
#         logger.info(RED + "------------------------ COCO Metrice Start ------------------------" + RESET)
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()    
#         if test_only:   
#             logger.info(ORANGE + f"Saving coco pred[{output_dir / 'pred.json'}] json..." + RESET)
#             with open(output_dir / 'pred_bbox.json', 'w') as f:
#                 json.dump(coco_det_pred_json, f)
#             logger.info(ORANGE + "save success." + RESET)     
            
#             for iouType in coco_evaluator.coco_gt:
#                 model_metrice_table = coco_evaluator_per_class(coco_evaluator, iouType)
#                 print(ORANGE, model_metrice_table, RESET)   

#             try:
#                 logger.info(RED + "------------------------ TIDE Metrice Start ------------------------" + RESET)
#                 tide = TIDE()
#                 if hasattr(data_loader.dataset, 'ann_file'):
#                     tide.evaluate_range(datasets.COCO(data_loader.dataset.ann_file), datasets.COCOResult(output_dir / 'pred_bbox.json'))
#                 else:
#                     tide.evaluate_range(datasets.COCO(coco_evaluator.coco_gt), datasets.COCOResult(output_dir / 'pred_bbox.json'))
                    
#                 tide.summarize() 
#                 tide.plot(out_dir=output_dir / 'tide_result')
#             except Exception as e:
#                 logger.error(f"{RED}TIDE failure... skip message: {e}{RESET}")
#                 logger.warning('------------------------ TIDE指标生成报错可以不用管 ------------------------') 
    
#     # stats = {}     
#     # if coco_evaluator is not None:     
#     #     if 'bbox' in iou_types: 
#     #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#     #     if 'segm' in iou_types:
#     #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

#     # # 🌟 将两项前沿空间拓扑指标完美塞入系统字典，交差给上层主 solver
#     # if topology_metric.total_images > 0:
#     #     stats['density_ssim_loss_val'] = float(final_ssim)
#     #     stats['density_psnr_loss_val'] = float(final_psnr)

#     # return stats, coco_evaluator
#     stats = {}     
#     if coco_evaluator is not None:     
#         # 🚨 框架的核心海选只认 'coco_eval_bbox' 或 'coco_eval_masks'
#         if 'bbox' in iou_types: 
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in iou_types:
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

#     # 🌟 核心破局：提取最终收敛的 SSIM 与 PSNR 指标，并执行高亮显现
#     final_ssim, final_psnr = topology_metric.compute()
#     if topology_metric.total_images > 0:
#         logger.info(ORANGE + f"------------------ DPDL Density Topology Metrice ------------------" + RESET)
#         logger.info(GREEN + f"[*] Density Structural Similarity (SSIM) : {final_ssim:.4f} (Topology Match)" + RESET)
#         logger.info(GREEN + f"[*] Density Peak Signal-Noise Ratio (PSNR): {final_psnr:.2f} dB (Background Suppression)" + RESET)
#         logger.info(ORANGE + f"-------------------------------------------------------------------" + RESET)
        
#         # 🚨 终极绝杀：利用嵌套键名 'density_metrics' 封装，框架在外层遍历时只会处理 bbox/masks，
#         # 绝对无法将 SSIM/PSNR 误加入 all_metrics 中去稀释、玷污你的真实 AP！
#         # 包装成列表格式 [v, v]，完美兼容后面 TensorBoard 和文本日志的索引解析。
#         stats['density_metrics'] = [float(final_ssim), float(final_psnr)]

#     return stats, coco_evaluator

# def distill_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, teahcer_model: torch.nn.Module, student_featureExt, teacher_featureExt,
#                     criterion: torch.nn.Module, feature_distill_criterion, logical_distill_criterion,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,     
#                     device: torch.device, epoch: int, max_norm: float = 0, **kwargs):  
#     model.train()  
#     teahcer_model.train() 
#     criterion.train()  
  
#     print_freq = kwargs.get('print_freq', 10)  
#     writer: SummaryWriter = kwargs.get('writer', None)  
#     ema: ModelEMA = kwargs.get('ema', None)  
#     scaler: GradScaler = kwargs.get('scaler', None)  
#     lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)  
#     plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)
#     output_dir = kwargs.get('output_dir', None)
#     epoches = kwargs.get('epoches', -1) 
#     verbose_type = kwargs.get('verbose_type', 'origin') 
#     feature_loss_ratio = kwargs.get('feature_loss_ratio', 1.0)
#     logical_loss_ratio = kwargs.get('logical_loss_ratio', 1.0)
#     distill_loss_decay = kwargs.get('distill_loss_decay', 'constant')    
#     header = 'Epoch: {}/{}'.format(epoch, epoches)  

#     cur_iters = epoch * len(data_loader)  
   
#     if verbose_type == 'origin':
#         metric_logger = MetricLogger(delimiter="  ")  
#     else:   
#         metric_logger = MetricLogger_progress(delimiter="  ")  
#     metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  
#     pbar = enumerate(metric_logger.log_every(data_loader, print_freq if verbose_type == 'origin' else 1, header))  
      
#     dt = [     
#         Profile(device=device), 
#         Profile(device=device), 
#         Profile(device=device),     
#         Profile(device=device), 
#         Profile(device=device)     
#     ] 
    
#     for i, (samples, targets) in pbar:    
#         if i % CLEAR_MEMORY_STEP == 0:  
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
  
#         if distill_loss_decay == 'constant': 
#             distill_decay = 1.0 
#         elif distill_loss_decay == 'cosine':
#             eta_min, base_ratio, T_max = 0.01, 1.0, 10
#             distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
#         elif distill_loss_decay == 'linear':  
#             distill_decay = ((1 - math.cos(i * math.pi / len(data_loader))) / 2) * (0.01 - 1) + 1    
#         elif distill_loss_decay == 'cosine_epoch':
#             eta_min, base_ratio, T_max = 0.01, 1.0, 10
#             distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * (cur_iters + i) / T_max)) / 2
#         elif distill_loss_decay == 'linear_epoch':    
#             distill_decay = ((1 - math.cos((cur_iters + i) * math.pi / (epoches * len(data_loader)))) / 2) * (0.01 - 1) + 1   

#         if epoch % plot_train_batch_freq == 0 and i == 0:    
#             _plot_training_modalities(samples, targets, data_loader, output_dir, epoch)  
#         with dt[0]: 
#             samples = move_samples_to_device(samples, device, non_blocking=True)  
#             model_inputs = select_model_input_for_model(samples, model=model, key='rgb')
#             targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets] 
        
#         global_step = epoch * len(data_loader) + i  
#         metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))  
 
#         if feature_distill_criterion:    
#             student_featureExt.clear_features()
#             teacher_featureExt.clear_features()  
   
#         with dt[1]:    
#             outputs = model(model_inputs, targets=targets)  
#             if feature_distill_criterion or logical_distill_criterion:  
#                 with torch.no_grad():    
#                     teacher_outputs = teahcer_model(model_inputs, targets=targets)

#         with dt[2]:    
#             loss_dict = criterion(outputs, targets, **metas)  
 
#             if feature_distill_criterion:
#                 feature_distill_loss = feature_distill_criterion(student_featureExt.get_features_in_order(), teacher_featureExt.get_features_in_order()) * feature_loss_ratio * distill_decay 
#                 loss_dict['fea_loss'] = feature_distill_loss
#             else:  
#                 loss_dict['fea_loss'] = torch.zeros(1, device=device)    
   
#             if logical_distill_criterion:    
#                 logical_distill_loss = logical_distill_criterion(outputs, teacher_outputs, targets) * logical_loss_ratio * distill_decay
#                 loss_dict['log_loss'] = logical_distill_loss
#             else:
#                 loss_dict['log_loss'] = torch.zeros(1, device=device)   

#             loss: torch.Tensor = sum(loss_dict.values())  
   
#         with dt[3]:     
#             optimizer.zero_grad()  
#             loss.backward()  
#             if max_norm > 0:    
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()    
     
#         with dt[4]:
#             if ema is not None:
#                 ema.update(model)
 
#             if self_lr_scheduler:
#                 optimizer = lr_scheduler.step(cur_iters + i, optimizer)  
#             else: 
#                 if lr_warmup_scheduler is not None:  
#                     lr_warmup_scheduler.step()

#             loss_dict_reduced = dist_utils.reduce_dict(loss_dict) 
#             loss_value = sum(loss_dict_reduced.values())  
#             if not math.isfinite(loss_value):
#                 print("Loss is {}, stopping training".format(loss_value))   
#                 print(loss_dict_reduced)
#                 sys.exit(1)  
   
#             metric_logger.update(loss=loss_value, **loss_dict_reduced)    
#             metric_logger.update(lr=optimizer.param_groups[0]["lr"])   

#             if writer and dist_utils.is_main_process() and global_step % 10 == 0:  
#                 writer.add_scalar('Loss/total', loss_value.item(), global_step)
#                 writer.add_scalar('Distill/Decay', distill_decay, global_step)
#                 for j, pg in enumerate(optimizer.param_groups):
#                     writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
#                 for k, v in loss_dict_reduced.items():
#                     writer.add_scalar(f'Loss/{k}', v.item(), global_step) 

#     metric_logger.synchronize_between_processes()
#     logger.info(f'Averaged stats:{metric_logger}')
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.   
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""     


import os, sys   
import math
import json  
import gc 
import numpy as np
from typing import Iterable
from tqdm import tqdm   
from tidecv import TIDE, datasets
    
import torch 
import torch.amp  
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from .density_metrics import DensityCountRMSE, DensityKL, DensityNSS
from ..deim.utils import DensityMapGenerator
from ..optim import ModelEMA, Warmup     
from ..data import CocoEvaluator
from ..misc import MetricLogger, MetricLogger_progress, SmoothedValue, dist_utils, plot_sample
from ..misc.modality_utils import normalize_tensor_minmax_per_sample    
from ..logger_module import get_logger
from ..extre_module.ops import Profile   
from ..extre_module.utils import TQDM, RANK
# from ..extre_module.yolo_metrice import get_yolo_det_metrice, get_yolo_seg_metrice
from ..deim.utils import coco_evaluator_per_class
from .sample_adapter import (
    move_samples_to_device,  
    select_model_input_for_model,    
    select_plot_samples_for_logging,  
)  

from pycocoeval.yoloeval import get_yolo_det_metrice, get_yolo_seg_metrice
     
CLEAR_MEMORY_STEP = 100
TIME_DEBUG = False     
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
logger = get_logger(__name__)
     
 
def _plot_training_modalities(samples, targets, data_loader, output_dir, epoch):     
    modality_samples = select_plot_samples_for_logging(samples, keys=("rgb", "npy")) 
    is_multimodal_plot = len(modality_samples) > 1    
     
    for modality, plot_samples in modality_samples:
        if modality == "npy":    
            plot_samples = normalize_tensor_minmax_per_sample(plot_samples)    
        if modality == "rgb" and not is_multimodal_plot: 
            suffix = ""
        else:
            suffix = f"_{modality}"
        save_path = output_dir / f"train_batch_{epoch}{suffix}.png"    
        if data_loader.dataset.remap_mscoco_category:  
            plot_sample((plot_samples, targets), data_loader.dataset.category2name, save_path, data_loader.dataset.label2category)     
        else:    
            plot_sample((plot_samples, targets), data_loader.dataset.category2name, save_path)  

# 训练单个 epoch
# self_lr_scheduler: 是否使用自定义学习率调度器
# lr_scheduler: 学习率调度器实例    
# model: 训练的 PyTorch 模型
# criterion: 损失计算函数   
# data_loader: 训练数据加载器     
# optimizer: 优化器     
# device: 训练设备（CPU 或 GPU）
# epoch: 当前 epoch 计数
# max_norm: 梯度裁剪的最大范数
# **kwargs: 其他参数，例如日志记录等    
def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()  # 设置模型为训练模式
    criterion.train()  # 设置损失函数为训练模式    

    print_freq = kwargs.get('print_freq', 10)  # 日志打印频率  
    writer: SummaryWriter = kwargs.get('writer', None)  # TensorBoard 记录器 
    ema: ModelEMA = kwargs.get('ema', None)  # 指数移动平均模型  
    scaler: GradScaler = kwargs.get('scaler', None)  # 混合精度训练的梯度缩放器 
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)  # 预热学习率调度器    
    plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)
    output_dir = kwargs.get('output_dir', None)     
    epoches = kwargs.get('epoches', -1) # 总的训练次数
    verbose_type = kwargs.get('verbose_type', 'origin') # 显示方式 
    header = 'Epoch: {}/{}'.format(epoch, epoches)  # 训练过程的日志标题

    cur_iters = epoch * len(data_loader)  # 计算当前 epoch 的起始迭代数    
    
    if verbose_type == 'origin':     
        metric_logger = MetricLogger(delimiter="  ")  # 记录训练过程中的度量信息
    else:
        metric_logger = MetricLogger_progress(delimiter="  ")  # 记录训练过程中的度量信息     
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率变化
    pbar = enumerate(metric_logger.log_every(data_loader, print_freq if verbose_type == 'origin' else 1, header))    
   
    dt = [     
        Profile(device=device), 
        Profile(device=device),
        Profile(device=device),
        Profile(device=device),  
        Profile(device=device)
    ] 

    for i, (samples, targets) in pbar:    
        if i % CLEAR_MEMORY_STEP == 0:
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()

        if epoch % plot_train_batch_freq == 0 and i == 0:
            _plot_training_modalities(samples, targets, data_loader, output_dir, epoch)
        with dt[0]:  
            samples = move_samples_to_device(samples, device, non_blocking=True)  # 将输入数据移动到指定设备   
            model_inputs = select_model_input_for_model(samples, model=model, key='rgb')
            targets = move_samples_to_device(targets, device, non_blocking=True)  # 目标数据也移动到设备
    
        global_step = epoch * len(data_loader) + i  # 计算全局训练步数
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))  # 训练元数据     

        # 使用混合精度训练
        if scaler is not None:     
            with dt[1]:
                with torch.autocast(device_type=str(device), cache_enabled=True):  
                    outputs = model(model_inputs, targets=targets)
            
            # 处理异常情况，避免 NaN 或 Inf 影响训练
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():     
                # print(outputs['pred_boxes'])     
                logger.warning(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}   
                for key, value in model.state_dict().items(): 
                    new_key = key.replace('module.', '')  # 兼容多 GPU 训练的情况   
                    state[new_key] = value  
                new_state['model'] = state    
                dist_utils.save_on_master(new_state, "./NaN.pth")  # 保存异常模型状态   
            
            with dt[2]:     
            # 计算损失
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)   
                loss = sum(loss_dict.values())  # 总损失
 
            with dt[3]:
                scaler.scale(loss).backward()  # 反向传播
                
                # 进行梯度裁剪（如果 max_norm > 0）
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)     
  
                scaler.step(optimizer)  # 更新参数    
                scaler.update()  # 更新梯度缩放因子
                optimizer.zero_grad()  # 清空梯度
   
        else:
            with dt[1]:
                outputs = model(model_inputs, targets=targets)  # 前向传播     
            with dt[2]:    
                loss_dict = criterion(outputs, targets, **metas)  # 计算损失     
                loss: torch.Tensor = sum(loss_dict.values())  # 总损失 
            with dt[3]:
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播 
                
                # 进行梯度裁剪
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()  # 更新参数 
   
        with dt[4]:
            # 更新 EMA（指数移动平均）   
            if ema is not None:
                ema.update(model)     

            # 更新学习率
            if self_lr_scheduler:     
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)    
            else:     
                if lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()     

            # 计算损失并检查是否异常
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())    
            if not math.isfinite(loss_value):  
                # print("Loss is {}, stopping training".format(loss_value))    
                # print(loss_dict_reduced)    
                logger.warning("Loss is {}, stopping training".format(loss_value))
                logger.info(loss_dict_reduced)
                sys.exit(1)    

            # 记录日志   
            metric_logger.update(loss=loss_value, **loss_dict_reduced) 
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
   
            # 记录到 TensorBoard    
            if writer and dist_utils.is_main_process() and global_step % 10 == 0:   
                writer.add_scalar('Loss/total', loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)    
                for k, v in loss_dict_reduced.items():    
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)
 
    # 统计并打印训练结果
    metric_logger.synchronize_between_processes()   
    logger.info(f'Averaged stats:{metric_logger}')     
    if TIME_DEBUG: 
        time_data = [x.t / len(data_loader) for x in dt]   
        # print(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)     
        logger.debug(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  
    
 
@torch.no_grad()  
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, test_only=False, output_dir=None, yolo_metrice=False, other_platform_model=None):  
    # 评估函数，禁用梯度计算以减少内存占用并提高推理速度    
    if model is not None:
        model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    dc_rmse_metric = DensityCountRMSE()
    from .density_metrics import DensityKL
    from .density_metrics import DensityNSS

    density_kl_metric = DensityKL()
    density_nss_metric = DensityNSS()
     
    metric_logger = MetricLogger_progress(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))    
    header = 'Test:'  

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    # 获取 IoU 计算类型（如 'bbox' 或 'segm'）
    iou_types = coco_evaluator.iou_types  
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
     
    # 初始化时间记录器   
    dt = [
        Profile(device=device),
        Profile(device=device)  
    ]    

    # 遍历数据集进行评估    
    coco_det_pred_json, coco_seg_pred_json = [], []    
    for samples, targets in metric_logger.log_every(data_loader, 1, header):
        samples = move_samples_to_device(samples, device, non_blocking=True)  # 将样本数据移动到指定设备（如 GPU）
        model_inputs = select_model_input_for_model(samples, model=model, key='rgb')
        targets = move_samples_to_device(targets, device, non_blocking=True)  # 目标数据也移动到设备   
   
        if model is not None:
            with dt[0]:  
                outputs = model(model_inputs)  # 前向传播，获取模型输出     

                if 'pred_densitymap' in outputs and outputs['pred_densitymap'] is not None:

                    pred_density = outputs['pred_densitymap']
                    _, _, h, w = pred_density.shape

                    dmg = DensityMapGenerator([h, w])

                    for b in range(pred_density.shape[0]):

                        boxes = (
                            targets[b]["boxes"]
                            .cpu()
                            .numpy()
                            .copy()
                        )

                        boxes[:, [0,2]] *= w
                        boxes[:, [1,3]] *= h

                        gt_densitymap = torch.from_numpy(
                            dmg.generate_from_boxes(
                                boxes,
                                method='gaussian'
                            )
                        ).float().to(pred_density.device)

                        gt_densitymap /= (
                            gt_densitymap.max() + 1e-6
                        )

                        density_kl_metric.update(
                            pred_density[b].squeeze(),
                            gt_densitymap
                        )

                        density_nss_metric.update(
                            pred_density[b].squeeze(),
                            gt_densitymap
                        )

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # 获取原始目标尺寸  

            with dt[1]:  
                results = postprocessor(outputs, orig_target_sizes, for_eval=True)  # 通过后处理器处理模型输出   
        else:
            if 'onnx' in other_platform_model:
                with dt[0]:   
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # 获取原始目标尺寸
                    onnx_result = other_platform_model['onnx'].run(
                        output_names=None,   
                        input_feed={'images': model_inputs.cpu().detach().numpy(), "orig_target_sizes": orig_target_sizes.cpu().detach().numpy()}   
                    )

                    results = []  
                    if len(onnx_result) == 3:
                        labels, boxes, scores = onnx_result
                        for lab, box, sco in zip(labels, boxes, scores):
                            result = dict(labels=torch.from_numpy(lab), boxes=torch.from_numpy(box), scores=torch.from_numpy(sco))
                            results.append(result)   
                    elif len(onnx_result) == 4: 
                        labels, boxes, scores, masks = onnx_result
                        for lab, box, sco, mask in zip(labels, boxes, scores, masks):
                            result = dict(labels=torch.from_numpy(lab), boxes=torch.from_numpy(box), scores=torch.from_numpy(sco), masks=torch.from_numpy(mask)) 
                            results.append(result)
                        
            elif 'engine' in other_platform_model: 
                with dt[0]:
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # 获取原始目标尺寸  
                    output = other_platform_model['engine']({'images': model_inputs,     
                                                             'orig_target_sizes': orig_target_sizes.to(device)})
                    labels, boxes, scores, masks = output['labels'], output['boxes'], output['scores'], output.get('masks', None)     

                    results = []
                    if masks is None:    
                        for lab, box, sco in zip(labels, boxes, scores):
                            result = dict(labels=lab, boxes=box, scores=sco)
                            results.append(result)
                    else:   
                        for lab, box, sco, mask in zip(labels, boxes, scores, masks):
                            result = dict(labels=lab, boxes=box, scores=sco, masks=mask)    
                            results.append(result)
  
        res = {target['image_id'].item(): output for target, output in zip(targets, results)} # 将评估结果与图像 ID 关联
        if coco_evaluator is not None:     
            coco_evaluator.update(res) # 更新 COCO 评估器
            coco_det_pred_json.extend(list(coco_evaluator.coco_eval['bbox'].cocoDt.anns.values()))     
            if 'segm' in coco_evaluator.coco_eval:
                coco_seg_pred_json.extend(list(coco_evaluator.coco_eval['segm'].cocoDt.anns.values()))   

    # gather the stats from all processes 在多进程环境下同步评估数据
    metric_logger.synchronize_between_processes()   
    dc_rmse = dc_rmse_metric.compute()
    density_kl = density_kl_metric.compute()

    density_nss = density_nss_metric.compute()

    density_dds = 1.0 / (1.0 + density_kl)

    logger.info(
        GREEN +
        f"Density Count RMSE: {dc_rmse:.4f}" +
        RESET
    )
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()     
    
    # 统计耗时   
    if test_only:
        if model is not None:   
            speed = dict(zip(['inference', 'postprocess'], (x.t / len(data_loader.dataset) * 1e3 for x in dt))) 
            logger.info(GREEN + f'Test On BatchSize:{data_loader.batch_size}' + RESET)
            logger.info(GREEN + f"Speed: {speed['inference']:.4f}ms inference, {speed['postprocess']:.4f}ms postprocess per image" + RESET)
            logger.info(GREEN + f"FPS(inference+postprocess): {1000 / (speed['inference'] + speed['postprocess']):.2f}" + RESET)  
        else: 
            inference_speed = dt[0].t / len(data_loader.dataset) * 1e3
            logger.info(GREEN + f'Test On BatchSize:{data_loader.batch_size}' + RESET)   
            logger.info(GREEN + f"Speed: {inference_speed:.4f}ms inference per image" + RESET)
            logger.info(GREEN + f"FPS(inference): {1000 / inference_speed:.2f}" + RESET)
  
    if yolo_metrice: 
        get_yolo_det_metrice(logger, coco_evaluator, coco_det_pred_json, output_dir if test_only else None) 
        if 'segm' in coco_evaluator.coco_eval:
            get_yolo_seg_metrice(logger, coco_evaluator, coco_seg_pred_json, save_vis=False)
   

    # Density RMSE
    dc_rmse = dc_rmse_metric.compute()

    logger.info(
        RED +
        "------------------------ Density Metrics Start ------------------------"
        + RESET
    )

    logger.info(
        GREEN +
        f"Density Count RMSE: {dc_rmse:.4f}"
        + RESET
    )

    logger.info(
        RED +
        "------------------------- Density Metrics End -------------------------"
        + RESET
    )

    logger.info(
        RED +
        "------------------------ Density Metrics Start ------------------------"
        + RESET
    )

    logger.info(
        GREEN +
        f"Density KL Divergence: {density_kl:.6f}"
        + RESET
    )

    logger.info(
        GREEN +
        f"Density DDS: {density_dds:.6f}"
        + RESET
    )

    # accumulate predictions from all images 累积并计算最终评估结果   
    if coco_evaluator is not None:  
        logger.info(RED + "------------------------ COCO Metrice Start ------------------------" + RESET) 
        coco_evaluator.accumulate()
        coco_evaluator.summarize()  
        if test_only:     
            logger.info(ORANGE + f"Saving coco pred[{output_dir / 'pred.json'}] json..." + RESET)
            with open(output_dir / 'pred_bbox.json', 'w') as f:    
                json.dump(coco_det_pred_json, f)    
            # if 'segm' in coco_evaluator.coco_eval:
            #     with open(output_dir / 'pred_segm.json', 'w') as f:
            #         json.dump(coco_seg_pred_json, f)
            logger.info(ORANGE + "save success." + RESET)
            
            for iouType in coco_evaluator.coco_gt:
                model_metrice_table = coco_evaluator_per_class(coco_evaluator, iouType)
                print(ORANGE, model_metrice_table, RESET) 
 
            try:   
                logger.info(RED + "------------------------ TIDE Metrice Start ------------------------" + RESET)
                tide = TIDE()     
                tide.evaluate_range(datasets.COCO(data_loader.dataset.ann_file), datasets.COCOResult(output_dir / 'pred_bbox.json'))     
                tide.summarize()     
                tide.plot(out_dir=output_dir / 'tide_result')
            except Exception as e:   
                logger.error(RED, 'TIDE failure... skip message:', e, RESET)
                logger.warning('------------------------ TIDE指标生成报错可以不用管 ------------------------')   
 
    stats = {}
    stats['dc_rmse'] = float(dc_rmse)
    stats['density_kl'] = float(density_kl)

    stats['density_nss'] = float(density_nss)
    if coco_evaluator is not None: 
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()  
        if 'segm' in iou_types:     
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()  
     
    return stats, coco_evaluator

def distill_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, teahcer_model: torch.nn.Module, student_featureExt, teacher_featureExt,
                    criterion: torch.nn.Module, feature_distill_criterion, logical_distill_criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()  # 设置模型为训练模式
    teahcer_model.train() # 设置模型为训练模式 
    criterion.train()  # 设置损失函数为训练模式

    print_freq = kwargs.get('print_freq', 10)  # 日志打印频率
    writer: SummaryWriter = kwargs.get('writer', None)  # TensorBoard 记录器
    ema: ModelEMA = kwargs.get('ema', None)  # 指数移动平均模型 
    scaler: GradScaler = kwargs.get('scaler', None)  # 混合精度训练的梯度缩放器 
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)  # 预热学习率调度器   
    plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)
    output_dir = kwargs.get('output_dir', None)     
    epoches = kwargs.get('epoches', -1) # 总的训练次数
    verbose_type = kwargs.get('verbose_type', 'origin') # 显示方式 
    feature_loss_ratio = kwargs.get('feature_loss_ratio', 1.0)
    logical_loss_ratio = kwargs.get('logical_loss_ratio', 1.0)   
    distill_loss_decay = kwargs.get('distill_loss_decay', 'constant')
    header = 'Epoch: {}/{}'.format(epoch, epoches)  # 训练过程的日志标题    

    cur_iters = epoch * len(data_loader)  # 计算当前 epoch 的起始迭代数  
   
    if verbose_type == 'origin':     
        metric_logger = MetricLogger(delimiter="  ")  # 记录训练过程中的度量信息   
    else:    
        metric_logger = MetricLogger_progress(delimiter="  ")  # 记录训练过程中的度量信息    
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率变化  
    pbar = enumerate(metric_logger.log_every(data_loader, print_freq if verbose_type == 'origin' else 1, header))    

    dt = [    
        Profile(device=device), 
        Profile(device=device), 
        Profile(device=device), 
        Profile(device=device),  
        Profile(device=device)
    ]  
   
    for i, (samples, targets) in pbar:    
        if i % CLEAR_MEMORY_STEP == 0:     
            if torch.cuda.is_available():
                torch.cuda.empty_cache()    

        # -------------- 蒸馏损失的调度因子  可视化文件在tools/visualization/distill_decay_visual.py内
        if distill_loss_decay == 'constant':
            # 特点：蒸馏损失权重保持不变
            # 适用场景：希望蒸馏损失在整个训练过程中保持恒定影响
            distill_decay = 1.0
        elif distill_loss_decay == 'cosine':
            # 特点：在每个epoch内进行余弦衰减，epoch间重置
            # 衰减曲线：平滑的余弦曲线，先快后慢  
            # 适用场景：希望在每个epoch内逐渐减少蒸馏损失的影响     
            eta_min, base_ratio, T_max = 0.01, 1.0, 10
            distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
        elif distill_loss_decay == 'linear':
            # 特点：在每个epoch内进行线性衰减
            # 衰减曲线：均匀的线性下降 
            # 适用场景：希望蒸馏损失在epoch内均匀递减     
            distill_decay = ((1 - math.cos(i * math.pi / len(data_loader))) / 2) * (0.01 - 1) + 1     
        elif distill_loss_decay == 'cosine_epoch':
            # 特点：跨epoch的连续余弦衰减
            # 衰减曲线：整个训练过程的平滑余弦衰减
            # 适用场景：希望蒸馏损失在整个训练过程中平滑递减
            eta_min, base_ratio, T_max = 0.01, 1.0, 10   
            distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * (cur_iters + i) / T_max)) / 2 
        elif distill_loss_decay == 'linear_epoch': 
            # 特点：跨epoch的连续线性衰减     
            # 衰减曲线：整个训练过程的均匀线性下降
            # 适用场景：希望蒸馏损失在整个训练过程中均匀递减  
            distill_decay = ((1 - math.cos((cur_iters + i) * math.pi / (epoches * len(data_loader)))) / 2) * (0.01 - 1) + 1 
     
        if epoch % plot_train_batch_freq == 0 and i == 0:  
            _plot_training_modalities(samples, targets, data_loader, output_dir, epoch) 
        with dt[0]:
            samples = move_samples_to_device(samples, device, non_blocking=True)  # 将输入数据移动到指定设备     
            model_inputs = select_model_input_for_model(samples, model=model, key='rgb')     
            targets = move_samples_to_device(targets, device, non_blocking=True)  # 目标数据也移动到设备  
        
        global_step = epoch * len(data_loader) + i  # 计算全局训练步数  
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))  # 训练元数据     

        if feature_distill_criterion:  
            student_featureExt.clear_features() 
            teacher_featureExt.clear_features()
  
        with dt[1]:     
            outputs = model(model_inputs, targets=targets)  # 前向传播
            if feature_distill_criterion or logical_distill_criterion:  
                with torch.no_grad():
                    teacher_outputs = teahcer_model(model_inputs, targets=targets)

        with dt[2]:     
            loss_dict = criterion(outputs, targets, **metas)  # 计算损失
  
            if feature_distill_criterion:   
                feature_distill_loss = feature_distill_criterion(student_featureExt.get_features_in_order(), teacher_featureExt.get_features_in_order()) * feature_loss_ratio * distill_decay
                loss_dict['fea_loss'] = feature_distill_loss
            else: 
                loss_dict['fea_loss'] = torch.zeros(1, device=device)  
            
            if logical_distill_criterion:
                logical_distill_loss = logical_distill_criterion(outputs, teacher_outputs, targets) * logical_loss_ratio * distill_decay     
                loss_dict['log_loss'] = logical_distill_loss
            else:
                loss_dict['log_loss'] = torch.zeros(1, device=device)

            loss: torch.Tensor = sum(loss_dict.values())  # 总损失

        with dt[3]:
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
  
            # 进行梯度裁剪  
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()  # 更新参数  

        with dt[4]:
            # 更新 EMA（指数移动平均） 
            if ema is not None:
                ema.update(model) 
    
            # 更新学习率   
            if self_lr_scheduler:     
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
            else:
                if lr_warmup_scheduler is not None:     
                    lr_warmup_scheduler.step()
    
            # 计算损失并检查是否异常  
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)   
            loss_value = sum(loss_dict_reduced.values())
            if not math.isfinite(loss_value):   
                print("Loss is {}, stopping training".format(loss_value)) 
                print(loss_dict_reduced)   
                sys.exit(1)
     
            # 记录日志  
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])  
    
            # 记录到 TensorBoard
            if writer and dist_utils.is_main_process() and global_step % 10 == 0:     
                writer.add_scalar('Loss/total', loss_value.item(), global_step)     
                writer.add_scalar('Distill/Decay', distill_decay, global_step)     
                for j, pg in enumerate(optimizer.param_groups):  
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)  
    
    # 统计并打印训练结果  
    metric_logger.synchronize_between_processes()
    logger.info(f'Averaged stats:{metric_logger}') 
    if TIME_DEBUG:
        time_data = [x.t / len(data_loader) for x in dt]  
        print(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)  
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
