"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import argparse, thop
from calflops import calculate_flops
from engine.core import YAMLConfig
from engine.logger_module import get_logger

import torch
import torch.nn as nn

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


class _TensorInputDict(dict):
    """Dict input wrapper compatible with profilers that call `.to(device)` on each arg."""

    @staticmethod
    def _move(value, device):
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _TensorInputDict._move(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [_TensorInputDict._move(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(_TensorInputDict._move(v, device) for v in value)
        return value

    def to(self, device):
        return _TensorInputDict({k: self._move(v, device) for k, v in self.items()})


def _unwrap_profile_target(model: nn.Module) -> nn.Module:
    core_model = model.module if hasattr(model, "module") else model
    wrapped_model = getattr(core_model, "model", None)
    if isinstance(wrapped_model, nn.Module):
        return wrapped_model
    return core_model


def _is_multimodal_profile_model(model: nn.Module) -> bool:
    core_model = _unwrap_profile_target(model)
    backbone = getattr(core_model, "backbone", None)
    if backbone is None or not hasattr(backbone, "children"):
        return False

    for module in list(backbone.children())[:4]:
        if module.__class__.__name__ in {"GetRGBModalityTensor", "GetNPYModalityTensor"}:
            return True
    return False


def _build_profile_inputs(input_shape, device, multimodal):
    b, _, h, w = input_shape
    if multimodal:
        return (_TensorInputDict({
            "rgb": torch.randn(size=(b, 3, h, w), device=device),
            "npy": torch.randn(size=(b, 1, h, w), device=device),
        }),)
    return (torch.randn(size=input_shape, device=device),)


def profile_model_complexity(model: nn.Module, input_shape, device, print_detailed=True):
    is_multimodal_model = _is_multimodal_profile_model(model)
    profile_inputs = _build_profile_inputs(input_shape, device, is_multimodal_model)
    flops, macs = None, None

    try:
        if is_multimodal_model:
            flops, macs, _ = calculate_flops(
                model=model,
                input_shape=None,
                args=list(profile_inputs),
                kwargs={},
                output_as_string=True,
                output_precision=4,
                print_detailed=print_detailed,
            )
        else:
            flops, macs, _ = calculate_flops(
                model=model,
                input_shape=input_shape,
                output_as_string=True,
                output_precision=4,
                print_detailed=print_detailed,
            )
    except Exception:
        logger.warning(RED + "calculate_flops failed.. using thop instead.." + RESET)

    if flops is None or macs is None:
        macs_val = thop.profile(model, inputs=profile_inputs, verbose=False)[0]
        macs, flops = thop.clever_format([macs_val, macs_val * 2], format="%.3f")

    params = sum(p.numel() for p in model.parameters())
    return flops, macs, params


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=None)
    class Model_for_flops(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            outputs = self.model(images)
            return outputs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model_for_flops().eval().to(device)
    input_shape = (1, 3, *cfg.yaml_cfg['eval_spatial_size'])

    flops, macs, params = profile_model_complexity(
        model=model,
        input_shape=input_shape,
        device=device,
        print_detailed=True,
    )
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default= "configs/dfine/dfine_hgnetv2_l_coco.yml", type=str)
    args = parser.parse_args()

    main(args)
