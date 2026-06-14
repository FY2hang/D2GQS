import sys
from pathlib import Path

import torch
import torch.nn as nn
import engine.misc.profiler_utils as profiler_utils

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.misc.profiler_utils import stats


class GetRGBModalityTensor(nn.Module):
    def forward(self, x):
        return x["rgb"]


class TinyMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(GetRGBModalityTensor(), nn.Conv2d(3, 4, kernel_size=1))

    def forward(self, x):
        return self.backbone(x)

    def deploy(self):
        return self


class _CfgStub:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"
        self.yaml_cfg = {"eval_spatial_size": [16, 16]}


def test_stats_supports_multimodal_dict_input_model():
    cfg = _CfgStub(TinyMultimodalModel())
    n_parameters, model_stats = stats(cfg)
    assert n_parameters > 0
    assert isinstance(model_stats, (dict, set))
    if isinstance(model_stats, dict):
        assert any("Model FLOPs" in k for k in model_stats.keys())
    else:
        assert any("Model FLOPs" in item for item in model_stats)


def test_stats_uses_calculate_flops_for_multimodal_model(monkeypatch):
    cfg = _CfgStub(TinyMultimodalModel())
    called = {"calculate_flops": False}

    def _fake_calculate_flops(*, model, input_shape=None, args=None, kwargs=None, **_):
        called["calculate_flops"] = True
        assert input_shape is None
        assert isinstance(args, list) and len(args) == 1
        mm_input = args[0]
        assert isinstance(mm_input, dict)
        assert set(mm_input.keys()) == {"rgb", "npy"}
        assert mm_input["rgb"].shape[1] == 3
        assert mm_input["npy"].shape[1] == 1
        return "1.0G", "0.5G", "1.0M"

    monkeypatch.setattr(profiler_utils, "calculate_flops", _fake_calculate_flops)

    n_parameters, model_stats = profiler_utils.stats(cfg)
    assert n_parameters > 0
    assert called["calculate_flops"] is True
    if isinstance(model_stats, dict):
        assert any("Model FLOPs:1.0G" in k for k in model_stats.keys())
    else:
        assert any("Model FLOPs:1.0G" in item for item in model_stats)
