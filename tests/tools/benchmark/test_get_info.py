import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.benchmark.get_info as get_info


class GetRGBModalityTensor(nn.Module):
    def forward(self, x):
        return x["rgb"]


class GetNPYModalityTensor(nn.Module):
    def forward(self, x):
        return x["npy"]


class TinyMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Identity(),
            GetRGBModalityTensor(),
            GetNPYModalityTensor(),
        )

    def forward(self, x):
        return x["rgb"]


class TinySingleModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Identity(), nn.Conv2d(3, 4, 1))

    def forward(self, x):
        return self.backbone(x)


def test_profile_model_complexity_uses_calflops_args_for_multimodal(monkeypatch):
    called = {"input_shape": None, "args": None}

    def _fake_calculate_flops(*, model, input_shape=None, args=None, kwargs=None, **_):
        called["input_shape"] = input_shape
        called["args"] = args
        return "1.0G", "0.5G", "1.0M"

    monkeypatch.setattr(get_info, "calculate_flops", _fake_calculate_flops)

    model = TinyMultimodalModel()
    flops, macs, params = get_info.profile_model_complexity(
        model=model,
        input_shape=(1, 3, 16, 16),
        device=torch.device("cpu"),
        print_detailed=False,
    )

    assert flops == "1.0G"
    assert macs == "0.5G"
    assert params >= 0
    assert called["input_shape"] is None
    assert isinstance(called["args"], list) and len(called["args"]) == 1
    mm_input = called["args"][0]
    assert isinstance(mm_input, dict)
    assert set(mm_input.keys()) == {"rgb", "npy"}
    assert mm_input["rgb"].shape[1] == 3
    assert mm_input["npy"].shape[1] == 1


def test_profile_model_complexity_keeps_input_shape_for_single_modal(monkeypatch):
    called = {"input_shape": None}

    def _fake_calculate_flops(*, model, input_shape=None, args=None, kwargs=None, **_):
        called["input_shape"] = input_shape
        assert args is None
        return "2.0G", "1.0G", "2.0M"

    monkeypatch.setattr(get_info, "calculate_flops", _fake_calculate_flops)

    model = TinySingleModalModel()
    flops, macs, params = get_info.profile_model_complexity(
        model=model,
        input_shape=(1, 3, 16, 16),
        device=torch.device("cpu"),
        print_detailed=False,
    )

    assert flops == "2.0G"
    assert macs == "1.0G"
    assert params > 0
    assert called["input_shape"] == (1, 3, 16, 16)
