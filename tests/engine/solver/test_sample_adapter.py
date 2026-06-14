import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.sample_adapter import (
    move_samples_to_device,
    select_model_input,
    select_model_input_for_model,
    select_plot_samples,
    select_plot_samples_for_logging,
)


def test_move_samples_to_device_tensor():
    samples = torch.randn(2, 3, 8, 8)
    out = move_samples_to_device(samples, torch.device("cpu"))
    assert isinstance(out, torch.Tensor)
    assert out.shape == samples.shape


def test_move_samples_to_device_dict():
    samples = {
        "rgb": torch.randn(2, 3, 8, 8),
        "npy": torch.randn(2, 1, 8, 8),
    }
    out = move_samples_to_device(samples, torch.device("cpu"))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"rgb", "npy"}
    assert out["rgb"].shape == (2, 3, 8, 8)
    assert out["npy"].shape == (2, 1, 8, 8)


def test_select_model_input_prefers_rgb_for_multimodal_dict():
    samples = {
        "rgb": torch.randn(2, 3, 8, 8),
        "npy": torch.randn(2, 1, 8, 8),
    }
    model_input = select_model_input(samples, key="rgb")
    assert model_input is samples["rgb"]


def test_select_model_input_raises_when_key_missing():
    samples = {"depth": torch.randn(2, 1, 8, 8)}
    with pytest.raises(KeyError, match="rgb"):
        select_model_input(samples, key="rgb")


def test_select_plot_samples_returns_rgb_branch():
    samples = {
        "rgb": torch.randn(2, 3, 8, 8),
        "npy": torch.randn(2, 1, 8, 8),
    }
    plot_samples = select_plot_samples(samples, key="rgb")
    assert plot_samples is samples["rgb"]


def test_select_plot_samples_for_logging_returns_rgb_and_npy_for_multimodal():
    samples = {
        "rgb": torch.randn(2, 3, 8, 8),
        "npy": torch.randn(2, 1, 8, 8),
    }
    out = select_plot_samples_for_logging(samples, keys=("rgb", "npy"))
    assert [k for k, _ in out] == ["rgb", "npy"]
    assert out[0][1] is samples["rgb"]
    assert out[1][1] is samples["npy"]


def test_select_plot_samples_for_logging_tensor_keeps_legacy_single_plot():
    samples = torch.randn(2, 3, 8, 8)
    out = select_plot_samples_for_logging(samples, keys=("rgb", "npy"))
    assert len(out) == 1
    assert out[0][0] == "rgb"
    assert out[0][1] is samples


class GetRGBModalityTensor(nn.Module):
    def forward(self, x):
        return x["rgb"]


class _MultimodalModelStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Identity(), GetRGBModalityTensor())

    def forward(self, x):
        return x


class _SingleModalModelStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()

    def forward(self, x):
        return x


def test_select_model_input_for_model_returns_dict_for_multimodal_model():
    samples = {
        "rgb": torch.randn(2, 3, 8, 8),
        "npy": torch.randn(2, 1, 8, 8),
    }
    model = _MultimodalModelStub()
    model_input = select_model_input_for_model(samples, model, key="rgb")
    assert model_input is samples


def test_select_model_input_for_model_returns_rgb_for_single_modal_model():
    samples = {
        "rgb": torch.randn(2, 3, 8, 8),
        "npy": torch.randn(2, 1, 8, 8),
    }
    model = _SingleModalModelStub()
    model_input = select_model_input_for_model(samples, model, key="rgb")
    assert model_input is samples["rgb"]
