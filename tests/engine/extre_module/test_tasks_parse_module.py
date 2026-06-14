import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.extre_module.tasks as tasks


class _FakeFeatureInfo:
    def channels(self):
        return [48, 96]


class _FakeTimmBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = _FakeFeatureInfo()


def test_parse_module_timm_string_defaults_pretrained_false_when_args_empty(monkeypatch):
    called = {}

    def _fake_create_model(name, pretrained, features_only):
        called["name"] = name
        called["pretrained"] = pretrained
        called["features_only"] = features_only
        return _FakeTimmBackbone()

    monkeypatch.setattr(tasks.timm, "create_model", _fake_create_model)

    module, channels, _, _ = tasks.parse_module(
        d={},
        i=0,
        f=-1,
        m="mobilenetv4_conv_small_050",
        args=[],
        ch=[3],
    )

    assert isinstance(module, _FakeTimmBackbone)
    assert channels == [48, 96]
    assert called["name"] == "mobilenetv4_conv_small_050"
    assert called["pretrained"] is False
    assert called["features_only"] is True
