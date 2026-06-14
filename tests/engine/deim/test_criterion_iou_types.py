import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.deim.box_ops import BBOX_IOU_TYPES, bbox_iou_aligned_xyxy
from engine.deim import deim_criterion as criterion_mod
from engine.deim.deim_criterion import DEIMCriterion


class _DummyMatcher(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("Not used in this unit test.")

    def set_epoch(self, epoch):
        return None


def _make_minimal_case():
    outputs = {
        "pred_boxes": torch.tensor(
            [[[0.50, 0.50, 0.40, 0.40], [0.25, 0.25, 0.20, 0.20]]],
            dtype=torch.float32,
            requires_grad=True,
        )
    }
    targets = [
        {
            "boxes": torch.tensor([[0.50, 0.50, 0.35, 0.35]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    indices = [(torch.tensor([0], dtype=torch.int64), torch.tensor([0], dtype=torch.int64))]
    return outputs, targets, indices


def test_bbox_iou_aligned_all_types_finite_and_shape():
    boxes1 = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    boxes2 = torch.tensor(
        [
            [0.5, 0.5, 2.5, 2.5],
            [2.0, 2.0, 3.0, 3.0],
            [1.2, 1.4, 2.8, 3.9],
        ],
        dtype=torch.float32,
    )

    for iou_type in BBOX_IOU_TYPES:
        iou = bbox_iou_aligned_xyxy(boxes1, boxes2, iou_type=iou_type)
        assert iou.shape == (3,)
        assert torch.isfinite(iou).all()

    bbox_iou_aligned_xyxy(boxes1, boxes2, iou_type="ciou").sum().backward()
    assert boxes1.grad is not None
    assert torch.isfinite(boxes1.grad).all()


def test_loss_boxes_dynamic_key_ciou():
    outputs, targets, indices = _make_minimal_case()
    criterion = DEIMCriterion(
        matcher=_DummyMatcher(),
        weight_dict={"loss_bbox": 5.0, "loss_ciou": 2.0},
        losses=["boxes"],
        boxes_iou_type="ciou",
    )

    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=1.0)
    assert "loss_bbox" in losses
    assert "loss_ciou" in losses
    assert "loss_giou" not in losses
    assert torch.isfinite(losses["loss_ciou"])

    (losses["loss_bbox"] + losses["loss_ciou"]).backward()
    assert outputs["pred_boxes"].grad is not None
    assert torch.isfinite(outputs["pred_boxes"].grad).all()


def test_get_loss_meta_info_extended_weight_format_siou():
    outputs, targets, indices = _make_minimal_case()
    criterion = DEIMCriterion(
        matcher=_DummyMatcher(),
        weight_dict={},
        losses=["vfl"],
        boxes_weight_format="siou",
    )

    meta_vfl = criterion.get_loss_meta_info("vfl", outputs, targets, indices)
    assert "values" in meta_vfl
    assert meta_vfl["values"].shape == (1,)
    assert torch.isfinite(meta_vfl["values"]).all()

    meta_boxes = criterion.get_loss_meta_info("boxes", outputs, targets, indices)
    assert "boxes_weight" in meta_boxes
    assert meta_boxes["boxes_weight"].shape == (1,)
    assert torch.isfinite(meta_boxes["boxes_weight"]).all()


def test_loss_config_log_only_once_on_set_epoch(monkeypatch):
    logged = []

    def _fake_info(msg):
        logged.append(str(msg))

    monkeypatch.setattr(criterion_mod.logger, "info", _fake_info)

    criterion = DEIMCriterion(
        matcher=_DummyMatcher(),
        weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0},
        losses=["boxes", "mal"],
        boxes_iou_type="giou",
    )

    criterion.set_epoch(0)
    criterion.set_epoch(1)
    criterion.set_epoch(2)

    config_logs = [m for m in logged if "Loss Config |" in m]
    assert len(config_logs) == 1
    assert "losses=['boxes', 'mal']" in config_logs[0]
    assert "weight_dict={'loss_bbox': 5.0, 'loss_giou': 2.0}" in config_logs[0]
    assert "boxes_iou_type=giou" in config_logs[0]
