from types import SimpleNamespace

import torch

import engine.solver.det_engine as det_engine


def _build_loader(remap=False):
    dataset = SimpleNamespace(
        remap_mscoco_category=remap,
        category2name={0: "obj"},
        label2category={0: 0},
    )
    return SimpleNamespace(dataset=dataset)


def test_plot_training_modalities_multimodal_uses_rgb_suffix(monkeypatch, tmp_path):
    calls = []

    def _fake_plot_sample(data, category2name, output_dir, coco2category=None):
        calls.append(str(output_dir))

    monkeypatch.setattr(det_engine, "plot_sample", _fake_plot_sample)

    samples = {
        "rgb": torch.rand(2, 3, 16, 16),
        "npy": torch.rand(2, 1, 16, 16),
    }
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]

    det_engine._plot_training_modalities(samples, targets, _build_loader(remap=False), tmp_path, epoch=3)

    assert calls == [
        str(tmp_path / "train_batch_3_rgb.png"),
        str(tmp_path / "train_batch_3_npy.png"),
    ]


def test_plot_training_modalities_single_modal_keeps_legacy_name(monkeypatch, tmp_path):
    calls = []

    def _fake_plot_sample(data, category2name, output_dir, coco2category=None):
        calls.append(str(output_dir))

    monkeypatch.setattr(det_engine, "plot_sample", _fake_plot_sample)

    samples = torch.rand(2, 3, 16, 16)
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]

    det_engine._plot_training_modalities(samples, targets, _build_loader(remap=False), tmp_path, epoch=4)

    assert calls == [str(tmp_path / "train_batch_4.png")]


def test_plot_training_modalities_normalizes_npy_before_plot(monkeypatch, tmp_path):
    captured = {}

    def _fake_plot_sample(data, category2name, output_dir, coco2category=None):
        images, _ = data
        captured[str(output_dir)] = images.clone()

    monkeypatch.setattr(det_engine, "plot_sample", _fake_plot_sample)

    samples = {
        "rgb": torch.rand(2, 3, 16, 16),
        "npy": torch.tensor(
            [
                [[[0.0, 5.0], [10.0, 20.0]]],
                [[[7.0, 7.0], [7.0, 7.0]]],
            ],
            dtype=torch.float32,
        ),
    }
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]

    det_engine._plot_training_modalities(samples, targets, _build_loader(remap=False), tmp_path, epoch=5)

    npy_out = captured[str(tmp_path / "train_batch_5_npy.png")]
    assert float(npy_out.min()) == 0.0
    assert float(npy_out.max()) == 1.0
    assert torch.allclose(npy_out[1], torch.zeros_like(npy_out[1]))
