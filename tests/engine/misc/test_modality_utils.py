import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.misc.modality_utils import normalize_tensor_minmax_per_sample


def test_normalize_tensor_minmax_per_sample_supports_non_contiguous_tensor():
    base = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
    cropped = base[:, 1:7, 1:7]
    assert cropped.is_contiguous() is False

    normalized = normalize_tensor_minmax_per_sample(cropped)

    assert normalized.shape == cropped.shape
    assert torch.isclose(normalized.min(), torch.tensor(0.0))
    assert torch.isclose(normalized.max(), torch.tensor(1.0))
