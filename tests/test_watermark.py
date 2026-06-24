"""Golden watermark tests: lock the deterministic signature->filter mapping.

These exercise the architecture registry end to end and require torch/torcheeg,
so they skip automatically where those are not installed (e.g. lint-only CI).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torcheeg")

import rsa  # noqa: E402
from eegwm.watermark.triggerset import (  # noqa: E402
    get_watermark,
    apply_true_embedding,
    Verifier,
)

ARCHITECTURES = ["CCNN", "EEGNet", "TSCeption"]


@pytest.fixture(scope="module")
def private_key():
    _, key = rsa.newkeys(1024)
    return key


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_watermark_is_deterministic(arch, private_key):
    f1, l1 = get_watermark(arch, Verifier.CORRECT.value, private_key, 16)
    f2, l2 = get_watermark(arch, Verifier.CORRECT.value, private_key, 16)
    assert torch.equal(f1, f2)
    assert l1 == l2
    assert f1.dtype == torch.float32
    assert 0 <= l1 < 16


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_true_embedding_relabels_sample(arch, private_key):
    wm_filter, wm_label = get_watermark(arch, Verifier.CORRECT.value, private_key, 16)
    sample = torch.zeros_like(wm_filter)
    out, label = apply_true_embedding(sample, wm_filter, wm_label)
    assert label == wm_label
    assert out.shape == sample.shape
