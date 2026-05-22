import numpy as np

from versa.definition import MetricRegistry
from versa.utterance_metrics import multigauss
from versa.utterance_metrics.multigauss import (
    MultiGaussMetric,
    register_multigauss_metric,
)


def test_multigauss_metric_class_returns_expected_keys(monkeypatch):
    expected = {
        "multigauss_mos": 1.0,
        "multigauss_noi": 2.0,
        "multigauss_col": 3.0,
        "multigauss_dis": 4.0,
        "multigauss_loud": 5.0,
        "multigauss_covariance": np.eye(5),
    }
    monkeypatch.setattr(
        multigauss,
        "multigauss_model_setup",
        lambda model_tag="probabilistic", cache_dir="versa_cache", use_gpu=False: {
            "model_tag": model_tag
        },
    )
    monkeypatch.setattr(
        multigauss,
        "multigauss_metric",
        lambda model, pred_x, fs: expected,
    )

    metric = MultiGaussMetric({"model_tag": "probabilistic"})
    result = metric.compute(
        np.zeros(16000, dtype=np.float32), metadata={"sample_rate": 16000}
    )

    assert result is expected


def test_register_multigauss_metric():
    registry = MetricRegistry()

    register_multigauss_metric(registry)

    assert registry.get_metric("multigauss") is MultiGaussMetric
    assert registry.get_metric("multi_gauss") is MultiGaussMetric
    assert registry.get_metadata("multigauss").requires_reference is False
