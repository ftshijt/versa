import numpy as np
import pytest
import torch

from versa.definition import MetricCategory, MetricRegistry, MetricType
from versa.utterance_metrics import songeval


class DummySongEvalModel:
    def __call__(self, hidden):
        return torch.tensor([[3.12345, 4.0, 2.5, 1.25, 5.0]], dtype=torch.float32)


class DummyMuQ:
    def __call__(self, audio, output_hidden_states=True):
        assert output_hidden_states
        hidden_states = [torch.zeros((1, 2, 3), device=audio.device) for _ in range(7)]
        return {"hidden_states": hidden_states}


def test_songeval_metric_returns_stable_prefixed_keys():
    model_dict = {
        "device": "cpu",
        "model": DummySongEvalModel(),
        "muq": DummyMuQ(),
    }

    scores = songeval.songeval_metric(
        model_dict,
        np.ones(16000, dtype=np.float32),
        fs=16000,
    )

    assert scores == {
        "songeval_coherence": 3.1235,
        "songeval_musicality": 4.0,
        "songeval_memorability": 2.5,
        "songeval_clarity": 1.25,
        "songeval_naturalness": 5.0,
    }


def test_songeval_metric_validates_predictions(monkeypatch):
    monkeypatch.setattr(
        songeval,
        "songeval_model_setup",
        lambda **kwargs: {
            "device": "cpu",
            "model": DummySongEvalModel(),
            "muq": DummyMuQ(),
        },
    )

    metric = songeval.SongEvalMetric()

    with pytest.raises(ValueError, match="Predicted signal"):
        metric.compute(None, metadata={"sample_rate": 16000})


def test_songeval_registration_metadata():
    registry = MetricRegistry()

    songeval.register_songeval_metric(registry)
    metadata = registry.get_metadata("song_eval")

    assert metadata.name == "songeval"
    assert metadata.category == MetricCategory.INDEPENDENT
    assert metadata.metric_type == MetricType.DICT
    assert not metadata.requires_reference
    assert metadata.gpu_compatible
    assert not metadata.auto_install
    assert registry.get_metric("SongEval") is songeval.SongEvalMetric


def test_songeval_setup_reports_missing_optional_dependencies(monkeypatch):
    monkeypatch.setattr(songeval, "MuQ", None)

    with pytest.raises(ImportError, match="SongEval requires optional dependencies"):
        songeval.songeval_model_setup()
