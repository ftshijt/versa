import pytest
import numpy as np
from pathlib import Path

from versa.corpus_metrics import fad as fad_module
from versa.corpus_metrics import individual_fad as individual_fad_module
from versa.corpus_metrics import kid as kid_module
from versa.definition import MetricCategory, MetricRegistry
from versa.scorer_shared import (
    VersaScorer,
    _create_populated_registry,
    load_score_modules,
)


class DummyFadScore:
    score = 0.25
    r2 = 0.75


class DummyFadBackend:
    def __init__(self, ml=None, load_model=True):
        self.ml = ml
        self.load_model = load_model
        self.cached = []

    def cache_embedding_file(self, key, path, cache_dir):
        self.cached.append((key, path, cache_dir))

    def score_inf(self, baseline_files, eval_files, cache_dir):
        return DummyFadScore()

    def score(self, baseline_files, eval_files, cache_dir):
        return 0.5


class DummyIndividualFadBackend(DummyFadBackend):
    def read_embedding_file(self, key, cache_dir):
        return {
            "ref1": np.array([[10.0]]),
            "ref2": np.array([[10.0]]),
            "utt1": np.array([[11.5]]),
            "utt2": np.array([[12.0]]),
        }[key]


class DummyKidBackend(DummyFadBackend):
    def score_kid(self, baseline_files, eval_files, cache_dir):
        return {"_mean": 0.1, "_std": 0.02}


def test_fad_metric_registers_and_preserves_output_keys(monkeypatch):
    monkeypatch.setattr(fad_module, "get_model", lambda embedding: f"model:{embedding}")
    monkeypatch.setattr(fad_module, "FrechetAudioDistance", DummyFadBackend)

    registry = MetricRegistry()
    fad_module.register_fad_metric(registry)

    assert registry.get_metric("fad") is fad_module.FadMetric
    assert registry.get_metadata("fad").category is MetricCategory.DISTRIBUTIONAL
    assert registry.get_metric("fad_metric") is fad_module.FadMetric

    metric = registry.get_metric("fad")({"io": "kaldi", "use_inf": True})
    scores = metric.compute(
        {"utt1": "pred1.wav", "utt2": "pred2.wav"},
        {"utt1": "ref1.wav", "utt2": "ref2.wav"},
    )

    assert scores == {"fad_overall": 0.25, "fad_r2": 0.75}


def test_kid_metric_registers_and_preserves_output_keys(monkeypatch):
    monkeypatch.setattr(kid_module, "get_model", lambda embedding: f"model:{embedding}")
    monkeypatch.setattr(kid_module, "FrechetAudioDistance", DummyKidBackend)

    registry = MetricRegistry()
    kid_module.register_kid_metric(registry)

    assert registry.get_metric("kid") is kid_module.KidMetric
    assert registry.get_metadata("kid").category is MetricCategory.DISTRIBUTIONAL
    assert registry.get_metric("kid_metric") is kid_module.KidMetric

    metric = registry.get_metric("kid")({"io": "kaldi"})
    scores = metric.compute(
        {"utt1": "pred1.wav", "utt2": "pred2.wav"},
        {"utt1": "ref1.wav", "utt2": "ref2.wav"},
    )

    assert scores == {"kid_mean": 0.1, "kid_std": 0.02}


def test_individual_fad_metric_registers_and_scores_each_prediction(monkeypatch):
    monkeypatch.setattr(
        individual_fad_module, "get_model", lambda embedding: f"model:{embedding}"
    )
    monkeypatch.setattr(
        individual_fad_module, "FrechetAudioDistance", DummyIndividualFadBackend
    )
    monkeypatch.setattr(
        individual_fad_module,
        "calc_embd_statistics",
        lambda embd: (float(np.mean(embd)), 1.0),
    )
    monkeypatch.setattr(
        individual_fad_module,
        "calc_frechet_distance",
        lambda mu_base, cov_base, mu_eval, cov_eval: mu_eval - mu_base,
    )

    registry = MetricRegistry()
    individual_fad_module.register_individual_fad_metric(registry)

    assert (
        registry.get_metric("individual_fad")
        is individual_fad_module.IndividualFadMetric
    )
    assert (
        registry.get_metadata("individual_fad").category
        is MetricCategory.DISTRIBUTIONAL
    )
    assert (
        registry.get_metric("individual_fad_metric")
        is individual_fad_module.IndividualFadMetric
    )

    metric = registry.get_metric("individual_fad")({"io": "kaldi"})
    scores = metric.compute(
        {"utt1": "pred1.wav", "utt2": "pred2.wav"},
        {"ref1": "ref1.wav", "ref2": "ref2.wav"},
    )

    assert scores == {"individual_fad": {"utt1": 1.5, "utt2": 2.0}}


def test_fad_kid_and_individual_fad_raise_clear_error_when_fadtk_missing(monkeypatch):
    monkeypatch.setattr(fad_module, "get_model", None)
    monkeypatch.setattr(fad_module, "FrechetAudioDistance", None)
    monkeypatch.setattr(kid_module, "get_model", None)
    monkeypatch.setattr(kid_module, "FrechetAudioDistance", None)
    monkeypatch.setattr(individual_fad_module, "get_model", None)
    monkeypatch.setattr(individual_fad_module, "FrechetAudioDistance", None)
    monkeypatch.setattr(individual_fad_module, "calc_embd_statistics", None)
    monkeypatch.setattr(individual_fad_module, "calc_frechet_distance", None)

    with pytest.raises(ModuleNotFoundError, match="FADTK is not installed"):
        fad_module.FadMetric()

    with pytest.raises(ModuleNotFoundError, match="FADTK is not installed"):
        kid_module.KidMetric()

    with pytest.raises(ModuleNotFoundError, match="FADTK is not installed"):
        individual_fad_module.IndividualFadMetric()


def test_default_registry_discovers_reviewed_metrics():
    registry = _create_populated_registry()

    for metric_name in ["fad", "individual_fad", "kid", "asvspoof_score", "emo_vad"]:
        assert registry.get_metric(metric_name) is not None
        assert registry.get_metadata(metric_name) is not None


def test_distributional_metrics_run_through_corpus_scorer(monkeypatch):
    monkeypatch.setattr(fad_module, "get_model", lambda embedding: f"model:{embedding}")
    monkeypatch.setattr(fad_module, "FrechetAudioDistance", DummyFadBackend)

    registry = MetricRegistry()
    fad_module.register_fad_metric(registry)
    scorer = VersaScorer(registry)
    suite = scorer.load_metrics(
        [{"name": "fad", "io": "kaldi"}],
        use_gt=True,
    )

    score_info = scorer.score_corpus(
        {"utt1": "pred1.wav", "utt2": "pred2.wav"},
        suite,
        {"utt1": "ref1.wav", "utt2": "ref2.wav"},
    )

    assert score_info == {"fad_overall": 0.25, "fad_r2": 0.75}


def test_load_score_modules_skips_distributional_metrics(monkeypatch):
    monkeypatch.setattr(fad_module, "get_model", lambda embedding: f"model:{embedding}")
    monkeypatch.setattr(fad_module, "FrechetAudioDistance", DummyFadBackend)

    suite = load_score_modules(
        [{"name": "fad", "io": "kaldi"}],
        use_gt=True,
    )

    assert len(suite.metrics) == 0


def test_corpus_metrics_do_not_include_empty_placeholders():
    corpus_metrics_dir = Path(__file__).parents[2] / "versa" / "corpus_metrics"
    empty_modules = sorted(
        path.name
        for path in corpus_metrics_dir.glob("*.py")
        if path.name != "__init__.py" and path.stat().st_size == 0
    )

    assert empty_modules == []
