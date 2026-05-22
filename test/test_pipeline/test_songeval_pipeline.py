import yaml

from versa.scorer_shared import VersaScorer, find_files, load_summary
from versa.utterance_metrics import songeval


def test_songeval_pipeline_with_registry_and_mocked_metric(monkeypatch):
    monkeypatch.setattr(
        songeval,
        "songeval_model_setup",
        lambda **kwargs: {"model": object()},
    )
    monkeypatch.setattr(
        songeval,
        "songeval_metric",
        lambda model_dict, pred_x, fs: {
            "songeval_coherence": 3.0,
            "songeval_musicality": 4.0,
        },
    )

    with open("egs/separate_metrics/songeval.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    gen_files = find_files("test/test_samples/test2")
    scorer = VersaScorer()
    metric_suite = scorer.load_metrics(score_config, use_gt=False, use_gpu=False)

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=None,
        io="soundfile",
    )
    summary = load_summary(score_info)

    assert score_info
    assert summary["songeval_coherence"] == 3.0
    assert summary["songeval_musicality"] == 4.0
