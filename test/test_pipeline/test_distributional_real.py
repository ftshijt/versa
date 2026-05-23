import importlib.util
import math
import os
from pathlib import Path

import pytest

from versa.scorer_shared import VersaScorer

RUN_REAL_MODEL_TESTS = os.environ.get("VERSA_RUN_REAL_MODEL_TESTS") == "1"


def _fadtk_is_available():
    try:
        return (
            importlib.util.find_spec("fadtk.fad_versa") is not None
            and importlib.util.find_spec("fadtk.model_loader") is not None
        )
    except ModuleNotFoundError:
        return False


def _sample_pair():
    pred = Path("test/test_samples/test2/test.wav")
    ref = Path("test/test_samples/test1/test.wav")
    if not pred.exists() or not ref.exists():
        pytest.skip("Required real sample audio files are not available")
    return str(pred), str(ref)


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real sample-backed checks",
)
@pytest.mark.skipif(
    not _fadtk_is_available(),
    reason="FADTK is not installed; run tools/install_fadtk.sh first",
)
def test_fad_pipeline_with_real_samples(tmp_path):
    """Run FAD through the registry/scorer path with real sample audio."""
    pred, ref = _sample_pair()
    scorer = VersaScorer()
    suite = scorer.load_metrics(
        [
            {
                "name": "fad",
                "io": "soundfile",
                "cache_dir": str(tmp_path / "fad"),
                "use_inf": True,
            }
        ],
        use_gt=True,
    )

    score_info = scorer.score_corpus(
        {"sample": pred},
        suite,
        {"sample": ref},
    )

    assert set(score_info) == {"fad_overall", "fad_r2"}
    assert math.isfinite(score_info["fad_overall"])
    assert isinstance(score_info["fad_r2"], float)


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real sample-backed checks",
)
@pytest.mark.skipif(
    not _fadtk_is_available(),
    reason="FADTK is not installed; run tools/install_fadtk.sh first",
)
def test_kid_pipeline_with_real_samples(tmp_path):
    """Run KID through the registry/scorer path with real sample audio."""
    pred, ref = _sample_pair()
    scorer = VersaScorer()
    suite = scorer.load_metrics(
        [
            {
                "name": "kid",
                "io": "soundfile",
                "cache_dir": str(tmp_path / "kid"),
            }
        ],
        use_gt=True,
    )

    score_info = scorer.score_corpus(
        {"sample_a": pred, "sample_b": pred},
        suite,
        {"sample_a": ref, "sample_b": ref},
    )

    assert score_info
    for key, value in score_info.items():
        assert key.startswith("kid")
        assert isinstance(value, float)


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real sample-backed checks",
)
@pytest.mark.skipif(
    not _fadtk_is_available(),
    reason="FADTK is not installed; run tools/install_fadtk.sh first",
)
def test_individual_fad_pipeline_with_real_samples(tmp_path):
    """Run individual FAD through the registry/scorer path with real sample audio."""
    pred, ref = _sample_pair()
    scorer = VersaScorer()
    suite = scorer.load_metrics(
        [
            {
                "name": "individual_fad",
                "io": "soundfile",
                "cache_dir": str(tmp_path / "individual_fad"),
            }
        ],
        use_gt=True,
    )

    score_info = scorer.score_corpus(
        {"sample": pred},
        suite,
        {"baseline": ref},
    )

    assert set(score_info) == {"individual_fad"}
    assert set(score_info["individual_fad"]) == {"sample"}
    assert isinstance(score_info["individual_fad"]["sample"], float)
    assert math.isfinite(score_info["individual_fad"]["sample"])
