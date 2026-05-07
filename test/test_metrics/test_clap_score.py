import os
from pathlib import Path
from importlib.util import find_spec

import numpy as np
import pytest

from versa.corpus_metrics import clap_score


class FakeCLAPScore:
    sample_rate = 16000

    def get_text_embeddings(self, text_data):
        return np.array(
            [[1.0, 0.0] if text == "low tone" else [0.0, 1.0] for text in text_data]
        )

    def get_audio_embeddings(self, audio_data, sr):
        return np.array(
            [[1.0, 0.0] if np.mean(audio) < 0.0 else [0.0, 1.0] for audio in audio_data]
        )

    def calculate_clap_score(self, text_embds, audio_embds, batch_size):
        text_norm = text_embds / np.linalg.norm(text_embds, axis=-1, keepdims=True)
        audio_norm = audio_embds / np.linalg.norm(audio_embds, axis=-1, keepdims=True)
        return np.mean(np.sum(text_norm * audio_norm, axis=-1)), 0.0


def test_clap_score_setup_requires_dependency(monkeypatch):
    monkeypatch.setattr(clap_score, "CLAPScore", None)
    monkeypatch.setattr(
        clap_score,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(ModuleNotFoundError, match="frechet_audio_distance"):
        clap_score.clap_score_setup()


def test_clap_score_scoring_matches_text_by_key(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    first = audio_dir / "b.wav"
    second = audio_dir / "a.wav"

    import soundfile as sf

    sf.write(first, -np.ones(16000, dtype=np.float32) * 0.1, 16000)
    sf.write(second, np.ones(16000, dtype=np.float32) * 0.1, 16000)

    clap_info = {
        "module": FakeCLAPScore(),
        "cache_dir": str(tmp_path / "cache"),
        "cache_embeddings": False,
        "io": "dir",
    }
    scores = clap_score.clap_score_scoring(
        str(audio_dir),
        clap_info,
        text_info={"a.wav": "high tone", "b.wav": "low tone"},
    )

    assert scores["clap_score"] == pytest.approx(1.0)


def test_clap_score_scoring_requires_text(tmp_path):
    clap_info = {
        "module": FakeCLAPScore(),
        "cache_dir": str(tmp_path / "cache"),
        "cache_embeddings": False,
        "io": "dir",
    }
    with pytest.raises(ValueError, match="requires text"):
        clap_score.clap_score_scoring(str(tmp_path), clap_info, text_info=None)


@pytest.mark.real_model
def test_clap_score_real_model_inference(tmp_path):
    if os.environ.get("VERSA_RUN_REAL_CLAP_SCORE_TEST") != "1":
        pytest.skip("Set VERSA_RUN_REAL_CLAP_SCORE_TEST=1 to run real CLAP inference")
    if find_spec("frechet_audio_distance") is None:
        pytest.skip("frechet_audio_distance is not installed")

    import soundfile as sf

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wav = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(audio_dir / "tone.wav", wav, sample_rate)

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        Path(xdg_cache_home, "torch", "hub").mkdir(parents=True, exist_ok=True)

    clap_info = clap_score.clap_score_setup(
        cache_dir=str(tmp_path / "cache"),
        cache_embeddings=False,
        io="dir",
    )
    scores = clap_score.clap_score_scoring(
        str(audio_dir),
        clap_info,
        text_info={"tone.wav": "a steady audio tone"},
        batch_size=1,
    )

    assert "clap_score" in scores
    assert np.isfinite(scores["clap_score"])
