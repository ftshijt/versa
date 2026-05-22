import numpy as np
import logging
import subprocess
import sys
from pathlib import Path

import torch

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

try:
    from hydra.utils import instantiate
except ImportError:
    instantiate = None

try:
    from muq import MuQ
except ImportError:
    MuQ = None

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 24000
SONGEVAL_OUTPUTS = (
    ("songeval_coherence", "Coherence"),
    ("songeval_musicality", "Musicality"),
    ("songeval_memorability", "Memorability"),
    ("songeval_clarity", "Clarity"),
    ("songeval_naturalness", "Naturalness"),
)


def _require_songeval_dependencies():
    missing = []
    if instantiate is None:
        missing.append("hydra-core")
    if MuQ is None:
        missing.append("muq")
    if OmegaConf is None:
        missing.append("omegaconf")
    if load_file is None:
        missing.append("safetensors")

    if missing:
        raise ImportError(
            "SongEval requires optional dependencies: {}. "
            "Install them with `tools/install_songeval.sh`.".format(", ".join(missing))
        )


def songeval_model_setup(cache_dir="versa_cache", use_gpu=False):
    """Set up SongEval classifier and MuQ encoder."""
    _require_songeval_dependencies()
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/ASLP-lab/SongEval.git"

    songeval_dir = cache_dir / "SongEval"

    if not songeval_dir.exists():
        logger.info(f"Cloning SongEval repository into {cache_dir}")
        subprocess.run(["git", "clone", repo_url, str(songeval_dir)], check=True)
    else:
        logger.info(f"Using existing SongEval repository in {cache_dir}")

    songeval_path = str(songeval_dir)
    if songeval_path not in sys.path:
        sys.path.insert(0, songeval_path)

    model_path = songeval_dir / "ckpt" / "model.safetensors"
    config_path = songeval_dir / "config.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in {config_path}")

    with torch.no_grad():
        train_config = OmegaConf.load(config_path)
        model = instantiate(train_config.generator)
        state_dict = load_file(model_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

    muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq_model = muq_model.to(device).eval()

    model_dict = {"model": model, "muq": muq_model, "device": device}
    return model_dict


def songeval_metric(model_dict, pred, fs):
    """
    pred: np.ndarray, original waveform
    fs: original sampling rate
    return: dict, metric results for five SongEval dimensions
    """
    device = model_dict["device"]
    model = model_dict["model"]
    muq_model = model_dict["muq"]

    pred = resample_audio(pred, fs, TARGET_SAMPLE_RATE)

    audio = torch.as_tensor(pred, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        output = muq_model(audio, output_hidden_states=True)
        hidden = output["hidden_states"][6]
        scores_g = model(hidden).squeeze(0)

    values = {}
    for index, (output_key, _) in enumerate(SONGEVAL_OUTPUTS):
        values[output_key] = round(scores_g[index].item(), 4)

    return values


class SongEvalMetric(BaseMetric):
    """SongEval song aesthetics predictor."""

    def _setup(self):
        self.cache_dir = self.config.get("cache_dir", "versa_cache")
        self.use_gpu = self.config.get("use_gpu", False)
        self.model_dict = songeval_model_setup(
            cache_dir=self.cache_dir,
            use_gpu=self.use_gpu,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        fs = metadata.get("sample_rate", TARGET_SAMPLE_RATE) if metadata else 24000
        return songeval_metric(self.model_dict, np.asarray(predictions), fs)

    def get_metadata(self):
        return _songeval_metadata()


def _songeval_metadata():
    return MetricMetadata(
        name="songeval",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=[
            "hydra",
            "muq",
            "numpy",
            "omegaconf",
            "safetensors",
            "torch",
        ],
        description="SongEval song aesthetics scores for generated songs",
        paper_reference="https://arxiv.org/abs/2505.10793",
        implementation_source="https://github.com/ASLP-lab/SongEval",
    )


def register_songeval_metric(registry):
    """Register SongEval with the registry."""
    registry.register(
        SongEvalMetric,
        _songeval_metadata(),
        aliases=["SongEval", "song_eval", "songeval_metric"],
    )


songeval_setup = songeval_model_setup


if __name__ == "__main__":
    a = np.random.rand(24000).astype(np.float32)
    model = songeval_model_setup(use_gpu=True)
    print("metrics:", songeval_metric(model, a, 24000))
