#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

from tqdm import tqdm

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType
from versa.scorer_shared import audio_loader_setup

try:
    from fadtk.fad_versa import FrechetAudioDistance
    from fadtk.model_loader import get_model
except ImportError:
    FrechetAudioDistance = None
    get_model = None
    logging.warning(
        "FADTK is not installed. Please install it following `tools/install_fadtk.sh`"
    )


def _load_audio_collection(audio, io):
    if isinstance(audio, dict):
        return audio
    return audio_loader_setup(audio, io)


def _kid_metadata():
    return MetricMetadata(
        name="kid",
        category=MetricCategory.DISTRIBUTIONAL,
        metric_type=MetricType.DICT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["fadtk"],
        description="Kernel distance metric over generated and reference audio embedding distributions",
        paper_reference="https://arxiv.org/abs/1812.08466",
        implementation_source="https://github.com/SonyCSLParis/audio-metrics",
    )


class KidMetric(BaseMetric):
    """Kernel distance metric over prediction and reference collections."""

    def _setup(self):
        if get_model is None or FrechetAudioDistance is None:
            raise ModuleNotFoundError(
                "FADTK is not installed. Please install it following `tools/install_fadtk.sh`"
            )

        self.cache_dir = self.config.get("cache_dir", "versa_cache/kid")
        self.io = self.config.get("io", "kaldi")
        self.baseline = self.config.get("baseline")
        self.embedding = self.config.get(
            "kid_embedding", self.config.get("fad_embedding", "default")
        )
        self.module = FrechetAudioDistance(
            ml=get_model(self.embedding),
            load_model=True,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("KID requires prediction audio files")

        metadata = metadata or {}
        baseline = references or metadata.get("baseline_files") or self.baseline
        if baseline is None:
            raise ValueError("KID requires reference or baseline audio files")

        baseline_files = _load_audio_collection(baseline, self.io)
        eval_files = _load_audio_collection(predictions, self.io)
        if len(baseline_files) < 2 or len(eval_files) < 2:
            raise ValueError("KID requires at least 2 files to compare.")

        logging.info("[KID] caching baseline embeddings...")
        for key in tqdm(baseline_files.keys()):
            self.module.cache_embedding_file(
                key, baseline_files[key], self.cache_dir + "/baseline"
            )
        logging.info("[KID] Finished caching baseline embeddings.")

        logging.info("[KID] caching eval embeddings...")
        for key in tqdm(eval_files.keys()):
            self.module.cache_embedding_file(
                key, eval_files[key], self.cache_dir + "/eval"
            )
        logging.info("[KID] Finished caching eval embeddings.")

        return {
            "kid" + key: value
            for key, value in self.module.score_kid(
                baseline_files, eval_files, self.cache_dir
            ).items()
        }

    def get_metadata(self):
        return _kid_metadata()


def register_kid_metric(registry):
    """Register KID with the registry."""
    registry.register(
        KidMetric,
        _kid_metadata(),
        aliases=["kernel_distance", "kernel_inception_distance", "kid_metric"],
    )
