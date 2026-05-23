# Existing Metrics Review Plan

This review uses `docs/contributing.md` as the checklist for existing metrics:
object-oriented implementation, registry exposure, metadata, optional dependency
handling, docs/examples, and focused tests. User-facing YAML names and report
keys should remain backward compatible unless a future migration explicitly
documents a breaking change.

## Ready / Keep

These metrics already follow the current `BaseMetric` and registry shape well
enough for normal maintenance. Future changes should stay limited to bug fixes,
metadata corrections, and real-model validation.

| Bucket | Metrics | Next move |
| --- | --- | --- |
| Sequence/dependent wrappers | `mcd_f0`, `signal_metric`, `warpq`, `stoi`, `pesq`, `log_wmse`, `pysepm`, `asr_match`, `chroma_alignment`, `dpam`, `cdpam` | Keep output keys and YAML aliases stable; add focused tests only when behavior changes. |
| Independent utterance metrics | `pseudo_mos`, `se_snr`, `sheet_ssqa`, `speaking_rate`, `squim_no_ref`, `srmr`, `vad`, `vqscore`, `wvmos`, `nisqa`, `pam`, `sigmos`, `audiobox_aesthetics`, `asvspoof`, `emo_vad` | Keep optional imports guarded and verify base package import remains safe. |
| ASR/text-assisted metrics | `espnet_wer`, `owsm_wer`, `whisper_wer`, `fwhisper_wer`, `nemo_wer`, `hubert_wer`, `owsm_lid` | Recheck `requires_text`, aliases, and installer notes whenever backend defaults change. |
| Multi-metric model families | `qwen2_audio`, `qwen_omni`, `scoreq`, `universa`, `arecho` | Keep generated registry names, aliases, and report keys synchronized with examples and docs. |

## Registry / Export Fix

These items were reviewed because the implementation existed but default
registry discovery was incomplete.

| Metric | Finding | Status |
| --- | --- | --- |
| `asvspoof` / `asvspoof_score` | `register_asvspoof_metric` existed but was not exposed from `versa/__init__.py`. | Fixed by adding `_optional_metric_import(...)`; covered by registry discovery test. |
| `emo_vad` | `register_emo_vad_metric` existed but was not exposed from `versa/__init__.py`. | Fixed by adding `_optional_metric_import(...)`; covered by registry discovery test. |
| `fad` | Legacy corpus helper was not registered in the default metric registry. | Fixed by adding `FadMetric`, `register_fad_metric`, and `versa/__init__.py` export. |
| `kid` | Legacy corpus helper was not registered in the default metric registry. | Fixed by adding `KidMetric`, `register_kid_metric`, and `versa/__init__.py` export. |

## Metadata / Documentation Fix

These items need metadata or docs attention separate from implementation shape.

| Metric | Requirement | Status |
| --- | --- | --- |
| `nomad` | Non-matching reference metric should use `MetricCategory.NON_MATCH` while still requiring reference audio. | Fixed in metadata and tests. |
| `noresqa_score`, `noresqa_mos` | Non-matching reference metrics should use `MetricCategory.NON_MATCH` while still requiring reference audio. | Fixed in metadata and tests. |
| `emo2vec_similarity` / `emotion` | Similarity uses reference audio that need not be matched; category should be `NON_MATCH`. | Fixed in metadata and tests. |
| `qwen_omni_*` | Code/tests existed but `docs/supported_metrics.md` and `egs/separate_metrics` lacked a dedicated entry/example. | Fixed with grouped docs row and `egs/separate_metrics/qwen_omni.yaml`. |
| `kid` | Distributional docs row had malformed table columns. | Fixed in `docs/supported_metrics.md`. |
| Auto-install marks | Several metrics rely on optional groups, model downloads, or external installers. | Still requires periodic review against `pyproject.toml` and installer scripts before releases. |

## Migration Settled

The metric object-oriented migration is complete for maintained metric modules.
FAD and KID now use `BaseMetric`, `MetricMetadata`, registry integration, and
distributional scorer tests. The empty `individual_fad` placeholder was removed
instead of registering a non-functional metric name.

## Needs Real-Model Verification

Lightweight tests use mocks for model-heavy metrics. Before a release, run
real-model checks only in environments with the matching dependencies,
checkpoints, and licenses.

| Metric family | Verification needed |
| --- | --- |
| Distributional audio metrics | Run real FAD/KID/CLAP scoring after `tools/install_fadtk.sh` or `tools/install_clap_score.sh`. |
| External speech quality models | Run `scoreq`, `nomad`, `noresqa`, `visqol`, `wvmos`, `srmr`, `pysepm`, `audiobox_aesthetics`, and `asvspoof` with their installer scripts. |
| Foundation-model profilers | Run `qwen2_audio`, `qwen_omni`, `universa`, and `arecho` in a GPU-capable environment and confirm output keys are stable. |
| ASR-backed metrics | Run WER/LID metrics with expected model caches and transcript inputs, including missing-text failure cases. |

## Static Audit Checks

Keep these checks as the minimum review checklist for future metric work:

- Every importable metric module has a `BaseMetric` subclass; empty placeholders should be removed instead of documented as pending/legacy.
- Every `register_*_metric` function is exposed through `versa/__init__.py`.
- Every documented `Key in config` resolves through `MetricRegistry` as a canonical name or alias.
- Every YAML name in `egs/separate_metrics` resolves through `MetricRegistry`.
- Optional dependencies do not break `import versa`; missing backends raise clear `ImportError` or `ModuleNotFoundError` from setup.
- Focused tests cover registration, aliases, metadata, input validation, output keys, and missing optional dependencies.
