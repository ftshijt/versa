import logging
import os

import yaml

from versa.scorer_shared import (
    find_files,
    list_scoring,
    load_score_modules,
    load_summary,
)


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/vqscore.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(score_config, use_gt=False, use_gpu=False)

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = list_scoring(
        gen_files, score_modules, gt_files=None, output_file=None, io="soundfile"
    )
    summary = load_summary(score_info)
    print("Summary: {}".format(load_summary(score_info)), flush=True)

    for key in summary:
        if summary[key] > 2:
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
