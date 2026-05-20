import logging
import math
import os

import yaml

from versa.scorer_shared import VersaScorer

TEST_INFO = {
    "fad_overall": 0.00753398077542222,
    "fad_r2": float("-inf"),
}


def info_update():

    with open("egs/separate_metrics/fad.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.safe_load(f)

    assert len(score_config) > 0, "no scoring function is provided"

    scorer = VersaScorer()
    score_info = scorer.score_corpus(
        "test/test_samples/test2",
        scorer.load_metrics([{**score_config[0], "io": "dir"}], use_gt=True),
        "test/test_samples/test1",
    )
    print("Summary: {}".format(score_info), flush=True)

    for key in score_info:
        if math.isinf(TEST_INFO[key]) and math.isinf(score_info[key]):
            # for sir"
            continue
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - score_info[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check dir IO successful", flush=True)

    score_info = scorer.score_corpus(
        "test/test_samples/test2.scp",
        scorer.load_metrics([{**score_config[0], "io": "kaldi"}], use_gt=True),
        "test/test_samples/test1.scp",
    )
    print("Summary: {}".format(score_info), flush=True)

    for key in score_info:
        if math.isinf(TEST_INFO[key]) and math.isinf(score_info[key]):
            # for sir"
            continue
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - score_info[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check kaldi IO successful", flush=True)

    score_info = scorer.score_corpus(
        "test/test_samples/test2.scp",
        scorer.load_metrics([{**score_config[0], "io": "soundfile"}], use_gt=True),
        "test/test_samples/test1.scp",
    )
    print("Summary: {}".format(score_info), flush=True)

    for key in score_info:
        if math.isinf(TEST_INFO[key]) and math.isinf(score_info[key]):
            # for sir"
            continue
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - score_info[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check Soundfile IO successful", flush=True)


if __name__ == "__main__":
    info_update()
