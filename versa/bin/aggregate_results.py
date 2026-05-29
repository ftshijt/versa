#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Aggregate and report VERSA results."""

import argparse
import json
import logging
import os

from tqdm import tqdm

from versa.reporting import (
    analyze_records,
    read_result_records,
    write_csv_report,
    write_html_report,
    write_markdown_report,
)


def get_parser() -> argparse.Namespace:
    """Get parser of aggregate results."""
    parser = argparse.ArgumentParser(
        description="Aggregate chunked results or generate polished result reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input JSONL result file or directory. If omitted, --logdir/--scoredir/--nj mode is used.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Input log directory.",
    )
    parser.add_argument(
        "--scoredir",
        type=str,
        default=None,
        help="Output scoring directory.",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=None,
        help="Number of sub jobs",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output report path for input mode.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "md", "html"],
        default="auto",
        help="Report format for input mode.",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        help="Optional record field used for per-metric ranking.",
    )
    return parser


def aggregate_results(logdir: str, scoredir: str, nj: int) -> None:
    """Aggregate results."""
    logging.info("Aggregating results...")
    score_info = []
    for i in range(nj):
        with open("{}/result.{}.txt".format(logdir, i + 1), "r") as f:
            for line in f:
                score_info.append(json.loads(line.strip()))
    with open("{}/utt_result.txt".format(scoredir), "w") as f, open(
        "{}/avg_result.txt".format(scoredir), "w"
    ) as f2:
        for info in tqdm(score_info):
            f.write("{}\n".format(info))
        for key in score_info[0].keys():
            if key == "key":
                continue
            avg = sum([info[key] for info in score_info]) / len(score_info)
            f2.write("{}: {}\n".format(key, avg))

    logging.info("Done.")


def generate_report(
    input_path: str,
    output_path: str,
    report_format: str = "auto",
    group_by: str = None,
) -> None:
    """Generate a CSV, Markdown, or HTML report from result records."""
    records = read_result_records(input_path)
    analysis = analyze_records(records, group_by=group_by)

    if report_format == "auto":
        suffix = os.path.splitext(output_path)[1].lower()
        report_format = {
            ".csv": "csv",
            ".md": "md",
            ".markdown": "md",
            ".html": "html",
            ".htm": "html",
        }.get(suffix, "csv")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if report_format == "csv":
        write_csv_report(analysis, output_path)
    elif report_format == "md":
        write_markdown_report(analysis, output_path)
    else:
        write_html_report(analysis, output_path)

    logging.info(
        "Wrote %s report for %s utterances and %s metrics to %s",
        report_format,
        analysis["record_count"],
        analysis["metric_count"],
        output_path,
    )


def main() -> None:
    """Run main function."""
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.input:
        output_path = args.out or "metrics_report.csv"
        generate_report(args.input, output_path, args.format, args.group_by)
        return

    if args.logdir is None or args.scoredir is None or args.nj is None:
        parser.error("either provide input, or provide --logdir, --scoredir, and --nj")

    aggregate_results(args.logdir, args.scoredir, args.nj)


if __name__ == "__main__":
    main()
