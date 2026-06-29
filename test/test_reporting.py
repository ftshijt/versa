import csv
import subprocess
import sys

from versa.definition import MetricCategory, MetricMetadata, MetricRegistry, MetricType
from versa.bin.scorer import _write_report
from versa.reporting import (
    analyze_records,
    metric_category,
    read_result_records,
    write_csv_report,
    write_html_report,
    write_markdown_report,
)


def test_analyze_records_computes_ci_failures_rankings_and_outliers():
    records = [
        {"key": "a", "model": "m1", "pesq": 2.0, "wer": 0.2},
        {"key": "b", "model": "m1", "pesq": 2.5, "wer": 0.1},
        {"key": "c", "model": "m2", "pesq": 4.0, "wer": 0.4},
        {"key": "d", "model": "m2", "pesq": "bad", "wer": 2.0},
        {"key": "e", "model": "m2", "wer": 0.3},
    ]

    analysis = analyze_records(records, group_by="model")
    summaries = {summary.name: summary for summary in analysis["metrics"]}

    assert analysis["record_count"] == 5
    assert summaries["pesq"].count == 3
    assert summaries["pesq"].missing == 1
    assert summaries["pesq"].invalid == 1
    assert summaries["pesq"].ci95_high > summaries["pesq"].ci95_low
    assert summaries["pesq"].best_key == "c"
    assert summaries["wer"].best_key == "b"
    assert summaries["wer"].worst_key == "d"
    assert analysis["groups"]["rankings"]["pesq"][0]["group"] == "m2"
    assert analysis["groups"]["rankings"]["wer"][0]["group"] == "m1"


def test_read_result_records_accepts_jsonl_and_python_literals(tmp_path):
    result_file = tmp_path / "results.txt"
    result_file.write_text(
        '{"key": "a", "pesq": 2.0}\n' "{'key': 'b', 'sir': inf}\n",
        encoding="utf-8",
    )

    records = read_result_records(str(result_file))

    assert [record["key"] for record in records] == ["a", "b"]
    assert records[0]["_source_file"] == "results.txt"


def test_report_exports(tmp_path):
    records = [
        {"key": "a", "pesq": 2.0, "wer": 0.2},
        {"key": "b", "pesq": 3.0, "wer": 0.1},
    ]
    analysis = analyze_records(records)
    csv_path = tmp_path / "report.csv"
    md_path = tmp_path / "report.md"
    html_path = tmp_path / "report.html"

    write_csv_report(analysis, str(csv_path))
    write_markdown_report(analysis, str(md_path))
    write_html_report(analysis, str(html_path))

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {"pesq", "wer"}
    assert "VERSA Results Report" in md_path.read_text(encoding="utf-8")
    assert "Radar Overview" in html_path.read_text(encoding="utf-8")
    assert "Category Sunburst" in html_path.read_text(encoding="utf-8")


def test_metric_category_strips_model_prefixes():
    assert metric_category("arecho_pesq") == "speech_enhancement"
    assert metric_category("custom_wer") == "asr_wer_cer"


def test_reporting_import_is_lightweight():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from versa.reporting import analyze_records; print(analyze_records)",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert "analyze_records" in result.stdout


def test_reporting_uses_registry_alias_metadata():
    registry = MetricRegistry()
    registry.register(
        object,
        MetricMetadata(
            name="pesq",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=False,
            auto_install=True,
            dependencies=["pesq"],
            description="PESQ",
        ),
        aliases=["custom_pesq_alias"],
    )

    assert (
        metric_category("custom_pesq_alias", registry=registry) == "speech_enhancement"
    )
    assert (
        metric_category("experiment_custom_pesq_alias", registry=registry)
        == "speech_enhancement"
    )


def test_scorer_report_writer_exports_html(tmp_path):
    registry = MetricRegistry()
    report_path = tmp_path / "score_report.html"

    _write_report(
        [{"key": "a", "pesq": 2.0}, {"key": "b", "pesq": 3.0}],
        str(report_path),
        report_format="auto",
        group_by=None,
        outlier_limit=3,
        registry=registry,
    )

    assert "VERSA Results Report" in report_path.read_text(encoding="utf-8")
