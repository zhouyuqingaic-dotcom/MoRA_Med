import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize SLAKE Router-only shuffle and RMS diagnostics."
        )
    )
    parser.add_argument(
        "--diagnostics-root",
        type=str,
        required=True,
        help=(
            "Root produced by stage2_test_checkpoints_slake.py, "
            "for example: .../test/diagnostics"
        ),
    )
    parser.add_argument(
        "--a2-diagnostics-root",
        type=str,
        default=None,
        help=(
            "Diagnostics root for the A2 no-RMS experiment. "
            "If omitted, defaults to --diagnostics-root for backward compatibility."
        ),
    )
    parser.add_argument(
        "--normal-tag",
        type=str,
        default="normal_rms",
        help="Run tag for A6 normal + RMS.",
    )
    parser.add_argument(
        "--a2-tag",
        type=str,
        default="a2_normal_rms",
        help="Run tag for A2 normal + RMS.",
    )
    parser.add_argument(
        "--shuffle-prefix",
        type=str,
        default="router_shuffle_",
        help="Prefix used by Router-only shuffle run tags.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint directory name. "
            "Needed only when a run tag contains multiple checkpoint outputs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Where summary CSV/PNG files are written. "
            "Default: <diagnostics-root>/summary"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise TypeError(
                    f"{path}:{line_number} must contain a JSON object."
                )

            records.append(value)

    return records


def find_run_checkpoint_dir(
    diagnostics_root: Path,
    run_tag: str,
    checkpoint: str | None,
) -> Path:
    run_dir = diagnostics_root / run_tag

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run tag directory does not exist: {run_dir}"
        )

    if checkpoint is not None:
        candidate = run_dir / checkpoint
        if not (candidate / "summary.json").is_file():
            raise FileNotFoundError(
                f"summary.json not found under: {candidate}"
            )
        return candidate

    candidates = sorted(
        {
            summary_path.parent
            for summary_path in run_dir.rglob("summary.json")
        },
        key=lambda path: str(path),
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No summary.json found under: {run_dir}"
        )

    if len(candidates) > 1:
        names = "\n".join(
            f"  - {path}"
            for path in candidates
        )
        raise RuntimeError(
            f"Run tag {run_tag!r} contains multiple checkpoint outputs.\n"
            f"Pass --checkpoint to choose one:\n{names}"
        )

    return candidates[0]


def discover_shuffle_tags(
    diagnostics_root: Path,
    prefix: str,
) -> list[str]:
    tags = sorted(
        path.name
        for path in diagnostics_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )

    if not tags:
        raise FileNotFoundError(
            f"No shuffle run tags found under {diagnostics_root} "
            f"with prefix {prefix!r}."
        )

    return tags


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def summarize_numeric(values: list[float]) -> dict:
    values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]

    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def get_metric(summary: dict, key: str) -> float:
    if key not in summary:
        raise KeyError(
            f"summary.json is missing required metric: {key}"
        )
    return float(summary[key])


def build_routing_control_table(
    normal_summary: dict,
    shuffle_summaries: list[tuple[str, dict]],
) -> list[dict]:
    normal_overall = get_metric(
        normal_summary,
        "overall_strict_acc",
    )
    normal_closed = get_metric(
        normal_summary,
        "closed_acc",
    )
    normal_open = get_metric(
        normal_summary,
        "open_strict_acc",
    )

    rows = [
        {
            "condition": "Normal",
            "run_tag": normal_summary.get(
                "run_tag",
                "normal",
            ),
            "checkpoint": normal_summary.get(
                "checkpoint",
            ),
            "shuffle_seed": None,
            "shuffle_map_sha256": None,
            "overall_acc": normal_overall,
            "closed_acc": normal_closed,
            "open_acc": normal_open,
            "delta_overall_vs_normal": 0.0,
            "delta_closed_vs_normal": 0.0,
            "delta_open_vs_normal": 0.0,
        }
    ]

    for order, (run_tag, summary) in enumerate(
        shuffle_summaries,
        start=1,
    ):
        overall = get_metric(
            summary,
            "overall_strict_acc",
        )
        closed = get_metric(
            summary,
            "closed_acc",
        )
        opened = get_metric(
            summary,
            "open_strict_acc",
        )

        rows.append(
            {
                "condition": f"Shuffle {order}",
                "run_tag": run_tag,
                "checkpoint": summary.get(
                    "checkpoint",
                ),
                "shuffle_seed": summary.get(
                    "shuffle_seed",
                ),
                "shuffle_map_sha256": summary.get(
                    "shuffle_map_sha256",
                ),
                "overall_acc": overall,
                "closed_acc": closed,
                "open_acc": opened,
                "delta_overall_vs_normal": (
                    overall - normal_overall
                ),
                "delta_closed_vs_normal": (
                    closed - normal_closed
                ),
                "delta_open_vs_normal": (
                    opened - normal_open
                ),
            }
        )

    shuffle_rows = rows[1:]

    rows.append(
        {
            "condition": "Shuffle Mean",
            "run_tag": "mean_of_shuffles",
            "checkpoint": normal_summary.get(
                "checkpoint",
            ),
            "shuffle_seed": None,
            "shuffle_map_sha256": None,
            "overall_acc": mean(
                row["overall_acc"]
                for row in shuffle_rows
            ),
            "closed_acc": mean(
                row["closed_acc"]
                for row in shuffle_rows
            ),
            "open_acc": mean(
                row["open_acc"]
                for row in shuffle_rows
            ),
            "delta_overall_vs_normal": mean(
                row["delta_overall_vs_normal"]
                for row in shuffle_rows
            ),
            "delta_closed_vs_normal": mean(
                row["delta_closed_vs_normal"]
                for row in shuffle_rows
            ),
            "delta_open_vs_normal": mean(
                row["delta_open_vs_normal"]
                for row in shuffle_rows
            ),
        }
    )

    return rows


def validate_rms_record(
    record: dict,
    label: str,
):
    required = {
        "stream_name",
        "ratio_raw",
        "ratio_used",
        "ratio_inject",
        "use_rms_norm",
    }

    missing = sorted(
        field
        for field in required
        if field not in record
    )

    if missing:
        raise KeyError(
            f"{label} RMS record is missing fields: {missing}"
        )


def group_rms_by_stream(
    records: list[dict],
    label: str,
) -> dict[str, list[dict]]:
    grouped = {}

    for record in records:
        validate_rms_record(record, label)
        stream = str(
            record.get(
                "stream_name",
                "unknown",
            )
        )
        grouped.setdefault(stream, []).append(record)

    return grouped


def summarize_rms_condition(
    records: list[dict],
    label: str,
) -> list[dict]:
    grouped = group_rms_by_stream(
        records,
        label,
    )
    rows = []

    for stream in sorted(grouped):
        stream_records = grouped[stream]

        row = {
            "condition": label,
            "stream": stream,
            "n": len(stream_records),
        }

        for metric in (
            "ratio_raw",
            "ratio_used",
            "ratio_inject",
            "rho_unclipped",
            "rho_applied",
            "routing_f3",
            "routing_f5",
            "routing_f7",
            "gate",
            "lambda",
        ):
            stats = summarize_numeric(
                [
                    safe_float(item.get(metric))
                    for item in stream_records
                ]
            )

            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]

        rho_flags = [
            item.get("rho_clipped")
            for item in stream_records
            if item.get("rho_clipped") is not None
        ]

        row["rho_clipping_rate"] = (
            sum(bool(value) for value in rho_flags)
            / len(rho_flags)
            if rho_flags
            else None
        )

        rows.append(row)

    return rows


def assert_a2_consistency(records: list[dict]):
    checked = 0

    for record in records:
        raw = safe_float(
            record.get("ratio_raw")
        )
        used = safe_float(
            record.get("ratio_used")
        )

        if raw is None or used is None:
            continue

        tolerance = max(
            1e-8,
            1e-5 * max(
                abs(raw),
                abs(used),
                1.0,
            ),
        )

        if abs(raw - used) > tolerance:
            raise RuntimeError(
                "A2 no-RMS consistency check failed: "
                f"ratio_raw={raw}, ratio_used={used}, "
                f"stream={record.get('stream_name')}, "
                f"index={record.get('index')}."
            )

        if record.get("rho_unclipped") is not None:
            raise RuntimeError(
                "A2 no-RMS record unexpectedly contains rho_unclipped."
            )
        if record.get("rho_applied") is not None:
            raise RuntimeError(
                "A2 no-RMS record unexpectedly contains rho_applied."
            )
        if record.get("rho_clipped") is not None:
            raise RuntimeError(
                "A2 no-RMS record unexpectedly contains rho_clipped."
            )

        checked += 1

    if checked == 0:
        raise RuntimeError(
            "No usable A2 RMS records were found for consistency checking."
        )


def assert_a6_consistency(records: list[dict]):
    checked = 0

    for record in records:
        ratio_used = safe_float(
            record.get("ratio_used")
        )
        ratio_inject = safe_float(
            record.get("ratio_inject")
        )
        gate = safe_float(
            record.get("gate")
        )
        lambda_value = safe_float(
            record.get("lambda")
        )

        if (
            ratio_used is None
            or ratio_inject is None
            or gate is None
            or lambda_value is None
        ):
            continue

        expected = abs(
            lambda_value * gate
        ) * ratio_used

        tolerance = max(
            1e-6,
            5e-4 * max(
                abs(expected),
                abs(ratio_inject),
                1.0,
            ),
        )

        if abs(
            ratio_inject - expected
        ) > tolerance:
            raise RuntimeError(
                "A6 injection consistency check failed: "
                f"observed={ratio_inject}, expected={expected}, "
                f"stream={record.get('stream_name')}, "
                f"index={record.get('index')}."
            )

        rho_applied = safe_float(
            record.get("rho_applied")
        )
        if rho_applied is not None and rho_applied > 10.0001:
            raise RuntimeError(
                "A6 rho_applied exceeds expected clip=10: "
                f"{rho_applied}"
            )

        checked += 1

    if checked == 0:
        raise RuntimeError(
            "No usable A6 RMS records were found for consistency checking."
        )


def stream_sort_key(stream: str):
    if stream == "pooled":
        return (0, 0)

    if stream.startswith("deepstack_"):
        suffix = stream.split("_", 1)[1]
        try:
            index = int(suffix)
        except ValueError:
            index = 10**9

        return (1, index)

    return (2, stream)


def build_metric_lookup(
    rows: list[dict],
    condition: str,
    metric: str,
) -> dict[str, float]:
    output = {}

    for row in rows:
        if row["condition"] != condition:
            continue

        value = safe_float(
            row.get(
                f"{metric}_mean"
            )
        )

        if value is not None:
            output[row["stream"]] = value

    return output


def plot_raw_vs_used(
    rows: list[dict],
    output_path: Path,
):
    raw = build_metric_lookup(
        rows,
        "A6",
        "ratio_raw",
    )
    used = build_metric_lookup(
        rows,
        "A6",
        "ratio_used",
    )

    streams = sorted(
        set(raw) & set(used),
        key=stream_sort_key,
    )

    if not streams:
        raise RuntimeError(
            "No A6 stream data available for raw/used RMS plot."
        )

    x = list(range(len(streams)))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(
            max(8.0, 1.15 * len(streams)),
            5.2,
        )
    )

    ax.bar(
        [value - width / 2 for value in x],
        [raw[stream] for stream in streams],
        width=width,
        label="Raw residual",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [used[stream] for stream in streams],
        width=width,
        label="RMS-matched residual",
    )

    ax.set_title(
        "A6 Residual Scale Before and After RMS Matching"
    )
    ax.set_xlabel("Visual stream")
    ax.set_ylabel("RMS(residual) / RMS(input)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        streams,
        rotation=35,
        ha="right",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_injection_a2_vs_a6(
    rows: list[dict],
    output_path: Path,
):
    a2 = build_metric_lookup(
        rows,
        "A2",
        "ratio_inject",
    )
    a6 = build_metric_lookup(
        rows,
        "A6",
        "ratio_inject",
    )

    streams = sorted(
        set(a2) & set(a6),
        key=stream_sort_key,
    )

    if not streams:
        raise RuntimeError(
            "No shared A2/A6 stream data available for injection plot."
        )

    x = list(range(len(streams)))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(
            max(8.0, 1.15 * len(streams)),
            5.2,
        )
    )

    ax.bar(
        [value - width / 2 for value in x],
        [a2[stream] for stream in streams],
        width=width,
        label="A2: without RMS",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [a6[stream] for stream in streams],
        width=width,
        label="A6: with RMS",
    )

    ax.set_title(
        "Injected Residual Scale: A2 vs A6"
    )
    ax.set_xlabel("Visual stream")
    ax.set_ylabel("RMS(delta X) / RMS(input)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        streams,
        rotation=35,
        ha="right",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()

    a6_root = Path(
        args.diagnostics_root
    ).expanduser().resolve()

    a2_root = (
        Path(args.a2_diagnostics_root)
        .expanduser()
        .resolve()
        if args.a2_diagnostics_root
        else a6_root
    )

    if not a6_root.is_dir():
        raise FileNotFoundError(
            f"A6 diagnostics root does not exist: {a6_root}"
        )

    if not a2_root.is_dir():
        raise FileNotFoundError(
            f"A2 diagnostics root does not exist: {a2_root}"
        )

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
        if args.output_dir
        else a6_root / "summary"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Router control summary
    # ---------------------------------------------------------
    normal_dir = find_run_checkpoint_dir(
        a6_root,
        args.normal_tag,
        args.checkpoint,
    )
    normal_summary = read_json(
        normal_dir / "summary.json"
    )

    shuffle_tags = discover_shuffle_tags(
        a6_root,
        args.shuffle_prefix,
    )

    shuffle_summaries = []

    for run_tag in shuffle_tags:
        run_dir = find_run_checkpoint_dir(
            a6_root,
            run_tag,
            args.checkpoint,
        )
        summary = read_json(
            run_dir / "summary.json"
        )

        condition = summary.get(
            "evaluation_condition"
        )
        if condition != "router_shuffle":
            raise ValueError(
                f"{run_dir}/summary.json is not a router_shuffle run."
            )

        shuffle_summaries.append(
            (
                run_tag,
                summary,
            )
        )

    routing_rows = (
        build_routing_control_table(
            normal_summary,
            shuffle_summaries,
        )
    )

    routing_csv = (
        output_dir
        / "routing_control_summary.csv"
    )

    write_csv(
        routing_csv,
        routing_rows,
        [
            "condition",
            "run_tag",
            "checkpoint",
            "shuffle_seed",
            "shuffle_map_sha256",
            "overall_acc",
            "closed_acc",
            "open_acc",
            "delta_overall_vs_normal",
            "delta_closed_vs_normal",
            "delta_open_vs_normal",
        ],
    )

    # ---------------------------------------------------------
    # RMS summary
    # ---------------------------------------------------------
    a6_rms_path = (
        normal_dir
        / "rms_diagnostics.jsonl"
    )

    a2_dir = find_run_checkpoint_dir(
        a2_root,
        args.a2_tag,
        args.checkpoint,
    )
    a2_rms_path = (
        a2_dir
        / "rms_diagnostics.jsonl"
    )

    if not a6_rms_path.is_file():
        raise FileNotFoundError(
            f"A6 RMS file not found: {a6_rms_path}"
        )
    if not a2_rms_path.is_file():
        raise FileNotFoundError(
            f"A2 RMS file not found: {a2_rms_path}"
        )

    a6_records = read_jsonl(
        a6_rms_path
    )
    a2_records = read_jsonl(
        a2_rms_path
    )

    assert_a6_consistency(
        a6_records
    )
    assert_a2_consistency(
        a2_records
    )

    rms_rows = (
        summarize_rms_condition(
            a6_records,
            "A6",
        )
        + summarize_rms_condition(
            a2_records,
            "A2",
        )
    )

    rms_rows.sort(
        key=lambda row: (
            0 if row["condition"] == "A6" else 1,
            stream_sort_key(row["stream"]),
        )
    )

    rms_csv = (
        output_dir
        / "rms_summary_by_stream.csv"
    )

    rms_fieldnames = [
        "condition",
        "stream",
        "n",
        "ratio_raw_mean",
        "ratio_raw_std",
        "ratio_used_mean",
        "ratio_used_std",
        "ratio_inject_mean",
        "ratio_inject_std",
        "rho_unclipped_mean",
        "rho_unclipped_std",
        "rho_applied_mean",
        "rho_applied_std",
        "rho_clipping_rate",
        "routing_f3_mean",
        "routing_f3_std",
        "routing_f5_mean",
        "routing_f5_std",
        "routing_f7_mean",
        "routing_f7_std",
        "gate_mean",
        "gate_std",
        "lambda_mean",
        "lambda_std",
    ]

    write_csv(
        rms_csv,
        rms_rows,
        rms_fieldnames,
    )

    raw_used_png = (
        output_dir
        / "rms_raw_vs_used_by_stream.png"
    )
    injection_png = (
        output_dir
        / "rms_injection_a2_vs_a6_by_stream.png"
    )

    plot_raw_vs_used(
        rms_rows,
        raw_used_png,
    )
    plot_injection_a2_vs_a6(
        rms_rows,
        injection_png,
    )

    manifest = {
        "diagnostics_root": str(
            a6_root
        ),
        "a6_diagnostics_root": str(
            a6_root
        ),
        "a2_diagnostics_root": str(
            a2_root
        ),
        "normal_run": str(
            normal_dir
        ),
        "a2_run": str(
            a2_dir
        ),
        "shuffle_tags": shuffle_tags,
        "num_shuffle_runs": len(
            shuffle_summaries
        ),
        "outputs": {
            "routing_control_summary": str(
                routing_csv
            ),
            "rms_summary_by_stream": str(
                rms_csv
            ),
            "rms_raw_vs_used_plot": str(
                raw_used_png
            ),
            "rms_injection_plot": str(
                injection_png
            ),
        },
    }

    manifest_path = (
        output_dir
        / "summary_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSummary completed.")
    print(
        f"Routing CSV: {routing_csv}"
    )
    print(
        f"RMS CSV: {rms_csv}"
    )
    print(
        f"RMS raw/used plot: {raw_used_png}"
    )
    print(
        f"RMS injection plot: {injection_png}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()