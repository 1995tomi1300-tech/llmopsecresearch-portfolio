from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_node_label(value: int) -> str:
    if value >= 1_000_000:
        return "1M"
    if value >= 1_000:
        return f"{value // 1000}k"
    return str(value)


def render_metric_chart(
    rows: list[dict[str, object]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    logical_nodes = sorted({int(row["logical_nodes"]) for row in rows})
    x_positions = {node: idx for idx, node in enumerate(logical_nodes)}

    series_order = [
        ("strong_balanced", "hybrid_manifold_periodic_random"),
        ("strong_balanced", "hybrid_manifold_periodic_noisy"),
        ("critical_push", "hybrid_manifold_periodic_random"),
        ("critical_push", "hybrid_manifold_periodic_noisy"),
    ]
    styles = {
        ("strong_balanced", "hybrid_manifold_periodic_random"): ("#1b9e77", "o"),
        ("strong_balanced", "hybrid_manifold_periodic_noisy"): ("#d95f02", "o"),
        ("critical_push", "hybrid_manifold_periodic_random"): ("#7570b3", "s"),
        ("critical_push", "hybrid_manifold_periodic_noisy"): ("#e7298a", "s"),
    }

    for variant, scenario in series_order:
        points = [
            row for row in rows
            if row["variant"] == variant and row["scenario"] == scenario
        ]
        if not points:
            continue
        points.sort(key=lambda item: int(item["logical_nodes"]))
        color, marker = styles[(variant, scenario)]
        xs = [x_positions[int(point["logical_nodes"])] for point in points]
        ys = [float(point[metric_key]) for point in points]
        label = f"{variant} | {scenario.split('_')[-1]}"
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2.2, label=label)
        for x, y, point in zip(xs, ys, points):
            if point["status"] != "complete":
                ax.annotate(
                    "interim",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=color,
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Logical nodes")
    ax.set_xticks(list(x_positions.values()), [fmt_node_label(node) for node in logical_nodes])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_report(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Partial Dim-Narrow Interim Report")
    lines.append("")
    lines.append("This report combines:")
    lines.append("- calibrated 100k and 300k results")
    lines.append("- calibrated 1M strong_balanced interim results")
    lines.append("- note: 1M critical_push is still missing")
    lines.append("")
    lines.append("## Merged Table")
    lines.append("")
    lines.append("| Logical | Variant | Scenario | Coherence | False isolation | Status |")
    lines.append("|---|---|---|---:|---:|---|")
    for row in sorted(rows, key=lambda item: (int(item["logical_nodes"]), str(item["variant"]), str(item["scenario"]))):
        lines.append(
            f"| {fmt_node_label(int(row['logical_nodes']))} | {row['variant']} | {row['scenario']} | "
            f"{float(row['late_recovery_coherence_mean']):.4f} | "
            f"{float(row['final_false_isolation_rate_mean']):.4f} | {row['status']} |"
        )

    lines.append("")
    lines.append("## Interim Reading")
    lines.append("")

    lookup = {
        (int(row["logical_nodes"]), str(row["variant"]), str(row["scenario"])): row
        for row in rows
    }
    sr100 = lookup.get((100000, "strong_balanced", "hybrid_manifold_periodic_random"))
    sr300 = lookup.get((300000, "strong_balanced", "hybrid_manifold_periodic_random"))
    sr1m = lookup.get((1000000, "strong_balanced", "hybrid_manifold_periodic_random"))
    sn100 = lookup.get((100000, "strong_balanced", "hybrid_manifold_periodic_noisy"))
    sn300 = lookup.get((300000, "strong_balanced", "hybrid_manifold_periodic_noisy"))
    sn1m = lookup.get((1000000, "strong_balanced", "hybrid_manifold_periodic_noisy"))

    if sr100 and sr300 and sr1m:
        lines.append(
            "- Strong/random: coherence rises from 100k to 300k, then drops at 1M; "
            "false isolation improves monotonically."
        )
        lines.append(
            f"  Values: {float(sr100['late_recovery_coherence_mean']):.4f} -> "
            f"{float(sr300['late_recovery_coherence_mean']):.4f} -> "
            f"{float(sr1m['late_recovery_coherence_mean']):.4f} coherence, "
            f"{float(sr100['final_false_isolation_rate_mean']):.4f} -> "
            f"{float(sr300['final_false_isolation_rate_mean']):.4f} -> "
            f"{float(sr1m['final_false_isolation_rate_mean']):.4f} false isolation."
        )
    if sn100 and sn300 and sn1m:
        lines.append(
            "- Strong/noisy: coherence also peaks at 300k and softens at 1M; "
            "false isolation still improves as scale rises."
        )
        lines.append(
            f"  Values: {float(sn100['late_recovery_coherence_mean']):.4f} -> "
            f"{float(sn300['late_recovery_coherence_mean']):.4f} -> "
            f"{float(sn1m['late_recovery_coherence_mean']):.4f} coherence, "
            f"{float(sn100['final_false_isolation_rate_mean']):.4f} -> "
            f"{float(sn300['final_false_isolation_rate_mean']):.4f} -> "
            f"{float(sn1m['final_false_isolation_rate_mean']):.4f} false isolation."
        )

    lines.append(
        "- Working interpretation: larger scale does not simply strengthen the field; "
        "it appears to increase selectivity. Coherence becomes harder to preserve, "
        "while false-node isolation remains strong or improves."
    )
    lines.append(
        "- Architectural implication: this looks less like classic LLM scaling and more "
        "like a resonant filtering system where scale amplifies both opportunity and instability."
    )
    lines.append(
        "- Caution: 1M critical_push is still missing, so the high-stress regime is not complete yet."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `merged_partial_estimates.csv`")
    lines.append("- `coherence_vs_scale_interim.png`")
    lines.append("- `false_isolation_vs_scale_interim.png`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render interim dim-narrow report and charts.")
    parser.add_argument("--base-calibrated", type=Path, required=True)
    parser.add_argument("--one-million-calibrated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, object]] = []

    for row in read_csv(args.base_calibrated):
        merged_rows.append(
            {
                "logical_nodes": int(row["logical_nodes"]),
                "effective_nodes": int(row["effective_nodes"]),
                "variant": row["variant"],
                "scenario": row["scenario"],
                "late_recovery_coherence_mean": float(row["late_recovery_coherence_mean"]),
                "final_false_isolation_rate_mean": float(row["final_false_isolation_rate_mean"]),
                "status": "complete",
            }
        )

    for row in read_csv(args.one_million_calibrated):
        merged_rows.append(
            {
                "logical_nodes": int(row["logical_nodes"]),
                "effective_nodes": int(row["effective_nodes"]),
                "variant": row["variant"],
                "scenario": row["scenario"],
                "late_recovery_coherence_mean": float(row["late_recovery_coherence_mean"]),
                "final_false_isolation_rate_mean": float(row["final_false_isolation_rate_mean"]),
                "status": "interim",
            }
        )

    merged_rows.sort(key=lambda item: (int(item["logical_nodes"]), str(item["variant"]), str(item["scenario"])))
    write_csv(args.output_dir / "merged_partial_estimates.csv", merged_rows)

    render_metric_chart(
        merged_rows,
        metric_key="late_recovery_coherence_mean",
        title="Late Recovery Coherence vs Scale (Interim)",
        ylabel="Late recovery coherence",
        output_path=args.output_dir / "coherence_vs_scale_interim.png",
    )
    render_metric_chart(
        merged_rows,
        metric_key="final_false_isolation_rate_mean",
        title="False Isolation vs Scale (Interim)",
        ylabel="Final false isolation rate",
        output_path=args.output_dir / "false_isolation_vs_scale_interim.png",
    )

    report_text = build_report(merged_rows)
    (args.output_dir / "partial_dim_narrow_report.md").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
