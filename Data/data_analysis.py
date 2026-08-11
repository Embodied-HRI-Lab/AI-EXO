"""Plot angle, angular velocity, and torque from exoskeleton CSV logs.

Example:
    python Data/data_analysis.py Data/exo_logs/yth10.csv --show
    python Data/data_analysis.py Data/exo_logs --output Data/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


COLUMNS = {
    "angle": ("left_angle_x_deg", "right_angle_x_deg", "Hip angle (deg)"),
    "velocity": (
        "left_angular_velocity_x_dps",
        "right_angular_velocity_x_dps",
        "Angular velocity (deg/s)",
    ),
    "torque": (
        "left_actual_torque_nm",
        "right_actual_torque_nm",
        "Actual torque (Nm)",
    ),
}


def segment_and_normalize_gaits(
    data: pd.DataFrame,
    extrema: str = "max",
    min_cycle_samples: int = 60,
    prominence: float | None = None,
    normalized_points: int = 101,
) -> pd.DataFrame:
    """Segment left/right gait cycles and normalize each cycle to 0--100%.

    Consecutive hip-angle extrema define one gait cycle. All numeric signals are
    linearly interpolated to ``normalized_points`` samples. The returned table
    contains ``side``, ``gait_cycle``, and ``gait_percent`` identifier columns.

    Args:
        data: Exoskeleton log returned by :func:`load_log`.
        extrema: ``"max"`` to use angle maxima or ``"min"`` for minima.
        min_cycle_samples: Minimum spacing between two detected extrema.
        prominence: Optional minimum peak prominence. ``None`` selects it from
            10% of the hip-angle signal range for each side.
        normalized_points: Number of points per cycle; 101 represents 0--100%.
    """
    if extrema not in {"max", "min"}:
        raise ValueError("extrema must be either 'max' or 'min'")
    if min_cycle_samples < 2:
        raise ValueError("min_cycle_samples must be at least 2")
    if normalized_points < 2:
        raise ValueError("normalized_points must be at least 2")

    numeric_columns = list(data.select_dtypes(include=np.number).columns)
    new_percent = np.linspace(0.0, 100.0, normalized_points)
    normalized_cycles = []

    for side in ("left", "right"):
        angle_column = f"{side}_angle_x_deg"
        angle = data[angle_column].to_numpy(dtype=float)
        peak_signal = angle if extrema == "max" else -angle
        side_prominence = prominence
        if side_prominence is None:
            side_prominence = 0.1 * np.ptp(angle)

        extrema_indices, _ = find_peaks(
            peak_signal,
            distance=min_cycle_samples,
            prominence=side_prominence,
        )

        for gait_cycle, (start_index, end_index) in enumerate(
            zip(extrema_indices[:-1], extrema_indices[1:]), start=1
        ):
            # Include both boundary extrema in the cycle.
            cycle = data.iloc[start_index:end_index + 1]
            old_percent = np.linspace(0.0, 100.0, len(cycle))
            normalized = {
                column: np.interp(new_percent, old_percent,
                                  cycle[column].to_numpy(dtype=float))
                for column in numeric_columns
            }
            normalized["side"] = side
            normalized["gait_cycle"] = gait_cycle
            normalized["gait_percent"] = new_percent
            normalized_cycles.append(pd.DataFrame(normalized))

    if not normalized_cycles:
        raise ValueError(
            "No complete gait cycles were detected; reduce min_cycle_samples "
            "or prominence."
        )
    return pd.concat(normalized_cycles, ignore_index=True)


def load_log(
    csv_path: Path,
    start: float | None = None,
    end: float | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
) -> pd.DataFrame:
    """Load a log and optionally select a row-index and/or time interval."""
    data = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"elapsed_s"}
    required.update(column for pair in COLUMNS.values() for column in pair[:2])

    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    data = data.dropna(subset=list(required)).sort_values("elapsed_s")
    data = data.iloc[start_index:end_index]
    if start is not None:
        data = data[data["elapsed_s"] >= start]
    if end is not None:
        data = data[data["elapsed_s"] <= end]
    if data.empty:
        raise ValueError("The selected index/time range contains no data.")
    return data


def plot_log(data: pd.DataFrame, title: str) -> plt.Figure:
    """Create a three-panel figure for one exoskeleton log."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)  
    time = data["elapsed_s"]

    for axis, (_, (left, right, ylabel)) in zip(axes, COLUMNS.items()):
        axis.plot(time, data[left], label="Left", color="#0072B2", linewidth=1.1)
        axis.plot(time, data[right], label="Right", color="#D55E00", linewidth=1.1)
        axis.set_ylabel(ylabel)  
        axis.margins(x=0)   
        axis.grid(alpha=0.25)  

    axes[0].legend(frameon=False, ncols=2, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()  
    return fig  


def plot_normalized_gait_angles(
    normalized_gaits: pd.DataFrame,
    title: str,
    max_gaits: int = 10,
) -> plt.Figure:
    """Plot the first normalized hip-angle gait cycles for both sides."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    side_colors = {"left": "#0072B2", "right": "#D55E00"}

    for axis, side in zip(axes, ("left", "right")):
        side_data = normalized_gaits[normalized_gaits["side"] == side]
        gait_numbers = sorted(side_data["gait_cycle"].unique())[:max_gaits]
        angle_column = f"{side}_angle_x_deg"

        for gait_number in gait_numbers:
            gait = side_data[side_data["gait_cycle"] == gait_number]
            axis.plot(
                gait["gait_percent"],
                gait[angle_column],
                color=side_colors[side],
                linewidth=1.1,
                alpha=0.65,
                label=f"Gait {gait_number}",
            )

        axis.set_title(f"{side.capitalize()} hip ({len(gait_numbers)} gaits)")
        axis.set_xlabel("Normalized gait cycle (%)")
        axis.set_xlim(0, 100)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncols=2)

    axes[0].set_ylabel("Hip angle (deg)")
    fig.suptitle(f"{title}: normalized hip-angle gait cycles", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_dataset_summary(logs: dict[str, pd.DataFrame]) -> plt.Figure:
    """Compare all subjects at each walking-speed condition."""
    speeds = sorted({name[-2:] for name in logs})
    fig, axes = plt.subplots(3, len(speeds), figsize=(5 * len(speeds), 9),
                             sharex="col")
    colors = plt.get_cmap("tab10")

    for column_index, speed in enumerate(speeds):
        matching = [(name, data) for name, data in logs.items() if name.endswith(speed)]
        for row_index, (_, (left, right, ylabel)) in enumerate(COLUMNS.items()):
            axis = axes[row_index, column_index]
            for subject_index, (name, data) in enumerate(matching):
                # Each trial starts at zero so recordings align on the time axis.
                time = data["elapsed_s"] - data["elapsed_s"].iloc[0]
                color = colors(subject_index % colors.N)
                subject = name[:-2].upper()
                axis.plot(time, data[left], color=color, linewidth=0.8,
                          alpha=0.8, label=f"{subject} left")
                axis.plot(time, data[right], color=color, linewidth=0.8,
                          alpha=0.8, linestyle="--", label=f"{subject} right")
            if row_index == 0:
                axis.set_title(f"Condition {speed}", fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == 2:
                axis.set_xlabel("Elapsed time (s)")
            axis.grid(alpha=0.2)
            axis.margins(x=0)

    axes[0, -1].legend(frameon=False, fontsize=8, ncols=2,
                       bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Exoskeleton log comparison", fontweight="bold")
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    default_log = Path(__file__).parent / "exo_logs" / "yth10.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=default_log.parent,
        help="CSV file or directory of CSV logs (default: Data/exo_logs)",
    )
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument(
        "--start-index", type=int, help="First row to plot (included)"
    )
    parser.add_argument(
        "--end-index", type=int, help="Last row boundary (excluded)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path for a file, or output directory for a folder",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure")
    parser.add_argument(
        "--gaits", type=int, default=10,
        help="Number of normalized gait cycles to plot per side (default: 10)",
    )
    parser.add_argument(
        "--extrema", choices=("max", "min"), default="max",
        help="Hip-angle extrema used as gait boundaries (default: max)",
    )
    parser.add_argument(
        "--min-cycle-samples", type=int, default=60,
        help="Minimum samples between gait boundaries (default: 60)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_output_dir = Path(__file__).parent / "figures"
    if args.start is not None and args.end is not None and args.start >= args.end:
        raise ValueError("--start must be less than --end")
    if args.start_index is not None and args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < 0:
        raise ValueError("--end-index must be non-negative")
    if (args.start_index is not None and args.end_index is not None 
            and args.start_index >= args.end_index):  
        raise ValueError("--start-index must be less than --end-index")   
    if args.gaits < 1:
        raise ValueError("--gaits must be at least 1")

    if args.input.is_file():   
        data = load_log(
            args.input, args.start, args.end, args.start_index, args.end_index
        )
        figure = plot_log(data, args.input.stem)
        output = args.output or default_output_dir / f"{args.input.stem}_plot.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")
        normalized = segment_and_normalize_gaits(
            data, extrema=args.extrema,
            min_cycle_samples=args.min_cycle_samples,
        )  
        gait_figure = plot_normalized_gait_angles(
            normalized, args.input.stem, max_gaits=args.gaits
        )
        gait_output = output.parent / f"{args.input.stem}_gaits.png"
        gait_figure.savefig(gait_output, dpi=300, bbox_inches="tight")
        print(f"Saved {len(data):,} samples to {output}")
        print(f"Saved normalized gait angles to {gait_output}")
        figures = [figure, gait_figure]
    elif args.input.is_dir():
        csv_files = sorted(args.input.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {args.input}")
        output_dir = args.output or default_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        logs = {}
        figures = []
        for csv_path in csv_files:
            data = load_log(
                csv_path, args.start, args.end, args.start_index, args.end_index
            )
            logs[csv_path.stem] = data
            figure = plot_log(data, csv_path.stem)  
            figure.savefig(output_dir / f"{csv_path.stem}_plot.png", dpi=300,
                           bbox_inches="tight")
            figures.append(figure)
            normalized = segment_and_normalize_gaits(
                data, extrema=args.extrema,
                min_cycle_samples=args.min_cycle_samples,
            )
            gait_figure = plot_normalized_gait_angles(
                normalized, csv_path.stem, max_gaits=args.gaits
            )
            gait_figure.savefig(
                output_dir / f"{csv_path.stem}_gaits.png", dpi=300,
                bbox_inches="tight",
            )
            figures.append(gait_figure)
            if not args.show:
                plt.close(figure)
                plt.close(gait_figure)
        summary = plot_dataset_summary(logs)
        summary.savefig(output_dir / "all_logs_summary.png", dpi=300,
                        bbox_inches="tight")
        figures.append(summary)
        print(
            f"Saved {len(csv_files)} raw plots, {len(csv_files)} gait plots, "
            f"and one summary to {output_dir}"
        )
    else:
        raise FileNotFoundError(f"Input does not exist: {args.input}")

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()  