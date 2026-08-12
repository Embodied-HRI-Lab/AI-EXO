"""Extract normalized bilateral gait cycles from the ExpertData dataset.

Examples
--------
Process one trial and display its gait cycles::

    python ExpertData/data_analysis.py \
        ExpertData/BT01/normal_walk_1_1_1-2_on --show

Process every subject/trial under ExpertData::

    python ExpertData/data_analysis.py ExpertData
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ANGLE_COLUMNS = ("hip_flexion_l", "hip_flexion_r")
SIGNAL_COLUMNS = {
    "angle": ANGLE_COLUMNS,
    "velocity": ("hip_flexion_l_velocity", "hip_flexion_r_velocity"),
    "moment": ("hip_flexion_l_moment", "hip_flexion_r_moment"),
}
SIGNAL_LABELS = {
    "angle": "Hip flexion angle (deg)",
    "velocity": "Hip angular velocity (deg/s)",
    "moment": "Hip flexion moment",
}


def load_expert_angles(
    csv_path: Path,
    start_index: int | None = None,
    end_index: int | None = None,
) -> pd.DataFrame:
    """Load time and bilateral hip-flexion angles from an expert CSV."""
    data = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"time", *ANGLE_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    data = data.loc[:, ["time", *ANGLE_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna().iloc[start_index:end_index].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No valid samples selected from {csv_path}")
    return data


def segment_expert_gaits(
    data: pd.DataFrame,
    reference_side: str = "left",
    extrema: str = "max",
    min_cycle_samples: int = 100,
    prominence: float | None = None,
    normalized_points: int = 101,
    max_gaits: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment paired left/right hip angles and normalize cycles to 0--100%.

    Returns
    -------
    gait_data:
        Array shaped ``(n_gaits, normalized_points, 2)`` in left/right order.
    boundaries:
        Source-row extrema indices shaped ``(n_gaits, 2)``.
    """
    if reference_side not in {"left", "right"}:
        raise ValueError("reference_side must be 'left' or 'right'")
    if extrema not in {"max", "min"}:
        raise ValueError("extrema must be 'max' or 'min'")
    if min_cycle_samples < 2 or normalized_points < 2:
        raise ValueError("Cycle distance and normalized points must be at least 2")

    reference_column = ANGLE_COLUMNS[0 if reference_side == "left" else 1]
    reference = data[reference_column].to_numpy(dtype=float)
    peak_signal = reference if extrema == "max" else -reference
    if prominence is None:
        prominence = 0.1 * np.ptp(reference)

    extrema_indices, _ = find_peaks(
        peak_signal,
        distance=min_cycle_samples,
        prominence=prominence,
    )
    if len(extrema_indices) < 2:  
        raise ValueError(
            "No complete gait cycle detected; adjust the selected interval, "
            "min_cycle_samples, prominence, or extrema"
        )  

    target_phase = np.linspace(0.0, 100.0, normalized_points)
    gait_data = []
    boundaries = []
    bilateral = data.loc[:, ANGLE_COLUMNS].to_numpy(dtype=float)

    for start, end in zip(extrema_indices[:-1], extrema_indices[1:]):
        cycle = bilateral[start:end + 1]
        source_phase = np.linspace(0.0, 100.0, len(cycle))
        gait_data.append(
            np.column_stack(
                [
                    np.interp(target_phase, source_phase, cycle[:, side])
                    for side in range(2)
                ]
            )
        )
        boundaries.append((start, end))
        if max_gaits is not None and len(gait_data) >= max_gaits:
            break

    return np.asarray(gait_data), np.asarray(boundaries, dtype=int)


def load_trial_signal(
    angle_csv: Path,
    target_time: np.ndarray,
    signal: str,
) -> np.ndarray:
    """Load and time-align bilateral velocity or moment for one trial."""
    if signal not in {"velocity", "moment"}:
        raise ValueError("signal must be 'velocity' or 'moment'")
    stem = output_stem(angle_csv)
    suffixes = (
        ("_velocity.csv",) if signal == "velocity"
        else ("_moment_filt_bio.csv", "_moment_filt.csv", "_moment.csv")
    )
    source_path = next(
        (angle_csv.parent / f"{stem}{suffix}" for suffix in suffixes
         if (angle_csv.parent / f"{stem}{suffix}").is_file()),
        None,
    )
    if source_path is None:
        raise FileNotFoundError(f"No {signal} CSV found beside {angle_csv}")

    columns = SIGNAL_COLUMNS[signal]
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    missing = [column for column in ("time", *columns) if column not in source]
    if missing:
        raise ValueError(f"{source_path} is missing: {', '.join(missing)}")
    source = source.loc[:, ["time", *columns]].apply(pd.to_numeric, errors="coerce")
    source = source.dropna().sort_values("time")
    return np.column_stack(
        [
            np.interp(target_time, source["time"], source[column])
            for column in columns
        ]
    )


def normalize_signal_cycles(
    signal_data: np.ndarray,
    boundaries: np.ndarray,
    normalized_points: int,
) -> np.ndarray:
    """Normalize paired signals using gait boundaries detected from angles."""
    target_phase = np.linspace(0.0, 100.0, normalized_points)
    cycles = []
    for start, end in boundaries:
        cycle = signal_data[start:end + 1]
        source_phase = np.linspace(0.0, 100.0, len(cycle))
        cycles.append(
            np.column_stack(
                [
                    np.interp(target_phase, source_phase, cycle[:, side])
                    for side in range(2)
                ]
            )
        )
    return np.asarray(cycles)


def gait_array_to_dataframe(gait_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert ``(gait, phase, side)`` data into a tidy DataFrame."""
    gaits = gait_sets["angle"] 
    frames = [] 
    gait_percent = np.linspace(0.0, 100.0, gaits.shape[1]) 
    for gait_index, gait in enumerate(gaits, start=1): 
        values = {
            "gait_cycle": gait_index,
            "gait_percent": gait_percent,
        }
        for signal, signal_gaits in gait_sets.items():
            columns = SIGNAL_COLUMNS[signal]
            values[columns[0]] = signal_gaits[gait_index - 1, :, 0]
            values[columns[1]] = signal_gaits[gait_index - 1, :, 1]
        frames.append(pd.DataFrame(values))
    return pd.concat(frames, ignore_index=True)


def plot_expert_gaits(gait_sets: dict[str, np.ndarray], title: str) -> plt.Figure:
    """Plot normalized angle, velocity, and/or moment gait cycles."""
    first = next(iter(gait_sets.values()))
    phase = np.linspace(0.0, 100.0, first.shape[1])
    rows = len(gait_sets)
    figure, axes = plt.subplots(
        rows, 2, figsize=(12, 3.8 * rows), squeeze=False, sharex=True
    )
    colors = ("#0072B2", "#D55E00")

    for row, (signal, gaits) in enumerate(gait_sets.items()):
        for side, (color, side_name) in enumerate(zip(colors, ("Left", "Right"))):
            axis = axes[row, side]
            for gait in gaits:
                axis.plot(
                    phase, gait[:, side], color=color, alpha=0.22, linewidth=0.8
                )
            mean = np.mean(gaits[:, :, side], axis=0)
            std = np.std(gaits[:, :, side], axis=0)
            axis.plot(phase, mean, color=color, linewidth=2.5, label="Mean")
            axis.fill_between(
                phase, mean - std, mean + std, color=color, alpha=0.18,
                label="Mean ± SD",
            )
            if row == 0:
                axis.set_title(f"{side_name} hip ({len(gaits)} gaits)")
            if side == 0:
                axis.set_ylabel(SIGNAL_LABELS[signal])
            axis.set_xlim(0.0, 100.0)
            axis.grid(axis="y", alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(frameon=False)

    for axis in axes[-1]:
        axis.set_xlabel("Normalized gait cycle (%)")
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout()
    return figure


def walking_speed_from_trial(trial_name: str) -> str | None:
    """Return a numeric normal-walking speed label such as ``1.2``."""
    match = re.search(r"normal_walk_.*_(\d+)-(\d+)_on$", trial_name)
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def plot_walking_speed_comparison(
    speed_gaits: dict[str, dict[str, np.ndarray]],
    subject: str,
    gait_index: int,
) -> plt.Figure:
    """Overlay one normalized gait from every walking speed for one subject."""
    signals = list(next(iter(speed_gaits.values())).keys())
    rows = len(signals)
    figure, axes = plt.subplots(
        rows, 2, figsize=(12, 3.8 * rows), squeeze=False, sharex=True
    )
    phase = np.linspace(0.0, 100.0, next(iter(speed_gaits.values()))[signals[0]].shape[0])
    colors = plt.get_cmap("viridis")
    ordered_speeds = sorted(speed_gaits, key=float)

    for row, signal in enumerate(signals):
        for side, side_name in enumerate(("Left", "Right")):
            axis = axes[row, side]
            for speed_index, speed in enumerate(ordered_speeds):
                color = colors(speed_index / max(len(ordered_speeds) - 1, 1))
                axis.plot(
                    phase,
                    speed_gaits[speed][signal][:, side],
                    color=color,
                    linewidth=2.0,
                    label=f"{speed} m/s",
                )
            if row == 0:
                axis.set_title(f"{side_name} hip")
            if side == 0:
                axis.set_ylabel(SIGNAL_LABELS[signal])
            axis.set_xlim(0.0, 100.0)
            axis.grid(axis="y", alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    for axis in axes[-1]:
        axis.set_xlabel("Normalized gait cycle (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(ordered_speeds),
        frameon=False, bbox_to_anchor=(0.5, 0.96),
    )
    figure.suptitle(
        f"{subject}: gait {gait_index} across walking speeds",
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    return figure


def find_angle_files(
    input_path: Path,
    use_raw_angles: bool,
    activities: list[str] | None = None,
) -> list[Path]:
    """Resolve a CSV, trial folder, or ExpertData root into angle CSV files."""
    if input_path.is_file():
        files = [input_path]
    else:
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input does not exist: {input_path}")
        suffix = "_angle.csv" if use_raw_angles else "_angle_filt.csv"
        files = sorted(input_path.rglob(f"*{suffix}"))
        if not files:
            raise FileNotFoundError(f"No *{suffix} files found under {input_path}")

    if activities:
        search_terms = [term.casefold() for term in activities]
        files = [
            path for path in files
            if any(term in path.parent.name.casefold() for term in search_terms)
        ]
        if not files:
            raise FileNotFoundError(
                f"No angle files match activities: {', '.join(activities)}"
            )
    return files


def output_stem(csv_path: Path) -> str:
    """Remove the angle-file suffix from an expert trial filename."""
    for suffix in ("_angle_filt", "_angle"):
        if csv_path.stem.endswith(suffix):
            return csv_path.stem[:-len(suffix)]
    return csv_path.stem


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=root)
    parser.add_argument("--output", type=Path, default=root / "processed")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--gaits", type=int, help="Maximum gaits per trial")
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--reference-side", choices=("left", "right"), default="left")
    parser.add_argument("--extrema", choices=("max", "min"), default="max")
    parser.add_argument("--min-cycle-samples", type=int, default=100)
    parser.add_argument("--prominence", type=float)
    parser.add_argument("--raw-angles", action="store_true")
    parser.add_argument(
        "--signals",
        nargs="+",
        choices=("angle", "velocity", "moment"),
        default=("angle",),
        help="Signals to export and plot (default: angle)",
    )
    parser.add_argument(
        "--activities",
        nargs="+",
        metavar="NAME",
        help=(
            "Only process trial folders containing one of these names, e.g. "
            "--activities normal_walk stairs"
        ),
    )
    parser.add_argument(
        "--compare-speeds",
        action="store_true",
        help="Create one additional gait-overlay figure across walking speeds",
    )
    parser.add_argument(
        "--gait-index", type=int, default=1,
        help="One-based gait cycle used in the speed comparison (default: 1)",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_index is not None and args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < 0:
        raise ValueError("--end-index must be non-negative")
    if (args.start_index is not None and args.end_index is not None
            and args.start_index >= args.end_index):
        raise ValueError("--start-index must be less than --end-index")
    if args.gaits is not None and args.gaits < 1:
        raise ValueError("--gaits must be positive")
    if args.gait_index < 1:
        raise ValueError("--gait-index must be at least 1")

    angle_files = find_angle_files(
        args.input,
        args.raw_angles,
        activities=args.activities,
    )   
    if args.activities:  
        print(
            f"Selected {len(angle_files)} trial(s) matching: "
            f"{', '.join(args.activities)}"
        )    
    args.output.mkdir(parents=True, exist_ok=True)     
    succeeded = 0  
    speed_comparisons: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)    

    for csv_path in angle_files:
        try:
            data = load_expert_angles(csv_path, args.start_index, args.end_index) 
            gaits, boundaries = segment_expert_gaits(
                data,
                reference_side=args.reference_side,
                extrema=args.extrema,
                min_cycle_samples=args.min_cycle_samples,
                prominence=args.prominence,
                normalized_points=args.points,
                max_gaits=args.gaits,
            )
            gait_sets = {"angle": gaits}
            target_time = data["time"].to_numpy(dtype=float)
            for signal in args.signals:
                if signal == "angle":
                    continue
                aligned = load_trial_signal(csv_path, target_time, signal)
                gait_sets[signal] = normalize_signal_cycles(
                    aligned, boundaries, args.points
                )
        except (ValueError, FileNotFoundError) as error:
            print(f"Skipped {csv_path}: {error}")
            continue

        subject = csv_path.parents[1].name
        trial = csv_path.parent.name
        destination = args.output / subject / trial
        destination.mkdir(parents=True, exist_ok=True)
        name = output_stem(csv_path)

        # Preserve the CLI order, while angle remains mandatory for segmentation.
        gait_sets = {
            signal: gait_sets[signal]
            for signal in ("angle", *args.signals)
            if signal in gait_sets
        }
        speed = walking_speed_from_trial(trial)
        selected_index = args.gait_index - 1
        if (
            args.compare_speeds
            and speed is not None
            and selected_index < len(gaits)
        ):
            speed_comparisons[subject][speed] = {
                signal: signal_gaits[selected_index]
                for signal, signal_gaits in gait_sets.items()
            }
        tidy = gait_array_to_dataframe(gait_sets)
        tidy.to_csv(destination / f"{name}_gaits.csv", index=False)
        npz_values = {
            "gait_data": gaits,
            "gait_percent": np.linspace(0.0, 100.0, args.points),
            "boundaries": boundaries,
            "boundary_times": data["time"].to_numpy()[boundaries],
            "columns": np.asarray(ANGLE_COLUMNS),
        }
        for signal, signal_gaits in gait_sets.items():
            npz_values[f"{signal}_gaits"] = signal_gaits
        np.savez(destination / f"{name}_gaits.npz", **npz_values)  

        figure = plot_expert_gaits(gait_sets, f"{subject} — {trial}")
        figure.savefig(
            destination / f"{name}_gaits.png", dpi=300, bbox_inches="tight"
        )
        if not args.show:
            plt.close(figure)
        print(f"{csv_path}: extracted {len(gaits)} gaits -> {destination}")
        succeeded += 1

    if args.compare_speeds:
        for subject, speed_gaits in speed_comparisons.items():
            if len(speed_gaits) < 2:
                print(f"Skipped speed comparison for {subject}: fewer than 2 speeds")
                continue
            figure = plot_walking_speed_comparison(
                speed_gaits,
                subject=subject,
                gait_index=args.gait_index,
            )
            comparison_output = (
                args.output / subject /
                f"{subject}_walking_speeds_gait_{args.gait_index}.png"
            )
            comparison_output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(comparison_output, dpi=300, bbox_inches="tight")
            if not args.show:
                plt.close(figure)   
            print(
                f"Saved {subject} walking-speed comparison "  
                f"({', '.join(sorted(speed_gaits, key=float))} m/s) -> "
                f"{comparison_output}"
            )    

    if args.show:
        plt.show()   
    print(f"Completed {succeeded}/{len(angle_files)} expert angle files")


if __name__ == "__main__":
    main()
