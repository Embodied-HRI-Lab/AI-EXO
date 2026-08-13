from kmp.demo_GMR import GMR_pred, KMP_pred    
from kmp.GMRbasedGP.utils.gmr import plot_gmm, Gmr     
from kmp.plot_pred import plot_bilateral_hip_mean_variance
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks



def load_hip_angle_csv(csv_path, start_index=0, end_index=None):
    """Read elapsed time and bilateral hip angles from one experiment CSV.

    ``start_index`` is included and ``end_index`` is excluded. The returned
    ``angles_deg`` array has shape ``(n_samples, 2)`` in left/right order.
    """
    path = Path(csv_path).expanduser()   
    if not path.is_file():  
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index <= start_index:
        raise ValueError("end_index must be greater than start_index")

    elapsed_s = []
    left_angle_deg = []
    right_angle_deg = []
    required = ("elapsed_s", "left_angle_x_deg", "right_angle_x_deg")

    # utf-8-sig removes the BOM present in the experiment CSV headers.
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

        for data_index, row in enumerate(reader):
            if data_index < start_index:
                continue
            if end_index is not None and data_index >= end_index:
                break
            try:
                elapsed_s.append(float(row["elapsed_s"]))
                left_angle_deg.append(float(row["left_angle_x_deg"]))
                right_angle_deg.append(float(row["right_angle_x_deg"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid angle data at CSV row {data_index + 2}"
                ) from exc

    if not elapsed_s:
        raise ValueError(f"No samples found in interval [{start_index}:{end_index}]")

    left = np.asarray(left_angle_deg, dtype=float)
    right = np.asarray(right_angle_deg, dtype=float)
    return {
        "csv_path": path,
        "elapsed_s": np.asarray(elapsed_s, dtype=float),
        "left_angle_deg": left,
        "right_angle_deg": right,
        "angles_deg": np.column_stack((left, right)),
    }


def load_hip_angles_from_data_folder(
    data_folder=None,
    pattern="*.csv",
    start_index=0,
    end_index=None,
):
    """Load bilateral hip angles from every matching experiment CSV."""
    if data_folder is None:
        repository_root = Path(__file__).resolve().parents[1]
        candidates = (
            repository_root / "RealData" / "exo_logs",
            repository_root / "Data" / "exo_logs",  # Legacy layout.
            repository_root / "Higher_Controller" / "logs",
        )
        data_folder = next((path for path in candidates if path.is_dir()), None)
        if data_folder is None:
            raise FileNotFoundError(
                "Could not find RealData/exo_logs, Data/exo_logs, or "
                "Higher_Controller/logs"
            )
    folder = Path(data_folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"Data folder does not exist: {folder}")

    csv_files = sorted(folder.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")

    return {
        path.stem: load_hip_angle_csv(path, start_index, end_index)
        for path in csv_files
    }


def segment_bilateral_hip_gaits(
    angles_deg, 
    max_gaits=10, 
    normalized_points=101, 
    min_cycle_samples=60,  
    prominence=None, 
    extrema="max",   
):  
    """Split bilateral angles using left-hip extrema and normalize each gait."""
    angles = np.asarray(angles_deg, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError("angles_deg must have shape (n_samples, 2)")
    if extrema not in ("max", "min"):
        raise ValueError("extrema must be 'max' or 'min'")
    if max_gaits < 2:
        raise ValueError("max_gaits must be at least 2 for GMR")

    left_angle = angles[:, 0]  
    peak_signal = left_angle if extrema == "max" else -left_angle  
    if prominence is None:  
        prominence = 0.1 * np.ptp(left_angle)   
    boundaries, _ = find_peaks(
        peak_signal,
        distance=min_cycle_samples,
        prominence=prominence,
    )
    if len(boundaries) < 3:
        raise ValueError(
            "Fewer than two complete gaits detected; adjust the input range, "
            "min_cycle_samples, or prominence"
        )

    normalized_phase = np.linspace(0.0, 1.0, normalized_points)
    normalized_gaits = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        gait = angles[start:end + 1]
        original_phase = np.linspace(0.0, 1.0, len(gait))
        normalized_gaits.append(
            np.column_stack(
                [
                    np.interp(normalized_phase, original_phase, gait[:, side])
                    for side in range(2)
                ]
            )
        )
        if len(normalized_gaits) >= max_gaits:
            break
    return np.asarray(normalized_gaits)


def learn_gait_distribution_gmr_kmp(
    gait_data,
    nb_states=5,
    kh=6.0,
    lambda_mean=0.01,
    lambda_covariance=0.6,
    via_indices=None,
    via_points=None,
    via_variances=None, 
    via_variance=1.0e-8,  
    via_scale=None, 
):
    """Learn the mean and variance of multiple normalized gait cycles.

    Parameters
    ----------
    gait_data : array-like
        Shape ``(n_gaits, n_points)`` for one signal or
        ``(n_gaits, n_points, n_signals)`` for multiple signals. Every gait
        must already be sampled on the same normalized gait axis.
    nb_states : int
        Number of Gaussian components used by GMR.
    kh, lambda_mean, lambda_covariance : float
        KMP kernel width and regularization parameters.
    via_indices : sequence of int, optional
        Normalized sample indices at which KMP constraints are imposed. 
    via_points : array-like, optional
        Desired values with shape ``(n_via, n_signals)``.
    via_scale : float or sequence, optional
        If ``via_points`` is omitted, construct each via point by multiplying
        the GMR mean at ``via_indices`` by this scale.
    via_variances : sequence, optional
        Scalar covariance assigned to each via point.
    via_variance : float
        Shared scalar variance used for every via point when
        ``via_variances`` is not provided.

    Returns
    -------
    dict
        GMR/KMP means, covariance matrices, diagonal variances, normalized
        gait percentage, and the fitted GMR model.
    """
    gaits = np.asarray(gait_data, dtype=float)
    if gaits.ndim == 2:
        gaits = gaits[:, :, None]
    if gaits.ndim != 3:
        raise ValueError(
            "gait_data must have shape (n_gaits, n_points[, n_signals])"
        )

    n_gaits, n_points, n_signals = gaits.shape  
    if n_gaits < 2:  
        raise ValueError("At least two gait cycles are required")
    if n_points < 2 or n_signals < 1:
        raise ValueError("Each gait must contain at least two data points")
    if not np.isfinite(gaits).all():
        raise ValueError("gait_data contains NaN or infinite values")
    if not 1 <= int(nb_states) <= n_points:
        raise ValueError("nb_states must be between 1 and n_points")
    if lambda_mean <= 0 or lambda_covariance <= 0:
        raise ValueError("KMP regularization parameters must be positive")

    # Use 0--1 as the regression input and report 0--100% to the caller.
    phase = np.linspace(0.0, 1.0, n_points)[:, None]
    demonstrations = np.vstack(
        [np.hstack((phase, gait)) for gait in gaits]
    )
    x_train = demonstrations[:, :1]  
    y_train = demonstrations[:, 1:]   

    gmr_mean, gmr_covariance, gmr_model = GMR_pred(
        demos_np=demonstrations, 
        X=x_train,
        Xt=phase, 
        Y=y_train,
        nb_data=n_points, 
        nb_samples=n_gaits,
        nb_states=int(nb_states),
        input_dim=1,
        output_dim=n_signals,
    )   

    if via_indices is None:
        via_indices_array = np.empty(0, dtype=int)
        via_points_array = np.empty((0, n_signals), dtype=float)
        via_variance_list = []
    else:
        via_indices_array = np.atleast_1d(np.asarray(via_indices, dtype=int))
        if np.any(via_indices_array < 0) or np.any(via_indices_array >= n_points):
            raise ValueError("via_indices contains an out-of-range index")
        scale_vector = np.ones(n_signals, dtype=float)
        if via_points is None:
            if via_scale is None:
                raise ValueError(
                    "Set via_points or via_scale when via_indices is provided"
                )
            scale = np.asarray(via_scale, dtype=float)
            if scale.ndim > 1 or (scale.ndim == 1 and len(scale) != n_signals):
                raise ValueError("via_scale must be scalar or one value per signal")
            scale_vector = np.broadcast_to(scale, (n_signals,)).astype(float)
            via_points_array = gmr_mean[via_indices_array] * scale_vector
        else:
            via_points_array = np.asarray(via_points, dtype=float).reshape(
                len(via_indices_array), n_signals
            )
        if via_variances is None:
            if via_variance <= 0:
                raise ValueError("via_variance must be positive")
            via_variance_list = [float(via_variance)] * len(via_indices_array)
        else:
            supplied_variances = list(via_variances)
            if len(supplied_variances) != len(via_indices_array):
                raise ValueError("via_variances and via_indices must match")
            via_variance_list = []
            for variance in supplied_variances:
                covariance = np.asarray(variance, dtype=float)
                if covariance.ndim == 0:
                    covariance = float(covariance) * np.eye(n_signals)
                if covariance.shape != (n_signals, n_signals):
                    raise ValueError(
                        "Each via variance must be scalar or an "
                        "(n_signals, n_signals) covariance matrix"
                    )
                via_variance_list.append(covariance)
            print("I am here !!!")

    print("via_variance_list: ", via_variance_list)    
    _, _, kmp_trajectory = KMP_pred(
        Xt=phase, 
        mu_gmr=gmr_mean,  
        sigma_gmr=gmr_covariance,  
        viaNum=len(via_indices_array),
        viaFlag=np.ones(len(via_indices_array)),
        via_time=phase[via_indices_array, 0],
        via_points=via_points_array,
        via_var_list=via_variance_list,
        dt=1.0 / (n_points - 1),
        lamda_1=float(lambda_mean),
        lamda_2=float(lambda_covariance),
        kh=float(kh),
        output_dim=n_signals,
        dim=1,
    )

    kmp_covariance = np.asarray(kmp_trajectory["sigma"])
    return {
        "gait_percent": np.linspace(0.0, 100.0, n_points),
        "gmr_mean": np.asarray(gmr_mean),
        "gmr_covariance": np.asarray(gmr_covariance),
        "gmr_variance": np.diagonal(gmr_covariance, axis1=1, axis2=2),
        "kmp_mean": np.asarray(kmp_trajectory["mu"]),
        "kmp_covariance": kmp_covariance,
        "kmp_variance": np.diagonal(kmp_covariance, axis1=1, axis2=2),
        "via_indices": via_indices_array,
        "via_points": via_points_array,
        "via_covariance": np.asarray(via_variance_list),
        "gmr_model": gmr_model,
    }


def build_parser():   
    repository_root = Path(__file__).resolve().parents[1]  
    default_csv = repository_root / "RealData" / "exo_logs" / "hjc13.csv" 
    parser = argparse.ArgumentParser(
        description="Learn and plot bilateral hip gait mean/variance with GMR/KMP"
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument(
        "--output", type=Path,
        help="Output PNG path (default: RealData/figures/<csv>_gmr_kmp.png)",
    )
    parser.add_argument("--start-index", type=int, default=0)   
    parser.add_argument("--end-index", type=int)    
    parser.add_argument("--gaits", type=int, default=10)     
    parser.add_argument("--points", type=int, default=101)    
    parser.add_argument("--states", type=int, default=5)   
    parser.add_argument("--min-cycle-samples", type=int, default=60)     
    parser.add_argument("--prominence", type=float)   
    parser.add_argument("--extrema", choices=("max", "min"), default="max")
    parser.add_argument(
        "--via-indices", type=int, nargs="+", default=(0, 50, 100),
        help="KMP constraint indices (default: 0 50 100)",
    )
    parser.add_argument(
        "--via-scale", type=float, default=1.8,
        help="Stretch GMR means to form KMP via points (default: 1.8)",
    )
    parser.add_argument(
        "--via-variance", type=float, default=1.0e-8,
        help="Shared KMP via-point variance (default: 1e-8)",
    )
    parser.add_argument(
        "--kmp-lambda-covariance", type=float, default=0.6,
        help="KMP covariance regularization lambda (default: 0.6)",
    )
    parser.add_argument(
        "--font-size", type=float, default=14.0,
        help="Base plot font size in points (default: 14)",
    )
    parser.add_argument("--show", action="store_true")   
    return parser  


def main():   
    args = build_parser().parse_args()   
    if args.output is None:
        repository_root = Path(__file__).resolve().parents[1]
        args.output = (
            repository_root / "RealData" / "figures" /
            f"{args.csv.stem}_bilateral_gmr_kmp.png"
        )
    csv_data = load_hip_angle_csv(
        args.csv,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    gait_data = segment_bilateral_hip_gaits(
        csv_data["angles_deg"],
        max_gaits=args.gaits,
        normalized_points=args.points,
        min_cycle_samples=args.min_cycle_samples,
        prominence=args.prominence,
        extrema=args.extrema,
    )   
    print(
        f"Loaded {len(csv_data['elapsed_s'])} samples from {args.csv}; "
        f"using {len(gait_data)} normalized gaits with shape {gait_data.shape}"
    )

    result = learn_gait_distribution_gmr_kmp(
        gait_data,
        nb_states=args.states,
        via_indices=args.via_indices,
        via_scale=args.via_scale,
        via_variance=args.via_variance,
        lambda_covariance=args.kmp_lambda_covariance,
    )
    print(
        f"KMP via indices={list(result['via_indices'])}, "
        f"GMR stretch scale={args.via_scale}, "
        f"via variance={args.via_variance:.3e}"
    )
    print(f"Stretched via points (left/right deg):\n{result['via_points']}")
    plot_bilateral_hip_mean_variance(
        result,
        gait_data=gait_data,
        save_path=args.output,
        show=args.show,
        font_size=args.font_size,
    )

    result_output = args.output.with_suffix(".npz")
    np.savez(
        result_output,
        gait_percent=result["gait_percent"],
        gmr_mean=result["gmr_mean"],
        gmr_variance=result["gmr_variance"],
        gmr_covariance=result["gmr_covariance"],
        kmp_mean=result["kmp_mean"],
        kmp_variance=result["kmp_variance"],
        kmp_covariance=result["kmp_covariance"],
        via_indices=result["via_indices"],
        via_points=result["via_points"],
        via_covariance=result["via_covariance"],
    )
    print(f"Saved GMR/KMP arrays: {result_output}")


if __name__ == "__main__": 
    main()  