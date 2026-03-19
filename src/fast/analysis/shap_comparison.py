"""Helpers for comparative SHAP analysis across FAST experiment conditions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import pandas as pd
import scipy.signal
import shap
import torch
from scipy import stats
from transformers import PretrainedConfig

from fast.data import CLASSES, Electrodes, SUBJECTS, Zones
from fast.models import FAST


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 100.0),
}

LANGUAGE_ZONE_GROUPS: dict[str, tuple[str, ...]] = {
    "Language": ("Pre-frontal", "Frontal", "Temporal"),
    "Motor-Speech": ("Pre-central", "Central"),
    "Non-Language": ("Parietal", "Post-central", "Occipital"),
}

DEFAULT_FOCUS_RUNS = (
    "run1_original",
    "run2_condB",
    "run3_condC",
    "run6_A_B_C",
    "run7_B_C",
    "run9_augment_no_es",
    "run11_highpass4",
)

EXPERIMENT_NAME_ALIASES = {
    "Run 1: Original (A)": "run1_original",
    "Run 1: Original Data (A)": "run1_original",
    "Run 2: Cond B (No Delta)": "run2_condB",
    "Run 2: Condition B (No Delta)": "run2_condB",
    "Run 3: Cond C (No Artifacts)": "run3_condC",
    "Run 3: Condition C (No Artifacts)": "run3_condC",
    "Run 4: A + B": "run4_A_B",
    "Run 4: Original + Condition B": "run4_A_B",
    "Run 5: A + C": "run5_A_C",
    "Run 5: Original + Condition C": "run5_A_C",
    "Run 6: A + B + C": "run6_A_B_C",
    "Run 6: Original + Cond B + Cond C": "run6_A_B_C",
    "Run 7: B + C": "run7_B_C",
    "Run 7: Condition B + Condition C": "run7_B_C",
    "Run 8: Augmented + ES": "run8_augment",
    "Run 8: Augmentations + Cool Down": "run8_augment",
    "Run 9: Augmented No ES": "run9_augment_no_es",
    "Run 9: Augmentations (No Early Stopping)": "run9_augment_no_es",
    "Run 10: Kaggle Replication": "run10_kaggle",
    "Run 11: High-pass 4 Hz": "run11_highpass4",
}


@dataclass(frozen=True)
class ExperimentSpec:
    """Metadata for one trained FAST experiment."""

    run_key: str
    display_name: str
    results_dir_name: str
    checkpoint_name: str
    h5_paths: tuple[Path, ...]

    @property
    def condition_family(self) -> str:
        if self.run_key in {"run1_original", "run8_augment", "run9_augment_no_es", "run10_kaggle"}:
            return "Original-family"
        if self.run_key in {"run2_condB", "run3_condC", "run11_highpass4"}:
            return "Ablation"
        return "Mixed"


def default_model_config(sfreq: int = 250) -> PretrainedConfig:
    """Return the FAST model configuration used by the training runs."""

    return PretrainedConfig(
        electrodes=Electrodes,
        zone_dict=Zones,
        dim_cnn=32,
        dim_token=32,
        seq_len=800,
        window_len=sfreq,
        slide_step=sfreq // 2,
        head="Conv4Layers",
        n_classes=5,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
    )


def default_experiments(project_root: str | Path) -> dict[str, ExperimentSpec]:
    """Build the canonical experiment registry for comparative analysis."""

    root = Path(project_root).resolve()
    processed = root / "Processed"
    return {
        "run1_original": ExperimentSpec(
            run_key="run1_original",
            display_name="Run 1: Original (A)",
            results_dir_name="run1_original",
            checkpoint_name="best_subject.pth",
            h5_paths=(processed / "BCIC2020Track3.h5",),
        ),
        "run2_condB": ExperimentSpec(
            run_key="run2_condB",
            display_name="Run 2: Cond B (No Delta)",
            results_dir_name="run2_condB",
            checkpoint_name="best_subject.pth",
            h5_paths=(processed / "BCIC2020Track3_ICA_no_delta.h5",),
        ),
        "run3_condC": ExperimentSpec(
            run_key="run3_condC",
            display_name="Run 3: Cond C (No Artifacts)",
            results_dir_name="run3_condC",
            checkpoint_name="best_subject.pth",
            h5_paths=(processed / "BCIC2020Track3_ICA_no_artifacts.h5",),
        ),
        "run4_A_B": ExperimentSpec(
            run_key="run4_A_B",
            display_name="Run 4: A + B",
            results_dir_name="run4_A_B",
            checkpoint_name="best_subject.pth",
            h5_paths=(
                processed / "BCIC2020Track3.h5",
                processed / "BCIC2020Track3_ICA_no_delta.h5",
            ),
        ),
        "run5_A_C": ExperimentSpec(
            run_key="run5_A_C",
            display_name="Run 5: A + C",
            results_dir_name="run5_A_C",
            checkpoint_name="best_subject.pth",
            h5_paths=(
                processed / "BCIC2020Track3.h5",
                processed / "BCIC2020Track3_ICA_no_artifacts.h5",
            ),
        ),
        "run6_A_B_C": ExperimentSpec(
            run_key="run6_A_B_C",
            display_name="Run 6: A + B + C",
            results_dir_name="run6_A_B_C",
            checkpoint_name="best_subject.pth",
            h5_paths=(
                processed / "BCIC2020Track3.h5",
                processed / "BCIC2020Track3_ICA_no_delta.h5",
                processed / "BCIC2020Track3_ICA_no_artifacts.h5",
            ),
        ),
        "run7_B_C": ExperimentSpec(
            run_key="run7_B_C",
            display_name="Run 7: B + C",
            results_dir_name="run7_B_C",
            checkpoint_name="best_subject.pth",
            h5_paths=(
                processed / "BCIC2020Track3_ICA_no_delta.h5",
                processed / "BCIC2020Track3_ICA_no_artifacts.h5",
            ),
        ),
        "run8_augment": ExperimentSpec(
            run_key="run8_augment",
            display_name="Run 8: Augmented + ES",
            results_dir_name="run8_augment",
            checkpoint_name="best_subject.pth",
            h5_paths=(processed / "BCIC2020Track3.h5",),
        ),
        "run9_augment_no_es": ExperimentSpec(
            run_key="run9_augment_no_es",
            display_name="Run 9: Augmented No ES",
            results_dir_name="run9_augment_no_es",
            checkpoint_name="best_subject.pth",
            h5_paths=(processed / "BCIC2020Track3.h5",),
        ),
        "run10_kaggle": ExperimentSpec(
            run_key="run10_kaggle",
            display_name="Run 10: Kaggle Replication",
            results_dir_name="run10_kaggle_replication",
            checkpoint_name="final_model.pth",
            h5_paths=(processed / "BCIC2020Track3.h5",),
        ),
        "run11_highpass4": ExperimentSpec(
            run_key="run11_highpass4",
            display_name="Run 11: High-pass 4 Hz",
            results_dir_name="run11_highpass4",
            checkpoint_name="best_subject.pth",
            h5_paths=(processed / "BCIC2020Track3_highpass_4hz.h5",),
        ),
    }


def zone_indices(
    electrodes: Iterable[str] = Electrodes,
    zones: Mapping[str, Iterable[str]] = Zones,
) -> dict[str, list[int]]:
    """Map each zone to channel indices."""

    electrode_to_index = {electrode: idx for idx, electrode in enumerate(electrodes)}
    return {
        zone: [electrode_to_index[electrode] for electrode in members if electrode in electrode_to_index]
        for zone, members in zones.items()
    }


def summarise_language_groups(
    zone_values: Mapping[str, float],
    groups: Mapping[str, Iterable[str]] = LANGUAGE_ZONE_GROUPS,
) -> dict[str, float]:
    """Aggregate zone-level values into language, motor-speech, and non-language groups."""

    grouped = {}
    for group_name, group_zones in groups.items():
        values = [float(zone_values[zone]) for zone in group_zones if zone in zone_values]
        grouped[group_name] = float(np.mean(values)) if values else np.nan
    return grouped


def load_model(checkpoint_path: str | Path, config: PretrainedConfig, device: torch.device) -> FAST | None:
    """Load a FAST checkpoint and move it to the target device."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    model = FAST(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_subject_data(
    h5_paths: Iterable[str | Path],
    sid: str,
    *,
    n_bg: int,
    n_explain: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one subject from one or more H5 files and split into SHAP sets."""

    arrays_x: list[np.ndarray] = []
    arrays_y: list[np.ndarray] = []
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as handle:
            if sid not in handle:
                continue
            arrays_x.append(handle[sid]["X"][()].astype(np.float32))
            arrays_y.append(handle[sid]["Y"][()].astype(np.uint8))

    if not arrays_x:
        joined = ", ".join(str(path) for path in h5_paths)
        raise ValueError(f"No data found for subject {sid} in {joined}")

    x = np.concatenate(arrays_x, axis=0)
    y = np.concatenate(arrays_y, axis=0)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(x), generator=generator)
    bg_count = min(n_bg, max(1, len(x) // 2))
    explain_count = min(n_explain, max(1, len(x) - bg_count))

    x_bg = torch.tensor(x[permutation[:bg_count]], dtype=torch.float32)
    x_explain = torch.tensor(x[permutation[bg_count:bg_count + explain_count]], dtype=torch.float32)
    y_explain = torch.tensor(y[permutation[bg_count:bg_count + explain_count]], dtype=torch.long)
    return x_bg, x_explain, y_explain


def _normalise_shap_output(shap_values: object, expected_batch: int) -> list[np.ndarray]:
    if isinstance(shap_values, list):
        normalised = [
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for value in shap_values
        ]
    elif isinstance(shap_values, torch.Tensor):
        normalised = np.asarray(shap_values.detach().cpu().numpy())
    else:
        normalised = np.asarray(shap_values)

    if isinstance(normalised, np.ndarray):
        if normalised.ndim == 4 and normalised.shape[-1] == len(CLASSES):
            normalised = np.transpose(normalised, (3, 0, 1, 2))
            normalised = [normalised[idx] for idx in range(normalised.shape[0])]
        elif normalised.ndim == 4 and normalised.shape[1] == len(CLASSES):
            normalised = np.transpose(normalised, (1, 0, 2, 3))
            normalised = [normalised[idx] for idx in range(normalised.shape[0])]
        else:
            raise ValueError(f"Unexpected SHAP tensor shape: {normalised.shape}")

    first_shape = normalised[0].shape
    if first_shape[0] != expected_batch and first_shape[-1] == expected_batch:
        normalised = [np.transpose(value, (2, 0, 1)) for value in normalised]
    return [np.asarray(value, dtype=np.float32) for value in normalised]


def compute_shap_values(
    model: FAST,
    x_background: torch.Tensor,
    x_explain: torch.Tensor,
    *,
    device: torch.device,
) -> list[np.ndarray]:
    """Compute GradientExplainer SHAP values on the configured device."""

    x_background = x_background.to(device)
    x_explain = x_explain.to(device)
    explainer = shap.GradientExplainer(model, x_background)
    shap_values = explainer.shap_values(x_explain)
    return _normalise_shap_output(shap_values, expected_batch=len(x_explain))


def compute_zone_time_matrix(
    avg_shap: np.ndarray,
    *,
    electrodes: Iterable[str] = Electrodes,
    zones: Mapping[str, Iterable[str]] = Zones,
) -> pd.DataFrame:
    """Aggregate a channel x time matrix into a zone x time frame."""

    zone_index = zone_indices(electrodes, zones)
    records = []
    for zone_name, indices in zone_index.items():
        if not indices:
            continue
        zone_signal = np.mean(avg_shap[indices], axis=0)
        records.append(pd.DataFrame({
            "zone": zone_name,
            "time_idx": np.arange(zone_signal.shape[0], dtype=int),
            "shap_value": zone_signal,
            "abs_shap_value": np.abs(zone_signal),
        }))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def compute_band_time_matrix(
    avg_shap: np.ndarray,
    avg_eeg: np.ndarray,
    *,
    sfreq: int = 250,
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
    nperseg: int = 64,
    noverlap: int = 32,
) -> pd.DataFrame:
    """Compute SHAP-weighted band-time summaries using filtered EEG and STFT."""

    nyquist = sfreq / 2.0
    _, time_axis, _ = scipy.signal.stft(avg_eeg[0], fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    frames = []

    for band_name, (low_hz, high_hz) in bands.items():
        low = low_hz / nyquist
        high = min(high_hz / nyquist, 0.99)
        if low >= 1.0 or low >= high:
            continue
        try:
            sos = scipy.signal.butter(4, [low, high], btype="band", output="sos")
        except ValueError:
            continue

        band_profile = np.zeros(len(time_axis), dtype=np.float64)
        for channel_idx in range(avg_eeg.shape[0]):
            filtered = scipy.signal.sosfiltfilt(sos, avg_eeg[channel_idx])
            weighted = np.abs(filtered * avg_shap[channel_idx])
            _, _, spectrum = scipy.signal.stft(weighted, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            band_profile += np.mean(np.abs(spectrum), axis=0)

        band_profile = band_profile / avg_eeg.shape[0]
        frames.append(pd.DataFrame({
            "band": band_name,
            "time_bin": np.arange(len(time_axis), dtype=int),
            "time_ms": time_axis * 1000.0,
            "band_importance": band_profile,
        }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _subject_zone_rows(
    run_key: str,
    display_name: str,
    sid: str,
    class_name: str,
    class_index: int,
    avg_shap: np.ndarray,
    *,
    electrodes: Iterable[str] = Electrodes,
    zones: Mapping[str, Iterable[str]] = Zones,
) -> list[dict[str, object]]:
    zone_index = zone_indices(electrodes, zones)
    rows = []
    for zone_name, indices in zone_index.items():
        if not indices:
            continue
        values = avg_shap[indices]
        rows.append({
            "run_key": run_key,
            "experiment": display_name,
            "subject": sid,
            "class_index": class_index,
            "class_name": class_name,
            "zone": zone_name,
            "mean_signed_shap": float(np.mean(values)),
            "mean_abs_shap": float(np.mean(np.abs(values))),
        })
    return rows


def _subject_band_rows(
    run_key: str,
    display_name: str,
    sid: str,
    class_name: str,
    class_index: int,
    band_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    if band_frame.empty:
        return []

    rows = []
    for band_name, group in band_frame.groupby("band"):
        rows.append({
            "run_key": run_key,
            "experiment": display_name,
            "subject": sid,
            "class_index": class_index,
            "class_name": class_name,
            "band": band_name,
            "mean_band_importance": float(group["band_importance"].mean()),
            "max_band_importance": float(group["band_importance"].max()),
            "band_importance_share": float(group["band_importance"].sum()),
        })
    return rows


def build_subject_summary_tables(
    *,
    run_key: str,
    experiment: ExperimentSpec,
    sid: str,
    shap_values: list[np.ndarray],
    x_explain: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Build tidy per-subject summary tables for downstream caching and analysis."""

    class_rows = []
    zone_rows = []
    band_rows = []
    zone_time_frames = []
    band_time_frames = []

    for class_index, class_name in enumerate(CLASSES):
        class_mask = np.where(y_true == class_index)[0]
        if len(class_mask) == 0:
            continue

        avg_shap = np.mean(shap_values[class_index][class_mask], axis=0)
        avg_eeg = np.mean(x_explain[class_mask], axis=0)
        correct_count = int(np.sum(y_pred[class_mask] == class_index))
        error_count = int(np.sum(y_pred[class_mask] != class_index))

        class_rows.append({
            "run_key": run_key,
            "experiment": experiment.display_name,
            "condition_family": experiment.condition_family,
            "subject": sid,
            "class_index": class_index,
            "class_name": class_name,
            "n_samples": int(len(class_mask)),
            "n_correct": correct_count,
            "n_error": error_count,
            "mean_signed_shap": float(np.mean(avg_shap)),
            "mean_abs_shap": float(np.mean(np.abs(avg_shap))),
        })

        zone_rows.extend(
            _subject_zone_rows(
                run_key,
                experiment.display_name,
                sid,
                class_name,
                class_index,
                avg_shap,
            )
        )

        zone_time = compute_zone_time_matrix(avg_shap)
        if not zone_time.empty:
            zone_time = zone_time.assign(
                run_key=run_key,
                experiment=experiment.display_name,
                subject=sid,
                class_index=class_index,
                class_name=class_name,
            )
            zone_time_frames.append(zone_time)

        band_time = compute_band_time_matrix(avg_shap, avg_eeg)
        if not band_time.empty:
            band_time = band_time.assign(
                run_key=run_key,
                experiment=experiment.display_name,
                subject=sid,
                class_index=class_index,
                class_name=class_name,
            )
            band_time_frames.append(band_time)
            band_rows.extend(_subject_band_rows(run_key, experiment.display_name, sid, class_name, class_index, band_time))

    zone_frame = pd.DataFrame(zone_rows)
    if not zone_frame.empty:
        group_rows = []
        for (group_run, group_subject, group_class), subset in zone_frame.groupby(["run_key", "subject", "class_name"]):
            zone_scores = dict(zip(subset["zone"], subset["mean_abs_shap"]))
            group_scores = summarise_language_groups(zone_scores)
            for group_name, group_value in group_scores.items():
                group_rows.append({
                    "run_key": group_run,
                    "experiment": experiment.display_name,
                    "subject": group_subject,
                    "class_name": group_class,
                    "zone_group": group_name,
                    "mean_abs_shap": group_value,
                })
        zone_group_frame = pd.DataFrame(group_rows)
    else:
        zone_group_frame = pd.DataFrame()

    return {
        "class_summary": pd.DataFrame(class_rows),
        "zone_summary": zone_frame,
        "zone_group_summary": zone_group_frame,
        "band_summary": pd.DataFrame(band_rows),
        "zone_time": pd.concat(zone_time_frames, ignore_index=True) if zone_time_frames else pd.DataFrame(),
        "band_time": pd.concat(band_time_frames, ignore_index=True) if band_time_frames else pd.DataFrame(),
    }


def load_experiment_performance_table(csv_path: str | Path) -> pd.DataFrame:
    """Load the cross-experiment performance summary and attach run keys."""

    frame = pd.read_csv(csv_path)
    frame["run_key"] = frame["Experiment"].map(EXPERIMENT_NAME_ALIASES)
    if frame["run_key"].isna().any():
        missing = sorted(frame.loc[frame["run_key"].isna(), "Experiment"].unique())
        raise KeyError(f"Missing run-key aliases for: {missing}")
    return frame


def compare_matched_runs(
    frame: pd.DataFrame,
    *,
    run_a: str,
    run_b: str,
    value_col: str,
    subject_col: str = "subject",
    run_col: str = "run_key",
) -> dict[str, float]:
    """Run a paired comparison between two runs over matched subjects."""

    wide = (
        frame.loc[frame[run_col].isin([run_a, run_b]), [subject_col, run_col, value_col]]
        .dropna()
        .pivot_table(index=subject_col, columns=run_col, values=value_col, aggfunc="mean")
        .dropna()
    )
    if run_a not in wide.columns or run_b not in wide.columns or wide.empty:
        return {
            "run_a": run_a,
            "run_b": run_b,
            "n_subjects": 0,
            "mean_diff": np.nan,
            "std_diff": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "cohens_dz": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    diffs = wide[run_b] - wide[run_a]
    t_stat, p_value = stats.ttest_rel(wide[run_b], wide[run_a], nan_policy="omit")
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan
    stderr = std_diff / np.sqrt(len(diffs)) if len(diffs) > 1 and std_diff > 0 else np.nan
    ci_delta = float(stats.t.ppf(0.975, df=len(diffs) - 1) * stderr) if len(diffs) > 1 and np.isfinite(stderr) else np.nan
    cohens_dz = mean_diff / std_diff if std_diff and np.isfinite(std_diff) and std_diff > 0 else np.nan

    return {
        "run_a": run_a,
        "run_b": run_b,
        "n_subjects": int(len(diffs)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_dz": float(cohens_dz) if np.isfinite(cohens_dz) else np.nan,
        "ci_low": mean_diff - ci_delta if np.isfinite(ci_delta) else np.nan,
        "ci_high": mean_diff + ci_delta if np.isfinite(ci_delta) else np.nan,
    }