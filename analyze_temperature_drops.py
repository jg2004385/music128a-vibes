import argparse
import csv
import math
import os
import struct
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASETS = [
    {
        "label": "001cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T004725Z-3-001/5.12_001cold.csv",
        "windows_csv": "output/emg_exercise_analysis/001cold_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "001warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T004725Z-3-001/5.12_001warm.csv",
        "windows_csv": "output/emg_exercise_analysis/001warm_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "002cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001/5.12_002cold.csv",
        "windows_csv": "output/emg_exercise_analysis/002cold__002cold_01_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "002warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001/5.12_002warm.csv",
        "windows_csv": "output/emg_exercise_analysis/002warm__002warm_01_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "003cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001/5.12_003cold.csv",
        "windows_csv": "output/emg_exercise_analysis/003cold__003cold_01_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "003warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001/5.12_003warm.csv",
        "windows_csv": "output/emg_exercise_analysis/003warm__003warm_01_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "004cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001/5.14_004cold.csv",
        "windows_csv": "output/emg_exercise_analysis/004cold__004cold_01_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "004warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001/5.14_004warm_2.csv",
        "windows_csv": "output/emg_exercise_analysis/004warm__004warm_01_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "005cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T014102Z-3-001/5.19_005cold.csv",
        "windows_csv": "output/emg_exercise_analysis/005cold__midi_only_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "005warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T014102Z-3-001/5.19_005warm.csv",
        "windows_csv": "output/emg_exercise_analysis/005warm__midi_only_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "007cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T013347Z-3-001/5.20_007cold.plw",
        "windows_csv": "output/emg_exercise_analysis/007cold__007cold_01_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "007warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T013347Z-3-001/5.20_007warm.plw",
        "windows_csv": "output/emg_exercise_analysis/007warm__007warm_01_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "008cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T013347Z-3-001/5.21_008cold.plw",
        "windows_csv": "output/emg_exercise_analysis/008cold__008cold_01_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "008warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T013347Z-3-001/5.21_008warm.plw",
        "windows_csv": "output/emg_exercise_analysis/008warm__008warm_01_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "009cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001 3/009cold.csv",
        "windows_csv": "output/emg_exercise_analysis/009cold__009cold._2_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "009warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001 3/009warm.csv",
        "windows_csv": "output/emg_exercise_analysis/009warm__009warm_2_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "010warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001 3/010warm.csv",
        "windows_csv": "output/emg_exercise_analysis/010warm__010warm_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "011cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001 3/011cold.csv",
        "windows_csv": "output/emg_exercise_analysis/011cold__011cold_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "011warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T005346Z-3-001 3/011warm.csv",
        "windows_csv": "output/emg_exercise_analysis/011warm__011warm_exercise_windows.csv",
        "sync_lead": "lead4",
    },
    {
        "label": "012cold",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T012346Z-3-001/5,29_012cold.csv",
        "windows_csv": "output/emg_exercise_analysis/012cold__012cold_01_exercise_windows.csv",
        "sync_lead": "lead3",
    },
    {
        "label": "012warm",
        "temp_csv": "/Users/jeenahgwak/Downloads/drive-download-20260606T012346Z-3-001/5.29_012warm.csv",
        "windows_csv": "output/emg_exercise_analysis/012warm__012warm_01_2_exercise_windows.csv",
        "sync_lead": "lead4",
    },
]


def temperature_dict(rows: list[list[float]]) -> dict[str, np.ndarray]:
    data = np.asarray(rows, dtype=float)
    return {
        "time": data[:, 0],
        "lead1": data[:, 1],
        "lead2": data[:, 2],
        "lead3": data[:, 3],
        "lead4": data[:, 4],
        "cold_junction": data[:, 5],
    }


def parse_picolog_plw(path: Path) -> dict[str, np.ndarray]:
    # PicoLog for Windows .plw files in this dataset store records as:
    # lead1, lead2, lead3, lead4, cold junction, zero marker.
    # Lead 1 is sometimes disconnected/noisy, so validate the channels we use.
    data = path.read_bytes()
    data_offset = 1688
    record_size = 24
    rows = []
    for idx in range((len(data) - data_offset) // record_size):
        values = struct.unpack_from("<6f", data, data_offset + idx * record_size)
        usable_channels = values[1:5]
        is_record = all(math.isfinite(value) and -100 < value < 100 for value in usable_channels)
        is_record = is_record and abs(values[5]) < 1e-5
        if not is_record and idx > 10:
            break
        if is_record:
            rows.append([float(len(rows)), values[0], values[1], values[2], values[3], values[4]])

    if not rows:
        raise ValueError(f"No PicoLog float records found in {path}")
    return temperature_dict(rows)


def parse_temperature_csv(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".plw":
        return parse_picolog_plw(path)

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 6:
                continue
            cleaned = [cell.replace("\x00", "").strip() for cell in row[:6]]
            try:
                values = [float(cell) for cell in cleaned]
            except ValueError:
                continue
            rows.append(values)

    if not rows:
        raise ValueError(f"No numeric temperature rows found in {path}")

    return temperature_dict(rows)


def detect_sync_anchor(time: np.ndarray, sync_temp: np.ndarray) -> tuple[float, float]:
    diffs = np.diff(sync_temp)
    if diffs.size == 0:
        raise ValueError("Not enough samples to detect sync anchor")
    idx = int(np.argmax(diffs))
    return float(time[idx + 1]), float(diffs[idx])


def parse_block_windows(path: Path) -> list[dict[str, float]]:
    blocks: dict[int, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            block = int(row["block"])
            start = float(row["midi_start_from_anchor_sec"])
            end = float(row["midi_end_from_anchor_sec"])
            blocks.setdefault(block, []).append((start, end))

    return [
        {
            "block": block,
            "start_from_anchor": min(start for start, _end in windows),
            "end_from_anchor": max(end for _start, end in windows),
        }
        for block, windows in sorted(blocks.items())
    ]


def mean_window(time: np.ndarray, values: np.ndarray, center: float, seconds: float) -> float:
    if center < float(time[0]) or center > float(time[-1]):
        return float("nan")
    half = seconds / 2.0
    mask = (time >= center - half) & (time <= center + half)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(time - center)))
        return float(values[idx])
    return float(np.mean(values[mask]))


def summarize_file(
    temp_csv: Path,
    exercise_windows_csv: Path,
    sync_lead: str,
    label: str,
    out_dir: Path,
    average_window_sec: float,
) -> list[dict[str, object]]:
    temp = parse_temperature_csv(temp_csv)
    anchor_time, anchor_jump = detect_sync_anchor(temp["time"], temp[sync_lead])
    block_windows = parse_block_windows(exercise_windows_csv)
    finger_temp = temp["lead2"]

    rows = []
    for block in block_windows:
        start_abs = anchor_time + block["start_from_anchor"]
        end_abs = anchor_time + block["end_from_anchor"]
        start_temp = mean_window(temp["time"], finger_temp, start_abs, average_window_sec)
        end_temp = mean_window(temp["time"], finger_temp, end_abs, average_window_sec)
        drop = start_temp - end_temp if not np.isnan(start_temp) and not np.isnan(end_temp) else float("nan")
        rows.append(
            {
                "condition": label,
                "temperature_file": temp_csv.name,
                "block": block["block"],
                "temp_anchor_sec": f"{anchor_time:.3f}",
                "sync_lead": sync_lead,
                "sync_jump_c": f"{anchor_jump:.3f}",
                "block_start_temp_sec": f"{start_abs:.3f}",
                "block_end_temp_sec": f"{end_abs:.3f}",
                "finger_temp_start_c": f"{start_temp:.4f}",
                "finger_temp_end_c": f"{end_temp:.4f}",
                "finger_temp_drop_c": f"{drop:.4f}",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{label}_temperature_drop_by_block.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{label}: temp anchor {anchor_time:.3f}s on {sync_lead}, jump {anchor_jump:.3f} C")
    print(f"  Wrote {out_csv}")
    return rows


def plot_rows(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    conditions = []
    for row in rows:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])

    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        blocks = [int(row["block"]) for row in condition_rows]
        drops = [float(row["finger_temp_drop_c"]) for row in condition_rows]
        ax.plot(blocks, drops, marker="o", linewidth=2, label=condition)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Block")
    ax.set_ylabel("Finger temperature drop during block (C)")
    ax.set_title(title)
    ax.set_xticks(sorted({int(row["block"]) for row in rows}))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def write_combined(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align thermocouple CSVs to MIDI block windows and plot finger temperature drop."
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Dataset labels to process, e.g. 002cold 002warm. Defaults to all built-in 001-004 datasets.",
    )
    parser.add_argument(
        "--cold-temp-csv",
        type=Path,
        default=Path("/Users/jeenahgwak/Downloads/drive-download-20260606T004725Z-3-001/5.12_001cold.csv"),
    )
    parser.add_argument(
        "--warm-temp-csv",
        type=Path,
        default=Path("/Users/jeenahgwak/Downloads/drive-download-20260606T004725Z-3-001/5.12_001warm.csv"),
    )
    parser.add_argument(
        "--cold-windows-csv",
        type=Path,
        default=Path("output/emg_exercise_analysis/001cold_exercise_windows.csv"),
    )
    parser.add_argument(
        "--warm-windows-csv",
        type=Path,
        default=Path("output/emg_exercise_analysis/001warm_exercise_windows.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output/temperature_analysis"))
    parser.add_argument("--average-window-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels:
        selected = [item for item in DEFAULT_DATASETS if item["label"] in set(args.labels)]
        missing = sorted(set(args.labels) - {item["label"] for item in selected})
        if missing:
            raise SystemExit(f"Unknown labels: {', '.join(missing)}")
    else:
        selected = DEFAULT_DATASETS

    all_rows = []
    for item in selected:
        all_rows.extend(
            summarize_file(
                Path(item["temp_csv"]),
                Path(item["windows_csv"]),
                sync_lead=item["sync_lead"],
                label=item["label"],
                out_dir=args.out_dir,
                average_window_sec=args.average_window_sec,
            )
        )

    by_subject: dict[str, list[dict[str, object]]] = {}
    for row in all_rows:
        subject = str(row["condition"])[:3]
        by_subject.setdefault(subject, []).append(row)

    for subject, rows in sorted(by_subject.items()):
        write_combined(rows, args.out_dir / f"{subject}_temperature_drop_by_block.csv")
        plot_rows(
            rows,
            args.out_dir / f"{subject}_temperature_drop_by_block.png",
            f"{subject} finger temperature drop by block",
        )

    if len(by_subject) > 1:
        subject_suffix = f"{min(by_subject)}_{max(by_subject)}"
        write_combined(all_rows, args.out_dir / f"combined_{subject_suffix}_temperature_drop_by_block.csv")
        plot_rows(
            all_rows,
            args.out_dir / f"combined_{subject_suffix}_temperature_drop_by_block.png",
            "Finger temperature drop by block",
        )


if __name__ == "__main__":
    main()
