import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

SIGNALS = [
    ("EMG 1 Time Series (s)", "EMG 1 (mV)"),
    ("ACC X Time Series (s)", "ACC X (G)"),
    ("ACC Y Time Series (s)", "ACC Y (G)"),
    ("ACC Z Time Series (s)", "ACC Z (G)"),
    ("GYRO X Time Series (s)", "GYRO X (deg/s)"),
    ("GYRO Y Time Series (s)", "GYRO Y (deg/s)"),
    ("GYRO Z Time Series (s)", "GYRO Z (deg/s)"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Plot Trigno CSV signals vs time.")
    p.add_argument("input_csv_path", type=Path, help="Path to the Trigno CSV file.")
    p.add_argument("output_path", type=Path, help="Path to write the output PNG.")
    p.add_argument("--start-time-sec", type=float, default=None,
                   help="Start time in seconds (default: start of file).")
    p.add_argument("--end-time-sec", type=float, default=None,
                   help="End time in seconds (default: end of file).")
    return p.parse_args()


def main():
    args = parse_args()

    header = [c.strip() for c in pd.read_csv(args.input_csv_path, skiprows=5, nrows=0).columns.tolist()]
    df = pd.read_csv(
        args.input_csv_path,
        skiprows=8,
        header=None,
        names=header,
        usecols=range(len(header)),
        index_col=False,
        low_memory=False,
    )

    t_start = args.start_time_sec
    t_end = args.end_time_sec

    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(12, 2 * len(SIGNALS)), sharex=True)

    for ax, (t_col, y_col) in zip(axes, SIGNALS):
        t = pd.to_numeric(df[t_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        mask = t.notna() & y.notna()
        if t_start is not None:
            mask &= t >= t_start
        if t_end is not None:
            mask &= t <= t_end
        ax.plot(t[mask], y[mask], linewidth=0.5)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    title = args.input_csv_path.stem
    if t_start is not None or t_end is not None:
        title += f" [{t_start if t_start is not None else 'start'}–{t_end if t_end is not None else 'end'} s]"
    fig.suptitle(title)
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=150)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
