import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib").resolve()))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


EXERCISES = [
    "C major scale 1 octave up/down",
    "Max loudness chords",
    "Hold G, 321",
    "Chromatic octaves",
    "C major arpeggios",
    "Anchored trill",
]


@dataclass
class MidiSegment:
    start: float
    end: float
    note_count: int
    notes: set[int]

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ExerciseWindow:
    block: int
    exercise_index: int
    exercise: str
    start_sec: float
    end_sec: float
    source: str

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def normalize_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    normalized = re.sub(r"[^0-9a-z_]", "", normalized)
    match = re.match(r"^(\d{3})_?(cold|warm)", normalized)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return normalized


def safe_stem(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._")
    return cleaned or "recording"


def load_midi_note_events(path: Path) -> list[tuple[float, int, int]]:
    mid = mido.MidiFile(path)
    events = []
    for track in mid.tracks:
        ticks = 0
        tempo = 500000
        for msg in track:
            ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0:
                sec = mido.tick2second(ticks, mid.ticks_per_beat, tempo)
                events.append((sec, msg.note, msg.velocity))
    return sorted(events, key=lambda item: item[0])


def find_midi_anchor(events: list[tuple[float, int, int]], first_exercise_start: float) -> float:
    if not events:
        raise ValueError("MIDI has no note_on events")
    min_note = min(note for _, note, _ in events)
    low_note_times = [sec for sec, note, _ in events if note == min_note and sec <= first_exercise_start]
    if not low_note_times:
        low_note_times = [sec for sec, note, _ in events if note == min_note]
    return max(low_note_times)


def segment_midi_events(
    events: list[tuple[float, int, int]],
    gap_threshold_sec: float,
    min_duration_sec: float,
) -> list[MidiSegment]:
    if not events:
        return []

    segments = []
    start = events[0][0]
    end = events[0][0]
    notes = set()
    note_count = 0

    for sec, note, _velocity in events:
        if sec - end > gap_threshold_sec:
            if end - start >= min_duration_sec:
                segments.append(MidiSegment(start, end, note_count, notes))
            start = sec
            notes = set()
            note_count = 0
        end = sec
        notes.add(note)
        note_count += 1

    if end - start >= min_duration_sec:
        segments.append(MidiSegment(start, end, note_count, notes))
    return segments


def split_into_blocks(segments: list[MidiSegment], expected_blocks: int) -> list[list[MidiSegment]]:
    if len(segments) < expected_blocks:
        return [segments]

    gaps = [(segments[i].start - segments[i - 1].end, i) for i in range(1, len(segments))]
    boundaries = sorted(i for _gap, i in sorted(gaps, reverse=True)[: expected_blocks - 1])
    blocks = []
    start = 0
    for boundary in boundaries:
        blocks.append(segments[start:boundary])
        start = boundary
    blocks.append(segments[start:])
    return blocks


def is_chromatic_segment(segment: MidiSegment) -> bool:
    return len(segment.notes) >= 12 and segment.duration < 10


def is_arpeggio_segment(segment: MidiSegment) -> bool:
    c_major_arpeggio_notes = {72, 76, 79, 84, 88, 91, 96}
    return segment.duration < 10 and segment.notes.issubset(c_major_arpeggio_notes)


def block_to_exercise_windows(block_index: int, segments: list[MidiSegment]) -> list[ExerciseWindow]:
    if len(segments) < 3:
        raise ValueError(f"Block {block_index} has too few MIDI segments: {len(segments)}")

    long_segments = [seg for seg in segments if seg.duration >= 15]
    if len(long_segments) < 3:
        raise ValueError(f"Block {block_index} has too few long MIDI segments: {len(long_segments)}")

    scale = long_segments[0]
    hold_g = long_segments[1]
    trill = long_segments[-1]
    middle_segments = [seg for seg in segments if hold_g.end < seg.start < trill.start]

    chromatic = [seg for seg in middle_segments if is_chromatic_segment(seg)]
    arpeggios = [seg for seg in middle_segments if is_arpeggio_segment(seg)]

    if chromatic:
        chromatic_start = min(seg.start for seg in chromatic)
        chromatic_end = max(seg.end for seg in chromatic)
    else:
        chromatic_start = hold_g.end
        chromatic_end = hold_g.end

    if arpeggios:
        arpeggio_start = min(seg.start for seg in arpeggios)
        arpeggio_end = max(seg.end for seg in arpeggios)
    else:
        arpeggio_start = chromatic_end
        arpeggio_end = chromatic_end

    # The chord exercise is not represented by note events in the inspected MIDI files.
    # Preserve the requested exercise order by using the gap before Hold G.
    chord_start = scale.end
    chord_end = hold_g.start

    values = [
        ExerciseWindow(block_index, 1, EXERCISES[0], scale.start, scale.end, "midi_notes"),
        ExerciseWindow(block_index, 2, EXERCISES[1], chord_start, chord_end, "inferred_gap_no_midi_notes"),
        ExerciseWindow(block_index, 3, EXERCISES[2], hold_g.start, hold_g.end, "midi_notes"),
        ExerciseWindow(block_index, 4, EXERCISES[3], chromatic_start, chromatic_end, "midi_notes"),
        ExerciseWindow(block_index, 5, EXERCISES[4], arpeggio_start, arpeggio_end, "midi_notes"),
        ExerciseWindow(block_index, 6, EXERCISES[5], trill.start, trill.end, "midi_notes"),
    ]
    return values


def read_trigno_rows(path: Path) -> tuple[list[list[str]], list[str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 9:
        raise ValueError(f"Not enough rows in {path}")
    return rows, rows[5]


def parse_emg(rows: list[list[str]], header: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    time_cols = [idx for idx, col in enumerate(header) if col.strip() == "EMG 1 Time Series (s)"]
    emg_cols = [idx for idx, col in enumerate(header) if col.strip() == "EMG 1 (mV)"]
    if not time_cols or not emg_cols:
        raise ValueError("Could not find EMG columns")

    time_col = time_cols[0]
    max_col = max([time_col] + emg_cols)
    times = []
    emg_values = [[] for _ in emg_cols]
    for row in rows[8:]:
        if len(row) <= max_col:
            continue
        try:
            t = float(row[time_col])
        except ValueError:
            continue
        times.append(t)
        for dst, col in zip(emg_values, emg_cols):
            try:
                dst.append(float(row[col]))
            except ValueError:
                dst.append(0.0)

    return np.asarray(times), [np.asarray(values) for values in emg_values]


def parse_imu_magnitude(rows: list[list[str]], header: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gyro_time_cols = [idx for idx, col in enumerate(header) if col.strip() == "GYRO X Time Series (s)"]
    gyro_x_cols = [idx for idx, col in enumerate(header) if col.strip() == "GYRO X (deg/s)"]
    gyro_y_cols = [idx for idx, col in enumerate(header) if col.strip() == "GYRO Y (deg/s)"]
    gyro_z_cols = [idx for idx, col in enumerate(header) if col.strip() == "GYRO Z (deg/s)"]
    if not gyro_time_cols:
        raise ValueError("Could not find IMU gyroscope columns")

    best_by_time = {}
    for time_col, x_col, y_col, z_col in zip(gyro_time_cols, gyro_x_cols, gyro_y_cols, gyro_z_cols):
        max_col = max(time_col, x_col, y_col, z_col)
        for row in rows[8:]:
            if len(row) <= max_col:
                continue
            try:
                t = float(row[time_col])
                mag = math.sqrt(float(row[x_col]) ** 2 + float(row[y_col]) ** 2 + float(row[z_col]) ** 2)
            except ValueError:
                continue
            best_by_time[t] = max(best_by_time.get(t, 0.0), mag)

    times = np.asarray(sorted(best_by_time))
    mags = np.asarray([best_by_time[t] for t in times])
    return times, mags


def detect_imu_punch_time(rows: list[list[str]], header: list[str], search_sec: float) -> float:
    times, mags = parse_imu_magnitude(rows, header)
    mask = (times >= 0) & (times <= search_sec)
    if not np.any(mask):
        return 0.0
    early_times = times[mask]
    early_mags = mags[mask]
    return float(early_times[int(np.argmax(early_mags))])


def rms_for_window(times: np.ndarray, emg_values: list[np.ndarray], start: float, end: float) -> tuple[float, int]:
    mask = (times >= start) & (times <= end)
    sample_count = int(np.count_nonzero(mask))
    if sample_count == 0:
        return float("nan"), 0
    channel_rms = [math.sqrt(float(np.mean(np.square(values[mask])))) for values in emg_values]
    return float(np.mean(channel_rms)), sample_count


def write_windows(path: Path, windows: list[ExerciseWindow], midi_anchor: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "block",
                "exercise_index",
                "exercise",
                "midi_start_sec",
                "midi_end_sec",
                "midi_start_from_anchor_sec",
                "midi_end_from_anchor_sec",
                "duration_sec",
                "source",
                "midi_anchor_lowest_note_sec",
            ]
        )
        for window in windows:
            writer.writerow(
                [
                    window.block,
                    window.exercise_index,
                    window.exercise,
                    f"{window.start_sec:.6f}",
                    f"{window.end_sec:.6f}",
                    f"{window.start_sec - midi_anchor:.6f}",
                    f"{window.end_sec - midi_anchor:.6f}",
                    f"{window.duration:.6f}",
                    window.source,
                    f"{midi_anchor:.6f}",
                ]
            )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "block",
        "exercise_index",
        "exercise",
        "midi_start_from_anchor_sec",
        "midi_end_from_anchor_sec",
        "emg_start_sec",
        "emg_end_sec",
        "duration_sec",
        "emg_rms_mv",
        "sample_count",
        "source",
        "midi_anchor_lowest_note_sec",
        "emg_anchor_imu_punch_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, object]], out_path: Path, title: str) -> bool:
    if plt is None:
        print("  Skipping plot: matplotlib is not installed in this Python environment")
        return False

    blocks = sorted({int(row["block"]) for row in rows})
    fig, ax = plt.subplots(figsize=(11, 6))
    for exercise_index, exercise in enumerate(EXERCISES, start=1):
        values = []
        for block in blocks:
            match = next(
                (
                    row
                    for row in rows
                    if int(row["block"]) == block and int(row["exercise_index"]) == exercise_index
                ),
                None,
            )
            values.append(float(match["emg_rms_mv"]) if match else float("nan"))
        ax.plot(blocks, values, marker="o", linewidth=1.8, label=exercise)
    ax.set_xlabel("Block")
    ax.set_ylabel("Average contraction, EMG RMS (mV)")
    ax.set_title(title)
    ax.set_xticks(blocks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def analyze_pair(midi_path: Path, csv_path: Path, out_dir: Path, args) -> None:
    events = load_midi_note_events(midi_path)
    segments = segment_midi_events(events, args.gap_threshold_sec, args.min_duration_sec)
    blocks = split_into_blocks(segments, args.expected_blocks)
    windows = []
    for block_index, block in enumerate(blocks, start=1):
        windows.extend(block_to_exercise_windows(block_index, block))

    midi_anchor = find_midi_anchor(events, windows[0].start_sec)
    rows, header = read_trigno_rows(csv_path)
    emg_times, emg_values = parse_emg(rows, header)
    emg_anchor = detect_imu_punch_time(rows, header, args.imu_search_sec)

    summary_rows = []
    for window in windows:
        rel_start = window.start_sec - midi_anchor
        rel_end = window.end_sec - midi_anchor
        emg_start = rel_start + emg_anchor
        emg_end = rel_end + emg_anchor
        rms, sample_count = rms_for_window(emg_times, emg_values, emg_start, emg_end)
        summary_rows.append(
            {
                "file": csv_path.name,
                "block": window.block,
                "exercise_index": window.exercise_index,
                "exercise": window.exercise,
                "midi_start_from_anchor_sec": f"{rel_start:.6f}",
                "midi_end_from_anchor_sec": f"{rel_end:.6f}",
                "emg_start_sec": f"{emg_start:.6f}",
                "emg_end_sec": f"{emg_end:.6f}",
                "duration_sec": f"{window.duration:.6f}",
                "emg_rms_mv": f"{rms:.8f}" if not math.isnan(rms) else "nan",
                "sample_count": sample_count,
                "source": window.source,
                "midi_anchor_lowest_note_sec": f"{midi_anchor:.6f}",
                "emg_anchor_imu_punch_sec": f"{emg_anchor:.6f}",
            }
        )

    stem = normalize_name(midi_path.stem)
    output_stem = f"{stem}__{safe_stem(csv_path.stem)}"
    window_path = out_dir / f"{output_stem}_exercise_windows.csv"
    summary_path = out_dir / f"{output_stem}_emg_exercise_summary.csv"
    plot_path = out_dir / f"{output_stem}_emg_exercise_contractions.png"
    write_windows(window_path, windows, midi_anchor)
    write_summary(summary_path, summary_rows)
    wrote_plot = plot_summary(summary_rows, plot_path, f"{stem}: average EMG contraction by exercise")

    print(f"{stem}")
    print(f"  MIDI anchor lowest note: {midi_anchor:.3f}s")
    print(f"  EMG anchor IMU punch: {emg_anchor:.3f}s")
    print(f"  Wrote {window_path}")
    print(f"  Wrote {summary_path}")
    if wrote_plot:
        print(f"  Wrote {plot_path}")


def default_csv_for_midi(midi_path: Path, csv_dir: Path) -> Path | None:
    midi_base = normalize_name(midi_path.stem)
    matches = []
    for path in csv_dir.rglob("*.csv"):
        if normalize_name(path.stem) == midi_base:
            matches.append(path)
    return sorted(matches)[0] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align MIDI exercise windows to Trigno EMG and summarize contraction by block."
    )
    parser.add_argument("pairs", nargs="*", help="Pairs of MIDI and Trigno CSV paths.")
    parser.add_argument("--csv-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--out-dir", type=Path, default=Path("output/emg_exercise_analysis"))
    parser.add_argument("--gap-threshold-sec", type=float, default=0.5)
    parser.add_argument("--min-duration-sec", type=float, default=0.5)
    parser.add_argument("--expected-blocks", type=int, default=5)
    parser.add_argument(
        "--imu-search-sec",
        type=float,
        default=20.0,
        help="Search only the start of the Trigno file for the sync punch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs and len(args.pairs) % 2 != 0:
        raise SystemExit("Provide MIDI/CSV paths in pairs.")

    pairs = []
    if args.pairs:
        for idx in range(0, len(args.pairs), 2):
            pairs.append((Path(args.pairs[idx]), Path(args.pairs[idx + 1])))
    else:
        for midi_path in sorted(Path("data").glob("001*.mid")):
            csv_path = default_csv_for_midi(midi_path, args.csv_dir)
            if csv_path:
                pairs.append((midi_path, csv_path))

    if not pairs:
        raise SystemExit("No MIDI/CSV pairs found.")

    failures = []
    for midi_path, csv_path in pairs:
        try:
            analyze_pair(midi_path, csv_path, args.out_dir, args)
        except Exception as exc:
            failures.append((midi_path, csv_path, exc))
            print(f"FAILED {midi_path.name} + {csv_path.name}: {exc}")

    if failures:
        print("\nFailures:")
        for midi_path, csv_path, exc in failures:
            print(f"  {midi_path} + {csv_path}: {exc}")


if __name__ == "__main__":
    main()
