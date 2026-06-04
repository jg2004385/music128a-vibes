#!/usr/bin/env python3
"""
Multi-participant piano exercise analysis — MIDI input version.

Usage:
    python analyze_midi.py file1.mid file2.mid ...
    python analyze_midi.py path/to/folder/

Output structure inside midi_plots/:
    001warmplots/    — individual plots for 001warm
    001coldplots/    — individual plots for 001cold
    001comparison/   — cold vs warm side-by-side for participant 001
    ...
"""

import sys, re
import numpy as np
import pretty_midi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

N_BLOCKS    = 5
N_EXERCISES = 6

EXERCISE_NAMES = {
    1: "C major scale (30s)",
    3: "Sustained G + EDC (30s)",
    4: "Chromatic (max speed)",
    5: "Arpeggio 2 oct (max speed)",
    6: "Anchored trill (30s)",
}

EXERCISE_COLORS = {
    1: "#2196F3",
    3: "#E91E63",
    4: "#4CAF50",
    5: "#FF9800",
    6: "#9C27B0",
}

# ── Segment detection ──────────────────────────────────────────────────────────

def detect_segments_midi(notes, window_s=0.25, min_active_s=0.4):
    if not notes:
        return []
    end_time = max(n.end for n in notes)
    n_windows = int(np.ceil(end_time / window_s)) + 1
    active = np.zeros(n_windows, dtype=bool)
    for note in notes:
        s = int(note.start / window_s)
        e = int(np.ceil(note.end / window_s))
        active[s:e] = True
    for i in range(len(active) - 2):
        if active[i] and active[i + 2] and not active[i + 1]:
            active[i + 1] = True
    changes = np.diff(active.astype(int))
    starts  = list(np.where(changes == 1)[0] + 1)
    ends    = list(np.where(changes == -1)[0] + 1)
    if active[0]:  starts = [0] + starts
    if active[-1]: ends   = ends + [len(active)]
    return [
        (s * window_s, e * window_s)
        for s, e in zip(starts, ends)
        if (e - s) * window_s >= min_active_s
    ]


def _gap_based_blocks(segments, gap_s):
    """Split segments by gaps > gap_s, take first 3 long segs per raw block."""
    raw_blocks = [[segments[0]]]
    for seg in segments[1:]:
        if seg[0] - raw_blocks[-1][-1][1] > gap_s:
            raw_blocks.append([seg])
        else:
            raw_blocks[-1].append(seg)

    result = {}
    b_num = 0
    for block in raw_blocks:
        long_segs  = sorted([(s, e) for s, e in block if e - s >= 22.0], key=lambda x: x[0])
        short_segs = sorted([(s, e) for s, e in block if e - s <  22.0], key=lambda x: x[0])
        if len(long_segs) < 3:
            continue
        ex1, ex3, ex6 = long_segs[0], long_segs[1], long_segs[2]
        ex_map = {ex: None for ex in range(1, N_EXERCISES + 1)}
        ex_map[1] = ex1
        ex_map[3] = ex3
        ex_map[6] = ex6
        between = [(s, e) for s, e in short_segs if s >= ex3[1] and e <= ex6[0]]
        ex_map[4], ex_map[5] = _assign_speed_exercises(between)
        b_num += 1
        result[b_num] = ex_map
        if b_num == N_BLOCKS:
            break
    return result if result else None


def _triplet_based_blocks(segments):
    """Group all long segments globally into consecutive triplets (Ex1/Ex3/Ex6 per block)."""
    long_segs  = sorted([(s, e) for s, e in segments if e - s >= 22.0], key=lambda x: x[0])
    short_segs = sorted([(s, e) for s, e in segments if e - s <  22.0], key=lambda x: x[0])
    n_long   = len(long_segs)
    n_blocks = min(N_BLOCKS, n_long // 3)
    if n_long < 3:
        return None
    result = {}
    for b_idx in range(n_blocks):
        ex1, ex3, ex6 = long_segs[b_idx * 3 : b_idx * 3 + 3]
        ex_map = {ex: None for ex in range(1, N_EXERCISES + 1)}
        ex_map[1] = ex1
        ex_map[3] = ex3
        ex_map[6] = ex6
        between = [(s, e) for s, e in short_segs if s >= ex3[1] and e <= ex6[0]]
        ex_map[4], ex_map[5] = _assign_speed_exercises(between)
        result[b_idx + 1] = ex_map
    return result if result else None


def group_into_blocks(segments):
    """
    Adaptive block detection:

    1. Try gap-based splitting with thresholds 30–300s (ascending order).
       Track the BEST gap result = most complete blocks found (≤ N_BLOCKS),
       with ties broken by the LARGER threshold (more conservative split).

    2. If the best gap result has ≥ 3 complete blocks, use it.
       (This handles partial recordings correctly — e.g. 002warm has a genuine
       4-block recording; using the gap result avoids spurious triplet blocks.)

    3. Otherwise (< 3 blocks from any gap threshold), fall back to global
       triplet grouping. This handles:
         - Very long intra-block pauses (007warm has ~120s gaps within a block)
         - Recordings with scattered ambient MIDI events that bridge block gaps (008)
    """
    if not segments:
        return None

    # Step 1: find best gap-based result
    best_gap_result = None
    best_gap_count  = 0
    for gap_s in [30, 50, 80, 120, 200, 300]:
        result = _gap_based_blocks(segments, gap_s)
        n = len(result) if result else 0
        if n == N_BLOCKS:
            return result          # perfect — use immediately
        if n > best_gap_count:
            best_gap_count  = n
            best_gap_result = result

    # Step 2: if gap approach found a meaningful partial result, use it
    if best_gap_count >= 3:
        print(f"  Warning: found only {best_gap_count} complete blocks (expected {N_BLOCKS}).")
        return best_gap_result

    # Step 3: fall back to global triplet grouping
    result = _triplet_based_blocks(segments)
    if result:
        n = len(result)
        if n < N_BLOCKS:
            print(f"  Warning: found only {n} complete blocks (expected {N_BLOCKS}).")
        return result

    print(f"  Warning: could not identify any complete blocks.")
    return None


def _assign_speed_exercises(between):
    """
    Split short segments between Ex3 and Ex6 into Ex4 (chromatic) and Ex5 (arpeggio).
    Each may have 1 or 2 attempts; always use the LAST attempt as the datapoint.
    The largest gap between consecutive segments marks the boundary between Ex4 and Ex5.
    """
    n = len(between)
    if n == 0:
        return None, None
    if n == 1:
        return between[0], None
    if n == 2:
        # One attempt each — use each directly
        return between[0], between[1]
    # 3+ segments: split at the largest inter-segment gap
    gaps = [between[i + 1][0] - between[i][1] for i in range(n - 1)]
    split = gaps.index(max(gaps)) + 1
    ex4_group = between[:split]
    ex5_group = between[split:]
    ex4 = ex4_group[-1]                         # last attempt of Ex4
    ex5 = ex5_group[-1] if ex5_group else None  # last attempt of Ex5
    return ex4, ex5


# ── Metrics ────────────────────────────────────────────────────────────────────

def notes_in_window(all_notes, t_start, t_end):
    return [n for n in all_notes if t_start <= n.start < t_end]


def ioi_stats(notes):
    if len(notes) < 2:
        return None, None, None
    onsets = np.array(sorted(n.start for n in notes))
    iois   = np.diff(onsets)
    iois   = iois[iois > 0.01]
    if len(iois) < 2:
        return None, None, None
    mean = float(np.mean(iois))
    var  = float(np.var(iois))
    cv   = float(np.std(iois) / mean) if mean > 0 else None
    return mean, var, cv


def compute_metrics(notes, t_start, t_end):
    n       = len(notes)
    dur     = t_end - t_start
    vels    = [note.velocity for note in notes]
    avg_vel = float(np.mean(vels)) if vels else None
    ioi_mean, ioi_var, ioi_cv = ioi_stats(notes)
    return {
        "note_count":   n,
        "avg_velocity": avg_vel,
        "ioi_mean_s":   ioi_mean,
        "ioi_var":      ioi_var,
        "ioi_cv":       ioi_cv,
        "duration_s":   dur,
        "notes_per_s":  n / dur if dur > 0 else None,
    }


# ── Per-file analysis ──────────────────────────────────────────────────────────

def analyze_file(mid_path):
    mid_path = Path(mid_path)
    print(f"\n{'='*60}")
    print(f"Participant: {mid_path.stem}")
    print(f"{'='*60}")
    try:
        mid = pretty_midi.PrettyMIDI(str(mid_path))
    except Exception as e:
        print(f"  ERROR reading MIDI: {e}")
        return None
    all_notes = sorted(
        [n for inst in mid.instruments for n in inst.notes],
        key=lambda n: n.start
    )
    print(f"  Duration:    {mid.get_end_time():.1f}s  |  Total notes: {len(all_notes)}")
    if not all_notes:
        print("  ERROR: No notes found.")
        return None

    segments = detect_segments_midi(all_notes)
    blocks   = group_into_blocks(segments)
    if not blocks:
        print("  ERROR: Could not identify blocks.")
        return None
    print(f"  Segments: {len(segments)}  |  Blocks: {len(blocks)}\n")

    print(f"  {'Ex':<4} {'Description':<38}", end="")
    for b in blocks: print(f"  {'B'+str(b):>12}", end="")
    print()
    for ex_num in EXERCISE_NAMES:
        print(f"  {ex_num:<4} {EXERCISE_NAMES[ex_num]:<38}", end="")
        for b_num, ex_map in blocks.items():
            seg = ex_map.get(ex_num)
            print(f"  {seg[0]:5.0f}–{seg[1]:5.0f}s" if seg else f"  {'?':>11}", end="")
        print()

    results = {}
    for b_num, ex_map in blocks.items():
        results[b_num] = {}
        for ex_num in EXERCISE_NAMES:
            seg = ex_map.get(ex_num)
            if seg is None:
                results[b_num][ex_num] = None
            else:
                notes = notes_in_window(all_notes, seg[0], seg[1])
                results[b_num][ex_num] = compute_metrics(notes, seg[0], seg[1])

    return {"name": mid_path.stem, "blocks": blocks, "results": results}


# ── Plotting helpers ───────────────────────────────────────────────────────────

BLOCKS_X = list(range(1, N_BLOCKS + 1))

def val(results, b, ex, key):
    m = results.get(b, {}).get(ex)
    return m[key] if m and m.get(key) is not None else np.nan


def _style_ax(ax, title, ylabel=None):
    ax.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Block", fontsize=8)
    ax.set_xticks(BLOCKS_X)
    ax.set_xticklabels([f"B{b}" for b in BLOCKS_X], fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


# ── Per-participant plots ──────────────────────────────────────────────────────

PLOT_SPECS = [
    ("note_count",   "Note count",               [1, 3, 6]),
    ("avg_velocity", "Avg velocity (0–127)",      [1, 3, 4, 5, 6]),
    ("ioi_cv",       "IOI CV (lower = more even)",[1, 3, 4, 5, 6]),
    ("notes_per_s",  "Speed (notes/s)",           [4, 5]),
    ("ioi_var",      "IOI Variance (s²)",         [1, 3, 4, 5, 6]),
    ("ioi_mean_s",   "Avg IOI (s)",               [1, 3, 6]),
]

PLOT_FILENAMES = [
    "note_count.png",
    "avg_velocity.png",
    "ioi_cv.png",
    "speed.png",
    "ioi_variance.png",
    "ioi_mean.png",
]


def make_individual_plots(data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = data["name"]

    for (key, ylabel, exercises), filename in zip(PLOT_SPECS, PLOT_FILENAMES):
        n_ex = len(exercises)
        fig, axes = plt.subplots(1, n_ex, figsize=(4.0 * n_ex, 3.8), sharey=False)
        if n_ex == 1:
            axes = [axes]
        for ax, ex_num in zip(axes, exercises):
            vals = [val(data["results"], b, ex_num, key) for b in BLOCKS_X]
            ax.plot(BLOCKS_X, vals, marker="o", linewidth=2, markersize=6,
                    color=EXERCISE_COLORS[ex_num], label=f"Ex {ex_num}")
            _style_ax(ax, EXERCISE_NAMES[ex_num], ylabel if ax is axes[0] else None)
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"{name} — {ylabel}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Dashboard: {name}", fontsize=14, fontweight="bold")
    dash_specs = [
        ("note_count",   "Note count",              [1, 3, 6]),
        ("avg_velocity", "Avg velocity",             [1, 3, 4, 5, 6]),
        ("ioi_cv",       "IOI CV (lower = more even)",[1, 3, 4, 5, 6]),
        ("notes_per_s",  "Speed (notes/s)",          [4, 5]),
        ("ioi_var",      "IOI Variance",              [1, 3, 6]),
        ("ioi_mean_s",   "Avg IOI (s)",               [1, 3, 6]),
    ]
    for ax, (key, label, exs) in zip(axes.flat, dash_specs):
        for ex_num in exs:
            vals = [val(data["results"], b, ex_num, key) for b in BLOCKS_X]
            ax.plot(BLOCKS_X, vals, marker="o", linewidth=2, markersize=5,
                    label=f"Ex {ex_num}", color=EXERCISE_COLORS[ex_num])
        _style_ax(ax, label)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {len(PLOT_FILENAMES) + 1} plots → {out_dir}/")


# ── Cold vs warm comparison plots ─────────────────────────────────────────────

def make_comparison_plots(cold_data, warm_data, participant_id, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-metric: one subplot per exercise, warm (solid) + cold (dashed) overlaid
    for (key, ylabel, exercises), filename in zip(PLOT_SPECS, PLOT_FILENAMES):
        n_ex = len(exercises)
        fig, axes = plt.subplots(1, n_ex, figsize=(4.0 * n_ex, 3.8), sharey=False)
        if n_ex == 1:
            axes = [axes]

        for ax, ex_num in zip(axes, exercises):
            color = EXERCISE_COLORS[ex_num]
            for data, cond_label, ls, marker in [
                (warm_data, "Warm", "-",  "o"),
                (cold_data, "Cold", "--", "s"),
            ]:
                vals = [val(data["results"], b, ex_num, key) for b in BLOCKS_X]
                ax.plot(BLOCKS_X, vals, marker=marker, linewidth=2, markersize=6,
                        color=color, linestyle=ls, label=cond_label)
            _style_ax(ax, EXERCISE_NAMES[ex_num], ylabel if ax is axes[0] else None)
            ax.legend(fontsize=7, loc="best")

        fig.suptitle(f"Participant {participant_id}: Warm vs Cold — {ylabel}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Comparison dashboard: each row = one metric, left col = warm, right col = cold
    dash_specs = [
        ("note_count",   "Note count",               [1, 3, 6]),
        ("avg_velocity", "Avg velocity",              [1, 3, 4, 5, 6]),
        ("ioi_cv",       "IOI CV (lower = more even)",[1, 3, 4, 5, 6]),
        ("notes_per_s",  "Speed (notes/s)",           [4, 5]),
        ("ioi_var",      "IOI Variance",               [1, 3, 6]),
        ("ioi_mean_s",   "Avg IOI (s)",                [1, 3, 6]),
    ]
    fig, axes = plt.subplots(len(dash_specs), 2, figsize=(11, 3.5 * len(dash_specs)))
    fig.suptitle(f"Participant {participant_id}: Warm (left) vs Cold (right)",
                 fontsize=14, fontweight="bold")

    for row, (key, label, exs) in enumerate(dash_specs):
        for col, (data, cond) in enumerate([(warm_data, "Warm"), (cold_data, "Cold")]):
            ax = axes[row, col]
            for ex_num in exs:
                vals = [val(data["results"], b, ex_num, key) for b in BLOCKS_X]
                ax.plot(BLOCKS_X, vals, marker="o", linewidth=2, markersize=5,
                        label=f"Ex {ex_num}", color=EXERCISE_COLORS[ex_num])
            _style_ax(ax, f"{cond} — {label}")
            ax.set_ylabel(label, fontsize=8)
            ax.legend(fontsize=7, loc="best")

        # Match y-axis range so warm and cold are directly comparable
        y_min = min(axes[row, 0].get_ylim()[0], axes[row, 1].get_ylim()[0])
        y_max = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        axes[row, 0].set_ylim(y_min, y_max)
        axes[row, 1].set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.savefig(out_dir / "comparison_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {len(PLOT_FILENAMES) + 1} comparison plots → {out_dir}/")


# ── Filename parsing ───────────────────────────────────────────────────────────

def parse_participant(stem):
    """Extract (participant_id, condition) from filename stem.
    e.g. '001warm' → ('001', 'warm'), '006_cold' → ('006', 'cold')
    """
    m = re.search(r"(\d+).*?(cold|warm)", stem, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).lower()
    return None, None


# ── Main ───────────────────────────────────────────────────────────────────────

def collect_midi_files(args):
    files = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob("*.mid")) + sorted(p.glob("*.midi")))
        elif p.suffix.lower() in (".mid", ".midi"):
            files.append(p)
    return files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_midi.py file1.mid file2.mid ...")
        print("  python analyze_midi.py path/to/folder/")
        sys.exit(1)

    midi_files = collect_midi_files(sys.argv[1:])
    if not midi_files:
        print("No .mid files found.")
        sys.exit(1)

    print(f"Found {len(midi_files)} file(s).")
    base_out = Path("midi_plots")

    # Process each file and save to its own folder
    all_data = {}   # stem → data dict
    for path in midi_files:
        data = analyze_file(path)
        if data:
            all_data[path.stem] = data
            folder_name = path.stem + "plots"
            print(f"\nPlotting {path.stem}...")
            make_individual_plots(data, base_out / folder_name)

    # Pair cold/warm and make comparison plots
    by_participant = {}
    for stem, data in all_data.items():
        pid, cond = parse_participant(stem)
        if pid is None:
            continue
        by_participant.setdefault(pid, {})[cond] = data

    print("\n" + "="*60)
    print("Generating cold vs warm comparisons...")
    for pid in sorted(by_participant):
        pair = by_participant[pid]
        if "cold" in pair and "warm" in pair:
            print(f"\nParticipant {pid}:")
            make_comparison_plots(
                pair["cold"], pair["warm"],
                pid,
                base_out / f"{pid}comparison"
            )
        else:
            cond = list(pair.keys())[0]
            print(f"  Participant {pid}: only '{cond}' found, skipping comparison.")

    print(f"\nAll output saved under: {base_out.resolve()}/")
