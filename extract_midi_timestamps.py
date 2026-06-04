import argparse
from pathlib import Path

import mido


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MIDI note activity segments and write begin/end timestamps."
    )
    parser.add_argument("midi_path", type=Path, help="Path to the input MIDI file.")
    parser.add_argument("output_csv_path", type=Path, help="Path to write the output CSV.")
    parser.add_argument(
        "--gap-threshold-sec",
        type=float,
        default=1.0,
        help="Maximum silence gap (seconds) between MIDI events within the same segment.",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=3.0,
        help="Minimum segment duration to include in the output.",
    )
    parser.add_argument(
        "--block-gap-threshold-sec",
        type=float,
        default=60.0,
        help="Silence gap (seconds) that separates blocks of exercises.",
    )
    return parser.parse_args()


def load_midi_events(midi_path: Path):
    mid = mido.MidiFile(midi_path)
    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat

    events = []
    for i, track in enumerate(mid.tracks):
        absolute_ticks = 0
        current_tempo = tempo
        for msg in track:
            absolute_ticks += msg.time
            if msg.type == "set_tempo":
                current_tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0:
                seconds = mido.tick2second(absolute_ticks, ticks_per_beat, current_tempo)
                events.append((seconds, msg.note, msg.velocity, i))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                seconds = mido.tick2second(absolute_ticks, ticks_per_beat, current_tempo)
                events.append((seconds, msg.note, 0, i))
    events.sort(key=lambda x: x[0])
    return events


def segment_events(events, gap_threshold_sec: float, min_duration_sec: float):
    if not events:
        return []

    segments = []
    segment_start = events[0][0]
    segment_end = events[0][0]
    note_count = 0

    for seconds, note, velocity, track_index in events:
        if seconds - segment_end > gap_threshold_sec:
            duration = segment_end - segment_start
            if duration >= min_duration_sec:
                segments.append((segment_start, segment_end, duration, note_count))
            segment_start = seconds
            note_count = 0
        segment_end = max(segment_end, seconds)
        note_count += 1

    duration = segment_end - segment_start
    if duration >= min_duration_sec:
        segments.append((segment_start, segment_end, duration, note_count))
    return segments


def split_blocks(segments, block_gap_threshold_sec: float):
    if not segments:
        return []

    blocks = []
    current_block = [segments[0]]

    for seg in segments[1:]:
        prev_end = current_block[-1][1]
        if seg[0] - prev_end > block_gap_threshold_sec:
            blocks.append(current_block)
            current_block = [seg]
        else:
            current_block.append(seg)

    blocks.append(current_block)
    return blocks


def write_csv(output_path: Path, blocks):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("block_index,segment_index,start_sec,end_sec,duration_sec,note_event_count\n")
        segment_idx = 1
        for block_index, block in enumerate(blocks, start=1):
            for start, end, duration, note_count in block:
                fh.write(
                    f"{block_index},{segment_idx},{start:.6f},{end:.6f},{duration:.6f},{note_count}\n"
                )
                segment_idx += 1


def main():
    args = parse_args()
    if not args.midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {args.midi_path}")

    events = load_midi_events(args.midi_path)
    segments = segment_events(events, args.gap_threshold_sec, args.min_duration_sec)
    blocks = split_blocks(segments, args.block_gap_threshold_sec)
    write_csv(args.output_csv_path, blocks)

    print(f"Loaded MIDI: {args.midi_path}")
    print(f"Note events: {len(events)}")
    print(f"Detected segments: {len(segments)}")
    print(f"Detected blocks: {len(blocks)}")

    for block_index, block in enumerate(blocks, start=1):
        print(f"Block {block_index}: {len(block)} segments")
        for idx, (start, end, duration, note_count) in enumerate(block, start=1):
            print(
                f"  Segment {idx}: {start:.3f}–{end:.3f} sec, {duration:.3f} sec, {note_count} note events"
            )


if __name__ == "__main__":
    main()
