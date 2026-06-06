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
        default=0.25,
        help="Maximum silence gap (seconds) between MIDI events within the same segment.",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.5,
        help="Minimum segment duration to include in the output.",
    )
    parser.add_argument(
        "--block-gap-threshold-sec",
        type=float,
        default=60.0,
        help="Silence gap (seconds) that separates blocks of exercises.",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=0.25,
        help="Merge small gaps (seconds) between consecutive segments into one segment.",
    )
    parser.add_argument(
        "--expected-blocks",
        type=int,
        default=5,
        help="Expected number of blocks in each recording. If set, the largest inter-segment gaps are used to define block boundaries.",
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


def segment_events(events, gap_threshold_sec: float, min_duration_sec: float, merge_gap_sec: float):
    """Segment events into contiguous note-activity windows, then merge
    very small gaps between consecutive segments so short bursts (e.g. chords)
    are not dropped. Finally, filter by `min_duration_sec`.
    """
    if not events:
        return []

    # build raw segments based only on the gap threshold
    raw_segments = []
    seg_start = events[0][0]
    seg_end = events[0][0]
    note_count = 0
    for seconds, note, velocity, track_index in events:
        if seconds - seg_end > gap_threshold_sec:
            raw_segments.append((seg_start, seg_end, seg_end - seg_start, note_count))
            seg_start = seconds
            note_count = 0
        seg_end = max(seg_end, seconds)
        note_count += 1
    raw_segments.append((seg_start, seg_end, seg_end - seg_start, note_count))

    # merge tiny gaps between consecutive raw segments so short, closely spaced
    # bursts are treated as a single segment (helps capture chord sequences)
    merged = []
    cur_start, cur_end, cur_dur, cur_count = raw_segments[0]
    for s, e, d, nc in raw_segments[1:]:
        gap = s - cur_end
        if gap <= merge_gap_sec:
            # extend current segment to include this one
            cur_end = e
            cur_dur = cur_end - cur_start
            cur_count += nc
        else:
            merged.append((cur_start, cur_end, cur_dur, cur_count))
            cur_start, cur_end, cur_dur, cur_count = s, e, d, nc
    merged.append((cur_start, cur_end, cur_dur, cur_count))

    # finally, filter merged segments by min_duration_sec
    segments = [seg for seg in merged if seg[2] >= min_duration_sec]
    return segments


def split_blocks(segments, block_gap_threshold_sec: float, expected_blocks: int = None):
    if not segments:
        return []

    if expected_blocks and expected_blocks > 1 and len(segments) >= expected_blocks:
        gaps = [(segments[i][0] - segments[i - 1][1], i - 1) for i in range(1, len(segments))]
        largest = sorted(gaps, key=lambda x: x[0], reverse=True)[: expected_blocks - 1]
        boundaries = sorted(idx for _, idx in largest)
        blocks = []
        start = 0
        for boundary in boundaries:
            blocks.append(segments[start : boundary + 1])
            start = boundary + 1
        blocks.append(segments[start:])
        return blocks

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
    segments = segment_events(
        events, args.gap_threshold_sec, args.min_duration_sec, args.merge_gap_sec
    )
    blocks = split_blocks(segments, args.block_gap_threshold_sec, args.expected_blocks)
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
