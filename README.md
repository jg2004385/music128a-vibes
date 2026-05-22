# cold_study

## Installation

Requires Python 3.10+.

```bash
pip3 install pandas matplotlib
```

## Running the plot script

Arguments:

- `input_csv_path` (required) — path to the Trigno Discover CSV.
- `output_path` (required) — path to write the PNG.
- `--start-time-sec` (optional) — window start in seconds. Defaults to the start of the file.
- `--end-time-sec` (optional) — window end in seconds. Defaults to the end of the file.

Examples:

```bash
# Full recording
python3 plot_signals.py "data/Pilot Trial_01.csv" "output/Pilot Trial_01.png"

# 600–700 s window
python3 plot_signals.py "data/Pilot Trial_01.csv" "output/Pilot Trial_01_600-700s.png" --start-time-sec 600 --end-time-sec 700
```
