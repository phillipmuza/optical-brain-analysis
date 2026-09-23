# Title: Convert IVIS .csv exports (tab-separated) into real comma-separated CSVs
# Author: Hermes (for Phillip Muza)
# Date: 22.09.26

"""IVIS exports its "csv" files tab-separated, so Excel/pandas default parsing collapses
every field into a single column. This reads them with the right delimiter and writes
genuine CSVs — same name, output folder instead of the originals.

Usage:
    python ivis_csv_fix.py <file_or_folder> [output_folder]

Folder mode converts every .csv in it; a single file converts just that one.
Originals are never modified.
"""
import csv
import os
import sys


def convert(in_path, out_dir):
    with open(in_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        print(f"  {in_path}: empty, skipped")
        return False
    out_path = os.path.join(out_dir, os.path.basename(in_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    n_cols = len(rows[0])
    ragged = [i for i, r in enumerate(rows) if len(r) != n_cols]
    print(f"  {os.path.basename(in_path)}: {len(rows)} rows x {n_cols} cols"
          + (f"  WARNING ragged rows: {ragged[:5]}" if ragged else ""))
    return True


def main():
    src = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.path.dirname(os.path.abspath(src)) if os.path.isfile(src) else src,
            "fixed")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(src):
        files = sorted(os.path.join(src, f) for f in os.listdir(src)
                       if f.lower().endswith(".csv"))
    else:
        files = [src]
    if not files:
        sys.exit(f"No .csv files found in {src}")

    done = sum(1 for p in files if convert(p, out_dir))
    print(f"\n{done}/{len(files)} converted -> {out_dir}")


if __name__ == "__main__":
    main()
