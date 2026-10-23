#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Program 2: batch reading of EDS Excel composition data, area identification, element pivoting and threshold statistics.

The input format follows the original EDS export template: after the specified rows are skipped, rows 1 and 2 hold
the area and spot information, and the next 12 rows are the composition records of B, C, Al, Ta, W, Zr, Mo, Ti, Cr,
Co, Ni, Hf. If the export format changes, adjust read_one_eds_file() first instead of forcing the fixed order.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ELEMENTS = ["B", "C", "Al", "Ta", "W", "Zr", "Mo", "Ti", "Cr", "Co", "Ni", "Hf"]
SKIP_ROWS = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12]


def _extract_id(row: pd.Series, kind: str) -> int:
    """Prefer 'area'/'spot' followed by a number; otherwise accept only a single purely numeric metadata cell."""
    s = " ".join(str(v) for v in row.to_list() if pd.notna(v))
    # Non-ASCII header keywords are written as \u escapes and must be kept: they match headers of
    # Chinese-locale EDS exports (\u533a\u57df = "area", \u6d4b\u70b9 / \u70b9\u4f4d = "measurement point",
    # \u7f16\u53f7 = "number", \uff03 = full-width "#").
    pat = r"(?:area|\u533a\u57df)\s*(?:id|\u7f16\u53f7)?\s*[:=\uff03#\- ]*([0-9]+)" if kind == "area" else \
          r"(?:spot|point|\u6d4b\u70b9|\u70b9\u4f4d)\s*(?:id|\u7f16\u53f7)?\s*[:=\uff03#\- ]*([0-9]+)"
    m = re.search(pat, s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    candidates = [str(v).strip() for v in row.to_list() if pd.notna(v)]
    ids = [int(v) for v in candidates if re.fullmatch(r"\d+(?:\.0+)?", v)]
    if len(ids) == 1:
        return ids[0]
    raise ValueError(f"cannot unambiguously identify the {kind} ID in {s!r}; check the header of the original EDS file")


def read_one_eds_file(path: Path) -> pd.DataFrame:
    """Parse one EDS Excel file and return areaID/spotID/element/wt/at/err."""
    df = pd.read_excel(path, header=0, skiprows=SKIP_ROWS,
                       usecols="A:D", dtype=object)
    df.columns = ["element", "wt", "at", "err"]
    df = df.dropna(how="all").reset_index(drop=True)
    if len(df) < 2 + len(ELEMENTS):
        raise ValueError(f"{path}: expected at least 14 rows (2 metadata rows + 12 elements), got {len(df)}")
    area_id = _extract_id(df.iloc[0], "area")
    spot_id = _extract_id(df.iloc[1], "spot")
    records = df.iloc[2:2+len(ELEMENTS)].copy()

    # Some instruments export the element column directly; check it first, otherwise use the fixed order of the old script.
    existing = records["element"].astype(str).str.strip().tolist()
    if set(existing) == set(ELEMENTS) and len(existing) == len(set(existing)):
        records["element"] = existing
    else:
        records["element"] = ELEMENTS
    for col in ("wt", "at", "err"):
        records[col] = pd.to_numeric(records[col], errors="coerce")
    if records["wt"].isna().any():
        raise ValueError(f"{path}: wt% values could not be parsed for: {records.loc[records.wt.isna(),'element'].tolist()}")
    records.insert(0, "spotID", spot_id)
    records.insert(0, "areaID", area_id)
    records["source_file"] = str(path)
    return records


def read_eds_directory(root: Path, duplicate: str = "error") -> pd.DataFrame:
    """Read Excel files recursively; only .xls/.xlsx/.xlsm are processed, so other files are not misread."""
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".xls", ".xlsx", ".xlsm")
                   and not p.name.startswith("~$"))
    if not paths:
        raise FileNotFoundError(f"{root}: no Excel measurement files found")
    tables = []
    errors = []
    for path in paths:
        try:
            tables.append(read_one_eds_file(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError("The following files could not be parsed; statistics were not computed:\n" + "\n".join(errors))
    long_df = pd.concat(tables, ignore_index=True)
    keys = ["areaID", "spotID", "element"]
    dup = long_df.duplicated(keys, keep=False)
    if dup.any():
        if duplicate == "error":
            raise ValueError("duplicate (areaID, spotID, element) records; check for duplicated files or pass --duplicate mean/first")
        if duplicate == "mean":
            long_df = long_df.groupby(keys, as_index=False)[["wt", "at", "err"]].mean()
        else:
            long_df = long_df.drop_duplicates(keys, keep="first")
    return long_df


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per spot and one column per element; avoids the chained assignment of the old version."""
    table = long_df.pivot(index=["areaID", "spotID"], columns="element", values="wt")
    table = table.reindex(columns=ELEMENTS)
    return table.sort_index().reset_index()


def parse_area_order(text: str) -> list[int]:
    """Parse e.g. '17-22,10-16', preserving the manual measurement order of the experiment."""
    result: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(v) for v in part.split("-", 1))
            step = 1 if a <= b else -1
            result.extend(range(a, b + step, step))
        else:
            result.append(int(part))
    return result


def select_group(wide: pd.DataFrame, areas: list[int], w_threshold: float) -> pd.DataFrame:
    area_rank = {a: i for i, a in enumerate(areas)}
    sel = wide.loc[wide["areaID"].isin(areas)].copy()
    sel["_order"] = sel["areaID"].map(area_rank)
    sel = sel.sort_values(["_order", "spotID"]).drop(columns="_order")
    return sel.loc[sel["W"] > w_threshold].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="root directory of the EDS Excel files")
    p.add_argument("--order1", default="17-22,10-16", help="area order of group 1")
    p.add_argument("--order2", default="23-28,3-9", help="area order of group 2")
    p.add_argument("--w1", type=float, default=6.8, help="W wt% threshold of group 1")
    p.add_argument("--w2", type=float, default=6.9, help="W wt% threshold of group 2")
    p.add_argument("--duplicate", choices=["error", "mean", "first"], default="error")
    p.add_argument("--output", type=Path, default=Path("results/02_EDS"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    long_df = read_eds_directory(a.input, a.duplicate)
    wide = to_wide(long_df)
    group1 = select_group(wide, parse_area_order(a.order1), a.w1)
    group2 = select_group(wide, parse_area_order(a.order2), a.w2)
    for name, table in [("EDS_long", long_df), ("EDS_per_spot", wide),
                        ("group1_W_filtered", group1), ("group2_W_filtered", group2)]:
        table.to_csv(a.output / f"{name}.csv", index=False, encoding="utf-8-sig")
    with (a.output / "W_filter_statistics.txt").open("w", encoding="utf-8") as f:
        for name, table in [("group1", group1), ("group2", group2)]:
            f.write(f"{name}: {len(table)} spots\n")
            f.write(table[ELEMENTS].describe().to_string() + "\n\n" if len(table) else "no spots meet the criterion\n\n")
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    for name, table, style in [("Group 1", group1, "o-"), ("Group 2", group2, "s-")]:
        if len(table):
            ax.plot(np.arange(1, len(table)+1), table["W"], style, label=name, markersize=3)
    ax.set_xlabel("Filtered spot sequence")
    ax.set_ylabel("W concentration (wt%)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.output / "W_filtered_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Done: {len(wide)} spots; group 1 = {len(group1)}, group 2 = {len(group2)}. Results: {a.output.resolve()}")


if __name__ == "__main__":
    main()
