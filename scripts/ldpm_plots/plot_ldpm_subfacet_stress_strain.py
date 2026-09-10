#!/usr/bin/env python3
"""
Create LDPM subfacet stress/strain plots from one or more CSV files.

Expected CSV columns:
  time_s (or '# time_s')
  facet_<id>_stress_n, facet_<id>_stress_m, facet_<id>_stress_l
  facet_<id>_strain_n, facet_<id>_strain_m, facet_<id>_strain_l

For each facet, one PDF page is generated with a 3x2 layout:
  StrainN-time, StressN-time
  StrainT-time, StressT-time
  StressN-StrainN, StressT-StrainT
where tangent quantities are magnitudes:
  StrainT = sqrt(strain_m^2 + strain_l^2)
  StressT = sqrt(stress_m^2 + stress_l^2)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


TIME_ALIASES = ("time_s", "# time_s", "time", "t")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    return df


def find_time_column(df: pd.DataFrame) -> str:
    for name in TIME_ALIASES:
        key = name.strip().lstrip("#").strip()
        if key in df.columns:
            return key
    raise ValueError(f"Could not find a time column. Tried: {TIME_ALIASES}")


def find_facets(df: pd.DataFrame) -> List[int]:
    facets = set()
    pat = re.compile(r"facet_(\d+)_(stress|strain)_[nml]$")
    for col in df.columns:
        m = pat.match(col)
        if m:
            facets.add(int(m.group(1)))
    return sorted(facets)


def read_case(csv_path: Path, label: str | None = None) -> Tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df = clean_columns(df)
    label = label or csv_path.stem
    return label, df


def require_facet_columns(df: pd.DataFrame, facet: int) -> None:
    required = [
        f"facet_{facet}_stress_n", f"facet_{facet}_stress_m", f"facet_{facet}_stress_l",
        f"facet_{facet}_strain_n", f"facet_{facet}_strain_m", f"facet_{facet}_strain_l",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Facet {facet} is missing columns: {missing}")


def tangent_magnitude(df: pd.DataFrame, facet: int, kind: str) -> np.ndarray:
    a = df[f"facet_{facet}_{kind}_m"].to_numpy(dtype=float)
    b = df[f"facet_{facet}_{kind}_l"].to_numpy(dtype=float)
    return np.sqrt(a * a + b * b)


def nice_ylim(ax, values: List[np.ndarray], default=(-1.0, 1.0), pad=0.08) -> None:
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values if len(v)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ax.set_ylim(*default)
        return
    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin
    if span < 1e-12:
        center = 0.5 * (vmin + vmax)
        half = max(abs(center) * 0.1, 1.0 if default == (-1.0, 1.0) else 1e-3)
        ax.set_ylim(center - half, center + half)
    else:
        ax.set_ylim(vmin - pad * span, vmax + pad * span)


def plot_one_facet(pdf: PdfPages, cases: List[Tuple[str, pd.DataFrame]], facet: int, page_number: int) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(8.5, 11.0))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.93, wspace=0.32, hspace=0.42)

    axes = axs.ravel()
    collected: Dict[str, List[np.ndarray]] = {k: [] for k in ["time", "en", "sn", "et", "st"]}

    for label, df in cases:
        require_facet_columns(df, facet)
        tcol = find_time_column(df)
        t = df[tcol].to_numpy(dtype=float)
        en = df[f"facet_{facet}_strain_n"].to_numpy(dtype=float)
        sn = df[f"facet_{facet}_stress_n"].to_numpy(dtype=float)
        et = tangent_magnitude(df, facet, "strain")
        st = tangent_magnitude(df, facet, "stress")

        collected["time"].append(t)
        collected["en"].append(en)
        collected["sn"].append(sn)
        collected["et"].append(et)
        collected["st"].append(st)

        axes[0].plot(t, en, label=label, linewidth=1.2)
        axes[1].plot(t, sn, label=label, linewidth=1.2)
        axes[2].plot(t, et, label=label, linewidth=1.2)
        axes[3].plot(t, st, label=label, linewidth=1.2)
        axes[4].plot(en, sn, label=label, linewidth=1.2)
        axes[5].plot(et, st, label=label, linewidth=1.2)

    titles = [
        f"StrainN Facet{facet}", f"StressN Facet {facet}",
        f"StrainT Facet{facet}", f"StressT Facet{facet}",
        f"StressN-StrainN Facet {facet}", f"StressT-StrainT Facet{facet}",
    ]
    xlabels = ["Time [s]", "Time [s]", "Time [s]", "Time [s]", "Strain", "Strain"]
    ylabels = ["Strain", "Stress [MPa]", "Strain", "Stress [MPa]", "Stress [MPa]", "Stress [MPa]"]

    for ax, title, xlabel, ylabel in zip(axes, titles, xlabels, ylabels):
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best", frameon=True)

    for ax in axes[:4]:
        nice_ylim(ax, collected["en"] if ax is axes[0] else collected["sn"] if ax is axes[1] else collected["et"] if ax is axes[2] else collected["st"])
    nice_ylim(axes[4], collected["sn"])
    nice_ylim(axes[5], collected["st"])

    # Match the example's page number centered near the bottom.
    fig.text(0.5, 0.035, str(page_number), ha="center", va="center", fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LDPM subfacet stress/strain CSV data into a multi-page PDF.")
    parser.add_argument("csv", nargs="+", type=Path, help="Input CSV file(s). Multiple files are overlaid.")
    parser.add_argument("-o", "--output", type=Path, default=Path("ldpm_subfacet_plots.pdf"), help="Output PDF path.")
    parser.add_argument("--labels", nargs="*", help="Optional legend labels, one per CSV.")
    parser.add_argument("--facets", nargs="*", type=int, help="Optional list of facet ids to plot. Default: detected facets common to all CSVs.")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.csv):
        raise SystemExit("--labels must provide exactly one label per CSV file.")

    cases = [read_case(path, args.labels[i] if args.labels else None) for i, path in enumerate(args.csv)]
    facet_sets = [set(find_facets(df)) for _, df in cases]
    common_facets = sorted(set.intersection(*facet_sets))
    facets = args.facets or common_facets
    if not facets:
        raise SystemExit("No facet columns were detected.")

    with PdfPages(args.output) as pdf:
        for page_no, facet in enumerate(facets, start=1):
            plot_one_facet(pdf, cases, facet, page_no)

    print(f"Wrote {args.output} with {len(facets)} page(s).")


if __name__ == "__main__":
    main()
