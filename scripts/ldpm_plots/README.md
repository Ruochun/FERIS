# LDPM Single-Tet Plot Scripts

Small helpers for visualizing `ldpm_singletet` CSV output, especially
`*_subfacet_stress_strain.csv`.

## Requirements

Python packages:

```bash
python -m pip install numpy pandas matplotlib
```

## Generate Plot For One Case

Run from the directory that contains the `ldpm_singletet_caseN/` output folders,
usually the repository root if the example was launched there:

```bash
python scripts/ldpm_plots/plot_ldpm_subfacet_stress_strain.py \
  ldpm_singletet_case1/ldpm_singletet_case1_subfacet_stress_strain.csv \
  -o ldpm_plot_1.pdf
```

The PDF contains one page per facet. Each page shows normal/tangential strain
and stress histories, plus stress-strain plots.

## Overlay Multiple CSVs

Multiple CSV files can be passed to compare/overlay results:

```bash
python scripts/ldpm_plots/plot_ldpm_subfacet_stress_strain.py \
  run_a/ldpm_singletet_case1_subfacet_stress_strain.csv \
  run_b/ldpm_singletet_case1_subfacet_stress_strain.csv \
  --labels run_a run_b \
  -o case1_compare.pdf
```

To plot only selected facets:

```bash
python scripts/ldpm_plots/plot_ldpm_subfacet_stress_strain.py \
  ldpm_singletet_case1/ldpm_singletet_case1_subfacet_stress_strain.csv \
  --facets 1 2 7 \
  -o ldpm_plot_case1_facets_1_2_7.pdf
```

## Plot All Seven Cases

After running `ldpm_singletet` for cases 1 through 7, this loop creates
`ldpm_plot_1.pdf` through `ldpm_plot_7.pdf`:

```bash
for i in {1..7}; do
  python scripts/ldpm_plots/plot_ldpm_subfacet_stress_strain.py \
    build/ldpm_singletet_case${i}/ldpm_singletet_case${i}_subfacet_stress_strain.csv \
    -o ldpm_plot_${i}.pdf
done
```

`run_all.sh` is a shorthand version of this loop, but it assumes the plotting
script and the `ldpm_singletet_caseN/` output folders are visible from the
current working directory.
