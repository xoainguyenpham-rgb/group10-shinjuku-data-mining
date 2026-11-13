# M4 corrected analysis package

This package replaces the earlier M4 segment benchmark calculation and its figures. All narrative and labels are in English. The original submissions remain unchanged.

## Start here

1. Read M4_Fixed_Group10_EN.docx.
2. Use outputs/M4_ward_comparisons.csv for the ward headline figures.
3. Use outputs/M4_segment_benchmarks.csv for corrected segment gaps and coverage.
4. Read outputs/M4_revision_audit.csv before updating E2, E3 or slides.

## Reproduce

Use Python 3.12 and the versions in requirements.txt. From this folder:

```bash
python -m pip install -r requirements.txt
python scripts/m4_fixed.py
```

The script regenerates the numerical outputs and figures, not the Word narrative. If the inputs change, update and review the Word narrative against the new outputs. The script uses package-relative paths and does not print personal paths. IDs are read as strings. Source files have been copied into inputs without editing.

## Inputs

- T3_compare.csv: original DP4 Q2 baseline, 14,935 rows across four wards; Q3 presence from the full Tokyo ID index.
- Q1_id_cluster.csv: final Orange RQ1 assignments, 5,328 priced Shinjuku Q2 listings. Despite the legacy filename, these are the Q2 clustering labels. The final numeric cluster field is used; raw Orange labels are not treated as the same numbering.
- previous_segment_benchmarks.csv: previous values retained solely to document revisions. It does not determine the revised estimates.

## Methods

Whole-ward comparisons use all baseline listings for presence and observed prices for median prices. Supported ward cells require at least 30 listings and 30 observed prices on each side. The price index is the weighted geometric mean of within-cell median ratios minus one. The presence gap is a weighted mean of within-cell differences. Weights are Shinjuku listing shares within the supported cells.

Corrected segment comparisons use priced listings only. A supported segment cell requires at least 30 listings within the Shinjuku segment and 30 priced comparator listings. Both sides use the same supported room-type by capacity cells and the segment's weights. Coverage is relative to the full Shinjuku segment. This is an explicit change from the previous use of ward-level support and full-segment outcomes against restricted benchmarks.

The corrected segment price index uses the same within-cell-ratio formula as the ward index. It replaces the old ratio of a full-segment median to a geometric comparator benchmark. These are different estimands; do not present the numerical changes as a market trend.

No inferential confidence intervals, significance stars or exceeds-range flags are produced. These outputs describe observed snapshots. They do not demonstrate regulatory effects, occupancy, profitability or future business survival. The 30-observation rule is a descriptive support check. RQ2 predictions are not used.

## Output table guide

- M4_ward_overview.csv: ward totals, observed price counts, median price and observed presence.
- M4_structure_shares.csv: counts and listing shares by room type, capacity and Tokyo host scale.
- M4_price_availability_audit.csv: priced and price-missing cohorts separately by ward.
- M4_within_stratum_gaps.csv: all 12 strata per comparator, denominators, medians, presence and support flags.
- M4_ward_comparisons.csv: raw and weighted ward gaps and support coverage.
- M4_sensitivity_checks.csv: cell thresholds, priced-only cohorts and price-only tail filters.
- M4_segment_profiles.csv: full-cluster totals, full presence, median prices and both share denominators.
- M4_segment_cell_audit.csv: all segment by comparator by stratum counts, measures and support flags.
- M4_segment_benchmarks.csv: corrected covered-cohort gaps and exact retained/excluded counts.
- M4_segment_sensitivity.csv: support thresholds 20, 30 and 50, with recalculated coverage and gaps.
- M4_unpriced_comparison.csv: unadjusted price-missing cohort comparisons; no cluster labels or stratification.
- M4_revision_audit.csv: old and revised segment values, with the reason for the changes.
- M4_QA.json: executed integrity checks, scope and source hashes.

Percentages in CSV columns ending pct are stored on a 0–100 scale. A pp column is a percentage-point difference. Price columns ending jpy are JPY per night. Blank numeric cells mean unavailable or undefined, never zero. Counts must be read as counts, not rates. CSVs preserve calculation precision; round only for presentation. Capacity labels such as 1-2 and 3-4 are text, not dates.

## Figures

- M4_01_ward_comparisons.png: headline ward price and presence gaps.
- M4_02_market_structure.png: room, capacity and host-scale composition.
- M4_03_segment_presence.png: corrected segment gaps alongside coverage.
- M4_04_price_missing_audit.png: full-range priced versus price-missing rates, including Taito.

Taito remains a sensitivity comparator. Pooled comparator results are omitted from this revised main package; the primary comparisons are separate Shibuya and Toshima benchmarks. Small-cell rows remain in the audit tables even when excluded from an aggregate.

## Verification and handover

The run checks unique IDs, baseline and follow-up periods, the exact eligible cluster cohort, five cluster counts, no row inflation, coverage arithmetic and the presence-gap identity. It also reproduces the previous verified ward price indices. These checks do not certify causal validity or every possible data-collection issue.

An and Toan should replace old segment benchmark values, range flags and chart captions. Mai Anh should update E2 and E3 from the revised tables. In particular, C1 is no longer below the Toshima benchmark under the corrected design. DP4 preparation outputs have not been overwritten: the RQ1 join is local to this M4 analysis, and RQ2 integration is separate.
