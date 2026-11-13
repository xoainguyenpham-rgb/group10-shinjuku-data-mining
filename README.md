# Group 10 — Shinjuku Airbnb Data Mining

Business Analytics course project — Group 10, BA65, National Economics University.

This repository contains the reviewed analysis code, processed inputs, outputs and Orange workflows for the Shinjuku Airbnb project. It combines the supplied shared-preparation and RQ1/RQ2 sources with the corrected DP4/M4 package. Final report wording and team sign-off remain separate from code publication.

## Research questions
1. What operational segments can be identified among Shinjuku Airbnb listings?
2. How well do baseline listing and host characteristics predict listing-ID presence in the next Tokyo snapshot?
3. Which Shinjuku segments warrant operational review based on comparative market evidence and the announced regulatory context?

## Setup and run
Use Python 3.12. From this project folder:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python verify_results.py
```

Windows activation: `.venv\Scripts\activate`.
Raw CSV files are supplied separately. Place the four original snapshots in `Shared_Preparation_and_Audits/raw_data/` using filenames in RAW_DATA_MANIFEST.csv. Do not edit IDs or open/save the files through automatic Excel type conversion. Verify checksums before running.

Stages may be rerun with `--stage shared`, `--stage rq1`, `--stage rq2`, or `--stage rq3`, in that order. Scripts overwrite their named outputs. RQ3 requires shared outputs and RQ1 final labels. Model fitting may take several minutes.

## Structure and authority
- Shared_Preparation_and_Audits: original raw-data audit, deterministic DP1 cleaning, repaired DU4 comparator exploration. Runnable scripts are generated from supplied notebooks. Scripts drive the common runner.
- RQ1_Segmentation: DP2 transformation; M1 K comparison and sensitivity; M2 observed-value profiles. Supplied Orange K=5 labels remain authoritative. Python K=5 is a validation comparison and is not silently substituted.
- RQ2_Persistence: DP3 temporal assembly and official M3 Orange script. `reference_outputs` preserves earlier reviewed M3 results; `outputs/M3` contains the new run. GUI workflows illustrate the process; their random seeds differ from the authoritative script.
- RQ3_Ward_Comparison: previously corrected DP4 preparation and M4 comparisons; current DP1 outputs and final RQ1 labels are propagated by the runner. This is not a claim that a newly uploaded RQ3 version was received.

## Analytical limits
RQ1 models 5,328 priced Q2 Shinjuku listings; 361 price-missing listings remain in broader descriptive analyses. K=3 maximises tested silhouette; K=5 is a profiling tradeoff. Seed stability does not establish feature robustness. RQ2 uses Q1→Q2 development and Q2→Q3 temporal evaluation. The test has already been inspected: this run reproduces the fixed analysis, not a new untouched test. New tuning based on Q3 would be exploratory. The tested Random Forest is a 10-tree configuration, not an exhaustive optimisation of every method.

Presence means an ID is in the supplied next Tokyo snapshot. It does not establish operation, compliance or permanent closure. Missing collection-source metadata in Q3 and 37 reappearing Q1 IDs must be discussed. Availability is not occupancy, asking price is not revenue, and host listing count has Tokyo scope.

M4 uses supported room-type × capacity cells. Weighted price comparisons are geometric indices of stratum median ratios, not adjusted ward medians. Segment comparisons use the same covered cells and weights on both sides. No causal regulatory effects or significance claims are made.

## Data access and project status
Raw snapshots are distributed separately and excluded from Git. Their exact filenames and SHA-256 checksums are listed in [RAW_DATA_MANIFEST.csv](Shared_Preparation_and_Audits/RAW_DATA_MANIFEST.csv). Obtain the matching files from the project team before running the full pipeline. Processed inputs and reviewed outputs are included for inspection.

The validation recorded in [VALIDATION_NOTES.md](VALIDATION_NOTES.md) belongs to the reviewed execution; publishing this repository is not a new full model run. Orange GUI workflows have not been independently executed. Verified original dataset attribution/licensing, final member contributions and the final report link are still to be completed by the team. No licence for the underlying third-party data is asserted by this repository.
