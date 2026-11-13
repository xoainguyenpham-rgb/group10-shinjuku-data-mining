# DP4 Validation record

The supplied Python script was executed on the packaged inputs. All nine notebook code cells were executed sequentially in a Python namespace; their actual stdout is embedded in the notebook. This verifies cell logic and execution order, but does not claim that Jupyter or the user's VS Code/Mac kernel was tested.

## Runtime

Python 3.12.14; pandas 2.2.3; numpy 2.3.5. The QA JSON records the runtime of each subsequent pipeline run.

## Preparation checks

- 14,935 baseline rows; 42 output columns.
- All 19 source columns preserved, including missing values.
- 953 price-missing listings retained.
- 0 duplicate listing/snapshot keys; 0 rows added by joins.
- Full Tokyo follow-up index: 35,970 unique listing IDs, derived without ward filtering.
- 0 disagreements with the 5,689 official DP3 Shinjuku Q2-to-Q3 labels.
- 48 complete stratum cells and 36 focal–comparator support records.
- Input hashes validated against packaged provenance; output checksums recorded in DP4_manifest.csv.

## Automated checks

Nine unittest cases passed:

1. Original values, missing prices and ward counts are preserved.
2. Membership labels, empty cells, group counts and composition denominators reconcile.
3. Partial cluster joins preserve rows and distinguish missing results from excluded listings.
4. Duplicate cluster keys are rejected.
5. Wrong cluster baseline periods are rejected.
6. Comparator cluster assignments are rejected.
7. Cluster assignments to price-missing Shinjuku listings are rejected.
8. A true probability of zero is preserved; absent probabilities stay missing; probabilities outside [0,1] are rejected.
9. Wrong prediction follow-up periods and missing supplied result values are rejected.

Synthetic labels/probabilities used in these tests exist only in temporary test files and are not included in the final comparison outputs.

## Remaining work outside preparation

Final RQ1 cluster labels (5,328 eligible rows) and final RQ2 test probabilities (5,689 in-scope rows) have not been supplied. Their output columns are blank with explicit coverage statuses. Model provenance and whether the final test was kept out of tuning require review of the modelling workflow, not merely a successful data join. M4 interpretation, final visualizations and regulatory-source verification remain separate tasks.
