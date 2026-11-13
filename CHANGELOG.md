# Changes in this review candidate
- Restored raw-to-DP1 executable preparation from the newly supplied notebooks.
- Fixed DU4 NumPy import order and missing capacity_price aggregation (entire-home cohort, matching the following figure).
- Added a root runner to propagate freshly generated DP1 data into RQ1/RQ2/DP4 and RQ1 labels into M4.
- Preserved final Orange cluster assignments and original RQ2 selection/settings; no new model search.
- Reused previously corrected M4 because the new RQ3 upload was not received.
- Changed M2 axis labels to JPY/night and Tokyo host scope.
- Removed absolute input path from newly generated M3 selection metadata.
- Kept raw data separate from the code ZIP and documented required filenames.
- GUI workflows are retained as supplied references and may need File-widget rebinding on a new machine; the script runner is the reproducible route.
