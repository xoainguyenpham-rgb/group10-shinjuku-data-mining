# Executed validation

- 01_T1_RQ1_Q2_Shinjuku: reproduced supplied table
- 02_T2_RQ2_train_Q1_to_Q2: reproduced supplied table
- 03_T2_RQ2_test_Q2_to_Q3: reproduced supplied table
- 04_T3_RQ3_Q2_four_wards: reproduced supplied table
- Raw checksums verified for all four snapshots.
- Shared audit, DP1 and repaired DU4 executed from raw.
- DP2, M1 (including sensitivities) and M2 executed; final cluster counts preserved.
- DP3 and M3 executed with Orange 3.40.0 / scikit-learn 1.5.2; RF confusion matrix exactly reproduced.
- DP4 and corrected M4 executed from freshly propagated inputs; both report PASS.
- 21 root validation checks and 9 DP4 tests passed.

Remaining review: the group should confirm wording, dataset attribution/licence and workflow File-widget locations. These checks do not establish causal validity. GUI workflows have not been executed here. M4 uses RQ1 labels locally; DP4 preparation's pending-model flags are expected and are not a failed run. RQ2 predictions remain separate because M4's descriptive benchmarks do not require them.
