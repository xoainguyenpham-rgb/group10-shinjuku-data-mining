# RQ1: Segmentation

See the repository root README for install and run commands.

- `inputs/`: Shinjuku DP1 source, variable-role dictionary, and Orange K=5 output.
- `scripts/`: DP2, M1 and M2 scripts.
- `outputs/`: retained DP2/M1/M2 results; scripts regenerate them.
- `workflows/`: Orange workflow used in the modelling process.
- `docs/`: handover and method documents.

Run order: `dp2_prepare.py` → `m1_evaluate.py` → `m2_profile.py`.
