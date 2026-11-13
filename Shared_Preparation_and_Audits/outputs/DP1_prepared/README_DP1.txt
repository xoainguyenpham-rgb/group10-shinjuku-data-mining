# Group 10 - DP1 handover

## Files and ownership
- 01_T1_RQ1_Q2_Shinjuku.csv: Toan - RQ1 segmentation.
- 02_T2_RQ2_train_Q1_to_Q2.csv: Hoang - RQ2 model development.
- 03_T2_RQ2_test_Q2_to_Q3.csv: Hoang - final temporal evaluation.
- 04_T3_RQ3_Q2_four_wards.csv: Mai Anh / An - RQ3 comparisons.
- 05_variable_roles.csv: column roles for each research question.
- 06-14: summaries and preparation audits.
- 15_RQ2_label_audit_NOT_predictors.csv: label provenance only.

## Data status
These are cleaned analysis tables, not fully preprocessed model inputs.
Missing values are preserved. No imputation, scaling or winsorisation
has been applied. Blank CSV cells represent missing values.

## Import rules
Import listing_id and host_id as text, not numeric variables.
Import capacity_group and host_portfolio_group as categorical text.
Do not open and re-save these CSVs in Excel with automatic type detection:
long IDs and group labels may be changed.
In Orange, explicitly set present_next as a categorical target.
Use 05_variable_roles.csv to select features and metadata.

## RQ1
Use the six proposed clustering inputs documented in variable_roles.
Handle missingness and skewness, then standardise before K-means.
Record preprocessing choices and assess cluster stability.
Profile clusters using observed values in the original units.
Keep listing_id for matching cluster assignments back to T1.

## RQ2
Target: 1 = present in the next Tokyo snapshot; 0 = absent.
Match labels against the full next Tokyo snapshot, not a ward subset.
Fit learned preprocessing within training folds during model selection.
Use the temporal test set only after model and tuning decisions are fixed.
Do not combine train and test for random splitting or preprocessing.
Exclude identifiers, period fields, split, comparison groups and
follow-up source information from predictors.
Evaluate against a majority-class baseline and report minority-class
performance, not accuracy alone.
Listings can recur across baseline periods; this is a temporal evaluation,
not an unseen-listing evaluation.

## RQ3
Use Q2 observed values for all four wards.
Primary comparators: Shibuya and Toshima. Sensitivity comparator: Taito.
Do not impute or winsorise prices for primary descriptive comparisons.
Retain price-missing listings for summaries of other observed variables.
Report valid counts and missingness for every measure.
Portfolio groups use the supplied Tokyo-scoped host listing count.
Group counts describe listings, not distinct hosts.

## Cleaning decisions and limits
Bathroom conflicts are left missing; use audit files 11 and 12 for details.
Missing review frequency or ratings must not automatically become zero.
Availability is not occupancy; review frequency is not booking volume.
Snapshot presence does not establish continued operation or compliance.
Comparisons do not identify causal effects of regulatory changes.
