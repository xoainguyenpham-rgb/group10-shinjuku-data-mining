# DP4 Comparative data preparation

## Research scope and cohort

RQ3 examines how Shinjuku's Airbnb market structure differs from selected Tokyo wards in the context of announced regulatory changes. The comparative dataset uses the project's Q2 2026 snapshot and consistent DP1 variable definitions. It includes 14,935 listings: 5,689 in Shinjuku, 2,433 in Shibuya, 3,232 in Toshima and 3,581 in Taito. Shinjuku is the focal ward, Shibuya and Toshima are the primary comparators, and Taito is reserved as an additional comparator for sensitivity analysis. These comparator roles follow the project design; they do not establish causal comparability.

The unit of analysis is a listing at baseline, identified by listing_id and snapshot. Identifiers are read as strings to preserve their full precision. The project's quarter labels identify snapshot cohorts rather than exact calendar-quarter scrape boundaries. The source Q2 records were scraped from 30 June to 3 July 2026; the Q3 follow-up records were scraped from 28 September to 4 October 2026.

## Variable preservation and missingness

DP4 preserves all 19 variables from the prepared DP1 comparison table. No baseline rows are removed. Missing values remain missing, and observed prices are not imputed, winsorized, logged or standardized for descriptive comparison. In total, 953 listings have missing prices. Price summaries therefore use the observed-price subset, whereas listing composition and next-snapshot presence use the full applicable cohort. Each output reports its relevant count and denominator. Advertised price is expressed in JPY per night; it is not realized revenue.

The supplied calculated_host_listings_count_clean retains its Tokyo scope. Host portfolio categories are 1, 2–5 and 6+ listings, and portfolio distributions count listings rather than distinct hosts. Unique host counts are reported separately and should not be summed across wards or strata because hosts may appear in multiple groups.

## Comparison strata and sample support

Listings are grouped by room_type and capacity_group, with capacity categories of 1–2, 3–4 and 5+ guests. The preparation produces all 48 ward-by-stratum cells, including empty cells, and 36 focal–comparator support records. A descriptive minimum of 30 observations is used to flag small cells separately for total listings and observed prices. This flag is a reporting convention, not a significance test or evidence that a comparison is unbiased. It does not remove observations. Empty cells retain missing summary values rather than zero-valued prices or rates.

These broad strata support comparisons between more similar listing types, but do not control all differences in location, quality, amenities or host behavior. Overall medians and within-stratum medians answer different descriptive questions and should be reported separately. Averaging stratum medians does not recover the overall median.

## Retrospective next-snapshot presence

For each Q2 listing, present_next_observed equals one if its identifier appears anywhere in the complete Tokyo Q3 snapshot and zero otherwise. The lookup is not restricted to the listing's baseline ward. The Q3 index contains all 35,970 source listing identifiers. In Shinjuku, the derived labels exactly match the official DP3 Q2-to-Q3 test labels: 5,372 present and 317 absent.

This variable measures presence in a later data file. It does not establish continued operation, permanent exit, legal compliance, occupancy or a policy effect. It is a retrospective outcome and must not be included among baseline predictors. Descriptive Q3 results must not be used to tune the RQ2 model or alter model selection after inspecting the final test outcomes.

## Model-result linkage and quality assurance

Final RQ1 cluster labels can be joined by listing_id and snapshot. Under the current DP2 specification, 5,328 Shinjuku listings with observed prices are eligible; the 361 price-missing Shinjuku listings remain in the comparison table with an explicit exclusion status. Comparator listings are outside the RQ1 clustering scope. Final RQ2 test probabilities can be joined by listing_id, baseline period and follow-up period, with their interpretation explicitly fixed to P(present_next = 1).

Both optional joins validate key uniqueness, periods, eligible identifiers and row preservation. Unavailable results remain missing and receive explicit status labels. No cluster labels or model probabilities were supplied for this DP4 delivery. The quality checks confirm preservation of source values, unchanged cohort sizes, unique baseline keys, consistent group definitions, correct denominators and agreement with DP3 labels. Input/output checksums support reproducibility. The current delivery completes data preparation; final model integration and substantive RQ3 interpretation remain subsequent tasks.
