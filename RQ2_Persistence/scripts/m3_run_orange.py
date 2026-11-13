"""Run the official M3 temporal evaluation with Orange 3.

The required input is DP3's official T2_clean.csv. Model development uses
Shinjuku 2026_Q1 -> 2026_Q2 only. The 2026_Q2 -> 2026_Q3 cohort is held out
until one non-constant model has been selected by stratified 5-fold CV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import Orange
import sklearn
from Orange.classification import (
    LogisticRegressionLearner,
    MajorityLearner,
    RandomForestLearner,
    TreeLearner,
)
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
from Orange.evaluation import CrossValidation, TestOnTestData
from Orange.preprocess import Continuize, Impute, Normalize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


SEED = 20261031
TARGET = "present_next"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
EXPECTED_COHORTS = {
    TRAIN_SPLIT: {
        "baseline_period": "2026_Q1",
        "followup_period": "2026_Q2",
        "rows": 4166,
        "class_0": 266,
        "class_1": 3900,
    },
    TEST_SPLIT: {
        "baseline_period": "2026_Q2",
        "followup_period": "2026_Q3",
        "rows": 5689,
        "class_0": 317,
        "class_1": 5372,
    },
}

CATEGORICAL_PREDICTORS = ["room_type", "superhost_status"]
NUMERIC_PREDICTORS = [
    "price_numeric",
    "accommodates_clean",
    "bedrooms_clean",
    "beds_clean",
    "bathrooms_clean",
    "minimum_nights_clean",
    "availability_365_clean",
    "number_of_reviews_clean",
    "reviews_per_month_clean",
    "review_scores_rating_clean",
    "calculated_host_listings_count_clean",
]
PREDICTORS = CATEGORICAL_PREDICTORS + NUMERIC_PREDICTORS
REQUIRED_COLUMNS = {
    "listing_id",
    "host_id",
    "baseline_period",
    "followup_period",
    "split",
    TARGET,
    *PREDICTORS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def validate_input(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    observed_splits = set(frame["split"].dropna().unique())
    if observed_splits != {TRAIN_SPLIT, TEST_SPLIT}:
        raise ValueError(f"Expected train/test splits, found: {sorted(observed_splits)}")

    if set(frame[TARGET].dropna().unique()) != {0, 1}:
        raise ValueError(f"{TARGET} must contain exactly the labels 0 and 1.")
    if frame[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values.")
    if frame.duplicated(["listing_id", "followup_period"]).any():
        raise ValueError("listing_id + followup_period must be unique.")

    for split_name, expected in EXPECTED_COHORTS.items():
        subset = frame.loc[frame["split"].eq(split_name)].copy()
        if len(subset) != expected["rows"]:
            raise ValueError(
                f"{split_name} has {len(subset)} rows; expected {expected['rows']}."
            )
        for period_column in ("baseline_period", "followup_period"):
            values = set(subset[period_column].dropna().unique())
            if values != {expected[period_column]}:
                raise ValueError(
                    f"{split_name} {period_column} is {sorted(values)}; "
                    f"expected {expected[period_column]}."
                )
        class_counts = subset[TARGET].value_counts().to_dict()
        if class_counts.get(0, 0) != expected["class_0"] or class_counts.get(1, 0) != expected["class_1"]:
            raise ValueError(
                f"{split_name} class counts are {class_counts}; "
                f"expected 0={expected['class_0']}, 1={expected['class_1']}."
            )


def build_domain(train_frame: pd.DataFrame) -> tuple[Domain, dict[str, list[str]]]:
    category_levels: dict[str, list[str]] = {}
    attributes = []

    for column in CATEGORICAL_PREDICTORS:
        levels = sorted(train_frame[column].dropna().astype(str).unique().tolist())
        if not levels:
            raise ValueError(f"{column} has no observed train values.")
        category_levels[column] = levels
        attributes.append(DiscreteVariable(column, values=levels))

    attributes.extend(ContinuousVariable(column) for column in NUMERIC_PREDICTORS)
    target = DiscreteVariable(TARGET, values=("0", "1"))
    return Domain(attributes, target), category_levels


def frame_to_orange_table(
    frame: pd.DataFrame,
    domain: Domain,
    category_levels: dict[str, list[str]],
) -> tuple[Table, dict[str, int]]:
    feature_columns: list[np.ndarray] = []
    unseen_levels: dict[str, int] = {}

    for column in CATEGORICAL_PREDICTORS:
        codes = {level: index for index, level in enumerate(category_levels[column])}
        raw_values = frame[column].astype("string")
        encoded = raw_values.map(codes)
        unseen = raw_values.notna() & encoded.isna()
        unseen_levels[column] = int(unseen.sum())
        feature_columns.append(encoded.to_numpy(dtype=float, na_value=np.nan))

    for column in NUMERIC_PREDICTORS:
        feature_columns.append(pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float))

    x_values = np.column_stack(feature_columns)
    y_values = frame[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    return Table.from_numpy(domain, x_values, y_values), unseen_levels


def make_preprocessors() -> list[Any]:
    # These are fitted inside the learner call, so CV learns them per train fold.
    return [Impute(), Continuize(), Normalize()]


def learner_factories(seed: int) -> list[tuple[str, Callable[[], Any], dict[str, Any]]]:
    return [
        (
            "Constant",
            lambda: MajorityLearner(preprocessors=make_preprocessors()),
            {"learner": "MajorityLearner", "model_parameters": "Orange defaults"},
        ),
        (
            "Logistic Regression",
            lambda: LogisticRegressionLearner(
                random_state=seed,
                preprocessors=make_preprocessors(),
            ),
            {
                "learner": "LogisticRegressionLearner",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 100,
                "random_state": seed,
            },
        ),
        (
            "Decision Tree",
            lambda: TreeLearner(preprocessors=make_preprocessors()),
            {
                "learner": "TreeLearner",
                "max_depth": None,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
            },
        ),
        (
            "Random Forest",
            lambda: RandomForestLearner(
                random_state=seed,
                preprocessors=make_preprocessors(),
            ),
            {
                "learner": "RandomForestLearner",
                "n_estimators": 10,
                "max_features": "sqrt",
                "random_state": seed,
            },
        ),
    ]


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def probability_columns(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if probabilities.shape[1] < 2:
        raise ValueError("Expected probability outputs for classes 0 and 1.")
    return probabilities[:, 0].astype(float), probabilities[:, 1].astype(float)


def summarize_predictions(
    model_name: str,
    stage: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probability_0, _ = probability_columns(probabilities)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    auc_0 = np.nan
    if len(np.unique(y_true)) == 2:
        auc_0 = roc_auc_score((y_true == 0).astype(int), probability_0)

    metrics = {
        "stage": stage,
        "model": model_name,
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_0": float(precision[0]),
        "recall_class_0": float(recall[0]),
        "f1_class_0": float(f1[0]),
        "auc_class_0": float(auc_0),
        "precision_class_1": float(precision[1]),
        "recall_class_1": float(recall[1]),
        "f1_class_1": float(f1[1]),
    }
    per_class = [
        {
            "stage": stage,
            "model": model_name,
            "class": int(label),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate((0, 1))
    ]
    return metrics, per_class


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=pd.Index([0, 1], name="actual_present_next"),
        columns=pd.Index(["predicted_0", "predicted_1"]),
    ).reset_index()


def extract_feature_importance(model: Any, model_name: str) -> pd.DataFrame:
    attributes = [attribute.name for attribute in model.domain.attributes]
    estimator = getattr(model, "skl_model", None)
    method = None
    values: np.ndarray | None = None

    if estimator is not None and hasattr(estimator, "feature_importances_"):
        method = "native_tree_feature_importance"
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif estimator is not None and hasattr(estimator, "coef_"):
        method = "absolute_logistic_coefficient"
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coefficients[0]) if coefficients.ndim == 2 else np.abs(coefficients)

    if values is None:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "fitted_on": "full_train_Q1_to_Q2_only",
                    "method": "not_available_for_constant_baseline",
                    "feature": "",
                    "original_predictor": "",
                    "importance": np.nan,
                    "rank": np.nan,
                }
            ]
        )
    if len(values) != len(attributes):
        raise RuntimeError("Feature importance length does not match fitted model domain.")

    records = []
    for feature, importance in zip(attributes, values):
        original_predictor = next(
            (column for column in CATEGORICAL_PREDICTORS if feature.startswith(f"{column}=")),
            feature,
        )
        records.append(
            {
                "model": model_name,
                "fitted_on": "full_train_Q1_to_Q2_only",
                "method": method,
                "feature": feature,
                "original_predictor": original_predictor,
                "importance": float(importance),
            }
        )
    result = pd.DataFrame(records).sort_values("importance", ascending=False, kind="stable")
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def cv_fold_assignments(results: Any, train_frame: pd.DataFrame) -> pd.DataFrame:
    fold_numbers = np.empty(results.nrows, dtype=int)
    for fold_index, fold_slice in enumerate(results.folds, start=1):
        fold_numbers[fold_slice] = fold_index

    assignment = pd.DataFrame(
        {
            "source_row": results.row_indices.astype(int),
            "cv_fold": fold_numbers,
            "present_next": results.actual.astype(int),
        }
    ).sort_values("source_row", kind="stable")
    identifiers = train_frame.loc[
        assignment["source_row"].to_numpy(),
        ["listing_id", "host_id", "baseline_period", "followup_period"],
    ].reset_index(drop=True)
    return pd.concat([identifiers, assignment.reset_index(drop=True)], axis=1)


def cv_oof_predictions(
    results: Any,
    train_frame: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    """Return pooled out-of-fold predictions with exact record identifiers."""
    fold_numbers = np.empty(results.nrows, dtype=int)
    for fold_index, fold_slice in enumerate(results.folds, start=1):
        fold_numbers[fold_slice] = fold_index

    base = train_frame.loc[
        results.row_indices.astype(int),
        ["listing_id", "host_id", "baseline_period", "followup_period"],
    ].reset_index(drop=True)
    base["source_row"] = results.row_indices.astype(int)
    base["cv_fold"] = fold_numbers
    base[TARGET] = results.actual.astype(int)

    model_frames = []
    for model_index, model_name in enumerate(model_names):
        probability_0, probability_1 = probability_columns(
            results.probabilities[model_index]
        )
        model_frame = base.copy()
        model_frame["model"] = model_name
        model_frame["predicted_present_next"] = results.predicted[model_index].astype(int)
        model_frame["probability_present_next_0"] = probability_0
        model_frame["probability_present_next_1"] = probability_1
        model_frames.append(model_frame)

    return (
        pd.concat(model_frames, ignore_index=True)
        .sort_values(["model", "source_row"], kind="stable")
        .reset_index(drop=True)
    )


def cv_host_overlap_summary(assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure whether each validation row's host is present in another fold."""
    host_fold_counts = assignments.groupby("host_id", dropna=False)["cv_fold"].nunique()
    detailed = assignments.copy()
    detailed["host_seen_in_cv_training"] = (
        detailed["host_id"].map(host_fold_counts).gt(1)
    )

    rows = []
    for fold, subset in detailed.groupby("cv_fold", sort=True):
        seen = int(subset["host_seen_in_cv_training"].sum())
        total = int(len(subset))
        rows.append(
            {
                "scope": f"fold_{int(fold)}_validation",
                "cv_fold": int(fold),
                "validation_rows": total,
                "rows_with_host_seen_in_training": seen,
                "share_with_host_seen_in_training": seen / total,
            }
        )

    seen_overall = int(detailed["host_seen_in_cv_training"].sum())
    total_overall = int(len(detailed))
    rows.append(
        {
            "scope": "all_pooled_validation_rows",
            "cv_fold": "",
            "validation_rows": total_overall,
            "rows_with_host_seen_in_training": seen_overall,
            "share_with_host_seen_in_training": seen_overall / total_overall,
        }
    )
    summary = {
        "validation_rows": total_overall,
        "rows_with_host_seen_in_training": seen_overall,
        "share_with_host_seen_in_training": seen_overall / total_overall,
        "interpretation": (
            "CV is stratified by class, not grouped by host. A validation row is marked "
            "seen when its host_id occurs in at least one of the other four folds."
        ),
    }
    return pd.DataFrame(rows), summary


def write_environment_file(output_path: Path, seed: int) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "M3 official runtime environment",
        f"generated_at_utc={generated_at}",
        "official_results_source=M3_run_orange.py",
        f"python={sys.version.split()[0]}",
        f"Orange={Orange.__version__}",
        f"scikit-learn={sklearn.__version__}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"platform={platform.platform()}",
        f"official_seed={seed}",
        "official_cv=Stratified 5-fold cross-validation",
        "official_cv_metric_aggregation=pooled out-of-fold predictions",
        "gui_workflow_role=process illustration; not the source of official numeric outputs",
        "orange_gui_test_and_score_seed=42 (Orange 3.40 internal setting)",
        "orange_gui_random_forest_replicable_seed=0",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(input_path: Path, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path)
    validate_input(frame)

    train_frame = frame.loc[frame["split"].eq(TRAIN_SPLIT)].reset_index(drop=True)
    test_frame = frame.loc[frame["split"].eq(TEST_SPLIT)].reset_index(drop=True)
    domain, category_levels = build_domain(train_frame)
    train_table, train_unseen = frame_to_orange_table(train_frame, domain, category_levels)
    test_table, test_unseen = frame_to_orange_table(test_frame, domain, category_levels)
    if any(train_unseen.values()):
        raise RuntimeError("Train data contains category values outside its own domain.")
    train_table.save(str(output_dir / "M3_Orange_train_Q1_to_Q2.tab"))
    test_table.save(str(output_dir / "M3_Orange_test_Q2_to_Q3.tab"))

    factories = learner_factories(seed)
    cv_learners = [factory() for _, factory, _ in factories]
    cv_results = CrossValidation(
        k=5,
        stratified=True,
        random_state=seed,
    )(train_table, cv_learners, suppresses_exceptions=False)
    if any(cv_results.failed):
        raise RuntimeError(f"Cross-validation failed: {cv_results.failed}")

    cv_metrics: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    for index, (model_name, _, _) in enumerate(factories):
        metrics, rows = summarize_predictions(
            model_name,
            "stratified_5_fold_cv_train_Q1_to_Q2",
            cv_results.actual,
            cv_results.predicted[index],
            cv_results.probabilities[index],
        )
        cv_metrics.append(metrics)
        per_class.extend(rows)

    cv_metrics_frame = pd.DataFrame(cv_metrics)
    cv_metrics_frame["is_constant_baseline"] = cv_metrics_frame["model"].eq("Constant")
    cv_metrics_frame["eligible_for_selection"] = ~cv_metrics_frame["is_constant_baseline"]
    cv_metrics_frame.to_csv(output_dir / "M3_cv_5fold_metrics.csv", index=False)
    fold_assignments = cv_fold_assignments(cv_results, train_frame)
    fold_assignments.to_csv(output_dir / "M3_cv_fold_assignments.csv", index=False)
    cv_oof_predictions(
        cv_results,
        train_frame,
        [name for name, _, _ in factories],
    ).to_csv(output_dir / "M3_oof_predictions.csv", index=False)
    host_overlap_frame, host_overlap_summary = cv_host_overlap_summary(fold_assignments)
    host_overlap_frame.to_csv(output_dir / "M3_cv_host_overlap_summary.csv", index=False)

    selection_pool = cv_metrics_frame.loc[cv_metrics_frame["eligible_for_selection"]].copy()
    order = {model_name: index for index, (model_name, _, _) in enumerate(factories)}
    selection_pool["selection_order"] = selection_pool["model"].map(order)
    selected_row = selection_pool.sort_values(
        [
            "f1_class_0",
            "recall_class_0",
            "precision_class_0",
            "auc_class_0",
            "selection_order",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).iloc[0]
    selected_model_name = str(selected_row["model"])
    selected_factory = next(factory for name, factory, _ in factories if name == selected_model_name)

    final_model_names = ["Constant", selected_model_name]
    final_factories = [
        next(factory for name, factory, _ in factories if name == model_name)
        for model_name in final_model_names
    ]
    final_results = TestOnTestData()( 
        train_table,
        test_table,
        [factory() for factory in final_factories],
        suppresses_exceptions=False,
    )
    if any(final_results.failed):
        raise RuntimeError(f"Final temporal test failed: {final_results.failed}")

    final_metrics: list[dict[str, Any]] = []
    final_prediction_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, model_name in enumerate(final_model_names):
        metrics, rows = summarize_predictions(
            model_name,
            "final_temporal_test_Q2_to_Q3",
            final_results.actual,
            final_results.predicted[index],
            final_results.probabilities[index],
        )
        metrics["is_constant_baseline"] = model_name == "Constant"
        metrics["is_cv_selected_model"] = model_name == selected_model_name
        final_metrics.append(metrics)
        per_class.extend(rows)
        final_prediction_arrays[model_name] = (
            final_results.predicted[index].astype(int),
            final_results.probabilities[index],
        )
        confusion_table(final_results.actual, final_results.predicted[index]).to_csv(
            output_dir / f"M3_confusion_matrix_test_{model_slug(model_name)}.csv",
            index=False,
        )

    pd.DataFrame(final_metrics).to_csv(output_dir / "M3_final_test_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(output_dir / "M3_per_class_results.csv", index=False)

    selected_predictions, selected_probabilities = final_prediction_arrays[selected_model_name]
    probability_0, probability_1 = probability_columns(selected_probabilities)
    prediction_frame = test_frame[
        ["listing_id", "host_id", "baseline_period", "followup_period", TARGET]
    ].copy()
    prediction_frame["selected_model"] = selected_model_name
    prediction_frame["predicted_present_next"] = selected_predictions
    prediction_frame["probability_present_next_0"] = probability_0
    prediction_frame["probability_present_next_1"] = probability_1
    prediction_frame.to_csv(output_dir / "M3_test_predictions.csv", index=False)

    selected_model = selected_factory()(train_table)
    extract_feature_importance(selected_model, selected_model_name).to_csv(
        output_dir / "M3_feature_importance.csv",
        index=False,
    )

    final_metrics_frame = pd.DataFrame(final_metrics)
    selected_final = final_metrics_frame.loc[
        final_metrics_frame["model"].eq(selected_model_name)
    ].iloc[0].to_dict()
    summary = {
        "analysis": "M3 official temporal model development",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": input_path.name,
            "sha256": sha256_file(input_path),
            "rows": int(len(frame)),
        },
        "cohorts": {
            "train": {
                "definition": "Shinjuku listing features at 2026_Q1; label is listing ID observed anywhere in Tokyo at 2026_Q2.",
                "rows": int(len(train_frame)),
                "class_counts": {
                    str(key): int(value)
                    for key, value in train_frame[TARGET].value_counts().sort_index().items()
                },
            },
            "test": {
                "definition": "Shinjuku listing features at 2026_Q2; label is listing ID observed anywhere in Tokyo at 2026_Q3.",
                "rows": int(len(test_frame)),
                "class_counts": {
                    str(key): int(value)
                    for key, value in test_frame[TARGET].value_counts().sort_index().items()
                },
            },
        },
        "predictors": {
            "categorical": CATEGORICAL_PREDICTORS,
            "numeric": NUMERIC_PREDICTORS,
            "excluded": [
                "listing_id",
                "host_id",
                "baseline_period",
                "followup_period",
                "split",
                TARGET,
                "capacity_group",
                "host_portfolio_group",
                "bathrooms",
                "estimated_occupancy_l365d",
                "estimated_revenue_l365d",
            ],
        },
        "preprocessing": {
            "steps": ["Impute", "Continuize", "Normalize"],
            "fit_scope": "Inside each learner fit. Therefore, preprocessing is fit only on each CV training fold and on the full train cohort before final test prediction.",
            "unseen_test_category_values_mapped_to_missing_then_train_imputed": test_unseen,
        },
        "cv": {
            "strategy": "StratifiedKFold",
            "folds": 5,
            "seed": seed,
            "metric_aggregation": "Pooled out-of-fold predictions across all five validation folds; metrics are not the arithmetic mean of five fold-level metric values.",
            "selection_metric": "f1_class_0",
            "selection_tie_break_order": [
                "f1_class_0 descending",
                "recall_class_0 descending",
                "precision_class_0 descending",
                "auc_class_0 descending",
                "fixed model order",
            ],
            "selection_pool": ["Logistic Regression", "Decision Tree", "Random Forest"],
            "constant_role": "Reported as the baseline and not eligible for model selection.",
            "host_overlap_diagnostic": host_overlap_summary,
        },
        "models": [
            {"name": name, "configuration": configuration}
            for name, _, configuration in factories
        ],
        "selected_model": {
            "name": selected_model_name,
            "selection_metrics": {
                key: float(selected_row[key])
                for key in [
                    "f1_class_0",
                    "recall_class_0",
                    "precision_class_0",
                    "auc_class_0",
                    "accuracy",
                ]
            },
            "final_test_metrics": {
                key: float(selected_final[key])
                for key in [
                    "f1_class_0",
                    "recall_class_0",
                    "precision_class_0",
                    "auc_class_0",
                    "accuracy",
                ]
            },
        },
        "final_test_protocol": "The final temporal test was evaluated once, after CV selection, for Constant and the selected non-constant model only.",
        "official_results_source": {
            "source": "M3_run_orange.py",
            "seed": seed,
            "gui_workflow_role": "M3_Orange_Q2.ows illustrates the process. Orange GUI Test & Score uses seed 42 and GUI Random Forest replicable training uses seed 0, so GUI values are not claimed to be numerically identical to the official script outputs.",
        },
        "quality_checks": {
            "compound_record_key_unique": True,
            "train_test_temporal_split": True,
            "no_random_reshuffle_of_train_and_test": True,
            "test_not_used_for_selection": True,
            "preprocessing_fit_on_train_scope_only": True,
        },
        "runtime": {
            "orange": Orange.__version__,
            "scikit_learn": sklearn.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with (output_dir / "M3_selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, default=as_native)
    write_environment_file(output_dir / "M3_environment.txt", seed)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    default_input = project_dir / "outputs" / "DP3" / "T2_clean.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=project_dir / "outputs" / "M3")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.input, arguments.output, arguments.seed)
