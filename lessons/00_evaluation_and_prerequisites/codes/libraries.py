"""Lesson 00: evaluation using scikit-learn."""


from native import predict_classes
from pandas import read_sql
from numpy import dtype
import random

import numpy as np
import sklearn

from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. DISPLAY AND METRIC HELPERS
# ============================================================

def show_metrics(metrics):
    for name, value in metrics.items():
        if value is None or np.isnan(value):
            text = "undefined"
        else:
            text = f"{value:.4f}"

        print(f"  {name}: {text}")
 
def classification_metrics(actual, predicted):
    # Matrix layout:
    #
    #                Predicted 0    Predicted 1
    # Actual 0           TN             FP
    # Actual 1           FN             TP

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    counts = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=np.nan
        ),
        "recall / TPR": recall_score(
            actual, predicted, zero_division=np.nan
        ),
        "F1": f1_score(
            actual, predicted, zero_division=np.nan
        ),
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
    }

    return counts, metrics 

def probability_metrics(actual, probabilities):
    return {
        "Brier score": brier_score_loss(
            actual,
            probabilities,
            pos_label=1,
        ),
        "log loss": log_loss(
            actual,
            probabilities,
            labels=[0, 1],
        ),
    }

def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mse = mean_squared_error(actual, predicted)

    sse = mse * len(actual)
    sst = np.sum((actual - actual.mean()) ** 2)

    # Match our native policy: undefined R2 when SST is zero.
    if sst == 0:
        r_squared = None
    else:
        r_squared = r2_score(
            actual,
            predicted,
            force_finite=False,
        )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": root_mean_squared_error(actual, predicted),
        "SSE": sse,
        "SST": sst,
        "R2": r_squared,
    }


# ============================================================
# 2. CLASSIFICATION AND PROBABILITY METRICS
# ============================================================
 
def classification_demo():
    actual = np.array([1, 0, 1, 0, 1, 0])

    probabilities = np.array([
        0.90, 0.80, 0.60, 0.40, 0.30, 0.10
    ])

    predicted = (probabilities >= 0.50).astype(int)

    counts, metrics = classification_metrics(
        actual,
        predicted,
    )

    print("  Actual labels:", actual.tolist())
    print("  Predictions:", predicted.tolist())
    print("  Counts:", counts)

    show_metrics(metrics)

    losses = probability_metrics(
        actual,
        probabilities,
    )

    show_metrics(losses)

    # Compare library results with our known paper/native results.
    np.testing.assert_allclose(
        [
            metrics["accuracy"],
            losses["Brier score"],
            losses["log loss"],
        ],
        [
            4 / 6,
            0.245,
            0.6742971659346116,
        ],
        atol=1e-10,
    )

# ============================================================
# 3. ROC AUC
# ============================================================

def roc_demo():
    # A and B passed; C and D failed.
    actual = np.array([1, 1, 0, 0])
    scores = np.array([0.90, 0.60, 0.80, 0.40])

    fpr, tpr, thresholds = roc_curve(
        actual,
        scores,
        drop_intermediate=False,
    )

    print("  Threshold        FPR        TPR")

    for threshold, x, y in zip(thresholds, fpr, tpr):
        print(
            f"  {threshold:9.2f}"
            f"     {x:.2f}"
            f"       {y:.2f}"
        )
    
    ranking_auc = roc_auc_score(actual, scores)
    curve_area = auc(fpr, tpr)

    print(f"  ROC AUC from scores: {ranking_auc:.4f}")
    print(f"  Area under curve:   {curve_area:.4f}")

    for threshold in (0.70, 0.85):
        predicted = (scores >= threshold).astype(int)

        print(
            f"  Threshold {threshold:.2f} predictions:",
            predicted.tolist(),
        )

    print(
        "  Pass scores, not thresholded labels, "
        "to roc_auc_score."
    )

# ============================================================
# 4. MAJORITY-CLASS BASELINE
# ============================================================

def majority_baseline_demo():
    training_y = np.array([0] * 80 + [1] * 20)
    test_y = np.array([0] * 7 + [1] * 3)

    # DummyClassifier ignores feature values.
    # It still expects a two-dimensional feature matrix.
    training_X = np.zeros((len(training_y), 1))
    test_X = np.zeros((len(test_y), 1))

    model = DummyClassifier(strategy="most_frequent")
    model.fit(training_X, training_y)

    predicted = model.predict(test_X)

    counts, metrics = classification_metrics(
        test_y,
        predicted,
    )

    print("  Counts:", counts)
    show_metrics(metrics)

# ============================================================
# 5. REGRESSION BASELINE AND METRICS
# ============================================================

def regression_demo():
    training_y = np.array([2.0, 4.0, 9.0])
    test_y = np.array([3.0, 6.0, 9.0])

    training_X = np.zeros((len(training_y), 1))
    test_X = np.zeros((len(test_y), 1))

    model = DummyRegressor(strategy="mean")
    model.fit(training_X, training_y)

    predicted = model.predict(test_X)

    print("  Fitted training mean:", model.constant_[0, 0])
    print("  Predictions:", predicted.tolist())

    metrics = regression_metrics(test_y, predicted)
    show_metrics(metrics)

    np.testing.assert_allclose(
        [
            metrics["MAE"],
            metrics["MSE"],
            metrics["RMSE"],
            metrics["R2"],
        ],
        [
            7 / 3,
            7,
            np.sqrt(7),
            -1 / 6,
        ],
    )

# ============================================================
# 6. MEAN VERSUS MEDIAN
# ============================================================

def mean_median_demo():
    targets = np.array([1.0, 2.0, 9.0])
    X = np.zeros((len(targets), 1))

    for strategy in ("mean", "median"):
        model = DummyRegressor(strategy=strategy)
        model.fit(X, targets)

        predicted = model.predict(X)

        mae = mean_absolute_error(targets, predicted)
        mse = mean_squared_error(targets, predicted)

        print(
            f"  {strategy}={model.constant_[0, 0]}: "
            f"training MAE={mae:.4f}, "
            f"training MSE={mse:.4f}"
        )

# ============================================================
# 7. TRAINING-ONLY IMPUTATION
# ============================================================

def imputation_demo():
    training_X = np.array([
        [2.0],
        [4.0],
        [np.nan],
    ])

    validation_X = np.array([
        [100.0],
        [np.nan],
    ])

    imputer = SimpleImputer(strategy="mean")

    # fit() learns the replacement value from training only.
    imputer.fit(training_X)

    # transform() applies the already learned value.
    completed_training = imputer.transform(training_X)
    completed_validation = imputer.transform(validation_X)

    print("  Learned feature mean:", imputer.statistics_[0])
    print("  Training:", completed_training.ravel().tolist())
    print("  Validation:", completed_validation.ravel().tolist())

# ============================================================
# 8. FIVE-FOLD CROSS-VALIDATION 
# ============================================================

def cross_validation_demo():
    development_y = np.arange(1.0, 11.0)
    test_y = np.array([3.0, 6.0, 9.0])

    # Independent toy feature values.
    development_X = np.array([
        np.nan, 14.0, 11.0, 18.0, 13.0,
        16.0, 12.0, 19.0, 15.0, 17.0,
    ]).reshape(-1, 1)

    test_X = np.array([
        [20.0],
        [21.0],
        [22.0],
    ])

    # Reproduce the exact fold memberships from native.py.
    # Matching only a seed across libraries is insufficient.
    indices = list(range(len(development_y)))
    random.Random(42).shuffle(indices)

    folds = [
        indices[i::5]
        for i in range(5)
    ]

    splits = []

    for validation_ids in folds:
        validation_set = set(validation_ids)

        training_ids = [
            index
            for index in indices
            if index not in validation_set
        ]

        splits.append((training_ids, validation_ids))

    # Each CV round refits both steps on that round's training data.
    # DummyRegressor ignores X; the pipeline also demonstrates
    # correct fitting of preprocessing inside cross-validation.
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        DummyRegressor(strategy="mean"),
    )
    result = cross_validate(
        pipeline,
        development_X,
        development_y,
        cv=splits,
        scoring="neg_mean_squared_error",
        return_estimator=True, # =True	Saves the fitted pipeline from each fold in result["estimator"]. With five folds, you get five fitted pipelines.
        n_jobs=1, # Runs the folds one at a time. It still runs all five folds.
        error_score="raise", # Stops the call and shows the Python error if fitting or scoring a fold fails.
    )

    # sklearn scoring follows "higher is better".
    # Negate this score to recover ordinary positive MSE.
    #
    # "test_score" means the CV validation folds here.
    # It does NOT mean our separate final test set.
    validation_mse = -result["test_score"]

    for i, (model, mse) in enumerate(
        zip(result["estimator"], validation_mse)
    ):
        training_ids, validation_ids = splits[i]

        target_mean = (
            model.named_steps["dummyregressor"].constant_[0, 0]
        )

        feature_mean = (
            model.named_steps["simpleimputer"].statistics_[0]
        )

        print(
            f"  Round {i + 1}: "
            f"train={len(training_ids)}, "
            f"validation={len(validation_ids)}, "
            f"target mean={target_mean:.3f}, "
            f"validation MSE={mse:.3f}, "
            f"feature imputation mean={feature_mean:.3f}"
        )

    print(
        f"  Mean validation MSE: "
        f"{validation_mse.mean():.4f}"
    )

    # The mean-baseline recipe was fixed in advance.
    # Final test outcomes do not choose or change it.
    pipeline.fit(development_X, development_y)

    final_predictions = pipeline.predict(test_X)

    print("  Final held-out test metrics:")

    show_metrics(
        regression_metrics(test_y, final_predictions)
    )

 
# ============================================================
# 9. REAL DATA: IRIS TRAIN / VALIDATION / TEST WORKFLOW
# ============================================================

def iris_demo():
    iris = load_iris()

    # Binary task:
    # 0 = setosa
    # 1 = versicolor
    #
    # Exclude class 2, virginica, for this binary lesson.
    selected = iris.target < 2

    X = iris.data[selected]
    y = iris.target[selected]

    # Split 100 observations into 60 training and 40 remaining.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=42,
    )

    # Split the remaining observations into:
    # 20 validation and 20 test.
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=0.50,
        stratify=y_rest,
        random_state=43,
    )

    print("  Dataset: Iris")
    print("  Class 0 = setosa; class 1 = versicolor")

    print(
        f"  Rows: train={len(y_train)}, "
        f"validation={len(y_validation)}, "
        f"test={len(y_test)}"
    )

    candidates = []

    # Compare three settings of a simple classifier.
    # C controls regularization in logistic regression.
    for C in (0.1, 1.0, 10.0):
        pipeline = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=42,
            ),
        )

        # Imputation, scaling, and classifier fitting
        # all use training data only.
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        validation_loss = log_loss(
            y_validation,
            probabilities,
            labels=[0, 1],
        )

        print(
            f"  C={C}: "
            f"validation log loss={validation_loss:.4f}"
        )

        candidates.append(
            (validation_loss, C, pipeline)
        )

    # Choose the candidate with lowest validation log loss.
    _, chosen_C, chosen_pipeline = min(
        candidates,
        key=lambda candidate: candidate[0],
    )

    print("  Frozen choice of C:", chosen_C)

    # Choices are now frozen.
    # Refit a fresh copy on training + validation data.
    development_X = np.vstack([
        X_train,
        X_validation,
    ])

    development_y = np.concatenate([
        y_train,
        y_validation,
    ])

    final_model = clone(chosen_pipeline)

    final_model.fit(
        development_X,
        development_y,
    )

    # Fit the baseline on the same development observations.
    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(
        development_X,
        development_y,
    )

    # Access the test set after freezing the workflow.
    probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    predicted = (probabilities >= 0.50).astype(int)

    print("  Fixed threshold: 0.50")
    print("  Final model test metrics:")

    _, metrics = classification_metrics(
        y_test,
        predicted,
    )

    show_metrics(metrics)

    show_metrics(
        probability_metrics(y_test, probabilities)
    )

    print(
        f"  Test ROC AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    baseline_predictions = baseline.predict(X_test)

    print(
        "  Majority baseline test accuracy:",
        f"{accuracy_score(y_test, baseline_predictions):.4f}",
    )

    print(
        "  These results describe a small, easy "
        "two-class subset, not the complete Iris dataset."
    )


# ============================================================
# 10. RUN ALL EXAMPLES
# ============================================================

def main():
    print("scikit-learn:", sklearn.__version__)
    print("NumPy:", np.__version__)

    examples = [
        (
            "CLASSIFICATION AND PROBABILITY METRICS",
            classification_demo,
        ),
        (
            "ROC AUC: FOUR STUDENTS",
            roc_demo,
        ),
        (
            "MAJORITY-CLASS BASELINE",
            majority_baseline_demo,
        ),
        (
            "REGRESSION METRICS",
            regression_demo,
        ),
        (
            "MEAN VERSUS MEDIAN",
            mean_median_demo,
        ),
        (
            "TRAINING-ONLY IMPUTATION",
            imputation_demo,
        ),
        (
            "FIVE-FOLD CROSS-VALIDATION",
            cross_validation_demo,
        ),
        (
            "IRIS: COMPLETE TRAIN/VALIDATION/TEST WORKFLOW",
            iris_demo,
        ),
    ]

    for number, (title, example) in enumerate(
        examples,
        start=1,
    ):
        print(f"\n{number}. {title}")
        example()


if __name__ == "__main__":
    main()