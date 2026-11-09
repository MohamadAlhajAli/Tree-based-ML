"""Lesson 00: evaluation using native Python."""

import numbers
import math
import random


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def check_pairs(actual, predicted):
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError(
            "Inputs must be nonempty and have equal lengths."
        )

def mean(values):
    if not values:
        raise ValueError("Cannot calculate an empty list's mean.")

    return sum(values) / len(values)

def median(values):
    if not values:
        raise ValueError("Cannot calculate an empty list's median.")

    ordered = sorted(values)
    middle = len(ordered) // 2

    if len(ordered) % 2 == 1:
        return ordered[middle]

    return (ordered[middle - 1] + ordered[middle]) / 2

def safe_divide(numerator, denominator):
    # None means the metric is undefined.
    if denominator == 0:
        return None

    return numerator / denominator

def show_metrics(metrics):
    for name, value in metrics.items():
        if value is None:
            text = "undefined"
        else:
            text = f"{value:.4f}"

        print(f"  {name}: {text}")
    

# ============================================================
# 2. CLASSIFICATION
# ============================================================

def predict_classes(probabilities, threshold=0.5):
    """Predict 1 when probability is >= the threshold"""

    return [
        1 if probability >= threshold else 0 
        for probability in probabilities
    ]

def confusion_counts(actual, predicted): 
    check_pairs(actual, predicted) 

    if any(value not in (0,1) for value in actual + predicted): 
        raise ValueError("Class labels must be 0 or 1.") 
    
    counts = {
        "TP": 0,
        "TN": 0, 
        "FP": 0, 
        "FN": 0, 
    }

    for actual_value, predicted_value in zip(actual, predicted):
        if actual_value == 1 and predicted_value == 1: 
            counts['TP'] += 1 
        
        elif actual_value == 0 and predicted_value == 0:
            counts["TN"] += 1

        elif actual_value == 0 and predicted_value == 1:
            counts["FP"] += 1

        else:
            counts["FN"] += 1
    
    return counts 

def classification_metrics(actual, predicted):
    counts = confusion_counts(actual, predicted)

    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]


    return {
        "accuracy": (tp + tn) / len(actual),
        "precision": safe_divide(tp, tp + fp),
        "recall / TPR": safe_divide(tp, tp + fn),
        "F1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "FPR": safe_divide(fp, fp + tn),
    }

# ============================================================
# 3. PROBABILITY METRICS
# ============================================================

def probability_metrics(actual, probabilities):
    check_pairs(actual, probabilities)

    if any(value not in (0, 1) for value in actual):
        raise ValueError("Actual labels must be 0 or 1.")

    for probability in probabilities:
        if not math.isfinite(probability):
            raise ValueError("Probabilities must be finite.")

        if not 0 <= probability <= 1:
            raise ValueError("Probabilities must be between 0 and 1.")
    
    squared_errors = []
    log_losses = []

    for actual_value, probability in zip(actual, probabilities):
        squared_errors.append(
            (probability - actual_value) ** 2
        )

        if actual_value == 1:
            probability_of_actual_outcome = probability
        else:
            probability_of_actual_outcome = 1 - probability

        if probability_of_actual_outcome == 0:
            loss = math.inf
        else:
            loss = -math.log(probability_of_actual_outcome)

        log_losses.append(loss)
    
    return {
        "Brier score": mean(squared_errors),
        "log loss": mean(log_losses),
    }


# ============================================================
# 4. ROC AUC USING POSITIVE-NEGATIVE PAIRS
# ============================================================

def auc_from_pairs(actual, scores):
    check_pairs(actual, scores)

    if any(value not in (0, 1) for value in actual):
        raise ValueError("Actual labels must be 0 or 1.")

    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Scores must be finite.")

    positive_scores = [
        score
        for actual_value, score in zip(actual, scores)
        if actual_value == 1
    ]

    negative_scores = [
        score
        for actual_value, score in zip(actual, scores)
        if actual_value == 0
    ]

    if not positive_scores or not negative_scores:
        raise ValueError("ROC AUC requires both classes.")

    points = 0.0

    for positive_score in positive_scores:
        for negative_score in negative_scores:
            if positive_score > negative_score:
                points += 1

            elif positive_score == negative_score:
                points += 0.5

            # A lower positive score earns zero points.

    number_of_pairs = (
        len(positive_scores) * len(negative_scores)
    )

    auc = points / number_of_pairs

    return points, number_of_pairs, auc


# ============================================================
# 5. ROC POINTS AND AREA UNDER THE CURVE
# ============================================================

def roc_points(actual, scores):
    # Validate the inputs and require both classes.
    auc_from_pairs(actual, scores)

    # Infinity is above every finite score.
    # It produces the starting point (FPR=0, TPR=0).
    thresholds = [
        math.inf
    ] + sorted(set(scores), reverse=True)

    rows = []

    for threshold in thresholds:
        predicted = predict_classes(scores, threshold)
        metrics = classification_metrics(actual, predicted)

        fpr = metrics["FPR"]
        tpr = metrics["recall / TPR"]

        rows.append((threshold, fpr, tpr))

    return rows       

def auc_from_curve(rows):
    """Calculate area using trapezoids between successive points."""

    area = 0.0

    for current_point, next_point in zip(rows, rows[1:]):
        _, x0, y0 = current_point
        _, x1, y1 = next_point

        width = x1 - x0
        average_height = (y0 + y1) / 2

        area += width * average_height

    return area


# ============================================================
# 6. REGRESSION METRICS
# ============================================================

def regression_metrics(actual, predicted):
    check_pairs(actual, predicted)

    errors = [
        actual_value - predicted_value
        for actual_value, predicted_value in zip(actual, predicted)
    ]

    absolute_errors = [
        abs(error)
        for error in errors
    ]

    squared_errors = [
        error ** 2
        for error in errors
    ]

    mae = mean(absolute_errors)
    mse = mean(squared_errors)
    rmse = math.sqrt(mse)

    sse = sum(squared_errors)

    # This evaluation mean belongs to the score calculation.
    # It is not used to fit the predictor.
    evaluation_mean = mean(actual)

    sst = sum(
        (actual_value - evaluation_mean) ** 2
        for actual_value in actual
    )

    if sst == 0:
        r_squared = None
    else:
        r_squared = 1 - sse / sst

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "SSE": sse,
        "SST": sst,
        "R2": r_squared,
    }
 
# ============================================================
# 7. TRAINING-ONLY MISSING-VALUE IMPUTATION
# ============================================================

def fit_imputation_mean(training_values):
    observed_values = [
        value
        for value in training_values
        if value is not None
    ]

    return mean(observed_values)


def impute(values, training_mean):
    return [
        training_mean if value is None else value
        for value in values
    ]

# ============================================================
# 8. CROSS-VALIDATION WITH A MEAN BASELINE
# ============================================================

def cross_validation_demo():
    # Small synthetic example with independent observations 
    development_targets = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ]

    test_targets = [3, 6, 9] 

    indices = list(range(len(development_targets)))

    # A fixed seed makes this split reproducible.
    random.Random(42).shuffle(indices)

    # Five folds, each containing two observations.
    folds = [
        indices[fold_number::5]
        for fold_number in range(5)
    ]

    validation_scores = []

    for round_number, validation_ids in enumerate(folds, start=1):
        validation_set = set(validation_ids)

        training_ids = [
            index
            for index in indices
            if index not in validation_set
        ]

        training_targets = [
            development_targets[index]
            for index in training_ids
        ]

        validation_targets = [
            development_targets[index]
            for index in validation_ids
        ]

        # Fit a fresh mean baseline using this round's
        # training targets only.
        training_mean = mean(training_targets)

        predictions = [
            training_mean
        ] * len(validation_targets)

        metrics = regression_metrics(
            validation_targets,
            predictions,
        )

        validation_mse = metrics["MSE"]
        validation_scores.append(validation_mse)

        print(
            f"  Round {round_number}: "
            f"train={len(training_targets)}, "
            f"validation={len(validation_targets)}, "
            f"fitted mean={training_mean:.3f}, "
            f"validation MSE={validation_mse:.3f}"
        )

    # The mean-baseline recipe was fixed beforehand.
    # Test results do not select or change the recipe.
    final_mean = mean(development_targets)

    final_predictions = [
        final_mean
    ] * len(test_targets)

    print(
        "  Final mean fitted on all development targets:",
        final_mean,
    )

    print("  Final held-out test metrics:")

    show_metrics(
        regression_metrics(test_targets, final_predictions)
    )


# ============================================================
# 9. RUN THE LESSON EXAMPLES
# ============================================================

def main():
    print("\n1. CLASSIFICATION AND PROBABILITY METRICS")

    actual = [1, 0, 1, 0, 1, 0]
    probabilities = [0.90, 0.80, 0.60, 0.40, 0.30, 0.10]

    predicted = predict_classes(
        probabilities,
        threshold=0.50,
    )

    print("  Actual labels:", actual)
    print("  Predictions:", predicted)
    print("  Counts:", confusion_counts(actual, predicted))

    show_metrics(
        classification_metrics(actual, predicted)
    )

    show_metrics(
        probability_metrics(actual, probabilities)
    )

    print("\n2. ROC AUC: FOUR STUDENTS")

    # A and B passed. C and D failed.
    actual = [1, 1, 0, 0]
    scores = [0.90, 0.60, 0.80, 0.40]

    points, pairs, pair_auc = auc_from_pairs(actual, scores)

    print(
        f"  Ranking points={points}, "
        f"pairs={pairs}, "
        f"AUC={pair_auc:.4f}"
    )

    rows = roc_points(actual, scores)

    print("  Threshold        FPR        TPR")

    for threshold, fpr, tpr in rows:
        print(
            f"  {threshold:9.2f}"
            f"     {fpr:.2f}"
            f"       {tpr:.2f}"
        )

    curve_auc = auc_from_curve(rows)

    print(
        f"  Area under ROC curve: {curve_auc:.4f}"
    )

    # Check the known paper result and agreement between
    # two different ways of calculating AUC.
    assert math.isclose(pair_auc, 0.75)
    assert math.isclose(curve_auc, pair_auc)

    print(
        "  Threshold 0.70 predictions:",
        predict_classes(scores, 0.70),
    )

    print(
        "  Threshold 0.85 predictions:",
        predict_classes(scores, 0.85),
    )

    print(
        "  Both choices leave the full ROC AUC unchanged."
    )

    print("\n3. MAJORITY-CLASS BASELINE")

    training_labels = [0] * 80 + [1] * 20

    majority_class = max(
        (0, 1),
        key=training_labels.count,
    )

    test_labels = [0] * 7 + [1] * 3
    predictions = [majority_class] * len(test_labels)

    print("  Training majority class:", majority_class)

    show_metrics(
        classification_metrics(test_labels, predictions)
    )

    print("\n4. REGRESSION METRICS")

    training_targets = [2, 4, 9]
    training_mean = mean(training_targets)

    test_targets = [3, 6, 9]
    predictions = [training_mean] * len(test_targets)

    print("  Training mean:", training_mean)
    print("  Test actual values:", test_targets)
    print("  Test predictions:", predictions)

    show_metrics(
        regression_metrics(test_targets, predictions)
    )

    print("\n5. MEAN VERSUS MEDIAN")

    targets = [1, 2, 9]

    baselines = [
        ("Mean", mean(targets)),
        ("Median", median(targets)),
    ]

    for name, constant in baselines:
        predictions = [constant] * len(targets)
        metrics = regression_metrics(targets, predictions)

        print(
            f"  {name}={constant}: "
            f"training MAE={metrics['MAE']:.4f}, "
            f"training MSE={metrics['MSE']:.4f}"
        )

    print("\n6. TRAINING-ONLY IMPUTATION")

    training_values = [2, 4, None]
    validation_values = [100, None]

    training_mean = fit_imputation_mean(training_values)

    print(
        "  Mean learned from training only:",
        training_mean,
    )

    print(
        "  Training after imputation:",
        impute(training_values, training_mean),
    )

    print(
        "  Validation after imputation:",
        impute(validation_values, training_mean),
    )

    print("\n7. FIVE-FOLD CROSS-VALIDATION")

    cross_validation_demo()


if __name__ == "__main__":
    main()