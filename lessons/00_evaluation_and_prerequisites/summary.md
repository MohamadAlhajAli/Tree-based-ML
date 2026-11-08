# Lesson 00 — Evaluation Summary

## Prediction tasks

- Features X are inputs.
- Target y is the correct answer.
- Prediction ŷ is the model's answer.
- Classification predicts categories.
- Regression predicts numerical quantities.
- Class labels, ranking scores, and probabilities are different outputs.

Example: house size is a feature; house price is a regression target.

## Data roles

| Partition | Job |
| --- | --- |
| Training | Fit preprocessing and model |
| Validation | Choose settings and thresholds |
| Test | Evaluate the final frozen workflow |

Choose splits that match prediction conditions:

- Independent rows: random splitting may work.
- Imbalanced classes: consider stratification.
- New people or devices: split by group.
- Future events: split chronologically.

## Classification

With threshold t, predict class 1 when p ≥ t.

| Count | Actual | Predicted |
| --- | ---: | ---: |
| TP | 1 | 1 |
| TN | 0 | 0 |
| FP | 0 | 1 |
| FN | 1 | 0 |

| Metric | Formula |
| --- | --- |
| Accuracy | (TP + TN) / n |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2TP / (2TP + FP + FN) |

Example: TP = 3, FP = 1, FN = 2, TN = 4.

- Accuracy = 7/10 = 70%
- Precision = 3/4 = 75%
- Recall = 3/5 = 60%
- F1 = 6/9 ≈ 66.67%

State a convention when a denominator is zero.

On 100 observations with one positive, always predicting negative gives 99% accuracy but 0% positive recall.

## Probability and ranking metrics

- Binary Brier score = average of (p − y)².
- Binary log loss:
  - Actual 1: −ln(p)
  - Actual 0: −ln(1 − p)
- Lower probability loss is better.
- Confident wrong predictions can have large log loss.
- Changing only a threshold leaves probability losses unchanged.

Example: actual class 1, p = 0.90.

- Brier contribution = (0.90 − 1)² = 0.01
- Log-loss contribution = −ln(0.90) ≈ 0.1054

## ROC curves and AUC

At each threshold:

- Predict class 1 when its score is greater than or equal to the threshold.
- TPR = TP / (TP + FN), which is recall.
- FPR = FP / (FP + TN) = 1 − specificity.
- Plot the point (FPR, TPR): horizontal axis FPR, vertical axis TPR.

Vary the threshold to trace the ROC curve. The ideal corner is (0, 1).

**AUC is the area under this curve.** It also measures positive-negative pair ordering: award 1 for a higher positive score, 0.5 for a tie, and 0 for a lower positive score, then average over all pairs.

Example: positive scores [0.90, 0.60] and negative scores [0.80, 0.40].

- 0.90 beats both negatives: 2 points.
- 0.60 beats only 0.40: 1 point.
- AUC = 3 / 4 = **0.75**.

AUC 0.75 does not mean 75% accuracy or calibrated probabilities. Changing only the decision threshold leaves the full ROC curve, AUC, Brier score, and log loss unchanged. Binary ROC AUC requires both classes. Choose the serving threshold using validation data and mistake costs.


## Regression

| Metric | Formula |
| --- | --- |
| MAE | Average absolute error |
| MSE | Average squared error |
| RMSE | √MSE |
| R² | 1 − SSE/SST |

SSE = Σ(y − ŷ)²  
SST = Σ(y − evaluation mean)²

For targets [2, 3, 4, 6, 7, 8] and predictions all equal to 5:

- Absolute errors sum to 12.
- Squared errors sum to 28.
- MAE = 12/6 = 2.
- MSE = 28/6 ≈ 4.6667.
- RMSE ≈ 2.1602.
- Evaluation mean = 5, so SST = 28 and R² = 0.

R² can be negative. If SSE = 34 and SST = 28:

R² = 1 − 34/28 ≈ −0.2143.

If SST = 0, state the undefined-score convention.

## Baselines

Learn baseline values from training data:

- Majority class: classification baseline.
- Mean target: constant predictor minimizing squared error.
- Median target: constant predictor minimizing absolute error.

For training targets [1, 2, 9]:

- Mean = 4.
- Median = 2.

## Leakage and cross-validation

Leakage includes future information and preprocessing fitted using held-out observations.

If training values are [10, 20], their mean is 15. Use 15 for mean imputation; do not include validation values when calculating it.

In cross-validation, refit preprocessing and the model within each round's training folds.

Use validation results to choose the workflow. Keep the final test set outside those choices.

A model with 100% training accuracy and 65% validation accuracy may be overfitting. A simpler model with 85% training accuracy and 80% validation accuracy generalizes better on that validation split.