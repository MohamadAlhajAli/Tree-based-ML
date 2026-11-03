# Lesson 00 — Evaluating Machine Learning Models

## 1. Features, targets, and predictions

A supervised learning dataset contains:

- **Features (X):** information used to make a prediction.
- **Target (y):** the correct answer.
- **Prediction (ŷ):** the model’s answer.

Example:

| House size | Bedrooms | Actual price |
| ---: | ---: | ---: |
| 100 m² | 2 | $150,000 |
| 150 m² | 3 | $220,000 |
| 200 m² | 4 | $300,000 |

The features are size and bedrooms. The target is price.

### Classification versus regression

| Task | Predicts | Example |
| --- | --- | --- |
| Classification | A category | Spam or not spam |
| Regression | A numerical quantity | House price |

Encoding spam as 1 and not spam as 0 still makes this classification. The numbers represent categories.

## 2. Training, validation, and test data

These datasets have different jobs:

| Dataset | Purpose |
| --- | --- |
| Training | Fit the model and learned preprocessing |
| Validation | Choose model settings and decision thresholds |
| Test | Evaluate the final chosen model |

### Numerical example

You have 100 observations:

- 60 training observations.
- 20 validation observations.
- 20 test observations.

You compare two trees:

| Model | Training accuracy | Validation accuracy |
| --- | ---: | ---: |
| Shallow tree | 85% | 80% |
| Deep tree | 100% | 65% |

Choose the shallow tree using validation results.

The deep tree fits training examples perfectly but performs worse on unseen examples. This is evidence of **overfitting**.

After choosing the model, evaluate it on the test set. Do not keep changing the model after inspecting test results.

### Choose a realistic split

- **Independent observations:** a random split may work.
- **Imbalanced classification:** stratification approximately preserves class proportions.
- **Several rows per person:** keep each person entirely within one partition when predicting for new people.
- **Future predictions:** train on earlier observations and evaluate on later ones.

Example: if each patient has five images, scattering their images across training and test can make evaluation too easy. For prediction on new patients, split by patient.

## 3. Classification probabilities and thresholds

A classifier may produce a probability:

P(class 1) = 0.80

A threshold converts that probability into a class label.

With threshold 0.50:

- If p ≥ 0.50, predict class 1.
- If p < 0.50, predict class 0.

Example:

| Probability | Threshold | Predicted class |
| ---: | ---: | ---: |
| 0.80 | 0.50 | 1 |
| 0.60 | 0.50 | 1 |
| 0.60 | 0.70 | 0 |

Changing the threshold changes the decision, even when the probability stays the same.

Choose a threshold using validation data, according to the costs of mistakes.

## 4. Confusion matrix

Let class 1 be the positive class.

| Term | Meaning |
| --- | --- |
| True positive (TP) | Actual 1, predicted 1 |
| True negative (TN) | Actual 0, predicted 0 |
| False positive (FP) | Actual 0, predicted 1 |
| False negative (FN) | Actual 1, predicted 0 |

### Worked example

Use threshold 0.50:

| Observation | Actual class | Probability | Predicted class | Result |
| --- | ---: | ---: | ---: | --- |
| A | 1 | 0.90 | 1 | TP |
| B | 0 | 0.80 | 1 | FP |
| C | 1 | 0.60 | 1 | TP |
| D | 0 | 0.40 | 0 | TN |
| E | 1 | 0.30 | 0 | FN |
| F | 0 | 0.10 | 0 | TN |

Counts:

- TP = 2
- TN = 2
- FP = 1
- FN = 1

Confusion matrix:

| Actual / Predicted | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | TN = 2 | FP = 1 |
| Actual 1 | FN = 1 | TP = 2 |

Always label the axes and state which class is positive.

## 5. Accuracy, precision, recall, and F1

### Accuracy

What fraction of all predictions were correct?

Accuracy = (TP + TN) / Total

For the six observations:

Accuracy = (2 + 2) / 6 = 66.67%

### Precision

Among positive predictions, how many were correct?

Precision = TP / (TP + FP)

Precision = 2 / (2 + 1) = 66.67%

### Recall

Among actual positives, how many did we find?

Recall = TP / (TP + FN)

Recall = 2 / (2 + 1) = 66.67%

### F1 score

F1 combines precision and recall using their harmonic mean:

F1 = 2TP / (2TP + FP + FN)

F1 = 4 / (4 + 1 + 1) = 66.67%

Precision and recall happen to be equal in this example. They are usually different.

For TP = 3, FP = 1, and FN = 2:

- Precision = 3/4 = 75%
- Recall = 3/5 = 60%
- F1 = 6/9 = 66.67%

### Why accuracy can mislead

Suppose 100 observations contain:

- 99 negatives.
- 1 positive.

A model predicts negative for everyone:

- Accuracy = 99/100 = 99%
- Recall = 0/1 = 0%

The model misses every positive despite its high accuracy.

If a metric's denominator is zero, report it as undefined or state the convention used by your software.

## 6. Evaluating probabilities

Class metrics evaluate decisions after thresholding.

Probability metrics evaluate the probability values themselves.

### Brier score

For binary classification:

Brier = average of (p − y)²

Using the six observations:

| Observation | Actual y | Probability p | (p − y)² |
| --- | ---: | ---: | ---: |
| A | 1 | 0.90 | 0.01 |
| B | 0 | 0.80 | 0.64 |
| C | 1 | 0.60 | 0.16 |
| D | 0 | 0.40 | 0.16 |
| E | 1 | 0.30 | 0.49 |
| F | 0 | 0.10 | 0.01 |

Brier = 1.47 / 6 = 0.245

Lower is better. A perfect binary probability prediction has Brier score 0.

### Log loss

Use natural logarithms.

For one observation:

- If y = 1: loss = −ln(p)
- If y = 0: loss = −ln(1 − p)

Example when the actual class is 1:

| Predicted probability | Log loss |
| ---: | ---: |
| 0.90 | 0.1054 |
| 0.50 | 0.6931 |
| 0.10 | 2.3026 |

A confident wrong prediction receives a large penalty.

For the six observations:

Log loss = (0.1054 + 1.6094 + 0.5108 + 0.5108 + 1.2040 + 0.1054) / 6

Log loss ≈ 0.6743

Lower is better. Assigning zero probability to the actual outcome gives infinite theoretical loss; software typically clips probabilities to avoid infinities.

Changing only the classification threshold does not change Brier score or log loss.

## 7. Evaluating ranking with ROC AUC

ROC AUC measures whether positives receive higher scores than negatives.

For each positive-negative pair:

- Positive score higher: award 1.
- Scores tied: award 0.5.
- Positive score lower: award 0.

AUC = Total awarded points / Number of pairs

Using the six observations:

Positive scores: 0.90, 0.60, 0.30  
Negative scores: 0.80, 0.40, 0.10

- Positive 0.90 beats 3 negatives.
- Positive 0.60 beats 2 negatives.
- Positive 0.30 beats 1 negative.

AUC = (3 + 2 + 1) / (3 × 3) = 6/9 ≈ 0.6667

Interpretation:

- AUC = 1: perfect separation on these observations.
- AUC = 0.5: chance-level pair ordering.
- AUC < 0.5: worse than chance-level ordering.

AUC requires both positive and negative observations.

High AUC does not mean the probability values are accurate. Scores can rank examples correctly while being overconfident.

## 8. Baselines

A baseline is a simple predictor used for comparison.

### Classification baseline

Training data contain:

- 80 examples of class 0.
- 20 examples of class 1.

A majority-class baseline always predicts class 0.

Its training accuracy is 80%. Evaluate this frozen rule on held-out data to obtain its validation or test accuracy.

### Regression baselines

Training targets:

[1, 2, 9]

Two useful constant predictions are:

- Mean = (1 + 2 + 9) / 3 = 4
- Median = 2

The mean minimizes squared error among constant predictions.

The median minimizes absolute error among constant predictions.

| Constant prediction | Sum of absolute errors | Sum of squared errors |
| ---: | ---: | ---: |
| 4 | 3 + 2 + 5 = 10 | 9 + 4 + 25 = 38 |
| 2 | 1 + 0 + 7 = 8 | 1 + 0 + 49 = 50 |

Learn the baseline from training targets, not test targets.

## 9. Regression metrics: MAE, MSE, and RMSE

Suppose the training target mean is 5. Our baseline therefore predicts 5 for every new observation.

Held-out results:

| Actual y | Prediction ŷ | Error y − ŷ | Absolute error | Squared error |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 5 | −3 | 3 | 9 |
| 3 | 5 | −2 | 2 | 4 |
| 4 | 5 | −1 | 1 | 1 |
| 6 | 5 | 1 | 1 | 1 |
| 7 | 5 | 2 | 2 | 4 |
| 8 | 5 | 3 | 3 | 9 |
| **Total** | | | **12** | **28** |

### Mean absolute error

MAE = Sum of absolute errors / n

MAE = 12/6 = 2

### Mean squared error

MSE = Sum of squared errors / n

MSE = 28/6 ≈ 4.6667

### Root mean squared error

RMSE = √MSE

RMSE = √4.6667 ≈ 2.1602

Lower is better for all three.

If the target is measured in minutes:

- MAE and RMSE are measured in minutes.
- MSE is measured in minutes squared.

Squaring gives large errors more influence.

## 10. R²

R² compares prediction errors with variation around the evaluation targets' mean.

R² = 1 − SSE/SST

Where:

- SSE = Σ(y − ŷ)²
- SST = Σ(y − evaluation mean)²

For targets [2, 3, 4, 6, 7, 8]:

Evaluation mean = 5  
SST = 9 + 4 + 1 + 1 + 4 + 9 = 28

If every prediction is 5:

SSE = 28

R² = 1 − 28/28 = 0

If every prediction is 6:

SSE = 16 + 9 + 4 + 0 + 1 + 4 = 34

R² = 1 − 34/28 ≈ −0.2143

Interpretation when SST > 0:

- R² = 1: perfect predictions.
- R² = 0: SSE equals the evaluation-mean reference error.
- R² < 0: SSE exceeds that reference error.

The evaluation mean belongs to the score calculation. Do not use it to fit the predictor.

If all evaluation targets are identical, SST = 0. The formula is undefined; state the software convention used.

## 11. Data leakage

Leakage occurs when model development uses information that should be unavailable.

### Example 1: future information

Goal: predict cancellation before it happens.

Feature: refund issued after cancellation.

This feature reveals information from after the prediction time. Exclude it.

### Example 2: preprocessing before splitting

Training feature values:

[10, 20]

Validation feature value:

[90]

Training mean = (10 + 20)/2 = 15

Combined mean = (10 + 20 + 90)/3 = 40

If missing training values are filled with 40, validation information has influenced training.

Correct procedure:

1. Split the data.
2. Learn the imputation value from training data: 15.
3. Use that same value to fill missing values in training, validation, and test data.

Apply this rule to learned scaling, category encoding, feature selection, and other fitted preprocessing.

## 12. Cross-validation

Cross-validation repeats training and validation on different development partitions.

Example: divide development data into five folds.

| Round | Validation fold | Training folds |
| --- | --- | --- |
| 1 | 1 | 2, 3, 4, 5 |
| 2 | 2 | 1, 3, 4, 5 |
| 3 | 3 | 1, 2, 4, 5 |
| 4 | 4 | 1, 2, 3, 5 |
| 5 | 5 | 1, 2, 3, 4 |

Validation accuracies:

[0.80, 0.85, 0.75, 0.90, 0.80]

Mean accuracy = 4.10/5 = 0.82 = 82%

For every round:

1. Fit preprocessing on that round's training folds.
2. Fit the model on those training folds.
3. Evaluate on that round's validation fold.

Keep the final test set outside cross-validation. Use group-aware or time-aware folds when the prediction task requires them.

## 13. Practice

Try these before reading the answers.

1. In the six-observation classification example, raise the threshold to 0.70. Find TP, FP, FN, TN, precision, and recall.
2. Does changing only the threshold change Brier score?
3. A model predicts 0 for all 100 observations, including one actual positive. Find accuracy and recall.
4. For targets [2, 4] and predictions [3, 3], calculate MAE, MSE, and RMSE.
5. Each person contributes ten rows. You must predict for new people. How should you split?

### Answers

1. Only A and B are predicted positive:
   - TP = 1, FP = 1, FN = 2, TN = 2
   - Precision = 1/2 = 50%
   - Recall = 1/3 ≈ 33.33%
2. No. The probability values are unchanged.
3. Accuracy = 99%; recall = 0%.
4. Absolute errors are [1, 1], squared errors are [1, 1]:
   - MAE = 1
   - MSE = 1
   - RMSE = 1
5. Split by person. Keep every person's rows entirely within one partition.