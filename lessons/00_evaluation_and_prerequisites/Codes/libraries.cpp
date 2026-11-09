// Lesson 00: evaluation with mlpack (the C++ version of libraries.py).
//
// Build (mlpack 4 is header-only; Armadillo brings in BLAS/LAPACK):
//   g++ -std=c++17 -O2 libraries.cpp -o libraries -larmadillo
// Run from the Codes folder:
//   ./libraries ../Data/iris_binary.csv
//
// mlpack stores one example per COLUMN, so labels and targets are rows.

#include <mlpack/core.hpp>
#include <mlpack/core/cv/metrics/r2_score.hpp>
#include <mlpack/core/cv/metrics/roc_auc_score.hpp>
#include <mlpack/core/data/confusion_matrix.hpp>
#include <mlpack/methods/linear_regression.hpp>
#include <mlpack/methods/logistic_regression.hpp>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using arma::mat;
using arma::rowvec;
using Labels = arma::Row<std::size_t>;

// ============================================================
// 1. DISPLAY AND METRIC HELPERS
// ============================================================

void show(const std::string &name, double value) {
  std::cout << "  " << name << ": ";
  if (std::isnan(value))
    std::cout << "undefined\n";
  else
    std::cout << value << '\n';
}

template <typename Vector>
void show_list(const std::string &name, const Vector &values) {
  std::cout << "  " << name << ":";
  for (const auto value : values)
    std::cout << ' ' << value;
  std::cout << '\n';
}

// mlpack metrics score a model: Metric::Evaluate(model, data, truth) calls
// model.Classify(data, ...) or model.Predict(data, ...). This "model" just
// hands back predictions we already have.
template <typename Vector> struct FixedPredictions {
  Vector values;
  void Classify(const mat &, Vector &output) const { output = values; }
  void Predict(const mat &, Vector &output) const { output = values; }
};

// Score existing predictions with any mlpack metric, for example
// score<mlpack::Accuracy>(actual, predicted).
template <typename Metric, typename Truth, typename Prediction>
double score(const Truth &truth, const Prediction &predicted) {
  FixedPredictions<Prediction> model{predicted};
  const mat placeholder(1, truth.n_elem, arma::fill::zeros); // never read
  return Metric::Evaluate(model, placeholder, truth);
}

Labels threshold(const rowvec &scores, double cutoff) {
  return arma::conv_to<Labels>::from(scores >= cutoff);
}

// Binary labels: predict 1 only when more than half of y is 1.
// Ties give 0, like scikit-learn's "most_frequent" strategy.
std::size_t majority_class(const Labels &y) {
  return 2 * arma::accu(y) > y.n_elem;
}

struct Counts {
  int tp, tn, fp, fn;
};

Counts confusion_counts(const Labels &actual, const Labels &predicted) {
  arma::Mat<int> matrix; // rows = predicted label, columns = actual label
  mlpack::ConfusionMatrix(predicted, actual, matrix,
                          2); // 2 is the number of classes.
  return {matrix(1, 1), matrix(0, 0), matrix(1, 0), matrix(0, 1)};
}

void classification_metrics(const Labels &actual, const Labels &predicted) {
  const auto [tp, tn, fp, fn] = confusion_counts(actual, predicted);
  std::cout << "  Counts: TP=" << tp << " TN=" << tn << " FP=" << fp
            << " FN=" << fn << '\n';

  // With no positive predictions, precision is 0/0 (undefined) and
  // F1 = 2TP / (2TP + FP + FN) = 0 because TP = 0.
  const bool any_positive = tp + fp > 0;
  show("accuracy", score<mlpack::Accuracy>(actual, predicted));
  show("precision", any_positive ? score<mlpack::Precision<mlpack::Binary, 1>>(
                                       actual, predicted)
                                 : arma::datum::nan);
  show("recall / TPR",
       score<mlpack::Recall<mlpack::Binary, 1>>(actual, predicted));
  show("F1", any_positive
                 ? score<mlpack::F1<mlpack::Binary, 1>>(actual, predicted)
                 : 0.0);
  show("FPR", fp + tn > 0 ? double(fp) / (fp + tn) : arma::datum::nan);
}

double log_loss(const Labels &actual, const rowvec &probabilities) {
  const rowvec y = arma::conv_to<rowvec>::from(actual);
  const rowvec p =
      arma::clamp(probabilities, 1e-15,
                  1 - 1e-15); // Clip probabilities to [1e-15, 1 - 1e-15] so
                              // log(p) and log(1 - p) stay finite.
  return -arma::mean(y % arma::log(p) + (1 - y) % arma::log(1 - p));
}

void probability_metrics(const Labels &actual, const rowvec &probabilities) {
  // The Brier score is the MSE between the probabilities and the 0/1 labels.
  const rowvec y = arma::conv_to<rowvec>::from(actual);
  show("Brier score", score<mlpack::MSE>(y, probabilities));
  show("log loss", log_loss(actual, probabilities));
}

void regression_metrics(const rowvec &actual, const rowvec &predicted) {
  const double mse = score<mlpack::MSE>(actual, predicted);
  const double sst = arma::accu(arma::square(actual - arma::mean(actual)));

  show("MAE", arma::mean(arma::abs(actual - predicted)));
  show("MSE", mse);
  show("RMSE", std::sqrt(mse));
  show("SSE", mse * actual.n_elem);
  show("SST", sst);
  // R2 = 1 - SSE / SST is undefined when every target is equal (SST = 0).
  show("R2", sst > 0 ? score<mlpack::R2Score<false>>(actual, predicted)
                     : arma::datum::nan);
}

// ============================================================
// 2. CLASSIFICATION AND PROBABILITY METRICS
// ============================================================

void classification_demo() {
  const Labels actual{1, 0, 1, 0, 1, 0};
  const rowvec probabilities{0.90, 0.80, 0.60, 0.40, 0.30, 0.10};
  const Labels predicted = threshold(probabilities, 0.50);

  show_list("Actual labels", actual);
  show_list("Predictions", predicted);
  classification_metrics(actual, predicted);
  probability_metrics(actual, probabilities);
}

// ============================================================
// 3. ROC AUC
// ============================================================

void roc_demo() {
  // A and B passed; C and D failed.
  const Labels actual{1, 1, 0, 0};
  const rowvec scores{0.90, 0.60, 0.80, 0.40};

  // Every score is a threshold. +inf predicts nobody positive, which starts
  // the curve at (0, 0).
  std::cout << "  Threshold     FPR     TPR\n";
  for (double cutoff : {arma::datum::inf, 0.90, 0.80, 0.60, 0.40}) {
    const auto [tp, tn, fp, fn] =
        confusion_counts(actual, threshold(scores, cutoff));
    const double fpr = double(fp) / (fp + tn);
    const double tpr = double(tp) / (tp + fn);
    std::cout << "  " << std::setw(9) << cutoff << std::setw(8) << fpr
              << std::setw(8) << tpr << '\n';
  }

  show("ROC AUC", mlpack::ROCAUCScore<1>::Evaluate(actual, scores));
  show_list("Threshold 0.70 predictions", threshold(scores, 0.70));
  show_list("Threshold 0.85 predictions", threshold(scores, 0.85));
  std::cout << "  Pass scores, not thresholded labels, to ROCAUCScore.\n";
}

// ============================================================
// 4. MAJORITY-CLASS BASELINE
// ============================================================

void majority_baseline_demo() {
  const Labels training_y = arma::join_rows(Labels(80, arma::fill::zeros),
                                            Labels(20, arma::fill::ones));
  const Labels test_y{0, 0, 0, 0, 0, 0, 0, 1, 1, 1};

  // The baseline ignores the features: it always predicts the majority class
  // of the training labels.
  const Labels predicted(test_y.n_elem,
                         arma::fill::value(majority_class(training_y)));
  classification_metrics(test_y, predicted);
}

// ============================================================
// 5. REGRESSION BASELINE AND METRICS
// ============================================================

void regression_demo() {
  const rowvec training_y{2, 4, 9};
  const rowvec test_y{3, 6, 9};

  // A mean baseline learns one number: the training mean.
  const double mean = arma::mean(training_y);
  const rowvec predicted(test_y.n_elem, arma::fill::value(mean));

  show("Fitted training mean", mean);
  show_list("Predictions", predicted);
  regression_metrics(test_y, predicted);
}

// ============================================================
// 6. MEAN VERSUS MEDIAN
// ============================================================

void mean_median_demo() {
  const rowvec targets{1, 2, 9};

  auto report = [&](const std::string &name, double constant) {
    const rowvec predicted(targets.n_elem, arma::fill::value(constant));
    std::cout << "  " << name << '=' << constant
              << ": training MAE=" << arma::mean(arma::abs(targets - predicted))
              << ", training MSE=" << score<mlpack::MSE>(targets, predicted)
              << '\n';
  };
  report("mean", arma::mean(targets));
  report("median", arma::median(targets));
}

// ============================================================
// 7. TRAINING-ONLY IMPUTATION
// ============================================================

void imputation_demo() {
  const double missing = arma::datum::nan;
  rowvec training{2, 4, missing};
  rowvec validation{100, missing};

  // "fit": learn the mean from the observed TRAINING values only.
  const arma::vec observed = training.elem(arma::find_finite(training));
  const double learned_mean = arma::mean(observed);

  // "transform": fill both sets with that same learned value.
  training.replace(missing, learned_mean);
  validation.replace(missing, learned_mean);

  show("Learned feature mean", learned_mean);
  show_list("Training", training);
  show_list("Validation", validation);
}

// ============================================================
// 8. FIVE-FOLD CROSS-VALIDATION
// ============================================================

void cross_validation_demo() {
  // Without shuffling, mlpack's KFoldCV uses five contiguous folds of two
  // values. The targets 1..10 are listed in the order that reproduces the
  // shuffled folds of libraries.py: {8,7} {4,10} {3,5} {9,1} {6,2}.
  const rowvec development_y{8, 7, 4, 10, 3, 5, 9, 1, 6, 2};
  const rowvec test_y{3, 6, 9};

  // The mean baseline as an mlpack model: a linear regression with one
  // feature that is always 1 and no intercept learns y = b * 1, and least
  // squares picks b = mean(y). Each round refits it on its four training
  // folds.
  const mat always_one(1, development_y.n_elem, arma::fill::ones);
  mlpack::KFoldCV<mlpack::LinearRegression<>, mlpack::MSE> cv(
      5, always_one, development_y, /* shuffle = */ false);
  show("Mean validation MSE",
       cv.Evaluate(/* lambda = */ 0.0, /* intercept = */ false));

  // The recipe was fixed in advance; refit it on all development data.
  const rowvec predicted(test_y.n_elem,
                         arma::fill::value(arma::mean(development_y)));
  std::cout << "  Final held-out test metrics:\n";
  regression_metrics(test_y, predicted);
}

// ============================================================
// 9. REAL DATA: IRIS TRAIN / VALIDATION / TEST WORKFLOW
// ============================================================

mat standardize(mat x, const arma::vec &mean, const arma::vec &sd) {
  x.each_col() -= mean;
  x.each_col() /= sd;
  return x;
}

// Scale with the fitting data's statistics, fit logistic regression, and
// return P(class 1) for every column of score_x.
rowvec fit_and_predict(const mat &fit_x, const Labels &fit_y,
                       const mat &score_x, double lambda) {
  const arma::vec mean = arma::mean(fit_x, 1);
  /*
    Compute one mean per row, so you get one mean per feature.
    In this mlpack code, each column is one record, and each row is one feature.
  */
  arma::vec sd = arma::stddev(fit_x, 1, 1);
  // divide by N, like StandardScaler
  /*
  _x, 1, 1)	The first 1 says to divide by (N) when calculating the variance
  before taking the square root. The second 1 says to calculate it per row.
  */
  sd.replace(0, 1); // leave constant features as-is

  mlpack::LogisticRegression<> model(standardize(fit_x, mean, sd), fit_y,
                                     lambda);
  Labels labels;
  mat probabilities; // row 0 = P(class 0), row 1 = P(class 1)
  model.Classify(standardize(score_x, mean, sd), labels, probabilities);
  return probabilities.row(1);
}

void iris_demo(const std::string &path) {
  mat data; // 100 rows: 4 features, then the label
  if (!data.load(path, arma::csv_ascii) || data.n_cols != 5)
    throw std::runtime_error("Could not read the 5-column CSV file " + path);

  const mat x = data.cols(0, 3).t(); // one example per column
  const Labels y = arma::conv_to<Labels>::from(data.col(4).t());

  // Stratified 60 / 20 / 20 split keeps the class balance in every part.
  mlpack::RandomSeed(42);
  mat train, rest, valid, test;
  Labels train_y, rest_y, valid_y, test_y;
  mlpack::StratifiedSplit(x, y, train, rest, train_y, rest_y, 0.40);
  mlpack::StratifiedSplit(rest, rest_y, valid, test, valid_y, test_y, 0.50);

  std::cout << "  Dataset: Iris\n"
            << "  Class 0 = setosa; class 1 = versicolor\n"
            << "  Rows: train=" << train.n_cols
            << ", validation=" << valid.n_cols << ", test=" << test.n_cols
            << '\n';

  // lambda is an L2 penalty: larger lambda = stronger regularization
  // (the opposite direction of scikit-learn's C).
  double best_lambda = 0, best_loss = arma::datum::inf;
  for (double lambda : {0.1, 1.0, 10.0}) {
    const double loss =
        log_loss(valid_y, fit_and_predict(train, train_y, valid, lambda));
    std::cout << "  lambda=" << lambda << ": validation log loss=" << loss
              << '\n';
    if (loss < best_loss) {
      best_loss = loss;
      best_lambda = lambda;
    }
  }
  std::cout << "  Frozen choice of lambda: " << best_lambda << '\n';

  // Choices are frozen. Refit on training + validation, then use the test set.
  const mat development_x = arma::join_rows(train, valid);
  const Labels development_y = arma::join_rows(train_y, valid_y);
  const rowvec probabilities =
      fit_and_predict(development_x, development_y, test, best_lambda);

  std::cout << "  Fixed threshold: 0.50\n  Final model test metrics:\n";
  classification_metrics(test_y, threshold(probabilities, 0.50));
  probability_metrics(test_y, probabilities);
  show("Test ROC AUC", mlpack::ROCAUCScore<1>::Evaluate(test_y, probabilities));

  // The baseline is fitted on the same development labels.
  const Labels baseline(test_y.n_elem,
                        arma::fill::value(majority_class(development_y)));
  show("Majority baseline test accuracy",
       score<mlpack::Accuracy>(test_y, baseline));

  std::cout << "  These results describe a small, easy two-class subset,"
               " not the complete Iris dataset.\n";
}

// ============================================================
// 10. RUN ALL EXAMPLES
// ============================================================

int main(int argc, char **argv) {
  const std::string iris_path = argc > 1 ? argv[1] : "../Data/iris_binary.csv";
  std::cout << std::fixed << std::setprecision(4);

  const std::vector<std::pair<std::string, std::function<void()>>> examples = {
      {"CLASSIFICATION AND PROBABILITY METRICS", classification_demo},
      {"ROC AUC: FOUR STUDENTS", roc_demo},
      {"MAJORITY-CLASS BASELINE", majority_baseline_demo},
      {"REGRESSION METRICS", regression_demo},
      {"MEAN VERSUS MEDIAN", mean_median_demo},
      {"TRAINING-ONLY IMPUTATION", imputation_demo},
      {"FIVE-FOLD CROSS-VALIDATION", cross_validation_demo},
      {"IRIS: COMPLETE TRAIN/VALIDATION/TEST WORKFLOW",
       [&] { iris_demo(iris_path); }},
  };

  try {
    for (std::size_t i = 0; i < examples.size(); ++i) {
      std::cout << '\n' << i + 1 << ". " << examples[i].first << '\n';
      examples[i].second();
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
