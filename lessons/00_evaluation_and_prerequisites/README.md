# Lesson 00 — Evaluating Machine Learning Models

Before building any tree, you need a fair way to tell whether a model is good.
This lesson covers data splits, classification and regression metrics, ROC AUC,
baselines, data leakage, and cross-validation.

## What is in this folder

| File | What it is |
| --- | --- |
| [tutorial.md](tutorial.md) | The full lesson: theory with worked numerical examples |
| [summary.md](summary.md) | A one-page recap of the formulas and rules |
| [quiz.json](quiz.json) | 10 multiple-choice questions (pass mark: 8 / 10) |
| [Codes/native.py](Codes/native.py) | Every metric written from scratch in plain Python |
| [Codes/libraries.py](Codes/libraries.py) | The same examples using scikit-learn, plus a real Iris workflow |
| [Codes/libraries.cpp](Codes/libraries.cpp) | The same examples in C++ using mlpack |
| [Data/iris_binary.csv](Data/iris_binary.csv) | The Iris data read by the C++ program |

## How to study this lesson

1. Read [tutorial.md](tutorial.md) and solve its examples on paper.
2. Review [summary.md](summary.md).
3. Take the quiz in [quiz.json](quiz.json).
4. Run `native.py` to see each formula as code.
5. Run `libraries.py` and check that the library results match.
6. Run `libraries.cpp` to see the same workflow in C++.

## Lesson outline

[tutorial.md](tutorial.md) has 13 sections:

1. **Features, targets, and predictions** — X, y, and ŷ; classification versus regression.
2. **Training, validation, and test data** — the job of each split, and how to choose a realistic split.
3. **Classification probabilities and thresholds** — turning a probability into a class.
4. **Confusion matrix** — TP, TN, FP, and FN.
5. **Accuracy, precision, recall, and F1** — and why accuracy can mislead.
6. **Evaluating probabilities** — Brier score and log loss.
7. **ROC curves and AUC** — ROC points, tracing the curve, the area under it, and the ranking check.
8. **Baselines** — majority class for classification; mean and median for regression.
9. **Regression metrics** — MAE, MSE, and RMSE.
10. **R²** — what it measures and why it can be negative.
11. **Data leakage** — future information and preprocessing before splitting.
12. **Cross-validation** — refitting the whole workflow inside each round.
13. **Practice** — exercises with answers.

## What the code examples cover

The three programs run the same examples in the same order, so you can compare
their outputs line by line.

| # | Example | Tutorial sections | `native.py` | `libraries.py` | `libraries.cpp` |
| --- | --- | --- | :---: | :---: | :---: |
| 1 | Classification and probability metrics | 3–6 | ✓ | ✓ | ✓ |
| 2 | ROC AUC: four students | 7 | ✓ | ✓ | ✓ |
| 3 | Majority-class baseline | 5, 8 | ✓ | ✓ | ✓ |
| 4 | Regression metrics | 9, 10 | ✓ | ✓ | ✓ |
| 5 | Mean versus median | 8 | ✓ | ✓ | ✓ |
| 6 | Training-only imputation | 11 | ✓ | ✓ | ✓ |
| 7 | Five-fold cross-validation | 12 | ✓ | ✓ | ✓ |
| 8 | Iris: train / validation / test workflow | 2, 12 | — | ✓ | ✓ |

## Get the code

```bash
git clone https://github.com/MohamadAlhajAli/Tree-based-ML.git
cd Tree-based-ML
```

## Run the Python code

You need Python 3.10 or newer.

**1. Create a virtual environment** in the repository folder (recommended).

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install the libraries.**

```bash
pip install numpy scikit-learn
```

`libraries.py` needs scikit-learn 1.4 or newer. Tested with Python 3.10.11,
scikit-learn 1.7.2, and NumPy 2.2.6.

**3. Run the examples from the lesson folder.**

```bash
cd lessons/00_evaluation_and_prerequisites
python Codes/native.py
python Codes/libraries.py
```

- `native.py` uses only the Python standard library.
- `libraries.py` loads Iris from scikit-learn, which needs no download.

## Run the C++ code

You need a C++17 compiler and [mlpack](https://www.mlpack.org/). mlpack is a
header-only library; installing it also installs Armadillo (linear algebra),
ensmallen, and cereal. Tested with g++ 16.2 and mlpack 4.8.0. Older mlpack
releases may be missing functions this code uses, such as `ROCAUCScore`.

### Windows (tested): MSYS2

1. Install MSYS2 from [msys2.org](https://www.msys2.org/).
2. Open the **MSYS2 UCRT64** terminal from the Start menu. Run every command
   below in this terminal, not in Git Bash or PowerShell. Package setup scripts
   fail outside it.
3. Install the compiler and mlpack:

   ```bash
   pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-mlpack
   ```

### macOS and Linux (not tested here)

- macOS: `brew install mlpack`
- Linux: distribution packages such as `libmlpack-dev` are often older than
  4.8. Use `conda install -c conda-forge mlpack`, or build mlpack from source
  by following its documentation.

### Build and run

From the lesson folder. In the MSYS2 terminal, drive `D:` is `/d`, so a
Windows path such as `D:\Projects\Tree-based-ML` becomes
`/d/Projects/Tree-based-ML`.

```bash
cd lessons/00_evaluation_and_prerequisites
mkdir -p build
g++ -std=c++17 -O2 Codes/libraries.cpp -o build/libraries -larmadillo
./build/libraries Data/iris_binary.csv
```

- Compiling takes about 20 seconds.
- On MSYS2 the compiler prints a note,
  `WARNING: use of ARMA_BLAS_LONG_LONG in conjunction with ARMA_USE_WRAPPER is not supported`.
  It comes from MSYS2's Armadillo settings, and the program still runs correctly.
- The `build/` folder is ignored by Git.
- If you run the program from inside `Codes/`, you can leave out the CSV path.

## Check your output

All three programs should print these values:

| Example | Expected values |
| --- | --- |
| Classification | accuracy 0.6667, Brier score 0.2450, log loss 0.6743 |
| ROC AUC | AUC 0.7500 |
| Majority-class baseline | accuracy 0.7000, precision undefined, recall 0.0000 |
| Regression | MAE 2.3333, MSE 7.0000, RMSE 2.6458, R2 -0.1667 |
| Mean versus median | mean: MAE 3.3333, MSE 12.6667; median: MAE 2.6667, MSE 16.6667 |
| Training-only imputation | learned mean 3; validation becomes 100, 3 |
| Cross-validation | round MSEs 6.500, 12.516, 4.516, 16.391, 7.516 (mean 9.4875); final test R2 -0.0417 |

For cross-validation, `native.py` prints each round, `libraries.cpp` prints only
the mean, and `libraries.py` prints both.

The Iris results differ between Python and C++, and that is expected. The two
languages split the data differently, and scikit-learn's `C` and mlpack's
`lambda` are different regularization scales. Both models classify the test set
perfectly, because setosa versus versicolor is an easy task.

## The data

[Data/iris_binary.csv](Data/iris_binary.csv) holds the first 100 rows of the
classic Iris dataset: setosa (label 0) and versicolor (label 1). It has no
header row. Its 5 columns are:

1. sepal length (cm)
2. sepal width (cm)
3. petal length (cm)
4. petal width (cm)
5. label (0 or 1)

These are the same rows that `libraries.py` gets from scikit-learn's
`load_iris()` with `target < 2`.

## The quiz

[quiz.json](quiz.json) has 10 questions. Each one has the `question`, four
`options` (A–D), the `correct_option_id`, an `explanation`, a `difficulty`, and
a `topic`. Answer each question before you look at its `correct_option_id`. The
pass mark is 8 correct answers.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'sklearn'` | Activate the virtual environment, then run the `pip install` command again. |
| `ImportError: cannot import name 'root_mean_squared_error'` | Your scikit-learn is older than 1.4. Run `pip install -U scikit-learn`. |
| `fatal error: mlpack/core.hpp: No such file or directory` | mlpack is not installed, or you are not in the MSYS2 UCRT64 terminal. |
| `sed: can't read /ucrt64/...` while installing packages | pacman was run outside the MSYS2 UCRT64 terminal. Reinstall from inside it: `pacman -S mingw-w64-ucrt-x86_64-armadillo`. |
| `Error: Could not read the 5-column CSV file ...` | Run from the lesson folder and pass `Data/iris_binary.csv`. |
