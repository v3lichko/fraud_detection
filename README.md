# Credit Card Fraud Detection

Classic supervised ML project: flag fraudulent credit card transactions in a
highly imbalanced dataset (492 frauds out of 284,807 transactions, ~0.17%).

## Business problem

A card issuer wants to automatically flag suspicious transactions for manual
review. Two kinds of mistakes matter, and they are *not* equally bad:

- **False negative** (missed fraud): the cardholder gets charged back, the
  issuer eats the loss, and takes a reputational hit. Expensive.
- **False positive** (false alarm): a legit transaction gets a manual review
  or a declined swipe. Annoying, but cheap by comparison.

That asymmetry drives most of the modeling choices below: we don't just
optimize accuracy (which is a useless metric here — predicting "not fraud"
for every transaction already scores 99.8%), we optimize precision/recall
trade-offs and pick a decision threshold based on an explicit cost matrix.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
(Dal Pozzolo et al., Université Libre de Bruxelles), mirrored on
[OpenML](https://www.openml.org/d/1597). Real European card transactions
from September 2013, 284,807 rows over two days.

- `Time`: seconds elapsed since the first transaction in the dump.
- `V1`..`V28`: PCA components of the original features (anonymized for
  confidentiality — we don't know what they represent, which is realistic:
  in practice a lot of fraud signal comes from features you're not allowed
  to see directly).
- `Amount`: transaction amount.
- `Class`: 1 = fraud, 0 = legitimate (target).

Not committed to git (144 MB) — fetch it with:

```bash
python scripts/download_data.py
```

## Approach

1. **Stratified train/test split** (80/20) — the test set is touched exactly
   once, at the end, to report final numbers.
2. **Feature engineering**: derive `Hour` (time-of-day) from `Time`, scale
   `Time` / `Amount` / `Hour` with a `RobustScaler` (median/IQR-based, robust
   to the extreme outliers in `Amount`). `V1`-`V28` are already PCA
   components, so they're passed through untouched.
3. **Model comparison** via 5-fold stratified CV on the training set,
   scored on **PR-AUC (average precision)** rather than ROC-AUC — with this
   much imbalance, ROC-AUC stays deceptively high even for weak models,
   while PR-AUC is much more sensitive to how well the model does on the
   rare positive class:
   - Logistic Regression (`class_weight="balanced"`) — baseline.
   - Random Forest (`class_weight="balanced"`).
   - XGBoost (`scale_pos_weight` set to the train-set imbalance ratio).
4. **Threshold tuning**: the default 0.5 cutoff is not chosen for anything.
   The winning model's decision threshold is tuned on a validation split
   carved out of the *training* data (never the test set) by minimizing
   `false_positives * cost_fp + false_negatives * cost_fn` — the code ships
   with an illustrative 1:100 cost ratio, meant to be replaced with real
   business numbers.
5. **Final report** on the untouched test set: precision, recall, F1,
   ROC-AUC, PR-AUC, confusion matrix, plus PR/ROC curves and feature
   importance for the winning model.

## Results

See `reports/metrics.csv` and `reports/figures/` for the full output of the
last training run. Headline test-set numbers (regenerate with the command
below — exact numbers will vary slightly run to run):

**Model comparison, test set @ default 0.5 threshold:**

| model | CV PR-AUC | PR-AUC | ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.842 | 0.876 | 0.974 | 0.837 | 0.837 | 0.837 |
| Random Forest | 0.841 | 0.845 | 0.982 | 0.870 | 0.816 | 0.842 |
| Logistic Regression | 0.751 | 0.724 | 0.974 | 0.056 | 0.908 | 0.105 |

XGBoost wins on PR-AUC (the metric CV model selection is based on) and gets
picked as the final model. Logistic Regression's abysmal precision at 0.5 is
expected and not a bug: `class_weight="balanced"` reweights the loss as if
classes were 50/50, so its raw probabilities are pushed way up — it "thinks"
far more transactions are fraud than actually are. That's exactly why we
don't trust the 0.5 default and tune the threshold instead of using it as-is.

**XGBoost, test set @ cost-tuned threshold (0.012, vs. default 0.5):**

| | precision | recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| @ 0.5 (default) | 0.837 | 0.837 | 0.837 | 82 | 16 | 16 | 56848 |
| @ 0.012 (cost-tuned) | 0.182 | 0.908 | 0.303 | 89 | 400 | 9 | 56464 |

Under the illustrative 1:100 cost matrix (a missed fraud costs 100x a false
alarm), the tuned threshold trades a lot of precision for catching 7 more of
the 98 fraud cases in the test set — expected total cost (`FP + 100*FN`)
drops from **1,616** at the default threshold to **1,300** at the tuned one.
Whether that trade is actually worth it depends entirely on your real
cost numbers, which is the point: the threshold is a business decision, not
a modeling one.

*(An earlier version of this pipeline tuned the threshold on a single
train/val split and got a threshold that was worse than the default — see
the "Threshold tuning" section of the notebook for why, and how out-of-fold
predictions fixed it.)*

## Project structure

```
scripts/download_data.py       fetch the dataset from OpenML -> data/raw/creditcard.csv
src/fraud_detection/
  config.py                    paths & constants
  data.py                      load + stratified split
  features.py                  Hour feature, scaling pipeline
  evaluate.py                  metrics, plots, threshold search
  train.py                     trains/compares models, saves model + report
  predict.py                   CLI to score new transactions with the saved model
notebooks/01_eda_and_modeling.ipynb   exploratory analysis + walkthrough
tests/                         pytest unit tests for features.py / data.py
models/fraud_model.joblib      winning pipeline + tuned threshold (generated)
reports/                       metrics.csv + figures (generated)
```

## Setup & usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # makes `fraud_detection` importable

python scripts/download_data.py       # -> data/raw/creditcard.csv
python -m fraud_detection.train       # trains, evaluates, saves model + report
python -m fraud_detection.predict --input new_transactions.csv --output scored.csv

pytest                                # unit tests
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Possible extensions (not implemented here, on purpose)

- Resampling (SMOTE / undersampling) as an alternative to class weighting —
  worth comparing, but class weighting already gets a lot of the benefit
  with far less risk of synthetic-sample artifacts.
- Calibrating predicted probabilities (`CalibratedClassifierCV`) if the
  scores need to be interpretable as real probabilities, not just a ranking.
- A real serving path (FastAPI endpoint) instead of a batch CLI, plus
  monitoring for feature/label drift, since fraud patterns shift over time
  and this model would go stale.
