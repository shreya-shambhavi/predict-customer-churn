# Customer Churn Prediction

Predicting telecom customer churn probability using classical ML and gradient boosting models. Built for the [Kaggle Playground Series - Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3) competition.

## Overview

This project predicts the probability that a telecom customer will churn based on their account details, services subscribed, and billing information. The pipeline is split into three stages, each captured in its own notebook:

1. **Exploratory Data Analysis & Preprocessing** — cleans and encodes the raw data
2. **Training** — trains and tunes five candidate models, selects the best performer
3. **Inference** — loads the trained model and generates competition submissions

## Results

Five models were trained and tuned via `GridSearchCV`, `RandomizedSearchCV`, or Optuna (5-fold cross-validated ROC AUC):

| Rank | Model               | ROC AUC |
|------|---------------------|---------|
| 1    | LightGBM             | 0.9088  |
| 2    | XGBoost              | 0.9087  |
| 3    | MLP Classifier       | 0.9081  |
| 4    | Random Forest        | 0.9076  |
| 5    | Logistic Regression  | 0.9063  |

**LightGBM** was selected as the final model, though the top three models are within a fraction of a point of one another.

## Project Structure

```
.
├── exploratory-data-analysis-and-preprocessing.ipynb   # Data cleaning & feature engineering
├── training.ipynb                                       # Model training, tuning & selection
├── inference.ipynb                                      # Prediction & submission generation

```

## Pipeline

### 1. Exploratory Data Analysis & Preprocessing

- Drops the `id` column and renames fields for consistency (`gender` → `Gender`, `tenure` → `Tenure`)
- Encodes binary categorical fields (`Yes`/`No`) as `1`/`0`
- Collapses "No internet/phone service" categories into `0` for the relevant service columns
- One-hot encodes `InternetService` and `PaymentMethod` into explicit binary columns
- Bins `Tenure` into 6 groups (12-month intervals) and `MonthlyCharges` / `TotalCharges` into quartiles, fitting bin edges on train and applying them to test to avoid leakage
- Outputs cleaned `train.csv` (with `Churn` target) and `test.csv`

### 2. Training

- Splits features (`X`) and target (`y`) from the preprocessed training set
- Trains and tunes five models:
  - **Logistic Regression** — `GridSearchCV` over regularization strength and penalty
  - **Random Forest** — `RandomizedSearchCV` over depth, leaf size, and feature sampling
  - **XGBoost** — Optuna Bayesian search (20 trials) with class imbalance handled via `scale_pos_weight`
  - **LightGBM** — Optuna Bayesian search (20 trials) with `class_weight="balanced"`
  - **MLP Classifier** — `RandomizedSearchCV` over architecture, activation, and learning rate
- Compares all models by cross-validated ROC AUC and saves:
  - `all_models.pkl` — dictionary of all five fitted estimators
  - `best_model.pkl` — the single best-performing model (LightGBM)

### 3. Inference

- Loads `all_models.pkl` / `best_model.pkl` and the preprocessed test set
- Generates churn probabilities with the selected model
- Merges predictions back with the original test `id` column
- Writes the final `submission.csv` in the format `id, Churn`

## Requirements

```
numpy
pandas
scikit-learn
xgboost
lightgbm
optuna
joblib
scipy
```

## Usage

Run the notebooks in order:

```bash
jupyter nbconvert --to notebook --execute exploratory-data-analysis-and-preprocessing.ipynb
jupyter nbconvert --to notebook --execute training.ipynb
jupyter nbconvert --to notebook --execute inference.ipynb
```

This will produce `submission.csv`, ready for upload to the Kaggle competition.

## Notes

- Class imbalance (~78% no-churn / 22% churn) is handled via `class_weight="balanced"` (Logistic Regression, Random Forest, LightGBM, MLP) or `scale_pos_weight` (XGBoost).
- Numerical binning thresholds (`Tenure`, `MonthlyCharges`, `TotalCharges`) are fit exclusively on the training set and applied to test data to prevent data leakage.
