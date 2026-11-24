"""
CatBoost + Extra Trees blend for freight rate prediction.

Split: random 80:20 on train_test.csv.
  1. Carve a 20% tuning set out of the 80% development portion and
     choose the blend weight there (the test set is not used).
  2. Report MAE, RMSE, R2 on the 20% test set.
  3. Refit on all of train_test.csv and write the submission files.

Run from the repo root:
    python src/train_blend.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA = Path("data")
SEED = 42

NUMERIC = [
    "distance", "weight", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "market_index", "quote_signal",
    "month", "dayofweek", "dayofyear",
]
CATEGORICAL = ["pickup", "delivery", "equipment"]
IMPUTE_COLS = ["weight", "market_index", "quote_signal"]


def load(path: Path) -> pd.DataFrame:
    """Read a CSV and add calendar features from the date column."""
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    df["month"] = dates.dt.month
    df["dayofweek"] = dates.dt.dayofweek
    df["dayofyear"] = dates.dt.dayofyear
    return df


def build_features(train: pd.DataFrame, other: pd.DataFrame):
    """
    Build feature matrices. Imputation medians come from `train` only,
    so no information leaks from the rows being predicted.
    """
    medians = train[IMPUTE_COLS].median()

    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        X = df[NUMERIC + CATEGORICAL].copy()
        X[IMPUTE_COLS] = X[IMPUTE_COLS].fillna(medians)
        for col in CATEGORICAL:
            X[col] = X[col].astype(str)  # CatBoost needs string categories
        return X

    return prepare(train), prepare(other)


def fit_predict(train: pd.DataFrame, other: pd.DataFrame):
    """Fit both models on `train` and return dollar predictions for `other`."""
    X_train, X_other = build_features(train, other)
    y_log = np.log1p(train["posted_rate"])  # train on log scale to tame skew

    # Model 1: CatBoost, handles pickup/delivery/equipment natively.
    cat = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        random_seed=SEED,
        verbose=False,
    )
    cat.fit(X_train, y_log, cat_features=CATEGORICAL)
    pred_cat = np.expm1(cat.predict(X_other))

    # Model 2: Extra Trees needs numeric input. City names are dropped
    # because coordinates already describe location, and equipment is one-hot encoded.
    def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=["pickup", "delivery"])
        return pd.get_dummies(X, columns=["equipment"], dtype=float)

    Z_train = to_numeric(X_train)
    Z_other = to_numeric(X_other).reindex(columns=Z_train.columns, fill_value=0.0)

    extra = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features=0.6,
        n_jobs=-1,
        random_state=SEED,
    )
    extra.fit(Z_train, y_log)
    pred_extra = np.expm1(extra.predict(Z_other))

    # Rates must be positive for the scorer.
    return np.clip(pred_cat, 1.0, None), np.clip(pred_extra, 1.0, None)


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    data = load(DATA / "train_test.csv")

    # ---- Split: 80% development / 20% test, random ----
    dev, test = train_test_split(data, test_size=0.20, random_state=SEED, shuffle=True)
    print(f"Random 80:20 split: dev={len(dev):,} rows, test={len(test):,} rows")

    # ---- 1. Tune the blend weight on a holdout carved from the dev set ----
    tune_train, tune_eval = train_test_split(dev, test_size=0.20, random_state=SEED, shuffle=True)
    pred_cat, pred_extra = fit_predict(tune_train, tune_eval)
    y_tune = tune_eval["posted_rate"].to_numpy()

    weights = np.linspace(0.0, 1.0, 21)  # weight on CatBoost
    maes = [mean_absolute_error(y_tune, w * pred_cat + (1 - w) * pred_extra) for w in weights]
    best_w = float(weights[int(np.argmin(maes))])
    print(f"[Tuning holdout] best CatBoost weight = {best_w:.2f}")

    # ---- 2. Evaluate on the 20% test set ----
    pred_cat, pred_extra = fit_predict(dev, test)
    y_test = test["posted_rate"].to_numpy()
    pred_blend = best_w * pred_cat + (1 - best_w) * pred_extra

    print("\n[Random 80:20 test evaluation]")
    for name, pred in [
        ("CatBoost", pred_cat),
        ("ExtraTrees", pred_extra),
        ("Blend", pred_blend),
    ]:
        m = metrics(y_test, pred)
        print(f"  {name:<10} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

    # ---- 3. Final fit on all of train_test.csv, predict validation and December ----
    validation = load(DATA / "validation.csv")
    december = load(DATA / "december_chart_inputs.csv")
    n_val = len(validation)

    to_predict = pd.concat([validation, december], ignore_index=True)
    pred_cat, pred_extra = fit_predict(data, to_predict)
    final = best_w * pred_cat + (1 - best_w) * pred_extra

    # Validation submission: exactly load_id,predicted_rate.
    submission = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": final[:n_val],
    })
    submission.to_csv("validation_predictions.csv", index=False)

    # December file: keep the original seven columns and fill predicted_rate.
    december_out = pd.read_csv(DATA / "december_chart_inputs.csv")
    december_out["predicted_rate"] = final[n_val:]
    december_out.to_csv(DATA / "december_chart_inputs.csv", index=False)

    print(f"\nWrote validation_predictions.csv ({len(submission)} rows)")
    print("Filled data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()