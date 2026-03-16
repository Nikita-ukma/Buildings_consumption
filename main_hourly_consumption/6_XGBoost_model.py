import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()

    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    return df


def add_lag_features(df, target="COMED_MW", lags=None):
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]

    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)

    return df


def add_rolling_features(df, target="COMED_MW"):
    df["rolling_mean_3"] = df[target].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df[target].shift(1).rolling(window=6).mean()
    df["rolling_max_12"] = df[target].shift(1).rolling(window=12).max()
    df["rolling_min_12"] = df[target].shift(1).rolling(window=12).min()
    return df


def prepare_features(df, target="COMED_MW", lags=None):
    df = add_lag_features(df, target=target, lags=lags)
    df = add_rolling_features(df, target=target)
    df = df.dropna()
    return df


def split_data(df, test_start="2017-01-01", val_days=30):
    test_start = pd.Timestamp(test_start)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df.loc[df.index < val_start].copy()
    val_df = df.loc[(df.index >= val_start) & (df.index < test_start)].copy()
    test_df = df.loc[df.index >= test_start].copy()

    return train_df, val_df, test_df

def train_model(X_train, y_train, X_val, y_val):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": 42,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "feature_fraction": 0.9,
        "lambda_l1": 0.1,
        "lambda_l2": 0.0,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "num_leaves": 63,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    return model

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def plot_predictions(index, y_true, y_pred, title):
    plt.figure(figsize=(14, 5))
    plt.plot(index, y_true, label="Actual", color="blue")
    plt.plot(index, y_pred, label="Predicted", color="red", alpha=0.7)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("MW")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, title):
    lgb.plot_importance(model, importance_type="gain", figsize=(12, 6), max_num_features=15)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(results_df["scenario"], results_df["RMSE"])
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(results_df["scenario"], results_df["MAE"])
    axes[1].set_title("MAE")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(results_df["scenario"], results_df["R2"])
    axes[2].set_title("R²")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()

def run_experiment(train_df, val_df, test_df, feature_cols, target_col, scenario_name, show_plots=False):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"\n{'='*70}")
    print(f"Running scenario: {scenario_name}")
    print(f"Number of features: {len(feature_cols)}")
    print("Features:", feature_cols)
    print(f"{'='*70}")

    model = train_model(X_train, y_train, X_val, y_val)

    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse, mae, r2 = evaluate(y_test, y_test_pred)

    print(f"\n{scenario_name} results:")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
    print(f"R²:   {r2:.4f}")

    if show_plots:
        plot_predictions(
            test_df.index,
            y_test,
            y_test_pred,
            title=f"Actual vs Predicted - {scenario_name}"
        )
        plot_feature_importance(model, title=f"Feature Importance - {scenario_name}")

    return {
        "scenario": scenario_name,
        "n_features": len(feature_cols),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "model": model,
        "predictions": y_test_pred
    }

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    filepath = os.path.join(BASE_DIR, "data", "FINAL_dataset.csv")

    target_col = "COMED_MW"
    lag_features = [1, 2, 3, 6, 12, 24]

    df = load_and_preprocess(filepath)
    df = prepare_features(df, target=target_col, lags=lag_features)

    train_df, val_df, test_df = split_data(df, test_start="2017-01-01", val_days=30)

    base_features = [
        "hour",
        "weekday",
        "month",
        "is_weekend",
        "is_holiday",
        "Chicago_temp",
        "Chicago_humidity",
        "Chicago_pressure",
        "Chicago_wind_speed",
    ]

    all_lag_features = [f"lag_{l}" for l in lag_features]

    rolling_features = [
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_max_12",
        "rolling_min_12"
    ]

    # Сценарій 1: з усіма lag/rolling ознаками
    features_full = base_features + all_lag_features + rolling_features

    # Сценарій 2: без lag_1
    features_without_lag1 = (
        base_features +
        [f for f in all_lag_features if f != "lag_1"] +
        rolling_features
    )

    features_no_lags = base_features.copy()

    results = []

    results.append(
        run_experiment(
            train_df, val_df, test_df,
            features_full, target_col,
            scenario_name="Full model (with all lags)",
            show_plots=False
        )
    )

    results.append(
        run_experiment(
            train_df, val_df, test_df,
            features_without_lag1, target_col,
            scenario_name="Without lag_1",
            show_plots=False
        )
    )

    results.append(
        run_experiment(
            train_df, val_df, test_df,
            features_no_lags, target_col,
            scenario_name="Without all lag/rolling features",
            show_plots=False
        )
    )

    results_df = pd.DataFrame([
        {
            "scenario": r["scenario"],
            "n_features": r["n_features"],
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "R2": r["R2"]
        }
        for r in results
    ])

    print("\n" + "="*90)
    print("COMPARISON OF SCENARIOS")
    print("="*90)
    print(results_df.sort_values("RMSE").to_string(index=False))

    plot_comparison(results_df)

    for r in results:
        plot_feature_importance(r["model"], f"Feature Importance - {r['scenario']}")

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_csv_path = os.path.join(results_dir, "lightgbm_scenarios_comparison.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nComparison table saved to: {results_csv_path}")

    for i, r in enumerate(results, start=1):
        safe_name = r["scenario"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(results_dir, f"{i}_{safe_name}.pkl")
        joblib.dump(r["model"], model_path)
        print(f"Saved model: {model_path}")
