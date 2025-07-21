"""
train_model_v2.py  (FINAL: older sklearn compatible, Excel-safe, debug prints)

Train the multi-feature house price prediction model used by app_v2.py.

Use either:
  --generate N         Generate N synthetic rows
  --data path.csv      Load existing dataset

Optional:
  --save-csv path.csv  Save generated synthetic data (UTF-8-SIG for Excel)
  --save-legacy        Also train & save an area-only fallback model.pkl

Outputs:
  model_v2.pkl         (multi-feature pipeline)
  model.pkl            (optional legacy area-only model)
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------
LOCATIONS = {
    "Vizag": 1.0,
    "Hyderabad": 1.2,
    "Bengaluru": 1.3,
    "Chennai": 1.1,
    "Mumbai": 1.8,
}

DEFAULT_MODEL_V2_PATH = "model_v2.pkl"
DEFAULT_LEGACY_MODEL_PATH = "model.pkl"


# ------------------------------------------------------------------
# Synthetic Data Generator (simple baseline)
# ------------------------------------------------------------------
def generate_synthetic(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = list(LOCATIONS.keys())
    rows = []

    for _ in range(n_rows):
        area = rng.integers(500, 5000)            # sq ft
        bedrooms = rng.integers(1, 6)             # 1-5
        bathrooms = rng.integers(1, 5)            # 1-4
        location = rng.choice(locations)

        # base structural value
        price = (area * 3000) + (bedrooms * 500_000) + (bathrooms * 300_000)

        # location premium
        price *= LOCATIONS[location]

        # noise (scale with price magnitude, cap at ±5% but at least ±300k)
        noise_scale = max(price * 0.05, 300_000)
        noise = rng.normal(0, noise_scale / 2)  # ~95% within ±noise_scale
        price = max(100_000, price + noise)     # floor

        rows.append([area, bedrooms, bathrooms, location, round(price)])

    return pd.DataFrame(rows, columns=["area", "bedrooms", "bathrooms", "location", "price"])


# ------------------------------------------------------------------
# Model Training (multi-feature)
# ------------------------------------------------------------------
def train_model_v2(df: pd.DataFrame, out_path: str = DEFAULT_MODEL_V2_PATH):
    # Basic clean
    needed = ["area", "bedrooms", "bathrooms", "location", "price"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    df = df[needed].dropna().copy()

    # Ensure valid dtypes
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], downcast="integer", errors="coerce")
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], downcast="integer", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["area", "bedrooms", "bathrooms", "price"])

    # Filter bad locations
    df = df[df["location"].isin(LOCATIONS.keys())]

    print(f"[MODEL_V2] Training rows: {len(df)}")

    X = df[["area", "bedrooms", "bathrooms", "location"]]
    y = df["price"]

    numeric_features = ["area", "bedrooms", "bathrooms"]
    categorical_features = ["location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    # Older sklearn versions don't support squared=
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)

    print(f"[MODEL_V2] Saved trained pipeline to {out_path}")
    print(f"[MODEL_V2] Test RMSE: {rmse:,.0f}")
    print(f"[MODEL_V2] Test R²:   {r2:.3f}")

    return pipe, {"rmse": rmse, "r2": r2}


# ------------------------------------------------------------------
# Legacy area-only model (optional)
# ------------------------------------------------------------------
def train_legacy_area_only(df: pd.DataFrame, out_path: str = DEFAULT_LEGACY_MODEL_PATH):
    """
    Train an area-only Linear Regression model (for fallback).
    """
    from sklearn.linear_model import LinearRegression

    if "price" not in df.columns or "area" not in df.columns:
        raise ValueError("DataFrame must contain 'area' and 'price' for legacy model.")

    # Coerce numeric to be safe
    d = df.dropna(subset=["area", "price"]).copy()
    d["area"] = pd.to_numeric(d["area"], errors="coerce")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.dropna(subset=["area", "price"])

    X = d[["area"]]
    y = d["price"]

    reg = LinearRegression()
    reg.fit(X, y)

    with open(out_path, "wb") as f:
        pickle.dump(reg, f)

    print(f"[LEGACY] Saved area-only model to {out_path}")
    return reg


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Area→Price prediction models.")
    p.add_argument("--data", type=str, default=None,
                   help="Path to CSV with columns: area,bedrooms,bathrooms,location,price.")
    p.add_argument("--generate", type=int, default=None,
                   help="Generate N synthetic rows instead of loading CSV.")
    p.add_argument("--save-csv", type=str, default=None,
                   help="If generating synthetic data, save the dataset to this CSV path.")
    p.add_argument("--model-out", type=str, default=DEFAULT_MODEL_V2_PATH,
                   help="Output path for multi-feature model (default model_v2.pkl).")
    p.add_argument("--save-legacy", action="store_true",
                   help="Also train & save an area-only fallback model.pkl.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data.")
    return p.parse_args()


def main():
    print(f"[INFO] Script path: {os.path.abspath(__file__)}")
    args = parse_args()

    # Get data
    if args.generate is not None:
        print(f"[DATA] Generating {args.generate} synthetic rows...")
        df = generate_synthetic(args.generate, seed=args.seed)
        print(f"[DATA] Generated rows: {len(df)}")
        if args.save_csv:
            # Excel-friendly encoding
            df.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
            print(f"[DATA] Synthetic dataset saved to {args.save_csv}")
    elif args.data:
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"CSV not found: {args.data}")
        print(f"[DATA] Loading dataset from {args.data}")
        df = pd.read_csv(args.data)
        print(f"[DATA] Loaded rows: {len(df)}")
    else:
        raise SystemExit("You must provide --data path or --generate N.")

    # Train multi-feature model
    train_model_v2(df, out_path=args.model_out)

    # Optional legacy model
    if args.save_legacy:
        train_legacy_area_only(df, out_path=DEFAULT_LEGACY_MODEL_PATH)


if __name__ == "__main__":
    main()
