# train_model_v2.py
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "data_v2.csv"
MODEL_PATH = "model_v2.pkl"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Basic sanity drop NA
    df = df.dropna(subset=["area", "bedrooms", "bathrooms", "location", "price"])
    return df

def build_pipeline():
    numeric_features = ["area", "bedrooms", "bathrooms"]
    categorical_features = ["location"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LinearRegression()

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipe

def train(df):
    X = df[["area", "bedrooms", "bathrooms", "location"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:,.0f}  |  R²: {r2:.3f}")

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Model pipeline saved to {MODEL_PATH}")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Please create the file.")

    df = load_data()
    print(f"Loaded {len(df)} rows.")
    train(df)

if __name__ == "__main__":
    main()
