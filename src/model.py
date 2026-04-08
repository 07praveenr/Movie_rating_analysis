# src/model.py
# PURPOSE: Predict a movie's rating from its features
# This is regression — predicting a continuous number (rating 1-10)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def train_and_evaluate(df: pd.DataFrame):
    """
    Build Linear Regression + Random Forest models.
    Evaluate both using MAE, RMSE, R².
    """
    print("\n" + "=" * 50)
    print("STEP 6: MACHINE LEARNING MODEL")
    print("=" * 50)

    # ── 1. Define Features & Target ──────────────────────
    FEATURES = ['log_votes', 'duration', 'year', 'movie_age',
                'genre_encoded', 'is_blockbuster']
    TARGET = 'rating'

    # Drop rows with any NaN in these columns
    model_df = df[FEATURES + [TARGET]].dropna()
    X = model_df[FEATURES]
    y = model_df[TARGET]

    print(f"📐 Training data: {X.shape[0]} movies, {X.shape[1]} features")

    # ── 2. Train/Test Split ──────────────────────────────
    # 80% train, 20% test — standard split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train set: {len(X_train)} | Test set: {len(X_test)}")

    # ── 3. Feature Scaling (for Linear Regression) ───────
    # Linear Regression is sensitive to scale — StandardScaler normalises
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    # NOTE: fit only on train data — prevents data leakage!

    # ── 4. Model 1: Linear Regression ────────────────────
    print("\n🔵 LINEAR REGRESSION:")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    print_metrics(y_test, y_pred_lr, "Linear Regression")

    # ── 5. Model 2: Random Forest ─────────────────────────
    print("\n🌲 RANDOM FOREST REGRESSOR:")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1        # use all CPU cores
    )
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    y_pred_rf = rf.predict(X_test)
    print_metrics(y_test, y_pred_rf, "Random Forest")

    # ── 6. Feature Importance (Random Forest) ────────────
    print("\n📊 FEATURE IMPORTANCE (Random Forest):")
    importance_df = pd.DataFrame({
        'feature': FEATURES,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.round(4).to_string(index=False))
    # Which features matter most for predicting rating?

    # ── 7. Visualise Predictions ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, y_pred, title in zip(axes,
                                  [y_pred_lr, y_pred_rf],
                                  ["Linear Regression", "Random Forest"]):
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', linewidth=1.5)
        ax.set_title(f"{title}\nActual vs Predicted")
        ax.set_xlabel("Actual Rating")
        ax.set_ylabel("Predicted Rating")

    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/07_model_predictions.png", bbox_inches='tight')
    plt.close()
    print("\n💾 Saved: outputs/plots/07_model_predictions.png")

    return rf, scaler


def print_metrics(y_true, y_pred, model_name: str):
    """Print MAE, RMSE, R² for a model."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    
    print(f"   MAE  = {mae:.4f}   ← avg error in rating points")
    print(f"   RMSE = {rmse:.4f}  ← penalises big errors more")
    print(f"   R²   = {r2:.4f}   ← % variance explained (1.0 = perfect)")


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/movies_features.csv")
    model, scaler = train_and_evaluate(df)