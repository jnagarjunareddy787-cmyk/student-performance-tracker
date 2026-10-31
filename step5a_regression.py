"""
STEP 5A: Train Random Forest Regression to predict final grade.
Saves: models/regression_model.pkl, scaler.pkl, features.pkl
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from step3_preprocess import preprocess_data

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_regression():
    print("=" * 60)
    print("📈 TRAINING REGRESSION MODEL")
    print("=" * 60)

    X_train, X_test, yr_train, yr_test, _, _, scaler, features = preprocess_data()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, yr_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
    mae = mean_absolute_error(yr_test, y_pred)
    r2 = r2_score(yr_test, y_pred)

    print(f"\n📊 Regression Metrics:")
    print(f"   RMSE : {rmse:.3f}")
    print(f"   MAE  : {mae:.3f}")
    print(f"   R²   : {r2:.3f}")

    joblib.dump(model, f"{MODEL_DIR}/regression_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(features, f"{MODEL_DIR}/features.pkl")

    print(f"\n✅ Saved: {MODEL_DIR}/regression_model.pkl")
    print(f"✅ Saved: {MODEL_DIR}/scaler.pkl")
    print(f"✅ Saved: {MODEL_DIR}/features.pkl")


if __name__ == "__main__":
    train_regression()