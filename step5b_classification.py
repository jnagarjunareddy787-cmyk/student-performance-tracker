"""
STEP 5B: Train XGBoost Classifier to detect at-risk students.
Saves: models/classification_model.pkl
"""
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from step3_preprocess import preprocess_data

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_classification():
    print("=" * 60)
    print("🎯 TRAINING CLASSIFICATION MODEL")
    print("=" * 60)

    X_train, X_test, _, _, yc_train, yc_test, _, _ = preprocess_data()

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, yc_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n📊 Classification Metrics:")
    print(f"   Accuracy : {accuracy_score(yc_test, y_pred):.3f}")
    print(f"   Precision: {precision_score(yc_test, y_pred, zero_division=0):.3f}")
    print(f"   Recall   : {recall_score(yc_test, y_pred, zero_division=0):.3f}")
    print(f"   F1-Score : {f1_score(yc_test, y_pred, zero_division=0):.3f}")
    print(f"   ROC-AUC  : {roc_auc_score(yc_test, y_prob):.3f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(yc_test, y_pred)}")

    joblib.dump(model, f"{MODEL_DIR}/classification_model.pkl")
    print(f"\n✅ Saved: {MODEL_DIR}/classification_model.pkl")


if __name__ == "__main__":
    train_classification()