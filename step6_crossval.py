"""
STEP 6: K-Fold cross-validation for both models.
"""
import joblib
from sklearn.model_selection import cross_val_score
from step3_preprocess import preprocess_data

MODEL_DIR = "models"


def cross_validate():
    print("=" * 60)
    print("📊 K-FOLD CROSS-VALIDATION (k=3)")
    print("=" * 60)

    X_train, _, yr_train, _, yc_train, _, _, _ = preprocess_data()

    reg = joblib.load(f"{MODEL_DIR}/regression_model.pkl")
    clf = joblib.load(f"{MODEL_DIR}/classification_model.pkl")

    reg_scores = cross_val_score(reg, X_train, yr_train, cv=3, scoring="r2")
    clf_scores = cross_val_score(clf, X_train, yc_train, cv=3, scoring="f1")

    print(f"\n📈 Regression (R² per fold): {reg_scores.round(3)}")
    print(f"   Mean R²  : {reg_scores.mean():.3f}")
    print(f"   Std      : {reg_scores.std():.3f}")

    print(f"\n🎯 Classification (F1 per fold): {clf_scores.round(3)}")
    print(f"   Mean F1  : {clf_scores.mean():.3f}")
    print(f"   Std      : {clf_scores.std():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    cross_validate()