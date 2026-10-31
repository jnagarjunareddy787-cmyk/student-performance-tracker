"""
STEP 3: Preprocess data — encoding, scaling, train/test split.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "data/student-mat.csv"


def preprocess_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Create targets
    df["final_grade"] = df["G3"]
    df["at_risk"] = (df["G3"] < 10).astype(int)

    # Features (drop grades to prevent data leakage)
    drop_cols = ["G1", "G2", "G3", "final_grade", "at_risk"]
    X = df.drop(columns=drop_cols)
    y_reg = df["final_grade"]
    y_clf = df["at_risk"]

    # Train/Test split
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Preprocessing complete")
    print(f"   Train samples: {X_train_scaled.shape[0]}")
    print(f"   Test samples : {X_test_scaled.shape[0]}")
    print(f"   Features     : {X_train_scaled.shape[1]}")
    print(f"   At-risk ratio: {y_clf.mean():.2%}")

    return (X_train_scaled, X_test_scaled,
            yr_train, yr_test, yc_train, yc_test,
            scaler, X.columns.tolist())


if __name__ == "__main__":
    preprocess_data()
    