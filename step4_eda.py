"""
STEP 4: Exploratory Data Analysis — generate plots.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/student-mat.csv"
OUT_DIR = "outputs"


def run_eda(path=DATA_PATH, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(path, sep=";")

    # ---- 4.1 Correlation Heatmap ----
    plt.figure(figsize=(16, 12))
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap — Student Performance")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png", dpi=100)
    plt.close()
    print(f"✅ Saved: {out_dir}/correlation_heatmap.png")

    # ---- 4.2 Distribution of Final Grades ----
    plt.figure(figsize=(8, 5))
    sns.histplot(df["G3"], bins=20, kde=True, color="steelblue")
    plt.title("Distribution of Final Grades (G3)")
    plt.xlabel("Final Grade")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/grade_distribution.png", dpi=100)
    plt.close()
    print(f"✅ Saved: {out_dir}/grade_distribution.png")

    # ---- 4.3 Study Time vs Grade ----
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="studytime", y="G3", data=df)
    plt.title("Study Time vs Final Grade")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/studytime_vs_grade.png", dpi=100)
    plt.close()
    print(f"✅ Saved: {out_dir}/studytime_vs_grade.png")

    # ---- 4.4 Absences vs Grade ----
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="absences", y="G3", data=df, alpha=0.6, color="crimson")
    plt.title("Absences vs Final Grade")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/absences_vs_grade.png", dpi=100)
    plt.close()
    print(f"✅ Saved: {out_dir}/absences_vs_grade.png")

    print("\n🎉 EDA complete — check outputs/ folder!")


if __name__ == "__main__":
    run_eda()