import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_data_profile(df, output_dir="output"):
    """Generate a profile report for any DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("AUTOMATED EDA PROFILE")
    lines.append("=" * 50)
    lines.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append("")
    lines.append("Data Types:")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("Missing Values:")

    missing_counts = df.isnull().sum()
    if len(df) > 0:
        missing_pct = (missing_counts / len(df)) * 100
    else:
        missing_pct = pd.Series(0, index=df.columns)

    for col in df.columns:
        lines.append(f"{col}: {missing_counts[col]} ({missing_pct[col]:.2f}%)")

    lines.append("")

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        lines.append("Descriptive Statistics (Numeric Columns):")
        lines.append(numeric_df.describe().to_string())
        lines.append("")

    with open(os.path.join(output_dir, "auto_data_profile.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_numeric_distributions(df, numeric_cols=None, output_dir="output", style="whitegrid"):
    """Generate histograms with KDE for numeric columns."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style(style)

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
            plt.close()


def plot_correlation_heatmap(df, numeric_cols=None, output_dir="output", style="whitegrid"):
    """Generate a correlation heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style(style)

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        return

    numeric_df = df[numeric_cols].select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return

    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()


def plot_missing_data(df, output_dir="output", style="whitegrid"):
    """Visualize missing data percentage by column."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style(style)

    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    missing_pct.plot(kind="bar")
    plt.title("Missing Data Percentage by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_data_visualization.png"))
    plt.close()


def generate_outlier_summary(df, numeric_cols=None, output_dir="output"):
    """Generate outlier summary using the IQR method."""
    os.makedirs(output_dir, exist_ok=True)

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    lines = []
    lines.append("OUTLIER SUMMARY (IQR METHOD)")
    lines.append("=" * 50)

    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()

            if series.empty:
                lines.append(f"{col}: 0 outliers")
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = series[(series < lower) | (series > upper)]
            lines.append(f"{col}: {len(outliers)} outliers")

    with open(os.path.join(output_dir, "outlier_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_eda_report(df, numeric_cols=None, output_dir="output", style="whitegrid"):
    """Generate a reusable full EDA report for any DataFrame."""
    generate_data_profile(df, output_dir=output_dir)
    plot_numeric_distributions(df, numeric_cols=numeric_cols, output_dir=output_dir, style=style)
    plot_correlation_heatmap(df, numeric_cols=numeric_cols, output_dir=output_dir, style=style)
    plot_missing_data(df, output_dir=output_dir, style=style)
    generate_outlier_summary(df, numeric_cols=numeric_cols, output_dir=output_dir)