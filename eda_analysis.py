"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded and cleaned dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
        - Handling decisions for missing values
    """
    df = pd.read_csv(filepath)

    report = []
    report.append("STUDENT PERFORMANCE DATA PROFILE")
    report.append("=" * 50)
    report.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    report.append("")

    report.append("Data Types:")
    report.append(df.dtypes.to_string())
    report.append("")

    report.append("Missing Values:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    for col in df.columns:
        report.append(f"{col}: {missing_counts[col]} ({missing_pct[col]:.2f}%)")
    report.append("")

    report.append("Descriptive Statistics (Numeric Columns):")
    report.append(df.describe().to_string())
    report.append("")

    report.append("Handling Decisions for Missing Values:")

    if "commute_minutes" in df.columns and df["commute_minutes"].isnull().sum() > 0:
        report.append(
            "- commute_minutes: imputed with median because around 10% of values are missing "
            "and median is robust to skewness and outliers."
        )
        df["commute_minutes"] = df["commute_minutes"].fillna(df["commute_minutes"].median())

    if "study_hours_weekly" in df.columns and df["study_hours_weekly"].isnull().sum() > 0:
        report.append(
            "- study_hours_weekly: rows with missing values were dropped because only around 5% "
            "are missing and the loss is limited."
        )
        df = df.dropna(subset=["study_hours_weekly"])

    with open("output/data_profile.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory.
    """
    numeric_cols = ["gpa", "study_hours_weekly", "attendance_pct"]

    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"output/{col}_distribution.png")
            plt.close()

    if "department" in df.columns and "gpa" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="department", y="gpa")
        plt.title("GPA Distribution Across Departments")
        plt.xlabel("Department")
        plt.ylabel("GPA")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("output/gpa_by_department_boxplot.png")
        plt.close()

    if "scholarship" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="scholarship", order=df["scholarship"].value_counts().index)
        plt.title("Distribution of Scholarship Types")
        plt.xlabel("Scholarship")
        plt.ylabel("Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("output/scholarship_distribution.png")
        plt.close()


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        tuple:
            - correlation matrix (DataFrame)
            - list of top 2 correlated variable pairs excluding self-correlation

    Side effects:
        Saves:
        - correlation heatmap
        - scatter plots for the two most correlated variable pairs
    """
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation Heatmap for Numeric Variables")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png")
    plt.close()

    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0]
    corr_pairs = corr_pairs.sort_values(ascending=False)

    top_pairs = []
    seen = set()

    for (var1, var2), corr_value in corr_pairs.items():
        pair_key = tuple(sorted([var1, var2]))
        if pair_key not in seen:
            seen.add(pair_key)
            top_pairs.append((var1, var2, corr_matrix.loc[var1, var2]))
        if len(top_pairs) == 2:
            break

    for idx, (var1, var2, corr_value) in enumerate(top_pairs, start=1):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=var1, y=var2)
        plt.title(f"Scatter Plot of {var1} vs {var2} (r = {corr_value:.2f})")
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.tight_layout()
        plt.savefig(f"output/top_correlation_pair_{idx}_{var1}_vs_{var2}.png")
        plt.close()

    return corr_matrix, top_pairs


def cohens_d(group1, group2):
    """Compute Cohen's d effect size for two independent samples."""
    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: results for the required hypothesis tests

    Side effects:
        Prints test results to stdout with interpretation.

    Tests:
        - t-test: GPA difference by internship status
        - chi-square: association between scholarship and department
    """
    results = {}

    # Hypothesis 1: Students with internships have a higher GPA
    if "has_internship" in df.columns and "gpa" in df.columns:
        gpa_yes = df[df["has_internship"] == "Yes"]["gpa"].dropna()
        gpa_no = df[df["has_internship"] == "No"]["gpa"].dropna()

        t_stat, p_val = stats.ttest_ind(gpa_yes, gpa_no, equal_var=False)
        d_value = cohens_d(gpa_yes, gpa_no)

        results["internship_ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": d_value,
            "mean_gpa_yes": gpa_yes.mean(),
            "mean_gpa_no": gpa_no.mean()
        }

        print("Hypothesis 1: Students with internships have a higher GPA than students without internships.")
        print(f"Mean GPA (Yes): {gpa_yes.mean():.4f}")
        print(f"Mean GPA (No): {gpa_no.mean():.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Cohen's d: {d_value:.4f}")
        if p_val < 0.05:
            print("Interpretation: There is a statistically significant GPA difference by internship status.\n")
        else:
            print("Interpretation: There is no statistically significant GPA difference by internship status.\n")

    # Hypothesis 2: Scholarship status is associated with department
    if "scholarship" in df.columns and "department" in df.columns:
        contingency_table = pd.crosstab(df["scholarship"], df["department"])
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

        results["scholarship_department_chi2"] = {
            "chi2_statistic": chi2_stat,
            "p_value": p_val,
            "degrees_of_freedom": dof
        }

        print("Hypothesis 2: Scholarship status is associated with department.")
        print(f"Chi-square statistic: {chi2_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Degrees of freedom: {dof}")

        if p_val < 0.05:
            print("Interpretation: Scholarship status and department are significantly associated.\n")
        else:
            print("Interpretation: Scholarship status and department are not significantly associated.\n")

    return results


def write_findings_report(df, corr_matrix, top_pairs, test_results):
    """Write a markdown findings report summarizing the analysis."""
    findings = []
    findings.append("# Findings Report")
    findings.append("")
    findings.append("## Overview")
    findings.append("This report summarizes exploratory data analysis on the student performance dataset.")
    findings.append("")

    findings.append("## Distribution Analysis")
    findings.append("- GPA, study hours, and attendance were visualized using histograms with KDE.")
    findings.append("- GPA was also compared across departments using a box plot.")
    findings.append("- Scholarship type frequencies were visualized using a bar chart.")
    findings.append("")

    findings.append("## Correlation Analysis")
    findings.append("Pearson correlations were computed for all numeric variables.")
    findings.append("")

    for idx, (var1, var2, corr_value) in enumerate(top_pairs, start=1):
        findings.append(
            f"- Top pair {idx}: **{var1}** and **{var2}** with correlation **{corr_value:.4f}**."
        )

    findings.append("")
    findings.append("These relationships may reflect how academic engagement or workload patterns relate to performance.")
    findings.append("")

    findings.append("## Hypothesis Testing")

    if "internship_ttest" in test_results:
        res = test_results["internship_ttest"]
        findings.append(
            f"- **Internship vs GPA t-test**: t = {res['t_statistic']:.4f}, "
            f"p = {res['p_value']:.4f}, Cohen's d = {res['cohens_d']:.4f}."
        )
        if res["p_value"] < 0.05:
            findings.append("  - Students with internships show a statistically significant difference in GPA.")
        else:
            findings.append("  - No statistically significant GPA difference was found by internship status.")

    if "scholarship_department_chi2" in test_results:
        res = test_results["scholarship_department_chi2"]
        findings.append(
            f"- **Scholarship vs Department chi-square test**: chi2 = {res['chi2_statistic']:.4f}, "
            f"p = {res['p_value']:.4f}, dof = {res['degrees_of_freedom']}."
        )
        if res["p_value"] < 0.05:
            findings.append("  - Scholarship status appears to be associated with department.")
        else:
            findings.append("  - No significant association was found between scholarship status and department.")

    findings.append("")
    findings.append("## Conclusion")
    findings.append(
        "The dataset shows meaningful structure across academic performance, engagement, and student characteristics. "
        "The strongest numeric relationships were visualized and statistically examined, and categorical associations "
        "were evaluated using appropriate hypothesis tests."
    )

    with open("output/FINDINGS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(findings))


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    df = load_and_profile("data/student_performance.csv")
    plot_distributions(df)
    corr_matrix, top_pairs = plot_correlations(df)
    test_results = run_hypothesis_tests(df)
    write_findings_report(df, corr_matrix, top_pairs, test_results)


if __name__ == "__main__":
    main()

    
