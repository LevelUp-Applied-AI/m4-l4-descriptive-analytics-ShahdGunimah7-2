import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from eda_analysis import load_and_profile


def run_department_anova(df):
    """Run one-way ANOVA on GPA across departments, then post-hoc if significant."""
    groups = []
    dept_names = []

    for dept, group in df.groupby("department"):
        gpa_vals = group["gpa"].dropna().values
        if len(gpa_vals) > 1:
            groups.append(gpa_vals)
            dept_names.append(dept)

    f_stat, p_val = stats.f_oneway(*groups)

    print("\n=== TIER 1: ANOVA ===")
    print("Testing whether GPA differs across departments")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        print("\nSignificant result -> running post-hoc pairwise t-tests with Bonferroni correction")
        pairs = list(itertools.combinations(dept_names, 2))
        m = len(pairs)

        for d1, d2 in pairs:
            g1 = df[df["department"] == d1]["gpa"].dropna()
            g2 = df[df["department"] == d2]["gpa"].dropna()

            t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
            p_adj = min(p * m, 1.0)

            print(f"{d1} vs {d2}: t={t_stat:.4f}, raw p={p:.4f}, Bonferroni p={p_adj:.4f}")
    else:
        print("No statistically significant GPA difference across departments, so post-hoc tests are not needed.")


def plot_violin(df):
    """Create violin plot for GPA across departments."""
    os.makedirs("output", exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="department", y="gpa", inner="box")
    plt.title("GPA Distribution Across Departments (Violin Plot)")
    plt.xlabel("Department")
    plt.ylabel("GPA")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("output/gpa_violin.png")
    plt.close()

    print("Saved: output/gpa_violin.png")


def bootstrap_ci(data, n_bootstrap=10000, random_state=42):
    """Bootstrap 95% confidence interval for the mean."""
    rng = np.random.default_rng(random_state)
    values = data.dropna().values
    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())

    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)

    return np.mean(means), lower, upper


def power_analysis(effect_size, alpha=0.05, power=0.80):
    """Estimate required sample size per group for a two-sample t-test."""
    analysis = TTestIndPower()
    n_required = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        alternative="two-sided"
    )
    return n_required


def simulate_false_positive(n_simulations=5000, sample_size=50, alpha=0.05, random_state=42):
    """Simulate false positive rate under the null hypothesis."""
    rng = np.random.default_rng(random_state)
    false_positives = 0

    for _ in range(n_simulations):
        g1 = rng.normal(0, 1, sample_size)
        g2 = rng.normal(0, 1, sample_size)

        _, p_val = stats.ttest_ind(g1, g2, equal_var=False)

        if p_val < alpha:
            false_positives += 1

    return false_positives / n_simulations


def main():
    os.makedirs("output", exist_ok=True)

    df = load_and_profile("data/student_performance.csv")

    # ===== Tier 1 =====
    run_department_anova(df)
    plot_violin(df)

    # ===== Tier 3 =====
    print("\n=== TIER 3: BOOTSTRAP CI ===")
    gpa_yes = df[df["has_internship"] == "Yes"]["gpa"]
    gpa_no = df[df["has_internship"] == "No"]["gpa"]

    mean_yes, low_yes, high_yes = bootstrap_ci(gpa_yes)
    mean_no, low_no, high_no = bootstrap_ci(gpa_no)

    print(f"Internship = Yes -> mean={mean_yes:.4f}, CI=({low_yes:.4f}, {high_yes:.4f})")
    print(f"Internship = No  -> mean={mean_no:.4f}, CI=({low_no:.4f}, {high_no:.4f})")

    print("\n=== TIER 3: POWER ANALYSIS ===")
    # استخدمنا effect size تقريبي من Cohen's d في التحليل الأساسي
    n_required = power_analysis(effect_size=0.69)
    print(f"Required sample size per group (80% power): {n_required:.2f}")

    print("\n=== TIER 3: FALSE POSITIVE SIMULATION ===")
    fp_rate = simulate_false_positive()
    print(f"Estimated false positive rate: {fp_rate:.4f}")


if __name__ == "__main__":
    main()