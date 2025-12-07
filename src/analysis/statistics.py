"""
Statistical Testing Framework

Provides rigorous statistical validation for all analyses:
- Bootstrap confidence intervals
- Hypothesis testing (t-tests, permutation tests)
- Effect size computation (Cohen's d, Glass's delta)
- Multiple testing correction (Bonferroni, FDR)

Principle: Every claim about GPN vs GAN differences must be statistically validated
with appropriate significance tests and effect sizes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
from dataclasses import dataclass


@dataclass
class StatisticalTest:
    """Results of a statistical hypothesis test."""
    statistic: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    significant: bool
    test_name: str

    def __str__(self) -> str:
        sig_str = "✓ SIGNIFICANT" if self.significant else "✗ NOT SIGNIFICANT"
        return (
            f"{self.test_name}: "
            f"stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, "
            f"d={self.effect_size:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"({sig_str})"
        )


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Data array [N]
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)
        random_state: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)

    # Compute point estimate
    point_estimate = statistic(data)

    # Bootstrap samples
    bootstrap_estimates = []
    n = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_estimates.append(statistic(sample))

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def compute_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size.

    d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - |d| ~ 0.5: medium
    - |d| > 0.8: large

    Args:
        group1: First group [N1]
        group2: Second group [N2]

    Returns:
        Cohen's d (can be negative)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return d


def paired_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
) -> StatisticalTest:
    """
    Paired t-test for matched samples.

    Use when comparing the same layers/conditions across GPN vs GAN.

    Args:
        group1: First group (e.g., GPN metrics) [N]
        group2: Second group (e.g., GAN metrics) [N]
        alpha: Significance level

    Returns:
        StatisticalTest object
    """
    assert len(group1) == len(group2), "Paired test requires equal sample sizes"

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(group1, group2)

    # Effect size
    effect_size = compute_cohens_d(group1, group2)

    # Bootstrap CI for difference
    differences = group1 - group2
    _, ci_lower, ci_upper = bootstrap_ci(differences)

    return StatisticalTest(
        statistic=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=(p_value < alpha),
        test_name="Paired t-test",
    )


def independent_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True,
) -> StatisticalTest:
    """
    Independent samples t-test.

    Use when comparing different runs/seeds.

    Args:
        group1: First group [N1]
        group2: Second group [N2]
        alpha: Significance level
        equal_var: Assume equal variances (Welch's t-test if False)

    Returns:
        StatisticalTest object
    """
    # Independent t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

    # Effect size
    effect_size = compute_cohens_d(group1, group2)

    # Bootstrap CI for mean difference
    def mean_diff(combined, n1):
        return np.mean(combined[:n1]) - np.mean(combined[n1:])

    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # Bootstrap
    rng = np.random.RandomState(42)
    boot_diffs = []
    for _ in range(10000):
        idx = rng.choice(len(combined), size=len(combined), replace=True)
        boot_sample = combined[idx]
        boot_diffs.append(mean_diff(boot_sample, n1))

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return StatisticalTest(
        statistic=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=(p_value < alpha),
        test_name="Independent t-test",
    )


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> StatisticalTest:
    """
    Permutation test for difference in means.

    Non-parametric alternative to t-test.
    Useful when distributions are not normal.

    Args:
        group1: First group [N1]
        group2: Second group [N2]
        n_permutations: Number of permutations
        alpha: Significance level
        random_state: Random seed

    Returns:
        StatisticalTest object
    """
    rng = np.random.RandomState(random_state)

    # Observed difference
    observed_diff = np.mean(group1) - np.mean(group2)

    # Combine groups
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        # Shuffle and split
        shuffled = rng.permutation(combined)
        perm_group1 = shuffled[:n1]
        perm_group2 = shuffled[n1:]
        perm_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))

    perm_diffs = np.array(perm_diffs)

    # P-value (two-tailed)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # Effect size
    effect_size = compute_cohens_d(group1, group2)

    # CI from permutation distribution
    ci_lower = np.percentile(perm_diffs, 2.5)
    ci_upper = np.percentile(perm_diffs, 97.5)

    return StatisticalTest(
        statistic=observed_diff,
        p_value=p_value,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=(p_value < alpha),
        test_name="Permutation test",
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], float]:
    """
    Bonferroni correction for multiple comparisons.

    Most conservative correction: reject if p < alpha / n_tests.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        Tuple of (significant_flags, corrected_alpha)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = [p < corrected_alpha for p in p_values]

    return significant, corrected_alpha


def fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.

    Less conservative than Bonferroni, controls false discovery rate.

    Args:
        p_values: List of p-values
        alpha: False discovery rate

    Returns:
        Tuple of (significant_flags, adjusted_p_values)
    """
    n_tests = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    # BH critical values
    critical_values = (np.arange(1, n_tests + 1) / n_tests) * alpha

    # Find largest i where p[i] <= critical_value[i]
    significant_sorted = sorted_p <= critical_values
    if np.any(significant_sorted):
        threshold_idx = np.where(significant_sorted)[0][-1]
        threshold = sorted_p[threshold_idx]
    else:
        threshold = 0.0

    # Adjusted p-values
    adjusted_p = np.minimum.accumulate(
        sorted_p * n_tests / (np.arange(1, n_tests + 1))[::-1]
    )[::-1]

    # Unsort
    original_order = np.argsort(sorted_idx)
    adjusted_p = adjusted_p[original_order]

    significant = [p <= threshold for p in p_values]

    return significant, adjusted_p.tolist()


def compare_metrics_across_layers(
    gpn_metrics: List[float],
    gan_metrics: List[float],
    metric_name: str,
    alpha: float = 0.05,
    correction: str = 'bonferroni',
) -> StatisticalTest:
    """
    Compare a metric across layers between GPN and GAN.

    Uses paired test since layers correspond.

    Args:
        gpn_metrics: GPN metric values per layer
        gan_metrics: GAN metric values per layer
        metric_name: Name of metric
        alpha: Significance level
        correction: Multiple testing correction ('bonferroni', 'fdr', or None)

    Returns:
        StatisticalTest object
    """
    gpn_array = np.array(gpn_metrics)
    gan_array = np.array(gan_metrics)

    # Paired t-test
    result = paired_t_test(gpn_array, gan_array, alpha=alpha)
    result.test_name = f"Paired t-test ({metric_name})"

    return result


def power_analysis(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute statistical power for detecting an effect.

    Power = probability of correctly rejecting null hypothesis
    when alternative is true.

    Args:
        effect_size: Cohen's d
        n_samples: Sample size
        alpha: Significance level

    Returns:
        Statistical power (0 to 1)
    """
    from scipy.stats import norm

    # Two-tailed test
    critical_value = norm.ppf(1 - alpha / 2)

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_samples / 2)

    # Power
    power = 1 - norm.cdf(critical_value - ncp) + norm.cdf(-critical_value - ncp)

    return power


def summarize_statistical_tests(
    tests: List[StatisticalTest],
    metric_names: List[str],
) -> str:
    """
    Generate formatted summary of statistical tests.

    Args:
        tests: List of test results
        metric_names: Names of metrics tested

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL SIGNIFICANCE TESTING")
    lines.append("=" * 80)
    lines.append("")

    # Extract p-values for multiple testing correction
    p_values = [test.p_value for test in tests]
    sig_bonf, alpha_bonf = bonferroni_correction(p_values)
    sig_fdr, _ = fdr_correction(p_values)

    lines.append(f"Tests conducted: {len(tests)}")
    lines.append(f"Bonferroni-corrected α: {alpha_bonf:.4f}")
    lines.append("")

    # Report each test
    for i, (test, metric_name) in enumerate(zip(tests, metric_names)):
        lines.append(f"Metric: {metric_name}")
        lines.append(f"  {test}")
        lines.append(f"  Bonferroni: {'✓ SIGNIFICANT' if sig_bonf[i] else '✗ NOT SIGNIFICANT'}")
        lines.append(f"  FDR: {'✓ SIGNIFICANT' if sig_fdr[i] else '✗ NOT SIGNIFICANT'}")
        lines.append("")

    # Summary
    n_sig_uncorr = sum(test.significant for test in tests)
    n_sig_bonf = sum(sig_bonf)
    n_sig_fdr = sum(sig_fdr)

    lines.append("-" * 80)
    lines.append("SUMMARY:")
    lines.append(f"  Significant (uncorrected): {n_sig_uncorr}/{len(tests)}")
    lines.append(f"  Significant (Bonferroni): {n_sig_bonf}/{len(tests)}")
    lines.append(f"  Significant (FDR): {n_sig_fdr}/{len(tests)}")
    lines.append("")

    # Effect sizes
    large_effects = [i for i, test in enumerate(tests) if abs(test.effect_size) > 0.8]
    if large_effects:
        lines.append("Metrics with LARGE effect sizes (|d| > 0.8):")
        for i in large_effects:
            lines.append(f"  - {metric_names[i]}: d = {tests[i].effect_size:.3f}")

    lines.append("=" * 80)

    return "\n".join(lines)


def validate_sample_size(
    effect_size: float,
    desired_power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """
    Compute required sample size for desired statistical power.

    Args:
        effect_size: Expected Cohen's d
        desired_power: Target power (default 0.8)
        alpha: Significance level

    Returns:
        Required sample size per group
    """
    # Binary search for required n
    n_low, n_high = 2, 10000

    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        power = power_analysis(effect_size, n_mid, alpha)

        if power < desired_power:
            n_low = n_mid
        else:
            n_high = n_mid

    return n_high
