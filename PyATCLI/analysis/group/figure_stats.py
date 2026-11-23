"""
analysis/group/figure_stats.py
------------------------------
The numbers the figures of a group analysis are made of, worked out here so that the window that draws them, a script and the data file beside the picture
all give the same: the mean each subject contributes to a group of markers, the mean, standard error and quartiles of a group's subjects, the points of a
normal Q-Q plot, and how a p-value is written as stars. (Drawing them is the viewer's: PyAT's figures package.)
"""

import numpy as np
import scipy.stats as st


def pooled_subject_values(means, measure, groups):
    """
    {label: {subject: value}} for one measure, each marker group pooled into one value per subject.

    `means` is the per-subject, per-marker table of a group analysis (columns "measure", "subject", "marker", "mean"; GroupResults.means), `measure` the measure
    asked about ("auc", "peak", ...), `groups` an ordered {label: [marker names]} of the markers pooled under each label. A subject contributes to a label the
    average of whichever of that label's markers it has a value for, and is left out of it when it has none; every label is in the answer, with no subjects
    if none had data.
    """
    sub = means[means["measure"] == measure] if len(means) else means
    by_subject = {}
    if len(sub):
        for _, row in sub.iterrows():
            by_subject.setdefault(row["subject"], {})[row["marker"]] = row["mean"]
    out = {label: {} for label in groups}
    for subject, per_marker in by_subject.items():
        for label, members in groups.items():
            values = [per_marker[m] for m in members if m in per_marker]
            if values:
                out[label][subject] = float(np.mean(values))
    return out


def summary_stats(values):
    """
    The descriptive statistics of a group of values (a measure's value for each subject, say), as plain floats:

        {"n", "mean", "sem", "q1", "median", "q3"}

    `sem` is the sample standard deviation (ddof=1) over the square root of n, None with fewer than two values; every other number is None for no values.
    """
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": None, "sem": None, "q1": None, "median": None, "q3": None}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"n": n, "mean": float(np.mean(values)), "sem": float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None,
            "q1": float(q1), "median": float(median), "q3": float(q3)}


def normal_qq(values):
    """
    What a normal Q-Q plot of `values` (a model's residuals) draws: {"theoretical", "ordered", "slope", "intercept"} — the theoretical quantiles of a normal
    distribution and the values in order (the points), and the least-squares line through them ("slope" * theoretical + "intercept": the reference line).
    """
    (theoretical, ordered), (slope, intercept, _r) = st.probplot(np.asarray(values, dtype=float), dist="norm")
    return {"theoretical": theoretical, "ordered": ordered, "slope": float(slope), "intercept": float(intercept)}


def significance_stars(p):
    """A p-value as the stars a figure prints: "***" below 0.001, "**" below 0.01, "*" below 0.05, "ns" otherwise, "" for a p that is not a number."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
