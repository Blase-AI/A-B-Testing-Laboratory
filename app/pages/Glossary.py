import streamlit as st


st.set_page_config(page_title="Glossary", layout="wide")

st.title("Glossary")
st.caption("A short, practical reference for the A/B Testing Laboratory.")

st.markdown("---")

with st.expander("What is an A/B test?", expanded=True):
    st.markdown(
        """
An **A/B test** compares two variants (A and B) of a product or process.
You collect a metric for each group and decide whether the observed difference is likely real.

Typical outcomes:
- **No clear difference**: keep the current version or keep experimenting
- **B improves the metric**: ship B (with appropriate risk checks)
- **B harms the metric**: stop or rollback
"""
    )

with st.expander("Key terms (practical)", expanded=True):
    st.markdown(
        """
- **Null hypothesis (H0)**: "no difference between groups"
- **Alternative hypothesis (H1)**: "there is a difference" (or "B > A")
- **p-value**: probability of observing results as extreme as yours **if H0 were true**
- **alpha (α)**: significance level; decision threshold for p-values (common: 0.05)
- **effect size**: how big the difference is (e.g. **Cohen’s d** for continuous metrics)
- **confidence interval (CI)**: plausible range for the estimate (e.g. difference in means)
"""
    )

st.subheader("Methods in this app")

with st.expander("Classic A/B (t-test / Mann-Whitney)", expanded=True):
    st.markdown(
        """
**Use when**
- You need a standard frequentist hypothesis test
- Your metric is continuous (revenue, time on page, etc.)

**What it does**
- Checks normality (Shapiro–Wilk) and variance equality (Levene)
- Uses:
  - **t-test** when assumptions look OK
  - **Mann–Whitney U** when normality is questionable

**Parameters**
- **α (significance level)**: smaller α → stricter evidence required
- **Minimum sample size**: UI hint for stability (not enforced by the math)
"""
    )

with st.expander("Bayesian (Beta-Binomial style demo)", expanded=False):
    st.markdown(
        """
**Use when**
- You want statements like "Probability that B is better than A"
- You have a binary metric (conversion), or a proxy for "success"

**What it does**
- Models conversion rate for each group with a **Beta** posterior
- Computes **P(B > A)** and an interval for the difference

**Parameters**
- **Alpha prior / Beta prior**: prior strength; larger values → stronger prior pull
"""
    )

with st.expander("Bootstrap", expanded=False):
    st.markdown(
        """
**Use when**
- You want robust intervals without distribution assumptions
- Your metric is skewed or has outliers

**What it does**
- Resamples with replacement many times
- Estimates distributions for means and differences

**Parameters**
- **Number of iterations**: more → smoother estimates, slower runtime
- **Confidence level**: e.g. 0.95 means 95% interval
"""
    )

with st.expander("Sequential (SPRT)", expanded=False):
    st.markdown(
        """
**Use when**
- You want the option to stop early when evidence is strong

**What it does**
- Updates the log-likelihood ratio after each observation
- Stops when boundaries are crossed (accept H0 or H1), otherwise continues

**Parameters**
- **Stop threshold**: maximum observations used in the simulation
- **μ0 / μ1, σ**: hypotheses and noise level for the simulated data
- **α**: type I error bound (false positive rate)
"""
    )

st.markdown("---")
st.subheader("How to read results")
st.markdown(
    """
- Prefer reporting **effect size + interval** over only p-values.
- If results are borderline, consider collecting more data or using sequential design.
- Always check practical significance: small p-value can still be a tiny change.
"""
)

