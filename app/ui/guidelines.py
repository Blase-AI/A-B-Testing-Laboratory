import streamlit as st


def show_guidelines() -> None:
    """Show guidelines for interpreting and choosing tests."""
    st.header("A/B Testing Guidelines")

    with st.expander("What is p-value?"):
        st.markdown(
            """
        - **p-value** - the probability of obtaining the observed results assuming that the null hypothesis (H0) is true.
        - **How to interpret**:
            - p-value < 0.05: statistically significant differences (reject H0)
            - p-value >= 0.05: differences are not significant
        - Not interpreted as the probability of the truth of hypotheses!
        """
        )

    with st.expander("Classic A/B Test"):
        st.markdown(
            """
        **When to use**:
        - Data is normally distributed (check with Shapiro-Wilk test)
        - Groups have approximately equal variances (Levene's test)

        **Method**:
        1. Normality check
        2. Variance equality check
        3. Choose between t-test (parametric) and Mann-Whitney U-test

        **Pros**:
        - Simple interpretation
        - Widely accepted in the scientific community
        """
        )

    with st.expander("Bayesian Test"):
        st.markdown(
            """
        **When to use**:
        - Need the probability of one group being superior to the other
        - Working with small samples
        - Interested in HDI (Highest Density Interval)

        **Method**:
        - Estimation of posterior distributions for groups
        - Calculation of P(B > A)

        **Pros**:
        - Intuitive interpretation ("Probability that B is better: 85%")
        - Does not require frequentist statistics
        """
        )

    with st.expander("Bootstrap Test"):
        st.markdown(
            """
        **When to use**:
        - Assumptions of parametric tests are violated
        - Need reliable confidence intervals
        - Analysis of non-parametric metrics (medians, percentiles)

        **Method**:
        - Repeated resampling with replacement
        - Construction of empirical distribution of statistics

        **Pros**:
        - No assumptions about distribution
        - Robust to outliers
        """
        )

    with st.expander("Sequential Test"):
        st.markdown(
            """
        **When to use**:
        - Test can be stopped early if a clear result is obtained
        - Limited resources for data collection

        **Method**:
        - Sequential calculation of the log-likelihood ratio (LR)
        - Checking for crossing decision boundaries

        **Pros**:
        - Saves time/money
        - Dynamic decision-making
        """
        )

    st.markdown("---")
    st.subheader("Recommendations for Choosing a Test")

    with st.expander("Usage Examples"):
        st.markdown(
            """
        | Situation                    | Recommended Method        |
        |------------------------------|---------------------------|
        | Large samples, normal data   | Classic t-test            |
        | Small samples, binary metrics| Bayesian test             |
        | Unknown distribution         | Bootstrap or U-test       |
        | Gradual data collection      | Sequential test           |
        """
        )

