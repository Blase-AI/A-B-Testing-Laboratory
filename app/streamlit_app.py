import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from ab_testing import ABTest, BayesianABTest, BootstrapABTest, Sequential_Testing, SequentialTest
from scipy import stats

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
st.set_page_config(page_title="A/B Testing Lab", layout="wide")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data
def load_data(uploader, group_name):
    """Load data from a CSV file with error handling and validation"""
    try:
        if uploader is not None:
            df = pd.read_csv(uploader)
            data = df.iloc[:, 0].values
            data = [float(x) for x in data]
            st.success(f"Data for group {group_name} successfully loaded!")
            logger.info(f"Data for group {group_name} loaded, count: {len(data)}")
            return np.array(data)
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data for group {group_name}: {str(e)}")
        return None

def validate_manual_input(raw_input, group_name):
    """Validate manual input: check that all values are numeric"""
    try:
        values = [float(x.strip()) for x in raw_input.split(",") if x.strip()]
        logger.info(f"Manual input for group {group_name} successfully processed, count: {len(values)}")
        return np.array(values)
    except ValueError as e:
        st.error(f"Invalid input data for group {group_name}: {str(e)}")
        logger.error(f"Validation error for manual input for group {group_name}: {str(e)}")
        return None

def display_results_table(results):
    """Display a summary table of experiment results and allow downloading as CSV"""
    df_results = pd.DataFrame(results)
    st.subheader("Summary Table of Experiment Results")
    st.dataframe(df_results)
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", data=csv, file_name='experiment_results.csv', mime='text/csv')

def show_guidelines():
    """Show guidelines for interpreting and choosing tests"""
    st.header("📖 A/B Testing Guidelines")

    with st.expander("❓ What is p-value?"):
        st.markdown("""
        - **p-value** - the probability of obtaining the observed results assuming that the null hypothesis (H0) is true.
        - **How to interpret**:
            - p-value < 0.05: statistically significant differences (reject H0)
            - p-value >= 0.05: differences are not significant
        - ⚠️ Not interpreted as the probability of the truth of hypotheses!
        """)
    
    with st.expander("📊 Classic A/B Test"):
        st.markdown("""
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
        """)
    
    with st.expander("🔮 Bayesian Test"):
        st.markdown("""
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
        """)
    
    with st.expander("🎢 Bootstrap Test"):
        st.markdown("""
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
        """)
    
    with st.expander("⏱ Sequential Test"):
        st.markdown("""
        **When to use**:
        - Test can be stopped early if a clear result is obtained
        - Limited resources for data collection
        
        **Method**:
        - Sequential calculation of the log-likelihood ratio (LR)
        - Checking for crossing decision boundaries
        
        **Pros**:
        - Saves time/money
        - Dynamic decision-making
        """)
    
    st.markdown("---")
    st.subheader("📝 Recommendations for Choosing a Test")

    
    with st.expander("📌 Usage Examples"):
        st.markdown("""
        | Situation                    | Recommended Method        |
        |------------------------------|---------------------------|
        | Large samples, normal data   | Classic t-test            |
        | Small samples, binary metrics| Bayesian test             |
        | Unknown distribution         | Bootstrap or U-test       |
        | Gradual data collection      | Sequential test           |
        """)


def main():
    st.title("🧪 A/B Testing Laboratory")
    st.sidebar.header("Experiment Settings")

    test_type = st.sidebar.selectbox(
        "Test Type",
        ["Classic A/B", "Bayesian", "Bootstrap", "Sequential", "Experiment Comparison"]
    )
    st.sidebar.header("Test Parameters")

    if test_type == "Classic A/B":
        alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.2, 0.05, key="alpha_classic")
        min_sample_size = st.sidebar.number_input("Minimum sample size", 10, 1000, 30, key="min_sample")

    elif test_type == "Bayesian":
        col1, col2 = st.sidebar.columns(2)
        alpha_prior = col1.number_input("Alpha prior", 0.1, 10.0, 1.0, key="alpha_prior")
        beta_prior = col2.number_input("Beta prior", 0.1, 10.0, 1.0, key="beta_prior")

    elif test_type == "Bootstrap":
        n_bootstrap = st.sidebar.number_input("Number of iterations", 100, 100000, 10000, key="n_bootstrap")
        ci_level = st.sidebar.slider("Confidence level", 0.8, 0.99, 0.95, key="ci_level")

    elif test_type == "Sequential":
        stop_threshold = st.sidebar.number_input("Stop threshold", 1, 100, 10)
        st.sidebar.number_input("Stop threshold", 1, 100, 10, key="stop_threshold")

    if test_type != "Sequential":
        data_input_method = st.sidebar.radio("Data input method:", ["Demo Data", "Upload CSV", "Manual Input"])
    
    if test_type == "Experiment Comparison":
        st.header("📊 Multiple Experiment Comparison")
        num_experiments = st.number_input("Number of experiments to compare", min_value=2, max_value=10, value=2, step=1)
        experiment_results = []
        
        for i in range(int(num_experiments)):
            st.subheader(f"Experiment {i+1}")
            exp_data_input_method = st.radio(
                f"Data input method for experiment {i+1}:",
                ["Demo Data", "Upload CSV", "Manual Input"],
                key=f"input_method_{i}"
            )
            data_a, data_b = None, None
            
            if exp_data_input_method == "Demo Data":
                mean_shift = i * 2
                data_a = np.random.normal(50, 10, 1000)
                data_b = np.random.normal(50 + mean_shift + 5, 10, 1000)
                st.info(f"Experiment {i+1}: Group A ~ N(50,10), Group B ~ N({50+mean_shift+5},10)")
                logger.info(f"Experiment {i+1}: demo data created with mean_shift={mean_shift}")
            
            elif exp_data_input_method == "Upload CSV":
                col1, col2 = st.columns(2)
                with col1:
                    a_file = st.file_uploader(f"CSV for group A of experiment {i+1}", type=["csv"], key=f"file_a_{i}")
                    data_a = load_data(a_file, f"A (experiment {i+1})")
                with col2:
                    b_file = st.file_uploader(f"CSV for group B of experiment {i+1}", type=["csv"], key=f"file_b_{i}")
                    data_b = load_data(b_file, f"B (experiment {i+1})")
            
            elif exp_data_input_method == "Manual Input":
                col1, col2 = st.columns(2)
                with col1:
                    raw_a = st.text_area(f"Data for group A (comma-separated) for experiment {i+1}", "50, 55, 60", key=f"raw_a_{i}")
                    data_a = validate_manual_input(raw_a, f"A (experiment {i+1})")
                with col2:
                    raw_b = st.text_area(f"Data for group B (comma-separated) for experiment {i+1}", "55, 60, 65", key=f"raw_b_{i}")
                    data_b = validate_manual_input(raw_b, f"B (experiment {i+1})")
            
            if data_a is not None and data_b is not None:
                if len(data_a) < 10 or len(data_b) < 10:
                    st.warning("⚠️ Recommended to use samples with ≥10 observations")
                    logger.warning(f"Experiment {i+1}: insufficient observations. Group A: {len(data_a)}, Group B: {len(data_b)}")
                else:
                    ab = ABTest(data_a, data_b)
                    test_res = ab.perform_t_test(alpha=alpha)
                    cohens_d = ab.calculate_cohens_d()
                    result_summary = {
                        "Experiment": f"Experiment {i+1}",
                        "Mean A": np.mean(data_a),
                        "Mean B": np.mean(data_b),
                        "Difference": np.mean(data_b) - np.mean(data_a),
                        "p-value": test_res['p_value'],
                        "Cohen's d": cohens_d
                    }
                    experiment_results.append(result_summary)
                    st.success(f"Experiment {i+1} completed successfully")
                    logger.info(f"Experiment {i+1}: p-value={test_res['p_value']:.4f}, Cohen's d={cohens_d:.3f}")
        
        if experiment_results:
            display_results_table(experiment_results)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=pd.DataFrame(experiment_results), x="Experiment", y="p-value", marker="o", ax=ax)
            ax.set_title("p-value Dynamics Across Experiments")
            st.pyplot(fig)
    
    else:
        data_a, data_b = None, None
        if test_type != "Sequential":
            with st.expander("⚙️ Data Settings", expanded=True):
                if data_input_method == "Demo Data":
                    data_a = np.random.normal(50, 10, 1000)
                    data_b = np.random.normal(55, 10, 1000)
                    st.info("Using demo data: Group A ~ N(50,10), Group B ~ N(55,10)")
                    logger.info("Demo data used for groups A and B")
                elif data_input_method == "Upload CSV":
                    col1, col2 = st.columns(2)
                    with col1:
                        a_file = st.file_uploader("CSV for group A", type=["csv"])
                        data_a = load_data(a_file, "A")
                    with col2:
                        b_file = st.file_uploader("CSV for group B", type=["csv"])
                        data_b = load_data(b_file, "B")
                elif data_input_method == "Manual Input":
                    col1, col2 = st.columns(2)
                    with col1:
                        raw_a = st.text_area("Data for group A (comma-separated)", "50, 55, 60")
                        data_a = validate_manual_input(raw_a, "A")
                    with col2:
                        raw_b = st.text_area("Data for group B (comma-separated)", "55, 60, 65")
                        data_b = validate_manual_input(raw_b, "B")
                if data_a is not None and data_b is not None:
                    if len(data_a) < 10 or len(data_b) < 10:
                        st.warning("⚠️ Recommended to use samples with ≥10 observations")
                        logger.warning("Insufficient observations for groups A or B")
        
        try:
            if test_type == "Classic A/B" and data_a is not None and data_b is not None:
                st.header("🔍 Classic A/B Test")
                with st.spinner("Performing analysis..."):
                    ab = ABTest(data_a, data_b)
                    with st.expander("📊 Descriptive Statistics", expanded=True):
                        cols = st.columns(3)
                        cols[0].metric("Mean A", f"{np.mean(data_a):.2f}")
                        cols[1].metric("Mean B", f"{np.mean(data_b):.2f}")
                        cols[2].metric("Difference", f"{np.mean(data_b)-np.mean(data_a):.2f}")

                    with st.expander("📈 Statistical Analysis"):
                        norm = ab.test_normality()
                        test_res = ab.perform_t_test()
                        col1, col2 = st.columns(2)
                        col1.metric("Normality A", "✅" if norm['Group A']['normal'] else "❌", f"p={norm['Group A']['p_value']:.4f}")
                        col2.metric("Normality B", "✅" if norm['Group B']['normal'] else "❌", f"p={norm['Group B']['p_value']:.4f}")
                        st.subheader("Test Results")
                        st.write(f"**Method:** {test_res['test']}")
                        st.metric("p-value", f"{test_res['p_value']:.4f}", help="Significance of differences")
                        d = ab.calculate_cohens_d()
                        effect_size = "small" if abs(d) < 0.2 else "medium" if abs(d) < 0.5 else "large"
                        st.metric("Cohen's d", f"{d:.3f} ({effect_size} effect)")

                        st.write(f"Used parameters:")
                        st.write(f"- Significance level: {alpha}")
                        st.write(f"- Minimum sample size: {min_sample_size}")

                    with st.expander("📉 Distribution Visualization"):
                        st.pyplot(ab.plot_distribution()) 

                    with st.expander("📉 Boxplot"):
                        st.pyplot(ab.plot_boxplot())

                    with st.expander("📉 Mean Difference Plot with Confidence Interval"):
                        st.pyplot(ab.plot_mean_difference_ci(confidence=0.95))

                    with st.expander("📉 Q-Q Plot for Normality Check"):
                        st.pyplot(ab.plot_qq())

            elif test_type == "Bayesian" and data_a is not None and data_b is not None:
                st.header("🔮 Bayesian Analysis")
                with st.spinner("Performing Bayesian analysis..."):
                    n_a, success_a = len(data_a), sum(data_a > np.mean(data_a))
                    n_b, success_b = len(data_b), sum(data_b > np.mean(data_b))
                    bab = BayesianABTest(n_a, success_a, n_b, success_b, alpha_prior=alpha_prior, beta_prior=beta_prior)
                    prob = bab.compute_prob_B_better()
                    diff_stats = bab.compute_difference_stats()
                    cols = st.columns(3)
                    cols[0].metric("Probability B > A", f"{prob*100:.1f}%")
                    cols[1].metric("HDI Lower", f"{diff_stats['hdi_interval'][0]:.3f}")
                    cols[2].metric("HDI Upper", f"{diff_stats['hdi_interval'][1]:.3f}")

                    with st.expander("📊 Posterior Distributions"):
                        st.pyplot(bab.plot_posteriors())

                    with st.expander("📊 Boxplot"):
                        st.pyplot(bab.plot_boxplot())

                    with st.expander("📊 KDE for Posterior Distributions"):
                        st.pyplot(bab.plot_kde())

                    with st.expander("📊 KDE Plot for Difference Between Groups"):
                        st.pyplot(bab.plot_difference_hist_kde())

                    with st.expander("📊 CDF Plot for Difference Between Groups"):
                        st.pyplot(bab.plot_difference_cdf())
                    
            elif test_type == "Bootstrap" and data_a is not None and data_b is not None:
                st.header("🎢 Bootstrap Analysis")
                with st.spinner("Performing bootstrap analysis..."):
                    boot = BootstrapABTest(data_a, data_b)
                    with st.expander("📈 Results"):
                        means_a, means_b = boot.compute_bootstrap_means(n_bootstrap=n_bootstrap)
                        ci_a = boot.percentile_ci(means_a, ci=ci_level*100)
                        ci_b = boot.percentile_ci(means_b)
                        diff = boot.compute_bootstrap_diff()
                        ci_diff = boot.percentile_ci(diff)
                        cols = st.columns(3)
                        cols[0].metric("CI Group A", f"{ci_a[0]:.2f} - {ci_a[1]:.2f}")
                        cols[1].metric("CI Group B", f"{ci_b[0]:.2f} - {ci_b[1]:.2f}")
                        cols[2].metric("Difference (B-A)", f"{ci_diff[0]:.2f} - {ci_diff[1]:.2f}")

                    with st.expander("📊 Distribution of the difference in averages"):
                        fig = boot.plot_bootstrap_distributions()
                        st.pyplot(fig)

                    with st.expander("📊 KDE Plot"):
                        fig = boot.plot_bootstrap_diff_hist()
                        st.pyplot(fig)
                    
                    with st.expander("📊 ECDF Plot"):
                        fig = boot.plot_bootstrap_diff_ecdf()
                        st.pyplot(fig)

            elif test_type == "Sequential":
                st.header("⏱ Sequential Analysis")
                with st.expander("⚙️ Simulation Parameters"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        mu0 = st.number_input("μ for H0", value=0.0)
                        mu1 = st.number_input("μ for H1", value=1.0)
                    with col2:
                        sigma = st.number_input("σ", value=1.0, min_value=0.1)
                        n = st.number_input("Max observations", value=100, min_value=10)
                    with col3:
                        true_state = st.radio("True hypothesis", ["H0", "H1"])
                        alpha = st.slider("Significance level", 0.01, 0.2, 0.05)
                if st.button("🚦 Start Simulation"):
                    with st.spinner("Running simulation..."):
                        try:
                            simulator = Sequential_Testing.SequentialTestSimulator(
                                mu0=mu0, mu1=mu1, sigma=sigma, n=n,
                                alpha=alpha, true_state=true_state, verbose=False, stop_threshold=stop_threshold
                            )

                            decision, n_used, _, history = simulator.run()
                            st.subheader("Test Results")
                            cols = st.columns(3)
                            cols[0].metric("Decision", decision)
                            cols[1].metric("Observations Used", n_used)
                            cols[2].metric("Efficiency", f"{(n_used/n)*100:.1f}%")
                            fig = simulator.plot_history()
                            st.pyplot(fig)
                            
                            sim_results = simulator.run_simulations(n_simulations=50)
                            fig_sim = simulator.plot_simulation_results(sim_results)
                            st.subheader("Aggregated Simulation Results")
                            st.pyplot(fig_sim)
                        except Exception as e:
                            st.error(f"Simulation error: {str(e)}")
                            logger.error(f"Simulation error: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"An error occurred: {str(e)}")
            st.error("Check the correctness of input data and parameters")


    st.markdown("---")
    col_help = st.columns([1, 2, 1])
    with col_help[1]:
        if st.button(
            "📚 How to interpret results?",
            help="Open guidelines for testing methods and interpretation",
            use_container_width=True
        ):
            show_guidelines()

if __name__ == "__main__":
    main()
