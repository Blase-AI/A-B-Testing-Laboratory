# A/B Testing Laboratory
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-2B7A78)](https://www.statsmodels.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)](https://seaborn.pydata.org/)
[![ArviZ](https://img.shields.io/badge/ArviZ-5C2D91)](https://python.arviz.org/)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Black](https://img.shields.io/badge/Black-000000?logo=black&logoColor=white)](https://black.readthedocs.io/)
[![Mypy](https://img.shields.io/badge/Mypy-2A6DB0)](https://mypy-lang.org/)

A/B Testing Laboratory is an interactive web app built with Streamlit that empowers users to perform **classic, Bayesian, bootstrap, and sequential A/B tests** with ease. Whether you're a data scientist, researcher, or marketer, this tool provides a one-stop solution for robust statistical analysis, dynamic visualizations, and actionable insights.  

---
## Key Features

### **1. Multiple Testing Methods**
- **Classic A/B Testing**: T-tests, Mann-Whitney U-tests, normality/variance checks, and effect size (Cohen’s d).  
- **Bayesian Testing**: Posterior distributions, HDI intervals, and probability of B > A.  
- **Bootstrap Analysis**: Non-parametric confidence intervals and difference distributions.  
- **Sequential Testing**: Early stopping with SPRT (Sequential Probability Ratio Test).  
- **Experiment Comparison**: Compare multiple A/B tests side-by-side.  

### **2. Interactive & User-Friendly**
- **Dynamic Visualizations**: QQ-plots, boxplots, KDEs, ECDFs, and confidence interval plots.
- **Flexible Data Input**: Upload CSV files, use demo data, or manually enter values.
- **Real-Time Results**: Instantly visualize distributions, p-values, and effect sizes.

### **3. Advanced Utilities**
- **Built-in Guidelines**: Interpret results with explanations of p-values, test selection tips, and statistical concepts.
- **Export Results**: Download summaries and graphs for reporting.
- **Power Analysis**: Estimate required sample sizes for desired test power.

### **4. Streamlit-Powered UI**
- Responsive, wide-mode layout optimized for data exploration.  
- Sidebar controls for test parameters and data input.  

---

## How to Use

### **Installation**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Blase-AI/A-B-Testing-Laboratory.git  
   cd A-B-Testing-Laboratory
   ```
2. Clone the repository:
   ```bash
   pip install -r requirements.txt  
   ```
## Launch the App
  ```bash
  streamlit run app/streamlit_app.py
  ```

## Development
```bash
pip install -r requirements.txt
pytest
ruff check .
black --check .
mypy app
```
## Steps to Run a Test
1. Select a test type (e.g., Bayesian, Bootstrap) from the sidebar.
2. Upload data (CSV), use demo data, or input values manually.
3. Adjust parameters (e.g., significance level, priors, bootstrap iterations).
4. Explore results: Interactive plots, metrics, and summaries will auto-update.
5. Compare experiments or export results as CSV.

## Examples
You can also explore examples in the examples folder. Inside, you'll find classic types of tests implemented in Jupyter notebooks using popular libraries. These examples demonstrate similar results and provide a practical reference for understanding how the test work in different contexts.

## Why This Project Stands Out
- All-in-One Testing: No need to switch between tools—compare methods in one interface.
- Educational Tool: Guidelines and visualizations make complex stats accessible.
- Production-Ready: Built with modular classes (ABTest, BayesianABTest, etc.) for easy extension.

## Screenshots
### Classic A/B Test  
![Classic A/B Test](screenshots/classic.png)  
*Classic A/B test*  

### Bayesian Results  
![Bayesian Results](screenshots/bayesian.png)  
*Bayesian*  

### Bootstrap Results  
![Bootstrap Results](screenshots/Bootstrap.png)  
*Bootstrap*  

### Sequential Results  
![Sequential Results](screenshots/Sequential.png)  
*Sequential*  
