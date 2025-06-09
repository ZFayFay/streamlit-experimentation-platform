# enhanced_ab_testing_portfolio.py
# Complete A/B Testing Portfolio Implementation with Expert Commentary
# Run: streamlit run enhanced_ab_testing_portfolio.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced A/B Testing Portfolio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main header style */
    .main-header {
        font-size: 3rem;
        color: #1E90FF; /* DodgerBlue */
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px #cccccc;
    }
    /* Subheader for more descriptive text */
    .subheader-text {
        font-size: 1.2rem;
        color: #4682B4; /* SteelBlue */
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Expander styling for a cleaner look */
    .stExpander {
        border: 1px solid #e6e9ef;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    /* Metric label styling */
    .stMetric > label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Core Functions ---

class MultiArmedBandit:
    """Thompson Sampling Multi-Armed Bandit Implementation"""
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.history = []

    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: int):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        self.history.append({'arm': arm, 'reward': reward})

    def get_estimated_rates(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def calculate_regret(self, true_rates: List[float]) -> float:
        history_df = pd.DataFrame(self.history)
        if history_df.empty:
            return 0
        
        optimal_rate = max(true_rates)
        # Calculate regret for each pull
        history_df['regret'] = optimal_rate - history_df['arm'].apply(lambda x: true_rates[x])
        return history_df['regret'].sum()

def design_binomial_experiment(mde: float, p_0: float, alpha: float, beta: float) -> int:
    """Calculate required sample size for a binomial test using variance of the mean."""
    if mde <= 0 or p_0 <= 0 or (p_0 + mde) >= 1:
        return float('inf')
    
    p_1 = p_0 + mde
    p_bar = (p_0 + p_1) / 2
    var = p_bar * (1 - p_bar)
    
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(1 - beta)
    
    n = ((z_alpha + z_beta)**2 * 2 * var) / (mde**2)
    return int(np.ceil(n))


@st.cache_data
def generate_synthetic_data(n_users: int, seed: int, base_ctr: float, uplift: float, population_params: Dict) -> pd.DataFrame:
    """Generate a realistic synthetic dataset for A/B testing."""
    np.random.seed(seed)
    
    users = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.normal(35, 12, n_users).clip(18, 75).astype(int),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_users, p=[0.48, 0.49, 0.03]),
        'device_type': np.random.choice(list(population_params['device_dist'].keys()), n_users, p=list(population_params['device_dist'].values())),
        'customer_segment': np.random.choice(list(population_params['segment_dist'].keys()), n_users, p=list(population_params['segment_dist'].values())),
    })
    
    # Pre-experiment covariate for CUPED
    users['pre_experiment_engagement'] = (users['age'] / 40 + np.random.normal(0, 0.2, n_users)).clip(0)

    users['treatment_group'] = np.random.choice(['Control', 'Treatment'], size=n_users, p=[0.5, 0.5])
    
    # Base probability with some noise from the covariate
    base_prob = base_ctr + (users['pre_experiment_engagement'] - users['pre_experiment_engagement'].mean()) * 0.05
    
    # Introduce interaction effect for HTE analysis (uplift is higher for younger users)
    age_effect = (40 - users['age']) / 1000
    treatment_effect = uplift + age_effect
    
    users['conversion_prob'] = np.where(users['treatment_group'] == 'Control', base_prob, base_prob + treatment_effect)
    
    noise = np.random.normal(0, 0.005, n_users)
    users['conversion_prob'] = (users['conversion_prob'] + noise).clip(0.001, 0.999)
    users['converted'] = np.random.binomial(1, users['conversion_prob'])
    
    start_date = datetime(2025, 1, 1)
    users['timestamp'] = start_date + pd.to_timedelta(np.random.randint(0, 30*24*60, n_users), unit='m')
    
    return users.drop(['conversion_prob'], axis=1)

def calculate_frequentist_results(df: pd.DataFrame, metric_col: str = 'converted') -> Dict:
    """Perform a comprehensive frequentist statistical analysis."""
    control = df[df['treatment_group'] == 'Control'][metric_col]
    treatment = df[df['treatment_group'] == 'Treatment'][metric_col]
    n_control, n_treatment = len(control), len(treatment)

    if n_control == 0 or n_treatment == 0: return {}

    conv_control, conv_treatment = control.mean(), treatment.mean()
    if (n_control + n_treatment) == 0: return {}
    
    pooled_prob = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    if pooled_prob <= 0 or pooled_prob >= 1: return {}
    
    # Z-test for proportions
    pooled_se = np.sqrt(pooled_prob * (1 - pooled_prob) * (1/n_control + 1/n_treatment))
    z_score = (conv_treatment - conv_control) / pooled_se if pooled_se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Confidence Interval
    diff = conv_treatment - conv_control
    se_diff = np.sqrt((conv_control * (1 - conv_control) / n_control) + (conv_treatment * (1 - conv_treatment) / n_treatment))
    ci_low, ci_high = stats.norm.interval(0.95, loc=diff, scale=se_diff)
    
    relative_lift = diff / conv_control if conv_control > 0 else float('inf')
    
    return {
        'control_conversion': conv_control, 'treatment_conversion': conv_treatment,
        'absolute_lift': diff, 'relative_lift': relative_lift,
        'z_score': z_score, 'p_value': p_value, 'significant': p_value < 0.05,
        'ci': [ci_low, ci_high]
    }

def run_multi_test_simulations(n_simulations: int, n_users: int, base_ctr: float, uplift: float, population_params: Dict) -> Dict:
    """Run multiple A/A or A/B test simulations to check power or false positive rate."""
    p_values = []
    prog_bar = st.progress(0, text=f"Running {n_simulations} simulations...")
    for i in range(n_simulations):
        df_sim = generate_synthetic_data(n_users, i, base_ctr, uplift, population_params)
        test_results = calculate_frequentist_results(df_sim)
        if test_results:
            p_values.append(test_results['p_value'])
        prog_bar.progress((i + 1) / n_simulations, text=f"Running {n_simulations} simulations... ({i+1}/{n_simulations})")
    prog_bar.empty()
    return {'p_values': p_values}

# --- Page Implementations ---

def show_overview(df, **kwargs):
    st.header("🏠 Dataset Overview & Sanity Checks")
    with st.expander("ℹ️ Understanding this Section: Exploratory Data Analysis (EDA)", expanded=True):
        st.markdown("""
        This initial step is crucial for any rigorous experiment. **Exploratory Data Analysis (EDA)** helps us:
        1.  **Verify Data Quality**: Ensure the data was logged correctly.
        2.  **Understand the Population**: Analyze the distribution of user segments (e.g., by device, location, or behavior).
        3.  **Check for Sample Ratio Mismatch (SRM)**: A critical sanity check. If the ratio of users in the control and treatment groups (e.g., 50/50) is not what we expect, it could signal a deep-seated bug in the randomization or assignment logic, which would invalidate the entire experiment.
        """)
    
    control_users = df['treatment_group'].value_counts().get('Control', 0)
    treatment_users = df['treatment_group'].value_counts().get('Treatment', 0)
    total_users = len(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{total_users:,}")
    col2.metric("Control Group", f"{control_users:,}", f"{control_users/total_users:.1%}" if total_users > 0 else "0.0%")
    col3.metric("Treatment Group", f"{treatment_users:,}", f"{treatment_users/total_users:.1%}" if total_users > 0 else "0.0%")
    
    if total_users > 0:
        srm_p_value = stats.chisquare([control_users, treatment_users]).pvalue
        if srm_p_value < 0.01:
            col4.error(f"SRM Alert! p={srm_p_value:.3f}")
        else:
            col4.success(f"No SRM. p={srm_p_value:.3f}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(df, names='customer_segment', title="Customer Segment Distribution", hole=0.3, color_discrete_sequence=px.colors.sequential.Blues_r), use_container_width=True)
    c2.plotly_chart(px.pie(df, names='device_type', title="Device Type Distribution", hole=0.3, color_discrete_sequence=px.colors.sequential.Oranges_r), use_container_width=True)
    st.dataframe(df.head())

def show_experiment_design(df, **kwargs):
    st.header("🎯 Experiment Design & Power Analysis")
    with st.expander("ℹ️ Understanding Experiment Design", expanded=True):
        st.markdown("""
        **Power analysis** is the process of determining the required sample size *before* an experiment begins. It's a crucial step to avoid running an underpowered test that has little chance of detecting a real effect, or an overpowered test that wastes resources.

        #### Key Levers:
        1.  **Significance Level (α)**: The probability of a false positive (Type I error). Typically set at 5%.
        2.  **Statistical Power (1-β)**: The probability of detecting a true effect (avoiding a false negative, or Type II error). Typically set at 80%.
        3.  **Minimum Detectable Effect (MDE)**: The smallest effect size you want to be able to detect. This is a business decision.
        
        **Real-World Scenario**: A company like **Amazon** wants to test if a new checkout button design increases conversion rate. The current rate is 10%. They decide they only care about detecting a relative lift of at least 2% (i.e., an absolute lift of 0.2%). Power analysis will tell them exactly how many millions of users they need to sample to confidently detect an effect of this size.
        """)

    col1, col2, col3 = st.columns(3)
    alpha = col1.slider("α (Significance Level)", 0.01, 0.20, 0.05, 0.01, key="alpha_design")
    power = col2.slider("Power (1-β)", 0.50, 0.99, 0.80, 0.01, key="power_design")
    mde = col3.slider("Minimum Detectable Effect (Relative Lift %)", 0.1, 10.0, 5.0, 0.1, key="mde_design") / 100

    base_ctr = kwargs['base_ctr']
    absolute_mde = base_ctr * mde
    min_samples_required = design_binomial_experiment(absolute_mde, base_ctr, alpha, 1 - power)

    st.subheader("Sample Size Calculation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Conversion Rate (CTR)", f"{base_ctr:.2%}")
    c2.metric("Required Sample Size (Per Group)", f"{min_samples_required:,}")
    c3.metric("Total Required Users", f"{min_samples_required*2:,}")

    with st.expander("🔍 Under the Hood: The Formula"):
        st.latex(r'''
        n = \frac{2 \cdot (Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \bar{p}(1-\bar{p})}{(p_1 - p_0)^2}
        ''')
        st.markdown(r"""
        Where:
        - $n$: sample size per group
        - $Z_{1-\alpha/2}$: Z-score for significance level (e.g., 1.96 for $\alpha=0.05$)
        - $Z_{1-\beta}$: Z-score for power (e.g., 0.84 for 80% power)
        - $p_0$: baseline conversion rate
        - $p_1$: conversion rate with MDE ($p_0 \cdot (1 + \text{MDE})$)
        - $\bar{p}$: average conversion rate ($(p_0+p_1)/2$)
        """)

    st.subheader("Simulating Statistical Power")
    if st.button("Run Power Simulation", key="run_power_sim"):
        st.info(f"Running 200 simulations with {min_samples_required*2} users each to see how often we correctly detect the MDE.")
        sim_results = run_multi_test_simulations(200, min_samples_required*2, base_ctr, absolute_mde, kwargs['population_params'])
        p_values_h1 = sim_results['p_values']
        empirical_power = np.mean(np.array(p_values_h1) < alpha)
        st.metric("Empirical Power (from simulation)", f"{empirical_power:.1%}", f"Target Power: {power:.1%}")
        fig = px.histogram(x=p_values_h1, nbins=30, title="P-value Distribution when H1 is True (should be skewed left)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Interpretation**: When a true effect exists (H1 is true), the p-values should be small. A left-skewed distribution indicates the test is working correctly. The empirical power is the percentage of these simulations where the p-value was below our significance level (α). It should be close to our target power.")

def show_aa_test_validation(df, **kwargs):
    st.header("🔬 A/A Test Validation")
    with st.expander("ℹ️ Understanding A/A Testing", expanded=True):
        st.markdown("""
        An **A/A test** involves showing identical versions of a product to two randomly assigned groups of users. The goal is to see *no statistical difference* between them. 
        
        #### Why is it important?
        1.  **Validates the Testing System**: It confirms that your randomization and assignment mechanisms are working correctly and are not biased.
        2.  **Checks for Data Quality Issues**: It can uncover problems in data logging or pipelines.
        3.  **Establishes a Baseline**: It helps you understand the natural variability in your metrics and ensures that your p-value distribution is uniform under the null hypothesis. If it's not uniform, your false positive rate isn't what you think it is.

        **Real-World Scenario**: Before launching a major new feature test, **Netflix** runs a large-scale A/A test. They check key metrics like "streaming hours per user." If the p-values from these tests are not uniformly distributed (e.g., too many are below 0.05), it signals a problem that must be fixed before any real A/B tests can be trusted.
        """)

    n_sims = st.slider("Number of A/A Simulations", 100, 1000, 500, 50, key="aa_sims")
    alpha_aa = st.slider("Significance Level (α)", 0.01, 0.20, 0.05, 0.01, key="alpha_aa")
    
    if st.button("Run A/A Test Validation", key="run_aa_validation"):
        st.info(f"Running {n_sims} A/A tests (uplift = 0) to check the false positive rate.")
        sim_results = run_multi_test_simulations(n_sims, len(df), kwargs['base_ctr'], 0, kwargs['population_params'])
        p_values_h0 = sim_results['p_values']
        
        fpr = np.mean(np.array(p_values_h0) < alpha_aa)
        st.metric("False Positive Rate (FPR)", f"{fpr:.2%}", f"Expected Rate: {alpha_aa:.2%}")
        
        fig = px.histogram(x=p_values_h0, histnorm='probability density', title="P-value Distribution under the Null Hypothesis (H0)")
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Uniform Distribution Line")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Interpretation**: Under the null hypothesis (no true effect), p-values should be uniformly distributed between 0 and 1. This means you should see a p-value < 0.05 about 5% of the time, a p-value < 0.10 about 10% of the time, and so on. The False Positive Rate (FPR) should be very close to your chosen significance level (α).")

def show_classic_ab_test(df, **kwargs):
    st.header("📈 Classic A/B Test Analysis (Frequentist)")
    with st.expander("ℹ️ Understanding Frequentist A/B Testing", expanded=True):
        st.markdown("""
        This is the most common method for online experimentation. It's based on the concept of hypothesis testing.

        1.  **Null Hypothesis (H₀)**: There is no difference between the control and treatment groups. The observed lift is due to random chance.
        2.  **Alternative Hypothesis (H₁)**: There is a real difference between the groups.
        
        We calculate a **p-value**, which is the probability of observing our data (or more extreme data) *if the null hypothesis is true*. If this probability is very low (typically < 0.05), we reject the null hypothesis and conclude that the treatment had a real effect.
        
        **Real-World Scenario**: **Spotify** tests a new recommendation algorithm. After collecting data for two weeks, they perform a Z-test. The result is a single p-value (e.g., 0.03). Because this is less than 0.05, they conclude the new algorithm is significantly better and roll it out.
        """)
    
    results = calculate_frequentist_results(df)
    if not results:
        st.error("Could not compute test results. Sample size might be zero for one group.")
        return

    st.subheader("Key Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Control Conversion Rate", f"{results['control_conversion']:.3%}")
    col2.metric("Treatment Conversion Rate", f"{results['treatment_conversion']:.3%}")
    if results['significant']:
        col3.success(f"P-value: {results['p_value']:.4f}")
    else:
        col3.warning(f"P-value: {results['p_value']:.4f}")
    
    st.subheader("Effect Size & Confidence Interval")
    lift_col, ci_col = st.columns(2)
    lift_col.metric("Relative Lift", f"{results['relative_lift']:.2%}", f"{results['absolute_lift']:.3%} absolute")
    ci_col.metric("95% CI for Absolute Lift", f"[{results['ci'][0]:.3%}, {results['ci'][1]:.3%}]")

    st.markdown("---")

    st.markdown(f"""
    **Interpretation**:
    The treatment group had a conversion rate of **{results['treatment_conversion']:.3%}** compared to the control group's **{results['control_conversion']:.3%}**, representing a relative lift of **{results['relative_lift']:.2%}**.

    The p-value is **{results['p_value']:.4f}**.
    - If **p < 0.05**: This is statistically significant. We can be reasonably confident that the observed lift is a real effect and not due to random chance.
    - If **p ≥ 0.05**: This result is not statistically significant. We do not have enough evidence to conclude that the treatment had an effect.
    
    The 95% Confidence Interval for the absolute lift is **[{results['ci'][0]:.3%}, {results['ci'][1]:.3%}]**.
    - This means we are 95% confident that the true absolute difference in conversion rates lies within this range.
    - If the interval **does not include 0**, it supports the conclusion of a significant result.
    - If the interval **includes 0**, it supports the conclusion of a non-significant result.
    """)

    with st.expander("🔍 Under the Hood: Formulas & Code"):
        st.latex(r'''
        \text{Z-score} = \frac{\hat{p}_{treat} - \hat{p}_{control}}{\sqrt{\hat{p}_{pool}(1-\hat{p}_{pool})(\frac{1}{n_{treat}} + \frac{1}{n_{control}})}}
        ''')
        st.latex(r'''
        \text{CI} = (\hat{p}_{treat} - \hat{p}_{control}) \pm Z_{1-\alpha/2} \cdot \sqrt{\frac{\hat{p}_{treat}(1-\hat{p}_{treat})}{n_{treat}} + \frac{\hat{p}_{control}(1-\hat{p}_{control})}{n_{control}}}
        ''')
        st.code("""
def calculate_frequentist_results(df):
    # ... (implementation as above) ...
        """, language='python')

def show_bayesian_platform(df, **kwargs):
    st.header("🔮 Bayesian Experimentation Platform")
    with st.expander("ℹ️ Understanding Bayesian Inference", expanded=True):
        st.markdown("""
        The Bayesian approach provides a more intuitive way to interpret experimental results. Instead of a p-value, it calculates the **posterior probability distribution** for the conversion rates of the control and treatment groups.
        
        #### Key Advantages:
        1.  **Direct Probability Statements**: You can make direct statements like, "There is a 98% chance that the treatment is better than the control."
        2.  **Continuous Monitoring**: It naturally allows for checking results at any time without the "peeking problem" that affects frequentist tests.
        3.  **Quantifying Risk**: It allows you to calculate the "Expected Loss," which is the potential downside of choosing a variant that isn't actually the best.

        **Real-World Scenario**: A company like **Booking.com** uses Bayesian methods to test new features. They monitor the "Probability to be Best" in real-time. Once this probability crosses a high threshold (e.g., 99%) and the "Expected Loss" of rolling out the feature drops below a small value, they can confidently stop the test and ship the winner, often weeks earlier than with a traditional fixed-horizon test.
        """)

    st.subheader("Live Experiment Simulation")
    total_days = st.slider("Experiment Duration (Days)", 1, 60, 14, key="bayesian_days")
    df_sorted = df.sort_values('timestamp').reset_index()
    
    history = []
    
    for day in range(1, total_days + 1):
        daily_data = df_sorted.iloc[:int(len(df_sorted) * day / total_days)]
        control = daily_data[daily_data['treatment_group'] == 'Control']['converted']
        treatment = daily_data[daily_data['treatment_group'] == 'Treatment']['converted']
        
        if len(control) > 1 and len(treatment) > 1:
            # Beta distribution parameters (with a non-informative prior of Beta(1,1))
            a_c, b_c = 1 + control.sum(), 1 + len(control) - control.sum()
            a_t, b_t = 1 + treatment.sum(), 1 + len(treatment) - treatment.sum()
            
            # Simulate draws from the posterior distributions
            control_samples = np.random.beta(a_c, b_c, 50000)
            treatment_samples = np.random.beta(a_t, b_t, 50000)
            
            prob_t_better = np.mean(treatment_samples > control_samples)
            history.append({'day': day, 'prob_t_better': prob_t_better})

    if history:
        history_df = pd.DataFrame(history)
        fig_prob = px.line(history_df, x='day', y='prob_t_better', title="Probability Treatment is Better Over Time")
        fig_prob.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="95% Threshold")
        fig_prob.update_layout(yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader(f"Results after {total_days} days")
    # Use data from the final day for plots
    final_day_data = df_sorted.iloc[:int(len(df_sorted) * total_days / total_days)]
    final_control = final_day_data[final_day_data['treatment_group'] == 'Control']['converted']
    final_treatment = final_day_data[final_day_data['treatment_group'] == 'Treatment']['converted']
    
    a_c, b_c = 1 + final_control.sum(), 1 + len(final_control) - final_control.sum()
    a_t, b_t = 1 + final_treatment.sum(), 1 + len(final_treatment) - final_treatment.sum()
    
    control_samples = np.random.beta(a_c, b_c, 50000)
    treatment_samples = np.random.beta(a_t, b_t, 50000)
    lift_samples = (treatment_samples - control_samples) / control_samples
    
    prob_t_better_final = np.mean(treatment_samples > control_samples)
    loss_control = np.maximum(0, treatment_samples - control_samples)
    loss_treatment = np.maximum(0, control_samples - treatment_samples)
    expected_loss_control = np.mean(loss_control)
    expected_loss_treatment = np.mean(loss_treatment)

    st.metric("Probability Treatment is Better", f"{prob_t_better_final:.2%}")
    
    col1, col2 = st.columns(2)
    with col1:
        x_range = np.linspace(min(control_samples.min(), treatment_samples.min()), max(control_samples.max(), treatment_samples.max()), 500)
        fig_posterior = go.Figure()
        fig_posterior.add_trace(go.Scatter(x=x_range, y=stats.beta.pdf(x_range, a_c, b_c), fill='tozeroy', name='Control Posterior', line_color='blue'))
        fig_posterior.add_trace(go.Scatter(x=x_range, y=stats.beta.pdf(x_range, a_t, b_t), fill='tozeroy', name='Treatment Posterior', line_color='orange'))
        fig_posterior.update_layout(title="Posterior Distributions of Conversion Rates", xaxis_title="Conversion Rate", yaxis_title="Density")
        st.plotly_chart(fig_posterior, use_container_width=True)

    with col2:
        fig_lift = px.histogram(lift_samples, nbins=100, title="Distribution of Relative Lift")
        fig_lift.add_vline(x=0, line_dash="dash", line_color="red")
        fig_lift.update_layout(xaxis_title="Relative Lift", yaxis_title="Frequency")
        st.plotly_chart(fig_lift, use_container_width=True)

    st.subheader("Decision Analysis")
    c1, c2 = st.columns(2)
    c1.metric("Expected Loss if we choose Control", f"{expected_loss_control:.4%}", "This is the average gain we'd miss out on if Treatment is actually better.")
    c2.metric("Expected Loss if we choose Treatment", f"{expected_loss_treatment:.4%}", "This is the average loss we'd incur if Control is actually better.")

    st.markdown("""
    **Interpretation**: Based on the data, there's a **{:.2%}** probability that the treatment is superior. If we decide to roll out the treatment, our expected loss (the risk we take) is **{:.4%}**. A common decision rule is to roll out a feature when the Probability to be Best is high (e.g., >95%) and the Expected Loss is below a small threshold.
    """.format(prob_t_better_final, expected_loss_treatment))

    with st.expander("🔍 Under the Hood: The Logic"):
        st.markdown(r"""
        We model the conversion rate $\theta$ for each group using a Beta distribution, which is updated with data (conversions `s` and non-conversions `f`).
        - **Prior**: $p(\theta) = \text{Beta}(\alpha_0, \beta_0)$ (e.g., $\alpha_0=1, \beta_0=1$ for a uniform prior)
        - **Likelihood**: $p(s, f | \theta) = \theta^s (1-\theta)^f$ (Binomial)
        - **Posterior**: $p(\theta | s, f) = \text{Beta}(\alpha_0 + s, \beta_0 + f)$
        
        The **Probability to be Best** is calculated by sampling from the two posterior distributions and counting how often the treatment samples are greater than the control samples.
        
        The **Expected Loss** for choosing Treatment is $\mathbb{E}[\max(0, \theta_C - \theta_T)]$.
        """)

def show_bandit_analysis(df, **kwargs):
    st.header("🎰 Multi-Armed Bandit Analysis")
    with st.expander("ℹ️ Understanding Multi-Armed Bandits (MAB)", expanded=True):
        st.markdown("""
        Multi-Armed Bandits (MAB) are algorithms that help make decisions under uncertainty. Unlike a traditional A/B test that allocates fixed traffic (e.g., 50/50), a bandit dynamically allocates more traffic to better-performing variations over time. This minimizes "regret" – the opportunity cost of not showing the best variant.
        
        It's an "explore-exploit" strategy:
        - **Explore**: Try out different arms to see which one is best.
        - **Exploit**: Use the arm that currently seems to be the best to maximize reward.
        
        The **Thompson Sampling** algorithm used here achieves this by modeling the expected reward of each arm as a probability distribution (a Beta distribution, just like in our Bayesian section). It then samples from these distributions and picks the arm with the highest sample, naturally balancing exploration and exploitation.

        **Real-World Scenario**: **Instagram** could use a bandit to test different ad creatives for a campaign. Instead of waiting two weeks for an A/B test result, the bandit quickly learns which creative has the highest click-through rate and starts showing it to more people within hours, maximizing the campaign's effectiveness in real-time.
        """)
    
    st.subheader("Bandit Simulation Setup")
    n_trials = st.slider("Number of Users (Trials)", 1000, 20000, 5000, 1000, key="bandit_trials")
    true_rates = [
        st.slider("Arm 1 (Control) True Rate (%)", 1.0, 20.0, 5.0, 0.1, key="b_rate1")/100,
        st.slider("Arm 2 (Variant A) True Rate (%)", 1.0, 20.0, 5.5, 0.1, key="b_rate2")/100,
        st.slider("Arm 3 (Variant B) True Rate (%)", 1.0, 20.0, 6.5, 0.1, key="b_rate3")/100,
    ]
    
    if st.button("Run Bandit Simulation", key="run_bandit_sim"):
        bandit = MultiArmedBandit(len(true_rates))
        history, selections, cumulative_regret_history = [], [], []
        
        with st.spinner(f"Running {n_trials} trials..."):
            for i in range(n_trials):
                arm = bandit.select_arm()
                selections.append(arm)
                reward = np.random.binomial(1, true_rates[arm])
                bandit.update(arm, reward)
                
                # Update history periodically for plotting
                if (i+1) % (n_trials//100) == 0 or i == n_trials-1:
                    history.append({'trial': i+1, **{f'Arm {j+1}': r for j, r in enumerate(bandit.get_estimated_rates())}})
                    cumulative_regret_history.append({'trial': i+1, 'regret': bandit.calculate_regret(true_rates)})

        st.subheader("Simulation Results")
        col1, col2 = st.columns(2)
        
        history_df = pd.DataFrame(history).melt(id_vars='trial', var_name='Arm', value_name='Estimated Rate')
        fig_beliefs = px.line(history_df, x='trial', y='Estimated Rate', color='Arm', title="Bandit's Beliefs Over Time")
        for i, rate in enumerate(true_rates):
            fig_beliefs.add_hline(y=rate, line_dash="dash", line_color="gray", annotation_text=f"True Rate Arm {i+1}")
        col1.plotly_chart(fig_beliefs, use_container_width=True)

        selections_df = pd.DataFrame(selections, columns=['arm_index'])
        selections_df['Arm'] = selections_df['arm_index'].apply(lambda x: f"Arm {x+1}")
        fig_alloc = px.histogram(selections_df, x='Arm', title="Traffic Allocation per Arm").update_xaxes(categoryorder='total descending')
        col2.plotly_chart(fig_alloc, use_container_width=True)
        
        st.markdown("---")
        regret_df = pd.DataFrame(cumulative_regret_history)
        fig_regret = px.line(regret_df, x='trial', y='regret', title="Cumulative Regret Over Time")
        st.plotly_chart(fig_regret, use_container_width=True)

        st.info(f"**Total Regret**: The bandit 'lost' **{bandit.calculate_regret(true_rates):.2f}** potential conversions compared to an oracle that always knew the best arm. The goal is a regret curve that flattens out as the bandit learns.")
        st.markdown("**Interpretation**: The bandit starts by exploring all arms equally. As it gathers data, its 'beliefs' (the estimated rates) converge towards the true rates. It then exploits this knowledge by allocating more traffic to the arm with the highest estimated rate, which causes the cumulative regret to flatten.")

def show_cuped_analysis(df, **kwargs):
    st.header("🚀 CUPED Variance Reduction")
    with st.expander("ℹ️ Understanding CUPED", expanded=True):
        st.markdown("""
        **CUPED** stands for **C**ontrolled-experiment **U**sing **P**re-**E**xperiment **D**ata. It's a powerful statistical technique to reduce variance in an A/B test, which leads to:
        1.  **Increased Statistical Power**: The ability to detect smaller effects.
        2.  **Smaller Required Sample Sizes**: You can conclude tests faster.
        
        It works by using data from *before* the experiment (a **covariate**) that is highly correlated with the metric you are measuring *during* the experiment. By subtracting out the portion of the variance that can be predicted by the pre-experiment data, we get a much "cleaner" and less noisy metric.
        
        **Real-World Scenario**: **Etsy** wants to test a new recommendation algorithm to increase the total purchase value per user. This metric is notoriously noisy (some users spend $5, others spend $500). By using each user's total purchase value from the month *before* the test as a covariate, CUPED can control for the pre-existing high variability in user spending habits. This makes it much easier to see the true, smaller impact of the new algorithm without needing to run the test for an excessively long time.
        """)

    # Metric Y and Covariate X
    Y = df['converted']
    X = df['pre_experiment_engagement']
    
    # Calculate theta
    theta = np.cov(Y, X)[0, 1] / np.var(X) if np.var(X) > 0 else 0
    
    # Create the CUPED-adjusted metric
    df['cuped_converted'] = Y - (X - np.mean(X)) * theta

    st.subheader("Variance Comparison")
    col1, col2 = st.columns(2)
    var_orig = df['converted'].var()
    var_cuped = df['cuped_converted'].var()
    col1.metric("Original Metric Variance", f"{var_orig:.4f}")
    if var_orig > 0:
        col2.metric("CUPED-Adjusted Variance", f"{var_cuped:.4f}", f"{-1 + var_cuped/var_orig:.1%} change")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['converted'], name='Original', histnorm='probability density', marker_color='#1f77b4'))
    fig.add_trace(go.Histogram(x=df['cuped_converted'], name='CUPED-Adjusted', histnorm='probability density', marker_color='#ff7f0e'))
    fig.update_layout(barmode='overlay', title="Distribution of Original vs. CUPED-Adjusted Metric")
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Impact on Test Results")
    st.info("Let's compare the results of a standard Z-test with and without the CUPED adjustment.")
    
    results_orig = calculate_frequentist_results(df, 'converted')
    results_cuped = calculate_frequentist_results(df, 'cuped_converted')
    
    if results_orig and results_cuped:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Standard Test Results")
            st.metric("P-value", f"{results_orig['p_value']:.4f}")
            st.metric("95% CI for Lift", f"[{results_orig['ci'][0]:.3%}, {results_orig['ci'][1]:.3%}]")
        with c2:
            st.markdown("#### CUPED-Adjusted Test Results")
            st.metric("P-value", f"{results_cuped['p_value']:.4f}")
            st.metric("95% CI for Lift", f"[{results_cuped['ci'][0]:.3%}, {results_cuped['ci'][1]:.3%}]")
    
        st.markdown("**Interpretation**: By reducing the variance, CUPED makes our measurements more precise. This results in a **narrower confidence interval** and a **lower p-value**, giving us more statistical power to detect the true effect.")

    with st.expander("🔍 Under the Hood: The Formula"):
        st.latex(r'''
        Y_{CUPED} = Y - \theta(X - \mu_X)
        ''')
        st.latex(r'''
        \theta = \frac{\text{cov}(Y, X)}{\text{var}(X)}
        ''')
        st.markdown(r"""
        Where:
        - $Y$: the metric of interest (e.g., conversions)
        - $X$: the pre-experiment covariate (e.g., past engagement)
        - $\mu_X$: the mean of the covariate across all users
        - $\theta$: a coefficient calculated to minimize the variance of $Y_{CUPED}$
        """)

def show_sequential_testing(df, **kwargs):
    st.header("⏳ Sequential Testing & Early Stopping")
    with st.expander("ℹ️ Understanding Sequential Testing", expanded=True):
        st.markdown("""
        In traditional A/B testing, repeatedly checking your results ("peeking") before the planned sample size is reached inflates the false positive rate. If you check 20 times, your chance of seeing a significant result just by luck is much higher than 5%!

        **Sequential testing** provides a statistically valid framework to check results at multiple, pre-defined points and stop early if the evidence is overwhelming. It does this by adjusting the significance boundaries at each "look."

        #### Common Approaches (Alpha-Spending Functions):
        - **Pocock Boundary**: Spends the alpha risk evenly across all looks. Uses the same strict significance boundary for each check. Good for stopping very early.
        - **O'Brien-Fleming (OF) Boundary**: Spends very little alpha at the beginning and most of it at the final look. This makes it hard to stop early but preserves almost all the power of a fixed-horizon test if you don't. This is often preferred in practice.
        
        **Real-World Scenario**: A pharmaceutical company is testing a new drug. If, after analyzing data from the first 20% of patients, the drug shows a massive, undeniable benefit (or harm), sequential testing provides a statistically valid framework to stop the trial early. For tech companies, this means shipping winning features weeks sooner or stopping a harmful experiment before it impacts too many users.
        """)
    
    st.subheader("Simulated Sequential Test")
    n_looks = st.slider("Number of 'Looks' at the Data", 2, 10, 5, key="seq_looks")
    
    # Calculate boundaries
    alpha = 0.05
    z_alpha_overall = stats.norm.ppf(1 - alpha / 2)
    
    # Pocock
    pocock_alpha = alpha / n_looks
    pocock_boundary = stats.norm.ppf(1 - pocock_alpha / 2)
    
    # O'Brien-Fleming
    k = np.arange(1, n_looks + 1)
    of_boundaries = z_alpha_overall * np.sqrt(n_looks / k)
    
    df_sorted = df.sort_values('timestamp').reset_index()
    z_scores_history = []
    
    for i in range(1, n_looks + 1):
        subset_df = df_sorted.iloc[:int(len(df_sorted) * i / n_looks)]
        results = calculate_frequentist_results(subset_df)
        if results and 'z_score' in results:
            z_scores_history.append({'look': i, 'z_score': results['z_score']})
    
    if z_scores_history:
        z_df = pd.DataFrame(z_scores_history)
        z_df['Pocock Boundary'] = pocock_boundary
        z_df['O-F Boundary'] = of_boundaries
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z_df['look'], y=z_df['z_score'], name='Z-score at Look', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=z_df['look'], y=z_df['Pocock Boundary'], name='Pocock Boundary', line=dict(dash='dash', color='orange')))
        fig.add_trace(go.Scatter(x=z_df['look'], y=z_df['O-F Boundary'], name="O'Brien-Fleming Boundary", line=dict(dash='dash', color='red')))
        fig.add_hline(y=z_alpha_overall, line_dash="dot", line_color="gray", annotation_text="Standard Z=1.96 Boundary (Naive)")
        
        fig.update_layout(title="Z-score Over Time vs. Stopping Boundaries",
                          xaxis_title="Look Number",
                          yaxis_title="Z-score")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Interpretation**: The graph shows the calculated Z-score at each 'look'. If the Z-score crosses one of the colored boundary lines, the test can be stopped early with a statistically significant result. Notice how the O'Brien-Fleming boundary starts very high (conservative) and gets closer to the standard 1.96 boundary at the final look, while the Pocock boundary is constant.")

def show_hte_analysis(df, **kwargs):
    st.header("🧬 Heterogeneous Treatment Effects (HTE)")
    with st.expander("ℹ️ Understanding HTE", expanded=True):
        st.markdown("""
        A standard A/B test measures the **Average Treatment Effect (ATE)** — the overall impact of the treatment on the entire population. However, this average can be misleading. A feature might be hugely beneficial for one user segment but slightly harmful to another, resulting in a flat or small ATE.

        **Heterogeneous Treatment Effects (HTE)** analysis aims to uncover these differences. The goal is to understand *how the treatment effect varies across users with different characteristics* (e.g., age, device, location, behavior). This is also known as looking for **interaction effects**.
        
        Instead of asking "Did the feature work?", we ask "Who did the feature work for?".
        
        **Real-World Scenario**: A fitness app like **Peloton** tests a new, intense workout program. The overall ATE on user engagement might be small. HTE analysis, however, might reveal that the program is highly effective for users under 30 but actually *reduces* engagement for users over 50. This insight is far more valuable than the simple average, allowing for a targeted rollout or personalization instead of a one-size-fits-all decision.
        """)
        
    st.info("Here, we'll use a logistic regression model to see if the treatment effect interacts with user age.")
    
    df['is_treatment'] = (df['treatment_group'] == 'Treatment').astype(int)
    df['age_centered'] = df['age'] - df['age'].mean()
    df['age_x_treatment'] = df['age_centered'] * df['is_treatment']

    # Define the model
    y = df['converted']
    X = df[['is_treatment', 'age_centered', 'age_x_treatment']]
    X = sm.add_constant(X) # Add an intercept
    
    # Fit the logistic regression model
    model = sm.Logit(y, X).fit(disp=0) # disp=0 suppresses convergence messages
    
    st.subheader("Regression Model Results")
    st.text(model.summary())
    
    # Extract the interaction term's results
    interaction_p_value = model.pvalues['age_x_treatment']
    interaction_coef = model.params['age_x_treatment']
    
    st.markdown("---")
    st.subheader("Interpretation of Results")
    st.markdown(f"The key term to look at is `age_x_treatment`, which represents the interaction effect.")
    st.metric("P-value for Interaction Term (`age_x_treatment`)", f"{interaction_p_value:.4f}")
    
    if interaction_p_value < 0.05:
        st.success("The interaction p-value is significant (< 0.05). This provides strong evidence that the effect of the treatment is not the same across different ages.")
        if interaction_coef > 0:
            st.markdown("The coefficient is **positive**, suggesting the treatment is **more effective for older users**.")
        else:
            st.markdown("The coefficient is **negative**, suggesting the treatment is **more effective for younger users**.")
    else:
        st.warning("The interaction p-value is not significant (≥ 0.05). We do not have strong evidence to conclude that the treatment's effect varies by age.")
    
    st.markdown("""
    A significant interaction term is the statistical foundation for personalization. It tells you that a "one-size-fits-all" approach is suboptimal.
    """)

    with st.expander("🔍 Under the Hood: The Model"):
        st.latex(r'''
        \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 \cdot \text{is\_treatment} + \beta_2 \cdot \text{age} + \beta_3 \cdot (\text{age} \times \text{is\_treatment})
        ''')
        st.markdown(r"""
        Where:
        - $\beta_0$: Baseline log-odds of conversion.
        - $\beta_1$: The main effect of the treatment for an average-aged user.
        - $\beta_2$: The effect of age on conversion for the control group.
        - $\beta_3$: **The interaction term**. This is the most important one. It measures how much the treatment effect ($\beta_1$) changes for each one-unit increase in age. A significant $\beta_3$ is evidence of HTE.
        """)

# --- Main App Structure ---
def main():
    st.markdown('<h1 class="main-header">🧪 Advanced A/B Testing Portfolio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">A Comprehensive and Interactive Guide to Modern Online Experimentation</p>', unsafe_allow_html=True)
    
    st.sidebar.title("📊 Navigation")
    analysis_type = st.sidebar.radio("Select Analysis Type:", [
        "🏠 Overview", "🎯 Experiment Design", "📈 Classic A/B Test",
        "🔬 A/A Test Validation", "🔮 Bayesian Platform",
        "🎰 Multi-Armed Bandit", "🚀 CUPED Variance Reduction",
        "⏳ Sequential Testing", "🧬 Heterogeneous Treatment Effects (HTE)"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Data Controls")
    sample_size = st.sidebar.slider("Sample Size (Users)", 1000, 100000, 20000, 1000)
    random_seed = st.sidebar.number_input("Random Seed", value=42)
    
    with st.sidebar.expander("⚙️ Population Distribution Controls"):
        mobile_perc = st.slider("Mobile User %", 0, 100, 60, 5)
        desktop_perc = st.slider("Desktop User %", 0, 100-mobile_perc, 35, 5)
        tablet_perc = 100 - mobile_perc - desktop_perc
        st.text(f"Tablet User %: {tablet_perc}")
        
        new_perc = st.slider("New User %", 0, 100, 40, 5)
        returning_perc = st.slider("Returning User %", 0, 100-new_perc, 50, 5)
        vip_perc = 100 - new_perc - returning_perc
        st.text(f"VIP User %: {vip_perc}")

        population_params = {
            'device_dist': {'Mobile': mobile_perc/100, 'Desktop': desktop_perc/100, 'Tablet': tablet_perc/100},
            'segment_dist': {'New': new_perc/100, 'Returning': returning_perc/100, 'VIP': vip_perc/100}
        }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Experiment Parameters")
    base_ctr = st.sidebar.slider("Base Conversion Rate (%)", 0.1, 20.0, 5.0, 0.1) / 100
    uplift = st.sidebar.slider("True Uplift (Absolute %)", 0.0, 5.0, 0.5, 0.05) / 100
    
    # Generate data once based on global controls
    df = generate_synthetic_data(sample_size, random_seed, base_ctr, uplift, population_params)
    
    # Page routing
    pages = {
        "🏠 Overview": show_overview,
        "🎯 Experiment Design": show_experiment_design,
        "📈 Classic A/B Test": show_classic_ab_test,
        "🔬 A/A Test Validation": show_aa_test_validation,
        "🔮 Bayesian Platform": show_bayesian_platform,
        "🎰 Multi-Armed Bandit": show_bandit_analysis,
        "🚀 CUPED Variance Reduction": show_cuped_analysis,
        "⏳ Sequential Testing": show_sequential_testing,
        "🧬 Heterogeneous Treatment Effects (HTE)": show_hte_analysis,
    }
    
    # Pass all relevant parameters to the page functions
    page_func = pages[analysis_type]
    page_func(df, base_ctr=base_ctr, uplift=uplift, population_params=population_params)

if __name__ == "__main__":
    main()
