# Experimentation lab (v2, November 2026)
# Every tab runs on a simulator where the true effect is known, so you can check the method, not just run it.
# Statistics live in experiments.py; audit/ holds the checks of the 2025 version and of this one.
# Run: streamlit run ab_testing_implementation_3.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import experiments as E

POST_URL = "https://medium.com/@faycal.zouine.usa/i-tested-my-own-a-b-testing-app-it-found-a-winner-in-38-of-tests-where-nothing-changed-d4b3a00ac9e6"
REPO_URL = "https://github.com/ZFayFay/streamlit-experimentation-platform"
BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#6b7280"

st.set_page_config(page_title="Experimentation lab", page_icon="🧪", layout="wide")


@st.cache_data
def get_data(n, base, lift, seed, slope):
    return E.simulate(n, base, lift, seed, age_slope=slope)


@st.cache_data
def get_bounds(k, kind):
    return E.gst_boundaries(k, kind=kind)


def pts(x, d=2):
    """Absolute difference in percentage points."""
    return f"{x * 100:+.{d}f} pts"


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Experimentation lab")
page = st.sidebar.radio("Tab", [
    "Start here", "Design", "A/A check", "Classic test", "CUPED",
    "Sequential test", "Bayesian", "Segments (HTE)", "Bandit"])

st.sidebar.markdown("---")
st.sidebar.subheader("The simulated test")
st.sidebar.caption("A subscription app tests a new onboarding screen. Metric: share of users who upgrade to premium.")
n_users = st.sidebar.slider("Users in the test", 2000, 100000, 20000, 1000)
base = st.sidebar.slider("Control upgrade rate (%)", 1.0, 20.0, 5.0, 0.5) / 100
lift = st.sidebar.slider("True average effect (points)", -1.0, 2.0, 0.5, 0.05) / 100
slope = st.sidebar.slider("How the effect changes per 10 years of age (points)", -2.0, 2.0, -1.0, 0.1,
                          help="0 means everyone gets the same effect. The average effect stays as set above.") / 100
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))

df, true_effect = get_data(n_users, base, lift, seed, slope)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**The truth in this sample**  \nTrue average effect: **{pts(true_effect)}** "
                    f"({true_effect / base:+.1%} relative)")
st.sidebar.caption("A real test never shows you this. Here it does, so you can check every method against it.")


# ------------------------------------------------------------------ pages
def start_here():
    st.title("Experimentation lab")
    st.markdown(f"""
This app runs A/B test methods on simulated data where the right answer is known. Set the true effect in the
sidebar, then see whether each method finds it.

This is version 2. Version 1 (June 2025) had bugs that I found by testing it the same way:
its A/A check flagged fake winners 38% of the time, its Bayesian tab said peeking was safe (it isn't),
its sequential boundaries were mislabeled, and its CUPED test lost power instead of gaining it.
The story and all the checks are in [the post]({POST_URL}) and in [the repo]({REPO_URL}) under `audit/`.
""")
    c1, c2, c3, c4 = st.columns(4)
    counts = df.treatment_group.value_counts()
    c1.metric("Users", f"{len(df):,}")
    c2.metric("Control", f"{counts.get('Control', 0):,}")
    c3.metric("Treatment", f"{counts.get('Treatment', 0):,}")
    p = E.srm_p_value(df)
    c4.metric("Sample ratio check (p)", f"{p:.3f}", "OK" if p >= 0.01 else "Mismatch", delta_color="off")
    st.caption("The sample ratio check tests whether the 50/50 split held. A p-value below 0.01 usually means "
               "a bug in assignment or logging, and the test results can't be trusted.")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x='age', nbins=40, title="Age of users", color_discrete_sequence=[BLUE]),
                   width='stretch')
    b.plotly_chart(px.histogram(df, x='past_activity', title="Premium-page visits in the 4 weeks before the test",
                                color_discrete_sequence=[BLUE]), width='stretch')
    st.dataframe(df.head(10), width='stretch')


def design():
    st.header("Design: how many users do you need?")
    st.markdown("Pick the smallest effect worth detecting before the test starts. The formula gives the users "
                "needed per group for a two-sided test. The simulation then checks that the formula delivers the power it promises.")
    c1, c2, c3 = st.columns(3)
    alpha = c1.slider("Significance level", 0.01, 0.10, 0.05, 0.01)
    power = c2.slider("Power", 0.50, 0.95, 0.80, 0.05)
    mde = c3.slider("Smallest effect worth detecting (points)", 0.1, 3.0, 0.5, 0.1) / 100
    n = E.sample_size_per_group(base, mde, alpha, power)
    a, b = st.columns(2)
    a.metric("Users per group", f"{n:,}")
    b.metric("Total users", f"{2 * n:,}")
    runs = st.select_slider("Simulated tests", [100, 200, 500], 200)
    if st.button("Check the power by simulation"):
        bar = st.progress(0.0)
        hits = []
        for i in range(runs):
            d, _ = E.simulate(2 * n, base, mde, 10_000 + i)
            hits.append(E.two_proportion_test(d)['p_value'] < alpha)
            bar.progress((i + 1) / runs)
        bar.empty()
        st.metric("Share of tests that found the effect", pct(np.mean(hits), 1), f"promised {power:.0%}", delta_color="off")
        st.caption("With a true effect equal to the one you designed for, about this share of tests should come out significant.")


def aa_check():
    st.header("A/A check: does the test cry wolf?")
    st.markdown("Both groups get the same experience. A fair test should flag a difference about as often as the "
                "significance level, 5% of the time. Version 1 of this app flagged 38%, because its simulator kept an "
                "age effect even with the lift set to zero.")
    runs = st.select_slider("Simulated A/A tests", [200, 500, 1000], 500)
    if st.button("Run the A/A tests"):
        bar = st.progress(0.0)
        p = []
        for i in range(runs):
            d, _ = E.simulate(n_users, base, 0.0, 20_000 + i)
            p.append(E.two_proportion_test(d)['p_value'])
            bar.progress((i + 1) / runs)
        bar.empty()
        p = np.array(p)
        st.metric("False positive rate", pct(np.mean(p < 0.05), 1), "should be close to 5%", delta_color="off")
        fig = px.histogram(x=p, nbins=20, histnorm='probability density', title="p-values when nothing changed",
                           color_discrete_sequence=[BLUE])
        fig.add_hline(y=1, line_dash="dash", line_color=GREY, annotation_text="flat = healthy")
        st.plotly_chart(fig, width='stretch')


def classic():
    st.header("Classic test")
    r = E.two_proportion_test(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control", pct(r['control_rate']))
    c2.metric("Treatment", pct(r['treatment_rate']))
    c3.metric("Difference", pts(r['diff']), f"{r['relative_lift']:+.1%} relative", delta_color="off")
    c4.metric("p-value", f"{r['p_value']:.4f}")
    lo, hi = r['ci']
    st.markdown(f"**95% confidence interval for the difference:** {pts(lo)} to {pts(hi)}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r['diff'] * 100], y=['estimate'], mode='markers', marker=dict(size=12, color=BLUE),
                             error_x=dict(type='data', symmetric=False, array=[(hi - r['diff']) * 100],
                                          arrayminus=[(r['diff'] - lo) * 100]), name='95% CI'))
    fig.add_vline(x=0, line_color=GREY)
    fig.add_vline(x=true_effect * 100, line_dash="dash", line_color=ORANGE, annotation_text="truth")
    fig.update_layout(height=220, xaxis_title="difference (points)", showlegend=False)
    st.plotly_chart(fig, width='stretch')
    if r['p_value'] >= 0.05:
        st.info(f"Not significant. That is not the same as \"no effect\". The data rule out effects above "
                f"{pts(hi)} and below {pts(lo)}. If an effect of, say, {pts(hi / 2)} would matter to the business, "
                "this test can't tell you whether it's there, and the honest answer is \"we don't know yet\".")
    else:
        st.success(f"Significant at 5%. The effect is most likely between {pts(lo)} and {pts(hi)}.")


def cuped_page():
    st.header("CUPED: less noise from data you already have")
    st.markdown("Users who were active before the test tend to be active during it. CUPED removes the part of the "
                "metric that past behaviour predicts, so the treatment effect stands out from less noise. It only helps "
                "as much as the past metric predicts the outcome.")
    d, theta = E.cuped(df)
    raw = E.mean_difference_test(d, 'converted')
    adj = E.mean_difference_test(d, 'converted_cuped')
    corr = np.corrcoef(d.converted, d.past_activity)[0, 1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Correlation, past activity vs outcome", f"{corr:.2f}")
    c2.metric("Variance removed", pct(1 - d.converted_cuped.var() / d.converted.var(), 1))
    c3.metric("Confidence interval width", f"{(adj['ci'][1] - adj['ci'][0]) / (raw['ci'][1] - raw['ci'][0]):.0%} of the original")
    t = pd.DataFrame({'': ['Without CUPED', 'With CUPED'],
                      'Difference': [pts(raw['diff']), pts(adj['diff'])],
                      '95% CI': [f"{pts(raw['ci'][0])} to {pts(raw['ci'][1])}", f"{pts(adj['ci'][0])} to {pts(adj['ci'][1])}"],
                      'p-value': [f"{raw['p_value']:.4f}", f"{adj['p_value']:.4f}"]})
    st.table(t.set_index(''))
    st.caption(f"Truth: {pts(true_effect)}. The test on the adjusted metric uses its own variance. Version 1 used the "
               "binomial formula p(1-p), which ignores the noise CUPED removed, so it gained nothing.")


def sequential():
    st.header("Sequential test: look early without fooling yourself")
    st.markdown("Checking a p-value every few days and stopping at the first p < 0.05 raises the false positive rate. "
                "A group sequential test sets a stricter bar at each look so the overall rate stays at 5%. "
                "O'Brien-Fleming is very strict early and close to 1.96 at the end. Pocock-type spreads the bar evenly.")
    k = st.slider("Number of looks", 2, 10, 5)
    of, po = get_bounds(k, 'obf'), get_bounds(k, 'pocock')
    z = E.z_at_looks(df, k)
    looks = np.arange(1, k + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=looks, y=z, mode='lines+markers', name='z at each look', line_color=BLUE))
    fig.add_trace(go.Scatter(x=looks, y=of, name="O'Brien-Fleming", line=dict(dash='dash', color=ORANGE)))
    fig.add_trace(go.Scatter(x=looks, y=po, name='Pocock-type', line=dict(dash='dot', color=GREY)))
    fig.add_hline(y=1.96, line_color="#d1d5db", annotation_text="1.96 (no correction)")
    fig.update_layout(xaxis_title="look", yaxis_title="z", height=380, xaxis_dtick=1)
    st.plotly_chart(fig, width='stretch')
    stop_of = next((i + 1 for i in range(k) if z[i] >= of[i]), None)
    stop_po = next((i + 1 for i in range(k) if z[i] >= po[i]), None)
    st.markdown(f"O'Brien-Fleming: {'stops at look ' + str(stop_of) if stop_of else 'no stop'}. "
                f"Pocock-type: {'stops at look ' + str(stop_po) if stop_po else 'no stop'}.")
    st.dataframe(pd.DataFrame({'look': looks, "O'Brien-Fleming": of.round(3), 'Pocock-type': po.round(3)}).set_index('look').T)
    st.caption("Boundaries are Lan-DeMets alpha spending for a two-sided 5% test, computed by numerical integration. "
               "Version 1 used 1.96 x sqrt(K/k), which lets through too much, and called a Bonferroni split \"Pocock\".")


def bayesian():
    st.header("Bayesian: probability that Treatment is better")
    st.markdown("With flat priors, the probability that Treatment beats Control is a clean way to report a result. "
                "It does **not** fix peeking. Under flat priors it moves almost one-for-one with the p-value, so "
                "checking it every day and stopping at 95% picks fake winners far more than 5% of the time "
                f"([Spotify explains why]({'https://engineering.atspotify.com/2026/9/why-spotify-is-not-using-bayesian-a-b-testing/'})).")
    daily = E.daily_prob_better(df)
    fig = px.line(daily, x='day', y='prob_better', markers=True, title="P(Treatment better), day by day",
                  color_discrete_sequence=[BLUE])
    fig.add_hline(y=0.975, line_dash="dash", line_color=ORANGE, annotation_text="97.5%")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig, width='stretch')
    c = df.loc[df.treatment_group == 'Control', 'converted']
    t = df.loc[df.treatment_group == 'Treatment', 'converted']
    pb, cs, ts = E.prob_treatment_better(c.sum(), len(c), t.sum(), len(t))
    loss_t = np.mean(np.maximum(cs - ts, 0))
    a, b = st.columns(2)
    a.metric("P(Treatment better) at the planned end", pct(pb, 1))
    b.metric("Expected loss if you ship Treatment", f"{loss_t * 100:.3f} pts")
    st.info("Rule this app recommends: decide once, at the planned end, and ship if the probability is above 97.5%. "
            "That matches a two-sided 5% test. If you need to stop early, use the Sequential tab.")
    if st.button("How often does daily checking fool you? (200 A/A tests)"):
        bar = st.progress(0.0)
        any95, end975 = [], []
        for i in range(200):
            d, _ = E.simulate(n_users, base, 0.0, 30_000 + i)
            pr = E.daily_prob_better(d)['prob_better'].values
            any95.append((pr > 0.95).any())
            end975.append(pr[-1] > 0.975)
            bar.progress((i + 1) / 200)
        bar.empty()
        x, y = st.columns(2)
        x.metric("Check daily, ship at 95%", pct(np.mean(any95), 1), "fake winners", delta_color="off")
        y.metric("Decide once at the end, 97.5%", pct(np.mean(end975), 1), "fake winners", delta_color="off")


def hte():
    st.header("Segments: who does the change work for?")
    st.markdown("The average can hide groups that react differently. The regression tests whether the effect "
                "changes with age. Pick segments before the test: if you search many after a flat result, one will "
                "look significant by chance.")
    m = E.age_interaction(df)
    a, b = st.columns(2)
    a.metric("Treatment x age, p-value", f"{m.pvalues['treatment_x_age']:.4f}")
    b.metric("True change per 10 years", pts(slope))
    bands = E.effect_by_age_band(df)
    fig = go.Figure(go.Scatter(x=bands['ages'], y=bands['diff'] * 100, mode='markers', marker=dict(size=11, color=BLUE),
                               error_y=dict(type='data', symmetric=False, array=(bands.ci_high - bands['diff']) * 100,
                                            arrayminus=(bands['diff'] - bands.ci_low) * 100)))
    fig.add_hline(y=0, line_color=GREY)
    fig.update_layout(title="Effect by age group, with 95% CI", yaxis_title="difference (points)", height=360)
    st.plotly_chart(fig, width='stretch')
    with st.expander("Regression output"):
        st.text(m.summary().as_text())


def bandit():
    st.header("Bandit: earn while you learn")
    st.markdown("Thompson sampling sends more traffic to whichever version looks best so far. It loses fewer "
                "conversions during the test than a 50/50 split, but its estimates for the losing versions stay rough, "
                "so it's better for short-lived choices (headlines, offers) than for measuring an effect.")
    n = st.slider("Users", 1000, 20000, 5000, 1000)
    rates = [st.slider(f"Version {j + 1} true rate (%)", 1.0, 20.0, v, 0.1) / 100 for j, v in enumerate([5.0, 5.5, 6.5])]
    if st.button("Run the bandit"):
        arms, regret, est = E.thompson_bandit(rates, n, seed)
        a, b = st.columns(2)
        f1 = px.line(est.melt('trial', var_name='version', value_name='estimate'), x='trial', y='estimate', color='version',
                     title="What the bandit believes")
        a.plotly_chart(f1, width='stretch')
        share = pd.Series(arms).value_counts(normalize=True).sort_index()
        b.plotly_chart(px.bar(x=[f"Version {i + 1}" for i in share.index], y=share.values, title="Share of traffic",
                              color_discrete_sequence=[BLUE]), width='stretch')
        st.metric("Conversions lost vs always showing the best version", f"{regret[-1]:.1f}",
                  f"a 50/50/... split would lose about {n * (max(rates) - np.mean(rates)):.1f}", delta_color="off")


{"Start here": start_here, "Design": design, "A/A check": aa_check, "Classic test": classic, "CUPED": cuped_page,
 "Sequential test": sequential, "Bayesian": bayesian, "Segments (HTE)": hte, "Bandit": bandit}[page]()
