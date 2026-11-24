"""Statistics behind the experimentation app (v2, Nov 2026).

Every number the app shows comes from a function in this file, and audit/audit_v2.py
checks these same functions. No Streamlit code here.

The simulated product: a subscription app tests a new onboarding screen.
Metric: share of users who upgrade to premium during the test.
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize

Z975 = stats.norm.ppf(0.975)


# ---------------------------------------------------------------- simulator
def simulate(n_users, base_rate, lift, seed, age_slope=0.0, past_weeks=4, heterogeneity=2.0):
    """Simulated experiment where the true answer is known.

    base_rate      share of control users who upgrade (e.g. 0.05)
    lift           true average effect, in absolute terms (0.005 = +0.5 points)
    age_slope      how the effect changes per 10 years of age, in absolute terms,
                   centred on the average user so the average effect stays `lift`
    past_weeks     weeks of pre-test history; a user's upgrades-page visits in that
                   period come from the same propensity as the outcome (the CUPED covariate)
    heterogeneity  lower = users differ more (Beta concentration)
    """
    rng = np.random.default_rng(seed)
    n = int(n_users)
    age = rng.normal(35, 12, n).clip(18, 75).round()
    device = rng.choice(['Mobile', 'Desktop', 'Tablet'], n, p=[0.60, 0.35, 0.05])
    segment = rng.choice(['New', 'Returning', 'Long-time'], n, p=[0.40, 0.50, 0.10])

    # each user has their own propensity to upgrade; mean = base_rate
    p = rng.beta(base_rate * heterogeneity, (1 - base_rate) * heterogeneity, n)
    past = rng.binomial(past_weeks, p)                       # pre-test behaviour

    treat = rng.integers(0, 2, n)
    # the effect scales each user's own propensity, so it never pushes a probability below 0,
    # and on average it equals lift + age_slope * (age - mean age) / 10
    shape = age_slope * (age - age.mean()) / 10

    def treated(c):
        return np.clip(p * (1 + (c + shape) / base_rate), 0, 1)

    # a few very likely users hit a 100% ceiling; pick the intercept so the average effect
    # in this sample is exactly `lift`
    if lift == 0 and age_slope == 0:
        p_treated = p
    else:
        c0 = optimize.brentq(lambda c: np.mean(treated(c) - p) - lift, -base_rate, 1.0)
        p_treated = treated(c0)
    converted = rng.binomial(1, np.where(treat == 1, p_treated, p))

    df = pd.DataFrame({
        'user_id': np.arange(n), 'age': age.astype(int), 'device_type': device,
        'customer_segment': segment, 'past_activity': past,
        'treatment_group': np.where(treat == 1, 'Treatment', 'Control'),
        'converted': converted,
        'day': rng.integers(0, 14, n),                       # day the user entered the test
    })
    true_effect_in_sample = float(np.mean(p_treated - p))
    return df, true_effect_in_sample


# ---------------------------------------------------------------- classic test
def two_proportion_test(df, metric='converted'):
    """Pooled z-test for the p-value, unpooled standard error for the 95% CI."""
    c = df.loc[df.treatment_group == 'Control', metric]
    t = df.loc[df.treatment_group == 'Treatment', metric]
    nc, nt = len(c), len(t)
    pc, pt = c.mean(), t.mean()
    pool = (c.sum() + t.sum()) / (nc + nt)
    se_pool = np.sqrt(pool * (1 - pool) * (1 / nc + 1 / nt))
    z = (pt - pc) / se_pool if se_pool > 0 else 0.0
    se = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    diff = pt - pc
    return {'n_control': nc, 'n_treatment': nt, 'control_rate': pc, 'treatment_rate': pt,
            'diff': diff, 'relative_lift': diff / pc if pc > 0 else np.nan,
            'z': z, 'p_value': 2 * stats.norm.sf(abs(z)),
            'ci': (diff - Z975 * se, diff + Z975 * se), 'se': se}


def mean_difference_test(df, metric):
    """Welch t-test on any per-user metric (used for the CUPED-adjusted metric)."""
    c = df.loc[df.treatment_group == 'Control', metric]
    t = df.loc[df.treatment_group == 'Treatment', metric]
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    z = diff / se
    return {'diff': diff, 'se': se, 'z': z, 'p_value': 2 * stats.norm.sf(abs(z)),
            'ci': (diff - Z975 * se, diff + Z975 * se)}


def srm_p_value(df):
    counts = df.treatment_group.value_counts()
    return stats.chisquare([counts.get('Control', 0), counts.get('Treatment', 0)]).pvalue


# ---------------------------------------------------------------- design
def sample_size_per_group(p0, mde, alpha=0.05, power=0.80):
    """Two-sided test of two proportions. mde is absolute (0.005 = 0.5 points)."""
    p1 = p0 + mde
    if mde <= 0 or not 0 < p1 < 1:
        return np.inf
    pbar = (p0 + p1) / 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    num = za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return int(np.ceil((num / mde) ** 2))


# ---------------------------------------------------------------- CUPED
def cuped(df, metric='converted', covariate='past_activity'):
    """Adds `<metric>_cuped`. Theta is estimated on both groups pooled."""
    y, x = df[metric].astype(float), df[covariate].astype(float)
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1) if np.var(x) > 0 else 0.0
    out = df.copy()
    out[metric + '_cuped'] = y - theta * (x - x.mean())
    return out, theta


# ---------------------------------------------------------------- Bayesian
def prob_treatment_better(conv_c, n_c, conv_t, n_t, draws=40000, seed=0):
    """P(rate_T > rate_C) under Beta(1,1) priors, by Monte Carlo."""
    rng = np.random.default_rng(seed)
    c = rng.beta(1 + conv_c, 1 + n_c - conv_c, draws)
    t = rng.beta(1 + conv_t, 1 + n_t - conv_t, draws)
    return float(np.mean(t > c)), c, t


def daily_prob_better(df, days=14):
    rows = []
    for d in range(days):
        sub = df[df.day <= d]
        c = sub.loc[sub.treatment_group == 'Control', 'converted']
        t = sub.loc[sub.treatment_group == 'Treatment', 'converted']
        pb, _, _ = prob_treatment_better(c.sum(), len(c), t.sum(), len(t), draws=20000, seed=d)
        rows.append({'day': d + 1, 'prob_better': pb, 'users': len(sub)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- sequential
def spending(t, alpha, kind):
    """Lan-DeMets alpha spending for one side. kind: 'obf' or 'pocock'."""
    t = np.asarray(t, float)
    if kind == 'obf':
        return 2 * stats.norm.sf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t))
    return alpha * np.log(1 + (np.e - 1) * t)


def gst_boundaries(n_looks, alpha=0.025, kind='obf', grid=0.01, lim=9.0):
    """One-sided efficacy boundaries (z scale) for equally spaced looks, by forward
    numerical integration of the Brownian motion. alpha=0.025 one-sided = 0.05 two-sided."""
    t = np.arange(1, n_looks + 1) / n_looks
    spent = spending(t, alpha, kind)
    x = np.arange(-lim, lim + grid, grid)
    f, prev_t, prev_s = None, 0.0, 0.0
    c = np.zeros(n_looks)
    for k in range(n_looks):
        dt, target = t[k] - prev_t, spent[k] - prev_s
        if k == 0:
            cross = lambda b: stats.norm.sf(b / np.sqrt(dt))
        else:
            cross = lambda b: np.sum(f * stats.norm.sf((b - x) / np.sqrt(dt))) * grid
        b = optimize.brentq(lambda b: cross(b) - target, -5, 20)
        c[k] = b / np.sqrt(t[k])
        if k == 0:
            f = np.where(x < b, stats.norm.pdf(x, 0, np.sqrt(dt)), 0.0)
        else:
            kern = stats.norm.pdf(x[:, None] - x[None, :], 0, np.sqrt(dt))
            f = np.where(x < b, kern @ f * grid, 0.0)
        prev_t, prev_s = t[k], spent[k]
    return c


def z_at_looks(df, n_looks):
    """z-score of the classic test using users who entered by each look (looks spread over 14 days)."""
    cut = np.linspace(14 / n_looks, 14, n_looks)
    return np.array([two_proportion_test(df[df.day < c])['z'] for c in cut])


# ---------------------------------------------------------------- HTE
def age_interaction(df):
    """Logistic regression: converted ~ treatment + age + treatment x age (age centred)."""
    import statsmodels.api as sm
    d = pd.DataFrame({'treatment': (df.treatment_group == 'Treatment').astype(int),
                      'age_c': (df.age - df.age.mean()) / 10})
    d['treatment_x_age'] = d.treatment * d.age_c
    model = sm.Logit(df.converted, sm.add_constant(d)).fit(disp=0)
    return model


def effect_by_age_band(df, bands=(18, 30, 40, 50, 76)):
    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        r = two_proportion_test(df[(df.age >= lo) & (df.age < hi)])
        rows.append({'ages': f'{lo}-{hi - 1}', 'users': r['n_control'] + r['n_treatment'],
                     'diff': r['diff'], 'ci_low': r['ci'][0], 'ci_high': r['ci'][1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- bandit
def thompson_bandit(true_rates, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    k = len(true_rates)
    a, b = np.ones(k), np.ones(k)
    best = max(true_rates)
    arms, regret, est = np.zeros(n_trials, int), np.zeros(n_trials), []
    for i in range(n_trials):
        arm = int(np.argmax(rng.beta(a, b)))
        r = rng.random() < true_rates[arm]
        a[arm] += r
        b[arm] += 1 - r
        arms[i] = arm
        regret[i] = best - true_rates[arm]
        if (i + 1) % max(1, n_trials // 100) == 0:
            est.append([i + 1, *(a / (a + b))])
    return arms, np.cumsum(regret), pd.DataFrame(est, columns=['trial'] + [f'Arm {j + 1}' for j in range(k)])
