"""Audit of the 2025 app, following PREREGISTRATION.md. Writes results_old.json.
Run: python audit.py
"""
import json, os
import numpy as np, pandas as pd, scipy.stats as stats
import statsmodels.api as sm
from multiprocessing import Pool
from old_app import generate_synthetic_data, calculate_frequentist_results, POP

N, BASE = 20000, 0.05
HERE = os.path.dirname(os.path.abspath(__file__))


def null_data(seed):
    """The app's generator with the hidden age effect removed: a true A/A test."""
    np.random.seed(seed)
    df = pd.DataFrame({'age': np.random.normal(35, 12, N).clip(18, 75).astype(int)})
    np.random.choice(['Male', 'Female', 'Other'], N, p=[0.48, 0.49, 0.03])  # keep the same random stream
    np.random.choice(list(POP['device_dist']), N, p=list(POP['device_dist'].values()))
    np.random.choice(list(POP['segment_dist']), N, p=list(POP['segment_dist'].values()))
    x = (df['age'] / 40 + np.random.normal(0, 0.2, N)).clip(0)
    df['treatment_group'] = np.random.choice(['Control', 'Treatment'], size=N, p=[0.5, 0.5])
    prob = BASE + (x - x.mean()) * 0.05 + np.random.normal(0, 0.005, N)
    df['converted'] = np.random.binomial(1, prob.clip(0.001, 0.999))
    df['timestamp'] = np.random.randint(0, 30 * 24 * 60, N)
    return df


# H1: the app's own A/A tab (uplift slider = 0)
def h1(seed):
    return calculate_frequentist_results(generate_synthetic_data(N, seed, BASE, 0.0, POP))['p_value']


# H2: Bayesian tab, 14 daily looks, ship if P(T > C) > 95% at any look
def h2(seed):
    df = null_data(seed).sort_values('timestamp').reset_index(drop=True)
    rng = np.random.default_rng(seed)
    probs = []
    for day in range(1, 15):
        d = df.iloc[:int(len(df) * day / 14)]
        c = d.loc[d.treatment_group == 'Control', 'converted']
        t = d.loc[d.treatment_group == 'Treatment', 'converted']
        cs = rng.beta(1 + c.sum(), 1 + len(c) - c.sum(), 20000)
        ts = rng.beta(1 + t.sum(), 1 + len(t) - t.sum(), 20000)
        probs.append(np.mean(ts > cs))
    return {'any_look': max(probs) > 0.95, 'final_only': probs[-1] > 0.95}


# H3: sequential tab, 5 looks, upper crossing of each boundary as coded
K = 5
POCOCK_AS_CODED = stats.norm.ppf(1 - 0.05 / K / 2)
OF_AS_CODED = stats.norm.ppf(0.975) * np.sqrt(K / np.arange(1, K + 1))


def h3(seed):
    df = null_data(seed).sort_values('timestamp').reset_index(drop=True)
    z = np.array([calculate_frequentist_results(df.iloc[:int(len(df) * k / K)])['z_score'] for k in range(1, K + 1)])
    return {'of': bool((z >= OF_AS_CODED).any()), 'pocock': bool((z >= POCOCK_AS_CODED).any()),
            'naive': bool((z >= 1.96).any())}


# H4a: CUPED variance reduction at default settings
def cuped(y, x):
    theta = np.cov(y, x)[0, 1] / np.var(x)
    return y - (x - np.mean(x)) * theta


def h4a(seed):
    df = generate_synthetic_data(N, seed, BASE, 0.005, POP)
    adj = cuped(df['converted'], df['pre_experiment_engagement'])
    return 1 - adj.var() / df['converted'].var()


# H4b: a predictive covariate (own conversions in 4 pre-period weeks), true lift 0.5 points
def strong_data(seed, lift=0.005):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.1, 1.9, N)                       # mean 5%, very uneven users
    pre = rng.binomial(4, p)                        # conversions in 4 weeks before the test
    t = rng.integers(0, 2, N)
    y = rng.binomial(1, np.clip(p + lift * t, 0, 1))
    return pd.DataFrame({'converted': y, 'pre': pre,
                         'treatment_group': np.where(t == 1, 'Treatment', 'Control')})


def h4b(seed):
    df = strong_data(seed)
    df['adj'] = cuped(df['converted'], df['pre'])
    raw = calculate_frequentist_results(df, 'converted')['p_value']
    app = calculate_frequentist_results(df, 'adj')['p_value']
    a = df.loc[df.treatment_group == 'Treatment', 'adj']; b = df.loc[df.treatment_group == 'Control', 'adj']
    correct = stats.ttest_ind(a, b, equal_var=False).pvalue
    return {'raw': raw, 'app': app, 'correct': correct,
            'var_red': 1 - df['adj'].var() / df['converted'].var(),
            'corr': np.corrcoef(df['converted'], df['pre'])[0, 1]}


# H5: HTE tab, logistic regression with an age x treatment term
def h5(seed):
    df = generate_synthetic_data(N, seed, BASE, 0.005, POP)
    df['t'] = (df.treatment_group == 'Treatment').astype(int)
    df['ac'] = df.age - df.age.mean()
    df['ax'] = df.ac * df.t
    m = sm.Logit(df.converted, sm.add_constant(df[['t', 'ac', 'ax']])).fit(disp=0)
    return m.pvalues['ax']


def wilson(k, n):
    lo, hi = sm.stats.proportion_confint(k, n, method='wilson')
    return [round(lo, 4), round(hi, 4)]


if __name__ == '__main__':
    out = {}
    with Pool() as pool:
        p = np.array(pool.map(h1, range(500)))
        k = int((p < 0.05).sum())
        out['H1'] = {'runs': 500, 'fpr': k / 500, 'ci': wilson(k, 500), 'pass': 0.035 <= k / 500 <= 0.065}

        r = pool.map(h2, range(2000))
        k = sum(x['any_look'] for x in r); kf = sum(x['final_only'] for x in r)
        out['H2'] = {'runs': 2000, 'fpr_any_look': k / 2000, 'ci': wilson(k, 2000),
                     'fpr_final_look_only': kf / 2000, 'pass': k / 2000 <= 0.05}

        r = pool.map(h3, range(5000))
        of = sum(x['of'] for x in r); po = sum(x['pocock'] for x in r); nv = sum(x['naive'] for x in r)
        out['H3'] = {'runs': 5000, 'nominal_one_sided': 0.025,
                     'of_fpr': of / 5000, 'of_ci': wilson(of, 5000),
                     'pocock_fpr': po / 5000, 'pocock_ci': wilson(po, 5000),
                     'naive_fpr': nv / 5000,
                     'of_boundaries': OF_AS_CODED.round(3).tolist(), 'pocock_boundary': round(POCOCK_AS_CODED, 3),
                     'pass_a': of / 5000 <= 0.0275, 'pass_b': 0.02 <= po / 5000 <= 0.03}

        vr = np.array(pool.map(h4a, range(200)))
        out['H4a'] = {'runs': 200, 'mean_variance_reduction': float(vr.mean()), 'pass': bool(vr.mean() >= 0.10)}

        r = pool.map(h4b, range(1000))
        pw = {m: float(np.mean([x[m] < 0.05 for x in r])) for m in ['raw', 'app', 'correct']}
        out['H4b'] = {'runs': 1000, 'lift_points': 0.5, 'power': pw,
                      'mean_variance_reduction': float(np.mean([x['var_red'] for x in r])),
                      'corr_pre_outcome': float(np.mean([x['corr'] for x in r])),
                      'pass': pw['app'] >= pw['correct'] - 0.02}

        p = np.array(pool.map(h5, range(500)))
        k = int((p < 0.05).sum())
        out['H5'] = {'runs': 500, 'power': k / 500, 'ci': wilson(k, 500), 'pass': k / 500 >= 0.80}

    json.dump(out, open(os.path.join(HERE, 'results_old.json'), 'w'), indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(json.dumps(out, indent=2, default=float))
