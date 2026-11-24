"""Same checks as audit.py (PREREGISTRATION.md, V1-V5), run on the v2 functions the app calls.
Writes results_v2.json. Run: python audit_v2.py"""
import json, os, sys
import numpy as np
import statsmodels.api as sm
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import experiments as E

N, BASE, LIFT, SLOPE = 20000, 0.05, 0.005, -0.01      # the app's default sidebar settings
HERE = os.path.dirname(os.path.abspath(__file__))
K = 5
OF, PO = E.gst_boundaries(K, kind='obf'), E.gst_boundaries(K, kind='pocock')


def v1(s):
    df, _ = E.simulate(N, BASE, 0.0, s)
    return E.two_proportion_test(df)['p_value']


def v2(s):
    df, _ = E.simulate(N, BASE, 0.0, s)
    pr = E.daily_prob_better(df)['prob_better'].values
    return {'end975': bool(pr[-1] > 0.975), 'daily95': bool((pr > 0.95).any())}


def v3(s):
    df, _ = E.simulate(N, BASE, 0.0, s)
    z = E.z_at_looks(df, K)
    return {'of': bool((z >= OF).any()), 'pocock': bool((z >= PO).any()), 'naive': bool((z >= 1.96).any())}


def v4a(s):
    df, _ = E.simulate(N, BASE, LIFT, s, age_slope=SLOPE)
    d, _ = E.cuped(df)
    return 1 - d.converted_cuped.var() / d.converted.var()


def v4b(s):
    df, _ = E.simulate(N, BASE, LIFT, s, age_slope=SLOPE)
    d, _ = E.cuped(df)
    raw = E.mean_difference_test(d, 'converted')['p_value']
    app = E.mean_difference_test(d, 'converted_cuped')['p_value']
    # independent reference: regression adjustment with robust standard errors
    X = np.column_stack([np.ones(len(d)), (d.treatment_group == 'Treatment').astype(float),
                         d.past_activity - d.past_activity.mean()])
    ref = sm.OLS(d.converted.astype(float), X).fit(cov_type='HC2').pvalues.iloc[1]
    return {'raw': raw, 'app': app, 'ref': ref}


def v5(s):
    df, _ = E.simulate(N, BASE, LIFT, s, age_slope=SLOPE)
    return E.age_interaction(df).pvalues['treatment_x_age']


def wilson(k, n):
    lo, hi = sm.stats.proportion_confint(k, n, method='wilson')
    return [round(lo, 4), round(hi, 4)]


if __name__ == '__main__':
    path = os.path.join(HERE, 'results_v2.json')
    out = json.load(open(path)) if os.path.exists(path) else {}   # blocks already run are kept

    def save():
        json.dump(out, open(path, 'w'), indent=2)

    with Pool() as pool:
        if 'V1' not in out:
            p = np.array(pool.map(v1, range(500))); k = int((p < 0.05).sum())
            out['V1'] = {'runs': 500, 'fpr': k / 500, 'ci': wilson(k, 500), 'pass': bool(0.035 <= k / 500 <= 0.065)}
            save()
        if 'V2' not in out:
            r = pool.map(v2, range(2000))
            k = sum(x['end975'] for x in r); kd = sum(x['daily95'] for x in r)
            out['V2'] = {'runs': 2000, 'fpr_recommended_rule': k / 2000, 'ci': wilson(k, 2000),
                         'fpr_daily_95': kd / 2000, 'ci_daily': wilson(kd, 2000), 'pass': bool(k / 2000 <= 0.05)}
            save()
        if 'V3' not in out:
            r = pool.map(v3, range(5000))
            of = sum(x['of'] for x in r); po = sum(x['pocock'] for x in r); nv = sum(x['naive'] for x in r)
            out['V3'] = {'runs': 5000, 'of_fpr': of / 5000, 'of_ci': wilson(of, 5000),
                         'pocock_fpr': po / 5000, 'pocock_ci': wilson(po, 5000), 'naive_fpr': nv / 5000,
                         'of_boundaries': OF.round(3).tolist(), 'pocock_boundaries': PO.round(3).tolist(),
                         'pass_a': bool(of / 5000 <= 0.0275), 'pass_b': bool(0.02 <= po / 5000 <= 0.03)}
            save()
        if 'V4a' not in out:
            vr = np.array(pool.map(v4a, range(200)))
            out['V4a'] = {'runs': 200, 'mean_variance_reduction': float(vr.mean()), 'pass': bool(vr.mean() >= 0.10)}
            save()
        if 'V4b' not in out:
            r = pool.map(v4b, range(1000))
            pw = {m: float(np.mean([x[m] < 0.05 for x in r])) for m in ['raw', 'app', 'ref']}
            out['V4b'] = {'runs': 1000, 'lift_points': 0.5, 'power': pw, 'pass': bool(pw['app'] >= pw['ref'] - 0.02)}
            save()
        if 'V5' not in out:
            p = np.array(pool.map(v5, range(500))); k = int((p < 0.05).sum())
            out['V5'] = {'runs': 500, 'power': k / 500, 'ci': wilson(k, 500), 'pass': bool(k / 500 >= 0.80)}
            save()
    print(json.dumps(out, indent=2))
