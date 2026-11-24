"""Fill the {placeholders} in medium_draft.md from the results files -> medium_post.md"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
o = json.load(open(os.path.join(H, 'results_old.json'))); n = json.load(open(os.path.join(H, 'results_v2.json')))
f = lambda x: f"{x * 100:.1f}%"
vals = {'H1': f(o['H1']['fpr']), 'H2': f(o['H2']['fpr_any_look']), 'H2final': f(o['H2']['fpr_final_look_only']),
        'H3of': f(o['H3']['of_fpr']), 'H3po': f(o['H3']['pocock_fpr']), 'H4a': f(o['H4a']['mean_variance_reduction']),
        'H4corr': f"{o['H4b']['corr_pre_outcome']:.2f}", 'H4correct': f(o['H4b']['power']['correct']),
        'H4raw': f(o['H4b']['power']['raw']), 'H4app': f(o['H4b']['power']['app']), 'H5': f(o['H5']['power']),
        'V1': f(n['V1']['fpr']), 'V2': f(n['V2']['fpr_recommended_rule']), 'V2daily': f(n['V2']['fpr_daily_95']),
        'V3of': f(n['V3']['of_fpr']), 'V3po': f(n['V3']['pocock_fpr']), 'V4a': f(n['V4a']['mean_variance_reduction']),
        'V4app': f(n['V4b']['power']['app']), 'V4raw': f(n['V4b']['power']['raw']), 'V5': f(n['V5']['power'])}
s = open(os.path.join(H, 'medium_draft.md')).read()
for k in sorted(vals, key=len, reverse=True):
    s = s.replace('{' + k + '}', vals[k])
assert '{' not in s, s[s.index('{') - 50: s.index('{') + 50]
open(os.path.join(H, 'medium_post.md'), 'w').write(s)
print(json.dumps(vals, indent=1))
