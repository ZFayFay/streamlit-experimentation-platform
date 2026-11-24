"""Figures for the post, from results_old.json and results_v2.json."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
old = json.load(open(os.path.join(HERE, 'results_old.json')))
new = json.load(open(os.path.join(HERE, 'results_v2.json')))
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
RED, GREEN, GREY = '#c53030', '#2f855a', '#94a3b8'


def pc(x, d=1):
    return f"{x * 100:.{d}f}%"


# ---- fig0: scorecard
rows = [
    ("A/A check", "Fake winners 3.5% to 6.5%", pc(old['H1']['fpr']), pc(new['V1']['fpr']), old['H1']['pass'], new['V1']['pass']),
    ("Bayesian tab", "Its rule keeps fake winners at 5% or less",
     pc(old['H2']['fpr_any_look']) + " (daily, 95%)", pc(new['V2']['fpr_recommended_rule']) + " (once, 97.5%)",
     old['H2']['pass'], new['V2']['pass']),
    ("Sequential: O'Brien-Fleming", "At most 2.75% (one-sided, nominal 2.5%)", pc(old['H3']['of_fpr']), pc(new['V3']['of_fpr']),
     old['H3']['pass_a'], new['V3']['pass_a']),
    ("Sequential: Pocock", "Between 2.0% and 3.0%", pc(old['H3']['pocock_fpr']), pc(new['V3']['pocock_fpr']),
     old['H3']['pass_b'], new['V3']['pass_b']),
    ("CUPED: variance", "Removes at least 10%", pc(old['H4a']['mean_variance_reduction']), pc(new['V4a']['mean_variance_reduction']),
     old['H4a']['pass'], new['V4a']['pass']),
    ("CUPED: power", "Within 2 pts of a correct analysis",
     f"{pc(old['H4b']['power']['app'])} vs {pc(old['H4b']['power']['correct'])}",
     f"{pc(new['V4b']['power']['app'])} vs {pc(new['V4b']['power']['ref'])}", old['H4b']['pass'], new['V4b']['pass']),
    ("Segments (HTE)", "Finds the age effect 80%+ of the time", pc(old['H5']['power']), pc(new['V5']['power']),
     old['H5']['pass'], new['V5']['pass']),
]
fig, ax = plt.subplots(figsize=(10.5, 4.4)); ax.axis('off')
xs = [0.0, 0.25, 0.60, 0.80]
heads = ['Tab', 'Pre-registered rule', '2025 app', 'Version 2']
y, step = 1.0, 1.0 / (len(rows) + 1.4)
for x, h in zip(xs, heads):
    ax.text(x, y, h, fontweight='bold', va='top', transform=ax.transAxes)
for r in rows:
    y -= step
    ax.plot([0, 1], [y + step * 0.3] * 2, color='#e2e8f0', lw=0.8, transform=ax.transAxes)
    ax.text(xs[0], y, r[0], va='top', transform=ax.transAxes, fontsize=9)
    ax.text(xs[1], y, r[1], va='top', transform=ax.transAxes, fontsize=8.5)
    for x, val, ok in [(xs[2], r[2], r[4]), (xs[3], r[3], r[5])]:
        ax.text(x, y, ('Pass  ' if ok else 'Fail  ') + val, va='top', transform=ax.transAxes, fontsize=8.5,
                color=GREEN if ok else RED, fontweight='bold')
ax.set_title('The same pre-registered checks on the 2025 app and on version 2', loc='left', fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'fig0_scorecard.png'), dpi=200, bbox_inches='tight')

# ---- fig1: fake winners when nothing changed
labels = ['A/A check\n(two-sided, 5%)', 'Bayesian tab\n(one-sided, 5%)', "Sequential, O'Brien-Fleming\n(one-sided, 2.5%)"]
o = [old['H1']['fpr'], old['H2']['fpr_any_look'], old['H3']['of_fpr']]
n = [new['V1']['fpr'], new['V2']['fpr_recommended_rule'], new['V3']['of_fpr']]
nominal = [0.05, 0.05, 0.025]
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(3)
ax.bar(x - 0.18, np.array(o) * 100, 0.36, color=RED, label='2025 app')
ax.bar(x + 0.18, np.array(n) * 100, 0.36, color=GREEN, label='Version 2')
for i, v in enumerate(nominal):
    ax.plot([i - 0.42, i + 0.42], [v * 100] * 2, color='black', ls='--', lw=1)
for i, (a, b) in enumerate(zip(o, n)):
    ax.text(i - 0.18, a * 100 + 0.8, pc(a), ha='center', fontsize=9)
    ax.text(i + 0.18, b * 100 + 0.8, pc(b), ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Tests that declared a winner (%)')
ax.set_title('When nothing changed, how often did each tab find a winner? Dashed line: what it should be', loc='left', fontsize=10)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'fig1_false_winners.png'), dpi=200)

# ---- fig2: CUPED power
fig, ax = plt.subplots(figsize=(7.5, 3.6))
lab = ['No CUPED', "2025 app's CUPED", 'Correct CUPED']
vals = [old['H4b']['power']['raw'], old['H4b']['power']['app'], old['H4b']['power']['correct']]
ax.barh(lab[::-1], np.array(vals[::-1]) * 100, color=[GREEN, RED, GREY])
for i, v in enumerate(vals[::-1]):
    ax.text(v * 100 + 0.5, i, pc(v), va='center')
ax.set_xlabel('Share of tests that found a real +0.5 point lift (%)')
ax.set_xlim(0, 55)
ax.set_title('Same data, same covariate. The old CUPED test lost power', loc='left', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'fig2_cuped_power.png'), dpi=200)
print('figures written')
