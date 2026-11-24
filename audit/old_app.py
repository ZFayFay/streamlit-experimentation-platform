"""Pure functions copied verbatim from the 2025 app (ab_testing_implementation_3.py), without Streamlit."""
import re, numpy as np, pandas as pd, scipy.stats as stats
from datetime import datetime
import os
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_2025.py')).read()
_ns = {'np': np, 'pd': pd, 'stats': stats, 'datetime': datetime, 'Dict': dict, 'List': list}
for _n in ['generate_synthetic_data', 'calculate_frequentist_results', 'design_binomial_experiment']:
    exec(re.search(r'(def ' + _n + r'\(.*?)(?=\n(?:def |@|class |# ---))', _src, re.S).group(1), _ns)
generate_synthetic_data = _ns['generate_synthetic_data']
calculate_frequentist_results = _ns['calculate_frequentist_results']
design_binomial_experiment = _ns['design_binomial_experiment']
POP = {'device_dist': {'Mobile': .6, 'Desktop': .35, 'Tablet': .05}, 'segment_dist': {'New': .4, 'Returning': .5, 'VIP': .1}}
