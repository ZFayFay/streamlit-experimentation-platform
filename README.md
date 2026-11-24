# Experimentation lab

A Streamlit app that runs A/B test methods on simulated data where the true effect is known, so you can check each method instead of trusting it.

**Live app:** https://app-experimentation-platform-p4mcqhkrtw5q3f72yr2wrw.streamlit.app/

Tabs: sample size and power, A/A check, classic test, CUPED, group sequential test (O'Brien-Fleming and Pocock-type Lan-DeMets boundaries), Bayesian probability to be best, segment effects by age, and a Thompson sampling bandit. The sidebar shows the true effect in each simulated sample.

## Version 2 (November 2026)

Version 1 (June 2025) had bugs I found by testing it against its own simulator, with the checks written down first in [`audit/PREREGISTRATION.md`](audit/PREREGISTRATION.md).

| Tab | Pre-registered rule | 2025 app | Version 2 |
|---|---|---|---|
| A/A check | fake winners between 3.5% and 6.5% | 37.6% (fail) | 3.6% (pass) |
| Bayesian | its rule keeps fake winners at or below 5% | 19.4% when checked daily (fail) | 1.9% deciding once at the end (pass) |
| Sequential, O'Brien-Fleming | at most 2.75% one-sided | 3.2% (fail) | 2.3% (pass) |
| Sequential, Pocock | between 2.0% and 3.0% | 1.6% (fail) | 2.5% (pass) |
| CUPED, variance | removes at least 10% | 0.3% (fail) | 23.2% (pass) |
| CUPED, power | within 2 points of a correct analysis | 35.5% vs 44.8% (fail) | 42.8% vs 42.8% (pass) |
| Segments by age | finds the built-in effect 80%+ of the time | 96.6% (pass) | 92.4% (pass) |

What was wrong in version 1:
- The simulator added an age effect to the treatment group even with the lift set to zero, about +0.5 points on average.
- The Bayesian tab said peeking was safe. It isn't.
- The sequential boundaries used a shortcut formula, and the "Pocock" line was a Bonferroni split.
- The simulated pre-period metric barely predicted the outcome, and the CUPED test used the binomial variance formula on the adjusted metric, which threw away the variance CUPED removed.

## Files

- `ab_testing_implementation_3.py`: the app (file name kept so the deployment keeps working)
- `experiments.py`: every statistical function the app uses
- `audit/`: pre-registration, the 2025 app (`app_2025.py`), the checks (`audit.py`, `audit_v2.py`), results and figures

Run locally: `pip install -r requirements.txt && streamlit run ab_testing_implementation_3.py`

References: CUPED, Deng, Xu, Kohavi and Walker (2013). Alpha spending, Lan and DeMets (1983). Bayesian peeking, Frånberg and Schultzberg (Spotify, 2026).
