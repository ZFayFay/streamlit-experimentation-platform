# Pre-registration: auditing my own 2025 experimentation app

Written before running the audit below. Committed to git so the date is visible.

## What is being tested

The Streamlit app in this repo (`ab_testing_implementation_3.py`, June 2025) and the Medium post that went with it. Each tab makes a claim, in its text or by what it shows. I test each claim inside the app's own simulator, where the true effect is known.

All runs use the app's own functions, copied without changes (`audit/old_app.py`), at the app's default settings unless stated: 20,000 users, 5% base conversion, seed i for run i.

## What I had already seen before writing this

While reading the code I noticed that the simulator adds an age effect to the treatment group even when the uplift slider is 0. I ran the app's A/A tab once (500 runs) and got a 37.6% false positive rate. I also ran one default dataset (seed 42) and saw a CUPED variance reduction of 0.3% and an age interaction p-value of 0.0002. So H1 is already known to fail, and H4a and H5 are not blind. They are listed anyway so every tab is covered.

## Hypotheses and pass rules

**H1. A/A tab.** With the uplift slider at 0, the share of runs with p < 0.05 is between 3.5% and 6.5% (500 runs). Expected: fail (seen).

For H2 and H3 I use a truly null version of the simulator (age effect removed, everything else identical), so they test the method and not the simulator bug.

**H2. Bayesian tab.** The tab says Bayesian testing "naturally allows for checking results at any time without the peeking problem". Rule tested: ship Treatment if P(Treatment better) > 95% at any of the 14 daily looks the tab plots. Pass if the false positive rate is at most 5% (2,000 null runs). Expected: fail.

**H3. Sequential tab.** The chart stops when the z-score crosses a boundary from above (one-sided, nominal 2.5%). With 5 looks, (a) the "O'Brien-Fleming" boundary as coded has a false positive rate of at most 2.75%; (b) the "Pocock" boundary as coded has a false positive rate between 2.0% and 3.0% (5,000 null runs). Expected: unsure.

**H4. CUPED tab.** (a) At default settings, CUPED cuts the variance of the metric by at least 10% (mean over 200 runs). Expected: fail (seen once). (b) With a pre-period covariate that is actually predictive (a user's own conversion in the 4 weeks before the test, from the same underlying propensity), the app's CUPED test has power within 2 points of a correct CUPED analysis (t-test on the adjusted metric) for a true lift of 0.5 points, 1,000 runs each. Expected: fail, because the app's z-test uses the binomial variance formula on the adjusted metric.

**H5. HTE tab.** At default settings the age interaction the simulator builds in is detected (p < 0.05) in at least 80% of 500 runs. Expected: pass.

## The fixed app (v2)

After the audit I will rebuild the app. V1 to V5 are the same checks run on v2, with the same pass rules where they apply (H2's rule becomes: the tab's recommended stopping rule keeps the false positive rate at or below 5%). v2 must pass all five before it replaces the live app.

## Reporting

Every result is reported, pass or fail, in the post and in `audit/results.json`. Any change to this plan after the first run is listed as a deviation.

## Deviations (added after the runs)

1. The first run of `audit_v2.py` crashed in the V4b block on a pandas indexing error (`pvalues[1]` instead of `pvalues.iloc[1]`). I fixed the line and reran every block from scratch with the same seeds and settings. No rule or setting changed.
2. For V4b the "correct analysis" reference is a regression adjustment with robust (HC2) standard errors, because v2's own CUPED test is now the Welch t-test I used as the reference for H4b. Same idea, an independent implementation.
3. Results are in `results_old.json` (the 2025 app) and `results_v2.json` (version 2) instead of one `results.json`.
