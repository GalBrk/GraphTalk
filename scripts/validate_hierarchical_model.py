"""Statistical validation for graphtalk/hierarchical_model.py -- not a
pytest test. Real MCMC at a trustworthy sample size is minutes, not
seconds; this belongs in an explicitly-run script, the same way
`scripts/characterize_non_termination.py` is "one-off diagnostics... not a
permanent pipeline stage" rather than part of `pytest`. `tests
/test_hierarchical_model.py` covers the fast, deterministic parts (see its
own module docstring) and one small, bounded end-to-end smoke test; this
script covers the three checks Phase 1.2.2's plan called for that don't
fit that mold:

1. **Prior predictive check** (`--check prior`): sample from the model's
   own prior, confirm it can generate data at all and that the implied
   baseline accuracy distribution is broad (weakly informative, not
   accidentally pinned near 0 or 1 by the prior alone).
2. **Posterior predictive check** (`--check posterior`, default; needs a
   real fit): after fitting on synthetic data with a known effect, resample
   from the posterior and confirm the simulated per-condition accuracy
   rates land close to the true generating rates -- the model fits what it
   was built to fit.
3. **Shrinkage demonstration** (`--check shrinkage`): the property Phase
   1.2.2 exists for. Construct two models sharing one true condition
   effect -- one with many instances (a precise per-model estimate), one
   with very few (a noisy one) -- and confirm the noisy model's `total_mean`
   sits closer to the shared, pooled `beta_condition` than its own raw
   per-model mean difference does. A model that doesn't shrink is not
   doing partial pooling; this is the one property that can't be inferred
   from `tests/test_hierarchical_model.py`'s synthetic-trace tests, since
   those hand the posterior in already fitted rather than fitting it.

`--check real-data` additionally fits on `analysis/sweep_frame.csv` at
real settings (the `fit()` defaults: 1000 draws/1000 tune/2 chains) and
prints convergence diagnostics -- off by default (this is the expensive
one; on a working compiled PyTensor backend it's tractable, on a
pure-Python fallback it is not, see the module's own performance note in
`graphtalk/hierarchical_model.py`).

  PYTHONPATH=. .venv/bin/python scripts/validate_hierarchical_model.py --check shrinkage
"""

import argparse
import random

import numpy as np
import pandas as pd

from graphtalk import hierarchical_model as hm


def _synthetic_frame(control_rate, treatment_rate, n, seed, model):
  rng = random.Random(seed)
  rows = []
  for i in range(n):
    rows.append({
        "model": model, "instance_id": f"task/{i}", "condition": "none",
        "exact": 1.0 if rng.random() < control_rate else 0.0,
    })
    rows.append({
        "model": model, "instance_id": f"task/{i}", "condition": "treat",
        "exact": 1.0 if rng.random() < treatment_rate else 0.0,
    })
  return pd.DataFrame(rows)


def check_prior(seed: int) -> None:
  print("=== Prior predictive check ===")
  frame = _synthetic_frame(0.5, 0.5, n=20, seed=seed, model="m")
  model, _ = hm.build_model(frame)
  import pymc as pm
  with model:
    prior = pm.sample_prior_predictive(samples=500, random_seed=seed)
  # Per-draw average accuracy (averaged over rows, not over draws) -- its
  # spread across draws is what shows whether the prior is weakly
  # informative (broad) or accidentally pinned near 0/1.
  per_draw_rate = prior.prior_predictive["y_obs"].mean(dim="y_obs_dim_0")
  print(f"  prior predictive per-row accuracy: generated {per_draw_rate.size} draws, "
        f"mean={float(np.asarray(per_draw_rate).mean()):.3f}, "
        f"std={float(np.asarray(per_draw_rate).std()):.3f}")
  spread = float(np.asarray(per_draw_rate).std())
  verdict = "OK -- broad, not pinned" if spread > 0.05 else "NARROW -- check priors"
  print(f"  spread across prior draws: {spread:.3f} ({verdict})")


def check_posterior(seed: int, draws: int, tune: int) -> None:
  print("=== Posterior predictive check ===")
  true_control, true_treatment = 0.3, 0.75
  frame = _synthetic_frame(true_control, true_treatment, n=40, seed=seed, model="m")
  trace, index = hm.fit(frame, draws=draws, tune=tune, chains=2, seed=seed,
                         progressbar=False)
  import pymc as pm
  model, _ = hm.build_model(frame)
  with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=seed, progressbar=False)
  simulated = np.asarray(ppc.posterior_predictive["y_obs"]).reshape(-1, len(frame))
  frame_reset = frame.reset_index(drop=True)
  control_mask = (frame_reset["condition"] == "none").to_numpy()
  sim_control_rate = simulated[:, control_mask].mean()
  sim_treat_rate = simulated[:, ~control_mask].mean()
  print(f"  true control rate {true_control:.2f}, simulated {sim_control_rate:.3f}")
  print(f"  true treatment rate {true_treatment:.2f}, simulated {sim_treat_rate:.3f}")
  ok = (abs(sim_control_rate - true_control) < 0.1
        and abs(sim_treat_rate - true_treatment) < 0.1)
  print(f"  verdict: {'OK -- within 0.1 of the generating rates' if ok else 'MISMATCH'}")


def _empirical_logit_delta(sub_frame: pd.DataFrame) -> float:
  """A naive, unpooled log-odds effect estimate from raw counts alone --
  the same scale `beta_condition`/`gamma`/`total` are fit on (see
  `graphtalk/hierarchical_model.py`'s "Why logit, not identity link"), so
  it's a fair baseline to compare the hierarchical model's *shrunk*
  per-model estimate against. Haldane-Anscombe continuity correction (+0.5
  successes, +1 to the denominator) avoids +/-inf at n=4 when a condition
  happens to land all-0 or all-1, which a tiny sample does easily."""
  def logit_rate(s: pd.Series) -> float:
    p = (s.sum() + 0.5) / (len(s) + 1.0)
    return float(np.log(p / (1 - p)))
  control = sub_frame[sub_frame["condition"] == "none"]["exact"]
  treat = sub_frame[sub_frame["condition"] == "treat"]["exact"]
  return logit_rate(treat) - logit_rate(control)


def check_shrinkage(seed: int, draws: int, tune: int) -> None:
  print("=== Shrinkage demonstration ===")
  true_effect_rate = 0.75  # shared true P(correct | treat) for both models
  control_rate = 0.5
  true_logit_delta = np.log(true_effect_rate / (1 - true_effect_rate)) - np.log(
      control_rate / (1 - control_rate)
  )
  # m_precise: 60 instances, a reliable per-model estimate on its own.
  # m_noisy: 4 instances, deliberately too few to estimate anything
  # reliably alone -- exactly the regime partial pooling is for. Disjoint
  # instance_id ranges (offset by 1000) so the two models don't
  # accidentally share `u_instance` graph-difficulty terms -- they're
  # meant to be independent synthetic datasets, not the same graphs.
  precise = _synthetic_frame(control_rate, true_effect_rate, n=60, seed=seed, model="m_precise")
  noisy = _synthetic_frame(control_rate, true_effect_rate, n=4, seed=seed + 1, model="m_noisy")
  noisy["instance_id"] = noisy["instance_id"].str.replace(
      r"task/(\d+)", lambda m: f"task/{int(m.group(1)) + 1000}", regex=True
  )
  frame = pd.concat([precise, noisy], ignore_index=True)

  raw_noisy_delta = _empirical_logit_delta(noisy)
  raw_precise_delta = _empirical_logit_delta(precise)
  print(f"  true log-odds effect: {true_logit_delta:+.3f}")
  print(f"  raw (unpooled) log-odds delta -- m_precise (n=60): {raw_precise_delta:+.3f}, "
        f"m_noisy (n=4): {raw_noisy_delta:+.3f}")

  trace, index = hm.fit(frame, draws=draws, tune=tune, chains=2, seed=seed,
                         progressbar=False)
  result = hm.fit_summary(trace, index).set_index("model")
  pooled_beta = result["beta_condition_mean"].iloc[0]
  precise_total = result.loc["m_precise", "total_mean"]
  noisy_total = result.loc["m_noisy", "total_mean"]
  print(f"  pooled (population-level) beta_condition: {pooled_beta:+.3f}")
  print(f"  shrunk total (same log-odds scale) -- m_precise: {precise_total:+.3f}, "
        f"m_noisy: {noisy_total:+.3f}")

  noisy_shrinkage = abs(raw_noisy_delta - noisy_total)
  precise_shrinkage = abs(raw_precise_delta - precise_total)
  print(f"  |raw - shrunk| (both log-odds scale) -- m_precise: {precise_shrinkage:.3f}, "
        f"m_noisy: {noisy_shrinkage:.3f}")
  verdict = ("OK -- the noisy (n=4) model's estimate moved further from its own "
             "raw log-odds delta than the precise (n=60) model's did"
             if noisy_shrinkage > precise_shrinkage else
             "UNEXPECTED -- the noisy model did not shrink more than the precise one")
  print(f"  verdict: {verdict}")
  print(f"  (also compare distance-to-truth: |raw - true| precise="
        f"{abs(raw_precise_delta - true_logit_delta):.3f} vs shrunk="
        f"{abs(precise_total - true_logit_delta):.3f}; noisy raw="
        f"{abs(raw_noisy_delta - true_logit_delta):.3f} vs shrunk="
        f"{abs(noisy_total - true_logit_delta):.3f} -- shrinkage should pull "
        f"m_noisy's shrunk estimate closer to the truth than its raw one was, "
        f"since its raw estimate is the noisier of the two)")


def check_real_data(frame_path: str, draws: int, tune: int, seed: int) -> None:
  print(f"=== Real-data fit ({frame_path}) ===")
  print("  WARNING: at real settings (draws/tune defaults) this is minutes on a "
        "working compiled PyTensor backend, and impractically slow on a "
        "pure-Python fallback (see graphtalk/hierarchical_model.py). "
        "Pass --draws/--tune to reduce for a quicker, lower-confidence check.")
  frame = pd.read_csv(frame_path)
  main_sweep = frame[(~frame["is_think"]) & (frame["failure_type"] != "non_terminating")]
  trace, index = hm.fit(main_sweep, draws=draws, tune=tune, chains=2, seed=seed,
                         progressbar=True)
  result = hm.fit_summary(trace, index)
  pd.set_option("display.width", 200)
  print(result.to_string(index=False))
  bad = result[(result["r_hat"] > 1.01) | (result["ess_bulk"] < 400)]
  if not bad.empty:
    print(f"\n  WARNING: {len(bad)} rows failed convergence thresholds:")
    print(bad[["model", "condition", "r_hat", "ess_bulk"]].to_string(index=False))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--check", choices=("prior", "posterior", "shrinkage", "real-data"),
                       default="posterior")
  parser.add_argument("--frame", default="analysis/sweep_frame.csv",
                       help="--check real-data only")
  parser.add_argument("--draws", type=int, default=300)
  parser.add_argument("--tune", type=int, default=300)
  parser.add_argument("--seed", type=int, default=1234)
  args = parser.parse_args()

  if args.check == "prior":
    check_prior(args.seed)
  elif args.check == "posterior":
    check_posterior(args.seed, args.draws, args.tune)
  elif args.check == "shrinkage":
    check_shrinkage(args.seed, args.draws, args.tune)
  elif args.check == "real-data":
    check_real_data(args.frame, args.draws, args.tune, args.seed)


if __name__ == "__main__":
  main()
