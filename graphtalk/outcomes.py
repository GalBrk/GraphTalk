"""The one rule for a response that hit the token budget.

Every response has exactly one outcome: correct, wrong or truncated. A response
that used the whole budget is truncated whatever its abandoned text says (the
text it stopped on can match the gold answer by accident), and it is never
labelled wrong or dropped. Analyses report the three outcomes as shares of all
responses and a primer's effect as the paired change in each share. Quantities
that describe an answer (error size, yes-rate, false alarms, response text) use
finished responses only. A cell whose truncated share is FLAG or more is
flagged. Rule R1 of docs/plans/2026-09-24-single-source-of-truth.md.
"""
import numpy as np

CORRECT, WRONG, TRUNCATED = "correct", "wrong", "truncated"
OUTCOMES = (CORRECT, WRONG, TRUNCATED)
FLAG = 0.15


def outcome(exact, hit_cap):
  """Elementwise outcome: `exact` is the scorer's 1/0, `hit_cap` whether the
  generation used the whole budget."""
  exact = np.asarray(exact, dtype=float)
  cap = np.asarray(hit_cap, dtype=bool)
  return np.where(cap, TRUNCATED, np.where(exact == 1, CORRECT, WRONG))


def shares(outcomes):
  """Share of each outcome among all the responses given; they sum to 1."""
  o = np.asarray(outcomes)
  if o.size == 0:
    raise ValueError("no responses to take shares of")
  return {k: float(np.mean(o == k)) for k in OUTCOMES}


def flagged(outcomes):
  """True when the truncated share reaches FLAG."""
  return shares(outcomes)[TRUNCATED] >= FLAG
