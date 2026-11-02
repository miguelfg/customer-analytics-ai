# Claude Desktop example 2 — high-value customers at risk of churn

**Question:** “Find five customers with predicted 90-day value above 500 who are at risk of churn. Explain the prioritisation.”

**Harness:** Claude Desktop, using the project-local `customer-analytics` skill and the `ca` CLI (JSON output only). Pipeline status was checked first: all 11 steps `ok`, last run 2026-11-02, 5,836 scored customers.

## Recommendation

Send a personal win-back contact to these five customers, in this order. All are Champions; they rank highest by **value at risk** (`churn_prob × ltv_90d`) among the 437 customers with `ltv_90d > 500`.

| Rank | Customer ID | Segment | Churn probability | Predicted 90-day value | Value at risk |
|---|---|---|---|---|---|
| 1 | 12477 | Champions (Germany) | 0.178 | 3,250.04 | 578.93 |
| 2 | 13081 | Champions (UK) | 0.051 | 10,550.39 | 536.06 |
| 3 | 14088 | Champions (UK) | 0.157 | 3,278.43 | 514.44 |
| 4 | 15838 | Champions (UK) | 0.041 | 9,857.76 | 406.47 |
| 5 | 17107 | Champions (UK) | 0.157 | 2,013.68 | 315.98 |

## Prioritisation

Ranking by value at risk weighs how likely a customer is to leave against how much they are expected to spend. That puts 13081 and 15838 (very high value, low churn probability) alongside 12477, 14088 and 17107 (moderate value, higher churn probability). Ranking by churn probability alone would send the team to lower-value accounts.

**Important:** the standard `ca at-risk` command requires `churn_prob >= 0.5`. Among customers with predicted 90-day value above 500, the highest churn probability is 0.251, so `ca --json at-risk --ltv-min 500 --limit 5` returns an empty list. The five customers above come from a read-only `ca --json query` on `customer_scores` ordered by `value_at_risk`. They are the *highest-exposure* high-value customers, not customers the model flags as likely churners (>= 0.5). The `top-at-risk` chart plots the same five.

## Charts

- [Customers with the most value at risk](claude-desktop-example-2/charts/top-at-risk.png) — the five customers above.
- [Value at risk by segment](claude-desktop-example-2/charts/churn-risk-by-segment.png) — Champions carry the largest segment exposure.
- [Churn reliability curve](claude-desktop-example-2/charts/calibration.png) — how far predicted probabilities match observed churn.

## Caveats on the 90-day prediction

- `churn_prob` and `ltv_90d` are 90-day estimates from a single snapshot; behaviour after the snapshot date is not reflected.
- Value at risk is a product of two independent model outputs, so errors in either carry through.
- Holdout quality is moderate: PR-AUC 0.823 for the HGB churn model versus 0.800 for the recency baseline (ROC-AUC 0.805 vs 0.763), so the gain over a simple recency rule is small.
- Calibration is imperfect: in the lowest-probability bin the mean prediction is 0.103 against an observed rate of 0.065, so probabilities are overstated at the low end.
- Churn probabilities for these five are low in absolute terms (0.04–0.18). They are prioritised by money exposed, and a discount offer may not be warranted for the lowest-probability accounts.
- Churn is a global statistical association, not a causal explanation for any individual customer.
