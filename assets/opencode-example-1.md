# OpenCode example 1 — retention priorities

**Question:** “Which segments should our retention team prioritise? Show evidence and a chart.”

**Harness:** OpenCode, using the project-local `customer-analytics` skill.

The seven screenshots record the skill selection, CLI calls, chart rendering and final business answer. The initial `segments` call encountered a transient DuckDB lock from a stale local process; the agent retried and completed the evidence-gathering flow. The final answer identifies Champions as the largest value-at-risk priority, Loyal as the best recovery target, and Needs Attention as the largest churn-risk pool, while stating the prediction and calibration limits.

## Charts used by this example

| Role | Chart | What it supports |
|---|---|---|
| Primary evidence | [Value at risk by segment](../charts/churn-risk-by-segment.png) | Which segment has the greatest retention exposure. |
| Supporting evidence | [Customers per RFM segment](../charts/segment-sizes.png) | The size of each retention audience. |
| Supporting evidence | [Customers with the most value at risk](../charts/top-at-risk.png) | Accounts to hand to the retention team. |
| Model evidence | [Churn permutation importance](../charts/feature-importance.png) | The global factors associated with churn predictions. |

## Capture sequence

1. [Skill and pipeline status](opencode-example-1-shot-1.png)
2. [Segment and risk evidence](opencode-example-1-shot-2.png)
3. [Chart rendering](opencode-example-1-shot-3.png)
4. [Model-report evidence](opencode-example-1-shot-4.png)
5. [Feature-importance evidence](opencode-example-1-shot-5.png)
6. [Final-answer preparation](opencode-example-1-shot-6.png)
7. [Decision, evidence and caveats](opencode-example-1-shot-7.png)
