# Claude Desktop example — at-risk customer prioritisation

**Question:** “Find five customers with predicted 90-day value above 500 who are at risk of churn. Explain the prioritisation.”

The recording shows the project-local `customer-analytics` skill guiding Claude Desktop through real CLI outputs. In this snapshot, no customer with predicted 90-day value above 500 meets the product’s hard `churn_prob >= 0.5` at-risk cutoff. The agent states that constraint, then provides the more useful fallback: the five qualifying customers ranked by `value_at_risk` and two charts that support the retention decision.

This is deliberate: the example demonstrates evidence-based analysis, including when a requested segment is empty, rather than inventing a result.

[Watch the 44-second recording](claude-desktop-at-risk-example.mp4)
