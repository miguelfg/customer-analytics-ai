---
name: customer-analytics
description: Answer business questions about customer churn, lifetime value (LTV), RFM segments, at-risk customers, single-customer profiles and pipeline health of the Online Retail II demo, by running the `ca` CLI and attaching its charts. Use when the user asks who is likely to churn, which customers or segments are most valuable or at risk, what offer to send, how big a segment is, or whether the analytics data is fresh.
---

# Customer analytics

Answers come from the `ca` CLI (run from the project root with `uv run ca ...`). It reads the
DuckDB file `data/demo.duckdb` and every command prints one JSON object:
`{"ok": true, "command": ..., "result": ...}` or `{"ok": false, "error": ...}` (exit code 2).

## Rules

1. Always pass `--json` (`uv run ca --json <command>`) and parse the JSON. Never scrape text output.
2. Run `uv run ca --json status` first. If a step is missing or `error`, run
   `uv run ca --json run-all`, then continue.
3. Only use numbers and customer ids that appear in command output. Never invent them.
4. Every business answer includes at least one chart from `ca --json chart <name>`. Show the
   PNG path from `result.path` and quote numbers from `result.data` / `result.summary`.
5. If a command returns `ok: false`, read `error`, fix the call (or run the step it names) and
   retry once. Report the error if it still fails.

6. When asked *why* a customer/segment is at risk or how good the model is, call `model-report`
   and the `feature-importance` chart. These are global associations, not individual or causal explanations.

## Command reference

| Intent | Command |
|---|---|
| Pipeline health, freshness, row counts | `ca --json status` |
| Rebuild everything / a missing step | `ca --json run-all [--from STEP]` |
| Segment sizes, shares, churn, value at risk | `ca --json segments` |
| Customers likely to churn, by value at risk | `ca --json at-risk [--ltv-min N] [--segment S] [--limit N]` |
| One customer's scores, features, last invoices | `ca --json customer ID` |
| Ad-hoc read-only SQL (one SELECT/WITH) | `ca --json query "SELECT ..."` |
| Model quality, selected configuration, global drivers | `ca --json model-report` |
| Search model configurations | `ca --json tune` (or `tune-churn` / `tune-ltv --n-iter N`) |
| Model search trials | `ca --json chart model-comparison` |
| Global feature importance | `ca --json chart feature-importance [--segment churn|ltv]` |
| Churn probability calibration | `ca --json chart calibration` |
| List charts | `ca --json chart --list` |
| Render a chart (PNG + plotted data) | `ca --json chart NAME [--segment S]` |

Charts: `segment-sizes`, `ltv-distribution`, `churn-risk-by-segment`, `churn-lift`,
`retention-cohorts`, `rfm-heatmap`, `top-at-risk`, `dq-report`, `pipeline-funnel`,
`model-comparison`, `feature-importance`, `calibration`.
Segments: `Champions`, `Loyal`, `New`, `At Risk`, `Needs Attention`, `Hibernating`.

Useful tables for `query`: `customer_scores` (`churn_prob`, `ltv_90d`, `value_at_risk`,
`rfm_segment`, `country`), `segment_profiles`, `churn_metrics` (model vs `recency_baseline`),
`churn_lift`, `ltv_metrics`, `dq_report`, `pipeline_runs`.

## Answer format

1. **Decision**: the recommendation in one or two sentences.
2. **Numbers**: the 3–5 key figures, with their source command.
3. **Chart**: image path(s), e.g. `![segment sizes](charts/segment-sizes.png)`.
4. **Caveats**: model limits (snapshot date, 90-day horizon, holdout PR-AUC), filters used.

## Worked examples

**"Which high-LTV customers are likely to churn and what offer would you send?"**
`ca --json status` → `ca --json at-risk --ltv-min 500 --limit 10` → `ca --json chart top-at-risk`.
Answer: target the top 10 by `value_at_risk` with a personal win-back offer; list ids, churn
probability and LTV; attach `charts/top-at-risk.png`; caveat that `churn_prob` is a 90-day estimate.

**"Which segment loses us the most money if they churn? Show me a chart."**
`ca --json segments` (sorted by `value_at_risk`) → `ca --json chart churn-risk-by-segment`.
Answer: name the first segment with its value at risk, size and average churn probability;
attach the chart; caveat that value at risk = churn_prob × predicted 90-day LTV.

**"How healthy is our data pipeline?"**
`ca --json status` → `ca --json chart pipeline-funnel` (and `dq-report` for cleaning losses).
Answer: all steps `ok` or which failed, last run time, row counts per stage; attach the funnel.

## Install

```bash
ln -s $(pwd)/skills/customer-analytics ~/.claude/skills/
```

The Python agent with the same tools: `uv run ca --json ask "QUESTION"`, and the eval set with
`uv run ca --json ask --eval` (needs `ANTHROPIC_API_KEY`; model from `CA_MODEL`).
