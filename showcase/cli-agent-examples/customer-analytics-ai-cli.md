---
marp: true
theme: default
paginate: true
size: 16:9
---

<style>
section { font-size: 23px; }
table { font-size: 18px; }
blockquote { font-size: 23px; }
header { font-size: 14px; color: #94a3b8; }
header strong { color: #2563eb; }
section.title-slide { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); color: white; }
section.title-slide h1, section.title-slide h2, section.title-slide p, section.part-product h1, section.part-product p, section.part-evidence h1, section.part-evidence p, section.part-trust h1, section.part-trust p { color: white; }
section.part-product, section.part-evidence, section.part-trust { color: white; }
section.part-product { background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 100%); }
section.part-evidence { background: linear-gradient(135deg, #064e3b 0%, #047857 100%); }
section.part-trust { background: linear-gradient(135deg, #3d1e5c 0%, #5a2d8e 100%); }
.columns { display: flex; gap: 28px; align-items: center; }
.columns > * { flex: 1; }
.visual { max-height: 430px; display: block; margin: auto; }
.shot { max-height: 440px; display: block; margin: auto; }
.small { font-size: 17px; color: #64748b; }
</style>

<!-- _class: lead title-slide -->

# Customer Analytics AI
## From retail data to defensible retention decisions

**One CLI**: reproducible pipeline and charts

**Two harnesses**: OpenCode and Claude Desktop
**Evidence**: live outputs, model diagnostics and stated limits

---

<!-- _header: "**The problem** > Evidence > Agent examples > Trust" -->

# What a retention team needs

1. **Which customer groups deserve attention first?**
2. **Which accounts are worth handing to a retention team?**
3. **How good is the model and where are its limits?**
4. **Can an AI agent answer using evidence rather than guesses?**

---

<!-- _header: "" -->
<!-- _class: lead part-product -->

# One inspectable analytics pipeline

**Load → clean → features → tune → model → segment → score → chart**

---

<!-- _header: "**The product** > Evidence > Agent examples > Trust" -->

# A CLI for the whole decision path

<div class="columns">
<div>

- `status` confirms pipeline health
- `segments` identifies customer groups
- `at-risk` ranks action candidates
- `model-report` exposes metrics and features
- `chart` creates evidence for a decision

</div>
<img class="visual" src="../../charts/pipeline-funnel.png" />
</div>

---

# Model selection is visible, not implicit

<div class="columns">
<img class="visual" src="../../charts/model-comparison.png" />
<div>

**Time-aware tuning** compares candidate models across expanding windows

**Churn quality**: PR-AUC 0.823 vs 0.800 recency baseline

**Business focus**: LTV selection optimises top-decile value capture

</div>
</div>

---

<!-- _header: "" -->
<!-- _class: lead part-evidence -->

# Two agents, one evidence contract

**Both examples use the same project-local skill and real CLI results**

---

<!-- _header: "The product > **OpenCode example** > Claude Desktop > Trust" -->

# OpenCode loads the skill and checks the pipeline

<img class="shot" src="../../assets/opencode-example-1-shot-1.png" />

<p class="small">Example question: Which segments should the retention team prioritise?</p>

---

# The recommendation is backed by charts

<div class="columns">
<img class="visual" src="../../charts/churn-risk-by-segment.png" />
<img class="visual" src="../../charts/feature-importance.png" />
</div>

Champions carry the largest value at risk; feature importance gives the global drivers behind the churn model.

---

<!-- _header: "The product > OpenCode example > **Claude Desktop** > Trust" -->

# Claude Desktop tests a customer-priority question

<img class="shot" src="../../assets/demos/claude-desktop-at-risk-example-poster.png" />

The agent uses the shared skill and live CLI outputs to answer a focused request.

---

# It handles an empty hard cutoff honestly

<div class="columns">
<div>

The requested high-value, high-risk filter was empty for this snapshot.

Instead of inventing customers, the agent:

- explains the cutoff result
- ranks the closest decision-ready customers by value at risk
- shows the supporting chart

</div>
<img class="visual" src="../../charts/top-at-risk.png" />
</div>

---

<!-- _header: "" -->
<!-- _class: lead part-trust -->

# Trust comes from traceable outputs

**Live data, explicit model quality and visible limitations**

---

<!-- _header: "The product > OpenCode example > Claude Desktop > **Trust**" -->

# Decision support with clear limits

<div class="columns">
<img class="visual" src="../../charts/calibration.png" />
<div>

- Predictions use a **90-day horizon**
- Calibration and feature importance are visible
- Importances are global associations, not causal claims
- Holdout metrics include a documented selection caveat

> The system helps prioritise action. It does not pretend to guarantee outcomes.

</div>
</div>

---

<!-- _class: lead title-slide -->

# Customer Analytics AI
## A reproducible CLI with agent-ready evidence

**See the full project**

github.com/miguelfg/customer-analytics-ai

<span class="small">Open retail demo data · live agent examples · documented model limits</span>
