# Customer Analytics AI

For customer-analytics business questions, load the `customer-analytics` skill before answering. Its canonical source is `skills/customer-analytics/SKILL.md` and its instructions require real JSON CLI results, a relevant chart, and stated limitations.

The local DuckDB database and trained models are intentionally ignored by Git. If they are absent, run `make data` followed by `uv run ca run-all --tune` before using the skill.

Do not add API keys to files or Git. The built-in Claude SDK agent requires `ANTHROPIC_API_KEY` in the invoking shell.
