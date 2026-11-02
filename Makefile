CHARTS := model-comparison feature-importance calibration segment-sizes ltv-distribution churn-risk-by-segment churn-lift retention-cohorts rfm-heatmap top-at-risk dq-report pipeline-funnel

.PHONY: install test lint format data pipeline run-all status charts ask load clean features features-ext tune train-churn fit-ltv segment score

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

data:
	uv run ca load && uv run ca clean

pipeline run-all:
	uv run ca run-all

status:
	uv run ca status

# Individual pipeline steps (in order)
load:
	uv run ca load

clean:
	uv run ca clean

features:
	uv run ca features

features-ext:
	uv run ca features-ext

tune:
	uv run ca tune

train-churn:
	uv run ca train-churn

fit-ltv:
	uv run ca fit-ltv

segment:
	uv run ca segment

score:
	uv run ca score

charts:
	@for c in $(CHARTS); do uv run ca chart $$c || exit 1; done

ask:
	uv run ca ask "$(Q)"
