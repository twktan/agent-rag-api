# Linux/macOS task runner. On Windows use the PowerShell equivalent: .\scripts\tasks.ps1 <task>
PY ?= python

.PHONY: install index run test lint eval-retrieval ablate eval-smoke eval eval-no-reflection readme \
        review failures calibrate-export calibrate-score docker-build docker-run monitoring-up deploy load-test

install:            ## CPU-only torch + runtime + dev dependencies
	$(PY) -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements-dev.txt

index:              ## build the FAISS index from data/docs
	$(PY) -m app.rag.build_index

run:                ## local dev server on :8081
	$(PY) -m uvicorn app.main:create_app --factory --reload --port 8081

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

eval-retrieval:     ## offline retrieval eval (free, no API key)
	$(PY) -m eval.retrieval_eval --split test
	$(PY) -m eval.retrieval_eval --split dev

ablate:             ## offline retrieval ablation (free)
	$(PY) -m eval.ablate_retrieval

eval-smoke:         ## 10-item end-to-end check (a few cents); writes eval_test_smoke.json
	$(PY) -m eval.run_eval --split test --limit 10 --tag smoke --yes

eval:               ## full agent eval on the test split (OpenAI usage: about US$1)
	$(PY) -m eval.run_eval --split test

eval-no-reflection: ## same, with the critic loop disabled
	$(PY) -m eval.run_eval --split test --max-refinements 0 --tag no-reflection

readme:             ## write eval/results/*.json into the README results block
	$(PY) -m eval.update_readme

review:             ## print the test split of the golden set for human review
	$(PY) -m eval.review_golden --split test

failures:           ## list misroutes, wrong answers and errors from the last eval
	$(PY) -m eval.inspect_failures

calibrate-export:   ## blind CSV of 50 judged answers for you to label
	$(PY) -m eval.judge_calibration export

calibrate-score:    ## agreement + Cohen's kappa between your labels and the judge
	$(PY) -m eval.judge_calibration score

docker-build:
	docker build -t agent-rag-api .

docker-run:         ## the container listens on 8080 (Cloud Run's $PORT); exposed on localhost:8081
	docker run --rm -p 8081:8080 --env-file .env agent-rag-api

monitoring-up:      ## API + MLflow + Prometheus + Grafana
	docker compose -f monitoring/docker-compose.yml up --build

deploy:
	./scripts/deploy.sh

load-test:          ## make load-test URL=https://<service>.run.app  (needs RAG_API_KEY)
	$(PY) -m eval.load_test --url $(URL) --n 30 --concurrency 2
