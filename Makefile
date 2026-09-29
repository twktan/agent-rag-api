PY ?= python

.PHONY: install index run test lint eval-retrieval ablate eval eval-no-reflection readme \
        docker-build docker-run monitoring-up deploy load-test

install:            ## CPU-only torch + runtime + dev dependencies
	$(PY) -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements-dev.txt

index:              ## build the FAISS index from data/docs
	$(PY) -m app.rag.build_index

run:                ## local dev server on :8080
	uvicorn app.main:create_app --factory --reload --port 8080

test:
	$(PY) -m pytest

lint:
	ruff check .

eval-retrieval:     ## offline retrieval eval (free, no API key)
	$(PY) -m eval.retrieval_eval --split test
	$(PY) -m eval.retrieval_eval --split dev

ablate:             ## offline retrieval ablation (free)
	$(PY) -m eval.ablate_retrieval

eval:               ## full agent eval on the test split (OpenAI usage: a few USD)
	$(PY) -m eval.run_eval --split test

eval-no-reflection: ## same, with the critic loop disabled
	$(PY) -m eval.run_eval --split test --max-refinements 0 --tag no-reflection

readme:             ## write eval/results/*.json into the README results block
	$(PY) -m eval.update_readme

docker-build:
	docker build -t agent-rag-api .

docker-run:
	docker run --rm -p 8080:8080 --env-file .env agent-rag-api

monitoring-up:      ## API + MLflow + Prometheus + Grafana
	docker compose -f monitoring/docker-compose.yml up --build

deploy:
	./scripts/deploy.sh

load-test:          ## make load-test URL=https://<service>.run.app  (needs RAG_API_KEY)
	$(PY) -m eval.load_test --url $(URL) --n 30 --concurrency 2
