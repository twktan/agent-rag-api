#!/usr/bin/env bash
# Build with Cloud Build and deploy to Cloud Run with cost guardrails.
#
#   PROJECT_ID=my-project ./scripts/deploy.sh
#
# One-time setup:
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
#       artifactregistry.googleapis.com secretmanager.googleapis.com
#   gcloud artifacts repositories create agent-rag --repository-format=docker --location="$REGION"
#   printf '%s' "$OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=-
#   python -c 'import secrets; print(secrets.token_urlsafe(32), end="")' | \
#       gcloud secrets create rag-api-keys --data-file=-          # comma-separate multiple keys
#   for s in openai-api-key rag-api-keys; do
#     gcloud secrets add-iam-policy-binding "$s" --role=roles/secretmanager.secretAccessor \
#         --member="serviceAccount:$(gcloud projects describe "$PROJECT_ID" \
#         --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
#   done
#   # If OPENAI_API_KEY was previously a plain env var on the service, remove it first:
#   gcloud run services update fastapi-rag --region "$REGION" --remove-env-vars OPENAI_API_KEY
#
# Cost controls applied here:
#   --max-instances 1     caps compute spend and makes the in-memory rate limiter global
#   --concurrency 8       the agent is I/O-bound on LLM calls; 8 in flight per instance is plenty
#   --timeout 60          worst-case request (7 LLM calls with 30 s timeouts) is cut off
#   --min-instances 0     scale to zero when idle
# Authentication is enforced by the app (X-API-Key), so the service itself is publicly routable.
# The OpenAI project budget (set in the OpenAI dashboard) is the final hard stop.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-fastapi-rag}"
REPO="${REPO:-agent-rag}"
TAG="$(git rev-parse --short HEAD)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}"

gcloud builds submit --project "$PROJECT_ID" --tag "$IMAGE" .

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi --cpu-boost \
  --min-instances 0 --max-instances 1 --concurrency 8 --timeout 60 \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest,API_KEYS=rag-api-keys:latest" \
  --set-env-vars "ENV=prod,METRICS_ENABLED=false,LOG_QUERY_TEXT=false,EMBED_THREADS=2,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"
echo "Deployed ${IMAGE} -> ${URL}"
curl -fsS "${URL}/ready" && echo
