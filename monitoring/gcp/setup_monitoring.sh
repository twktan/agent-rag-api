#!/usr/bin/env bash
# Production monitoring on Cloud Run: log-based metrics + alert policies in Cloud Monitoring.
#
# The API writes one structured `query_completed` JSON log line per request (app/main.py).
# Cloud Logging ingests it as jsonPayload.*; these log-based metrics turn it into time series
# with no extra infrastructure (no Prometheus, no MLflow server) and stay inside the free tier
# at this traffic level.
#
#   PROJECT_ID=my-project ALERT_EMAIL=me@example.com ./monitoring/gcp/setup_monitoring.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
SERVICE="${SERVICE:-fastapi-rag}"
DIR="$(cd "$(dirname "$0")" && pwd)"
gcloud config set project "$PROJECT_ID" >/dev/null

BASE="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE}\""

upsert_metric() {  # name, config file
  local name=$1 cfg=$2 tmp
  tmp=$(mktemp)
  sed "s|__BASE_FILTER__|${BASE//&/\\&}|" "$cfg" > "$tmp"
  gcloud logging metrics describe "$name" >/dev/null 2>&1 \
    && gcloud logging metrics update "$name" --config-from-file="$tmp" \
    || gcloud logging metrics create "$name" --config-from-file="$tmp"
  rm -f "$tmp"
}

upsert_metric rag_query_count      "$DIR/metrics/rag_query_count.yaml"
upsert_metric rag_query_latency_ms "$DIR/metrics/rag_query_latency_ms.yaml"
upsert_metric rag_query_cost_usd   "$DIR/metrics/rag_query_cost_usd.yaml"
upsert_metric rag_retrieval_top1   "$DIR/metrics/rag_retrieval_top1.yaml"
upsert_metric rag_guardrail_blocks "$DIR/metrics/rag_guardrail_blocks.yaml"
upsert_metric rag_critic_failures  "$DIR/metrics/rag_critic_failures.yaml"
upsert_metric rag_errors           "$DIR/metrics/rag_errors.yaml"

CHANNEL_ARGS=()
if [[ -n "${ALERT_EMAIL:-}" ]]; then
  CHANNEL=$(gcloud beta monitoring channels create --display-name="RAG API alerts" --type=email \
            --channel-labels="email_address=${ALERT_EMAIL}" --format='value(name)')
  CHANNEL_ARGS=(--notification-channels="$CHANNEL")
fi

for policy in "$DIR"/alerts/*.json; do
  gcloud alpha monitoring policies create --policy-from-file="$policy" "${CHANNEL_ARGS[@]}"
done

echo "Done. Logs Explorer query for raw events:"
echo "  ${BASE} AND jsonPayload.event=\"query_completed\""
