import mlflow
from datetime import datetime
from zoneinfo import ZoneInfo


class RAGMLflowLogger:

    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f'[MLFLOW INIT WARNING] {e}')

    def log_rag_event(self,
                      query,
                      answer,
                      retrieved_chunks,
                      prompt,
                      model_name,
                      latency,
                      score,
                      metadata=None):
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param('query', query)
                mlflow.log_param('model', model_name)
                mlflow.log_param('num_chunks', len(retrieved_chunks))
                if metadata:
                    for k, v in metadata.items():
                        mlflow.log_param(f'meta_{k}', v)

                # Log metrics
                mlflow.log_metric('latency', latency)
                mlflow.log_metric('score', score)

                # Log text
                mlflow.log_text(prompt, 'prompt.txt')
                mlflow.log_text(answer, 'answer.txt')

                # Log dict
                mlflow.log_dict({'chunks': retrieved_chunks}, 'retrieved.json')

                # Set tags
                mlflow.set_tag("system", "enterprise-rag")
                mlflow.set_tag(
                    "timestamp",
                    datetime.now(ZoneInfo("Asia/Singapore")).isoformat())

        except Exception as e:
            print(f'[MLFLOW LOG WARNING] {e}')
