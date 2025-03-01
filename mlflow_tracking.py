import mlflow
import pandas as pd

def log_run_info(df: pd.DataFrame, experiment_name: str = "SocialMediaNLP"):
    """
    Log parameters and metrics of the analysis run to MLflow.
    - Logs number of records, sentiment distribution, and topic counts as metrics.
    - Logs the processed DataFrame as an artifact (CSV).
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("num_records", len(df))
        # Example of logging an arbitrary parameter (could be model versions, etc.)
        mlflow.log_param("sentiment_model", "distilbert-sst2")
        mlflow.log_param("topic_model", "bart-mnli")
        # Log metrics: e.g., fraction of posts that are positive sentiment
        if 'sentiment_label' in df.columns:
            positive_frac = (df['sentiment_label'] == 'POSITIVE').mean()
            mlflow.log_metric("positive_fraction", positive_frac)
        # Log topic distribution as metrics
        if 'topic' in df.columns:
            topic_counts = df['topic'].value_counts()
            for topic, count in topic_counts.items():
                mlflow.log_metric(f"num_{topic}_posts", int(count))
        # Save the processed data as an artifact for review
        output_path = "processed_data.csv"
        df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)
        print(f"Logged run to MLflow with {len(df)} records.")

# The MLflow UI can be launched separately (e.g., via `mlflow ui`).
# The Streamlit app will provide a link to this UI for convenience.
if __name__ == "__main__":
    # Simple test with dummy data
    test_df = pd.DataFrame({"text": ["Hello World"], "sentiment_label": ["POSITIVE"], "topic": ["other"]})
    log_run_info(test_df)
