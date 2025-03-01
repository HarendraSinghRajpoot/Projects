from transformers import pipeline

# Load a zero-shot classification pipeline (BART MNLI)
zero_shot_classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli"
)

# Define some candidate topic labels for classification
DEFAULT_LABELS = ["sports", "politics", "technology", "health", "entertainment", "other"]

def classify_topics(texts, candidate_labels=DEFAULT_LABELS):
    """
    Classify a list of texts into one of the candidate topic labels using zero-shot learning.
    Returns a list of results where each result has 'labels' and 'scores' fields.
    """
    if isinstance(texts, str):
        texts = [texts]
    # Using Hugging Face zero-shot pipeline to get topic probabilities&#8203;:contentReference[oaicite:7]{index=7}.
    results = zero_shot_classifier(texts, candidate_labels=candidate_labels, truncation=True)
    # If a single text was provided, wrap the single result into a list for consistency
    if isinstance(results, dict):
        results = [results]
    return results

# Example usage
if __name__ == "__main__":
    example_text = "The team played a fantastic game last night!"
    topics = classify_topics(example_text, candidate_labels=DEFAULT_LABELS)
    top_label = topics[0]['labels'][0]  # most likely topic
    print(f"Text: '{example_text}' => Topic: {top_label} (score={topics[0]['scores'][0]:.2f})")
