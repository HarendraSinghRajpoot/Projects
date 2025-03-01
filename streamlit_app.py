import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from data_pipeline import load_and_clean_data
from sentiment_analysis import analyze_sentiment
from topic_classification import classify_topics, DEFAULT_LABELS
from response_generator import generate_response
from mlflow_tracking import log_run_info

st.title("🔎 Social Media NLP Dashboard")
st.write("Analyze sentiment and topics of social media posts and generate automated responses.")

# File uploader for CSV data
uploaded_file = st.file_uploader("Upload a CSV file of social media posts", type="csv")
if uploaded_file is not None:
    # Load and preprocess data
    df = load_and_clean_data(uploaded_file)
    st.success(f"Loaded {len(df)} posts.")
    st.subheader("Sample Data")
    st.dataframe(df[['text', 'text_clean']].head(5))  # show a preview of raw and cleaned text

    # Run sentiment analysis and topic classification
    sentiments = analyze_sentiment(df['text_clean'].tolist())
    topics = classify_topics(df['text_clean'].tolist(), candidate_labels=DEFAULT_LABELS)
    # Append results to DataFrame
    df['sentiment_label'] = [res['label'] for res in sentiments]
    df['sentiment_score'] = [res['score'] for res in sentiments]
    df['topic'] = [res['labels'][0] if res['labels'] else None for res in topics]
    df['topic_score'] = [res['scores'][0] if res['scores'] else None for res in topics]

    # Display classification results
    st.subheader("Classification Results")
    st.dataframe(df[['text', 'sentiment_label', 'topic']].head(10))

    # Visualize sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment_label'].value_counts()
    st.bar_chart(sentiment_counts)  # Streamlit will render a simple bar chart of sentiment counts

    # (Optional) If timestamp data is available in df, you could plot sentiment over time for trends.
    # For now, we just show a static distribution.

    # Visualize topic distribution
    st.subheader("Topic Distribution")
    topic_counts = df['topic'].value_counts()
    st.bar_chart(topic_counts)

    # LLM-based response generation
    st.subheader("LLM Response Generator")
    st.write("Select a post and generate an AI-based reply:")
    sample_post = st.selectbox("Choose a post", df['text'].tolist()[:20])
    if st.button("Generate Reply"):
        reply = generate_response(sample_post)
        st.write("**Original Post:**", sample_post)
        st.write("**AI Reply:**", reply)

    # Log the run info to MLflow (parameters and metrics)
    log_run_info(df)
    st.info("Run info logged to MLflow Tracking server. You can view detailed metrics in the MLflow UI.")

    # Pipeline Visualization (Graphviz flowchart)
    st.subheader("Pipeline Flowchart")
    flowchart_dot = """
        digraph pipeline {
            rankdir=LR;
            "Raw Data" -> "Preprocessing";
            "Preprocessing" -> "Sentiment Model";
            "Preprocessing" -> "Topic Model";
            "Sentiment Model" -> "Sentiment Output";
            "Topic Model" -> "Topic Output";
            "Sentiment Output" -> "Dashboard";
            "Topic Output" -> "Dashboard";
            "Dashboard" -> "User";
            "Sentiment Model" -> "MLflow";
            "Topic Model" -> "MLflow";
            "LLM Response" -> "Dashboard";
            "LLM Response" -> "API Service";
        }
    """
    st.graphviz_chart(flowchart_dot)

    # NetworkX graph to visualize model decisions for the first post
    st.subheader("Sample Decision Graph")
    if len(df) > 0:
        G = nx.DiGraph()
        # Use first post as an example node
        post_label = "Post_1"
        sentiment_node = f"Sentiment: {df['sentiment_label'][0]} ({df['sentiment_score'][0]:.2f})"
        topic_node = f"Topic: {df['topic'][0]} ({df['topic_score'][0]:.2f})"
        G.add_node(post_label)
        G.add_node(sentiment_node)
        G.add_node(topic_node)
        G.add_edge(post_label, sentiment_node)
        G.add_edge(post_label, topic_node)
        # Draw the graph
        fig, ax = plt.subplots()
        nx.draw_networkx(G, ax=ax, node_color="#8fbcd4", node_size=1500, font_size=10, font_color="black")
        ax.set_title("Post vs. Model Outputs")
        st.pyplot(fig)

    # Provide a link to MLflow UI for further inspection
    st.markdown("[🔗 Open MLflow UI](http://localhost:5000) for detailed experiment metrics and artifacts.")
else:
    st.info("Please upload a CSV file to start the analysis.")
