import streamlit as st
import pandas as pd
from nlp_models import SentimentAnalyzer, TopicClassifier
from llm_module import generate_reply

# Initialize models outside of main function so they load once
sent_analyzer = SentimentAnalyzer()
topic_classifier = TopicClassifier()

def classify_and_respond(text, candidate_labels):
    # Run sentiment
    sentiment_result = sent_analyzer.predict_sentiment(text)
    # Run topic
    topic_result = topic_classifier.predict_topic(text, candidate_labels)
    # Generate reply
    reply = generate_reply(text)

    return sentiment_result, topic_result, reply

def main():
    st.title("Social Media Analysis & GenAI Reply")

    st.write("## 1. Single Post Analysis")
    user_input = st.text_input("Enter a social media post")
    candidate_labels = ["shipping", "product quality", "billing", "returns", "general inquiries"]

    if st.button("Analyze"):
        sentiment, topic, reply = classify_and_respond(user_input, candidate_labels)
        st.write("**Sentiment**:", sentiment)
        st.write("**Topic**:", topic)
        st.write("**Generative Reply**:", reply)

    st.write("---")
    st.write("## 2. Batch Analysis from File (Optional)")

    uploaded_file = st.file_uploader("Upload a CSV with a 'text' column")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        results = []
        for idx, row in df.iterrows():
            sentiment, topic, reply = classify_and_respond(row['text'], candidate_labels)
            results.append({
                "text": row['text'],
                "sentiment_label": sentiment['label'],
                "sentiment_score": sentiment['score'],
                "topic_label": topic['label'],
                "topic_score": topic['score'],
                "reply": reply
            })
        out_df = pd.DataFrame(results)
        st.dataframe(out_df)
        
        # Quick summary stats
        sentiment_counts = out_df['sentiment_label'].value_counts()
        st.bar_chart(sentiment_counts)

if __name__ == "__main__":
    main()
