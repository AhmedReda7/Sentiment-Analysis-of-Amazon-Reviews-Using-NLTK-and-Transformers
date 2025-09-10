# app.py
 
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import nltk
 
# Download required NLTK data
nltk.download('vader_lexicon')
 
# Initialize VADER
sia = SentimentIntensityAnalyzer()
 
# Initialize RoBERTa
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
 
# Title
st.title("Amazon Review Sentiment Analyzer")
st.write("Analyze your Amazon review using VADER and RoBERTa models!")
 
# Input
user_input = st.text_area("Enter your review:", height=150)
 
# Analysis
if st.button("Analyze"):
    if user_input:
        # VADER
        vader_result = sia.polarity_scores(user_input)
        st.subheader("VADER Sentiment Analysis")
        st.write(vader_result)
 
        # RoBERTa
        encoded_input = tokenizer(user_input, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
 
        labels = ['Negative', 'Neutral', 'Positive']
        roberta_result = {f"RoBERTa {label}": float(score) for label, score in zip(labels, scores)}
        
        st.subheader("RoBERTa Sentiment Analysis")
        st.write(roberta_result)
    else:
        st.warning("Please enter a review before analyzing.")
