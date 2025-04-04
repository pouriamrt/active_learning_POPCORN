import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

@st.cache_data
def load_papers():
    df = pd.read_csv("papers.csv")
    df['abstract'] = df['abstract'].astype(str).fillna("")
    return df[['title', 'abstract']]

def load_model(df):
    try:
        with open("model.pkl", "rb") as f:
            model, vectorizer, labeled_data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        model = LogisticRegression()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        if df.empty:
            st.error("Dataset is empty. Please provide valid paper data.")
            st.stop()
        
        labeled_data = df.iloc[:2].copy()
        labeled_data['label'] = [1, 0]  # Ensure at least two classes
        
        X_labeled = vectorizer.fit_transform(labeled_data['abstract'].astype(str))
        y_labeled = labeled_data['label']
        model.fit(X_labeled, y_labeled)
        save_model(model, vectorizer, labeled_data)
    return model, vectorizer, labeled_data

def save_model(model, vectorizer, labeled_data):
    with open("model.pkl", "wb") as f:
        pickle.dump((model, vectorizer, labeled_data), f)

df = load_papers()
model, vectorizer, labeled_data = load_model(df)

if not labeled_data.empty and len(labeled_data['label'].unique()) > 1:
    X_labeled = vectorizer.fit_transform(labeled_data['abstract'].astype(str))
    y_labeled = labeled_data['label']
    model.fit(X_labeled, y_labeled)

unseen_papers = df[~df['title'].isin(labeled_data['title'])]
if unseen_papers.empty:
    st.write("All papers have been labeled. Retrain the model with new data!")
else:
    paper = unseen_papers.sample(1).iloc[0]
    
    st.subheader("Title: " + paper['title'])
    with st.expander("Show Abstract"):
        st.write("**Abstract:** " + str(paper['abstract']))
    
    col1, col2 = st.columns(2)
    with col1:
        include = st.button("✅ Include")
    with col2:
        exclude = st.button("❌ Exclude")
    
    st.markdown("---")
    st.write("### Model Statistics")
    st.write(f"Total Labeled Papers: {len(labeled_data)}")
    
    if not labeled_data.empty:
        label_counts = labeled_data['label'].value_counts()
        st.write(f"Included Papers: {label_counts.get(1, 0)}")
        st.write(f"Excluded Papers: {label_counts.get(0, 0)}")
    
    if include or exclude:
        new_label = 1 if include else 0
        new_entry = pd.DataFrame({'title': [paper['title']], 'abstract': [paper['abstract']], 'label': [new_label]})
        labeled_data = pd.concat([labeled_data, new_entry], ignore_index=True)
        
        if len(labeled_data['label'].unique()) > 1:
            X_labeled = vectorizer.fit_transform(labeled_data['abstract'].astype(str))
            y_labeled = labeled_data['label']
            model.fit(X_labeled, y_labeled)
            save_model(model, vectorizer, labeled_data)
        
        st.rerun()