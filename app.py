import streamlit as st
from utils.search_utils import search_papers
from utils.nlp_utils import summarize_paper, extract_concepts, explain_concept
import pandas as pd
import json

# Load arXiv data
@st.cache_data
def load_data():
          return pd.read_json('data/arxiv_cs.json', lines=True, encoding='utf-8')

df = load_data()

st.title("arXiv Expert Chatbot (Computer Science)")

# Search papers
query = st.text_input("Search for papers (title, author, keyword):")
if query:
    results = search_papers(df, query)
    st.write(results)

    selected = st.selectbox("Select a paper to summarize:", results['title'])
    if selected:
        paper = results[results['title'] == selected].iloc[0]
        st.subheader("Summary")
        st.write(summarize_paper(paper['abstract']))

        st.subheader("Ask about a concept in this paper:")
        concept = st.text_input("Concept/question:")
        if concept:
            st.write(explain_concept(concept, paper['abstract']))

# Concept visualization (e.g., topic map)
if st.button("Show Concept Map"):
    from utils.nlp_utils import visualize_concepts
    st.graphviz_chart(visualize_concepts(df))