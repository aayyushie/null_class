import streamlit as st
import yake
import spacy

def extract_and_visualize_concepts(text, topk=10):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    keywords = [kw[0] for kw in keywords[:topk]]
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents]
    if not keywords and not ents:
        st.write("No concepts or entities found.")
        return
    nodes = '\n'.join([f'"{kw}"' for kw in keywords] + [f'"{ent}"' for ent in ents])
    edges = '\n'.join([f'"{keywords[0]}" -> "{kw}"' for kw in keywords[1:]]) if keywords else ''
    edges += '\n' + '\n'.join([f'"{keywords[0]}" -> "{ent}"' for ent in ents]) if keywords else ''
    st.graphviz_chart(f'digraph G {{\n{nodes}\n{edges}\n}}') 