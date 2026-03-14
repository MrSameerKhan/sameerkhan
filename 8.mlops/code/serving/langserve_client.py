"""
Streamlit client for the LangServe API (langserve_api.py must be running).
Run: streamlit run langserve_client.py
"""
import requests
import streamlit as st


def get_essay(topic: str) -> str:
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={"input": {"topic": topic}},
    )
    return response.json()["output"]["content"]


def get_poem(topic: str) -> str:
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={"input": {"topic": topic}},
    )
    return response.json()["output"]


st.title("LangServe Client — Essay & Poem")

essay_topic = st.text_input("Write an essay on")
poem_topic = st.text_input("Write a poem on")

if essay_topic:
    st.subheader("Essay")
    st.write(get_essay(essay_topic))

if poem_topic:
    st.subheader("Poem")
    st.write(get_poem(poem_topic))
