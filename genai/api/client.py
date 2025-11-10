import requests
import streamlit as st

def get_openai_response_essay(input_text):
    response = requests.post("http://localhost:8010/essay/invoke",
                             json={'input':{'topic':input_text}})
    
    return response.json()['output']['content']

def get_openai_response_poem(input_text):
    response = requests.post("http://localhost:8010/poem/invoke",
                             json={'input':{'topic':input_text}})
    
    return response.json()['output']['content']


st.title("Langchain Demo with DeepSeek API")
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    st.write(get_openai_response_essay(input_text))
if input_text1:
    st.write(get_openai_response_poem(input_text1))
