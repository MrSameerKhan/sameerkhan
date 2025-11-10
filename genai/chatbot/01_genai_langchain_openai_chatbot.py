# ==============================
# 🚀 LangChain + Streamlit Demo
# ==============================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# ---------------------------------
# Prompt Template
# ---------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("🧠 LangChain Demo with OpenAI API")
input_text = st.text_input("🔍 Enter your question:")

# ---------------------------------
# LLM Setup
# ---------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
output_parser = StrOutputParser()

# Create a simple chain: prompt → LLM → parser
chain = prompt | llm | output_parser

# ---------------------------------
# Run Inference
# ---------------------------------
if input_text:
    response = chain.invoke({"question": input_text})
    st.write("💬 **Response:**")
    st.write(response)
