from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
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
st.title("🧠 LangChain Demo with Ollama (Llama2 Model)")
input_text = st.text_input("🔍 Enter your question:")


# ---------------------------------
# LLM Setup
# ---------------------------------
llm = OllamaLLM(model="llama2")
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


#Final