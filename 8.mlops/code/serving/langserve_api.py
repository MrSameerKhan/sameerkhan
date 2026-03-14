"""
FastAPI + LangServe: expose LangChain chains as REST endpoints.
Run: python langserve_api.py
Endpoints:
  POST /essay/invoke   → GPT-4o essay on {topic}
  POST /poem/invoke    → LLaMA2 (Ollama) poem on {topic}
  GET  /openai/playground → LangServe UI
Requires: OPENAI_API_KEY in .env, Ollama running locally with llama2
"""
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="LangChain LangServe Demo",
    version="1.0",
    description="Expose LangChain chains as REST endpoints via LangServe",
)

# Raw OpenAI model endpoint
add_routes(app, ChatOpenAI(), path="/openai")

# Essay chain (OpenAI)
model = ChatOpenAI()
llm = Ollama(model="llama2")  # requires Ollama running locally

prompt_essay = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words"
)
prompt_poem = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5 year old child with 100 words"
)

add_routes(app, prompt_essay | model, path="/essay")
add_routes(app, prompt_poem | llm, path="/poem")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
