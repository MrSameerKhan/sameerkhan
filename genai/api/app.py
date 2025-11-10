from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn
import os

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ---------------------------------
# Initialize FastAPI app
# ---------------------------------
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API Server exposing LLM chains",
)

# ---------------------------------
# Define models
# ---------------------------------
# OpenAI model (optional; only if you want it)
openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Ollama model (DeepSeek-R1 running locally via ollama)
ollama_model = OllamaLLM(model="deepseek-r1")   # or "deepseek-r1:8b" if that’s your tag

# ---------------------------------
# Define prompts
# ---------------------------------
prompt_essay = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words."
)

prompt_poem = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} with 100 words."
)

# ---------------------------------
# Add routes using LangServe
# ---------------------------------

# Essay using Ollama (DeepSeek-R1)
add_routes(
    app,
    prompt_essay | openai_model,
    path="/essay",
)

# Poem using Ollama (DeepSeek-R1)
add_routes(
    app,
    prompt_poem | openai_model,
    path="/poem",
)

# ---------------------------------
# Run server
# ---------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8010)
