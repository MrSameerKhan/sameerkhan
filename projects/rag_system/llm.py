"""
llm.py — LLM abstraction layer

Supports two providers:
  "ollama"      — local Ollama server (default, for development)
  "huggingface" — HuggingFace Inference API (for HuggingFace Spaces deployment)

Set LLM_PROVIDER env var to switch.
Set HF_TOKEN env var when using HuggingFace provider.
"""

import os

from config import HF_INFERENCE_MODEL, LLM_PROVIDER, OLLAMA_MODEL


def generate(system_prompt: str, user_message: str) -> str:
    if LLM_PROVIDER == "huggingface":
        return _generate_hf(system_prompt, user_message)
    return _generate_ollama(system_prompt, user_message)


def _generate_ollama(system_prompt: str, user_message: str) -> str:
    import ollama
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )
    return resp["message"]["content"]


def _generate_hf(system_prompt: str, user_message: str) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(
        model=HF_INFERENCE_MODEL,
        token=os.getenv("HF_TOKEN"),
    )
    # Mistral Instruct prompt format
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    return client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.1,
    )
