from langchain_experimental.llms.ollama_functions import OllamaFunctions
from src.logger import logging

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P


def get_llm_model():
    logging.info("Getting LLM model")
    return OllamaFunctions(
        model=LLM_MODEL,
        keep_alive=-1,
        format="json",
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
    )
