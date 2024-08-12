"""
This module contains the configuration settings for the project.
Attributes:
    ROOT_DIR (str): The root directory of the project.
    DATA_DIR (str): The directory where the data files are stored.
    MODEL_FLOPS_PATH (str): The path to the model flops file.
    GPUS_PATH (str): The path to the GPUs file.
    PRICING_PATH (str): The path to the GPU pricing file.
    EMISSIONS_PATH (str): The path to the emissions file.
    LLM_MODEL (str): The LLM model name.
    LLM_TEMPERATURE (int): The LLM temperature.
    LLM_TOP_P (float): The LLM top p value.
    MAX_SIMPLIFICATION_ATTEMPTS (int): The maximum number of simplification attempts.
"""

import os

from src.utils.get_root import get_root

ROOT_DIR = get_root()
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_FLOPS_PATH = os.path.join(DATA_DIR, "model_flops", "model_flops.xlsx")
GPUS_PATH = os.path.join(DATA_DIR, "gpus.csv")
PRICING_PATH = os.path.join(DATA_DIR, "pricing", "GCP gpus pricing.xlsx")
EMISSIONS_PATH = os.path.join(DATA_DIR, "impact.csv")

LLM_MODEL = "phi3:mini"
LLM_TEMPERATURE = 0
LLM_TOP_P = 0.2

MAX_SIMPLIFICATION_ATTEMPTS = 3
