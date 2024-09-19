from typing import List
from src.logger import logging
import pandas as pd

from src.config import MODEL_FLOPS_PATH


def get_all_models() -> List[str]:
    logging.info("Getting all models")
    models = pd.read_excel(MODEL_FLOPS_PATH)["Model"]
    return list(models)


def normalize_data(data):
    logging.info("Normalizing data")
    return (data - data.min()) / (data.max() - data.min())
