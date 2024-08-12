import os
from typing import Tuple

import numpy as np
import pandas as pd

from config import EMISSIONS_PATH, GPUS_PATH, MODEL_FLOPS_PATH, PRICING_PATH
from src.utils.get_root import get_root
from utils.data_processing import normalize_data
from utils.time_formatting import format_time
from src.logger import logging


def get_tflops_value(perf_data, tflops_type):
    logging.info(f"Getting TFLOPS value for GPU: {perf_data['name']}")
    if pd.notna(perf_data.get(tflops_type, np.nan)):
        return perf_data[tflops_type]
    elif pd.notna(perf_data.get("TFLOPS32", np.nan)):
        logging.warning(
            f"TFLOPS value not found for the GPU: {perf_data['name']}. Using TFLOPS32 value instead."
        )
        return perf_data["TFLOPS32"]
    elif pd.notna(perf_data.get("TFLOPS16", np.nan)):
        logging.warning(
            f"TFLOPS value not found for the GPU: {perf_data['name']}. Using TFLOPS16 value instead."
        )
        return perf_data["TFLOPS16"]
    else:
        raise ValueError(
            f"No valid TFLOPS value found for the GPU: {perf_data['name']}"
        )


def calculate_kwh_consumption(gpu_name, time_seconds, gpu_df):
    logging.info(f"Calculating energy consumption for GPU: {gpu_name}")
    tdp_watts = gpu_df.loc[gpu_df["name"] == gpu_name, "tdp_watts"].values[0]
    tdp_kw = tdp_watts / 1000
    time_hours = time_seconds / 3600
    energy_consumption_kwh = tdp_kw * time_hours
    logging.info(f"Energy consumption for GPU {gpu_name}: {energy_consumption_kwh} kWh")
    return energy_consumption_kwh


def estimate_flops(
    model: str,
    input_size: Tuple[int, int],
    training_strategy: str,
    sample_count: int,
    estimated_epochs: int,
    flops_df: pd.DataFrame,
) -> float:
    logging.info(f"Estimating FLOPs for model: {model}")
    model_info = flops_df[flops_df["Model"] == model]

    if model_info.empty:
        raise ValueError(f"Model {model} not found in the flops database")

    model_type = model_info["Type"].iloc[0]
    original_input_size = model_info["Input Size"].iloc[0].split()[0]

    if model_type == "Vision":
        width, height = map(int, original_input_size.split("x"))
        scaling = (input_size[0] * input_size[1]) / (width * height)
    else:
        scaling = input_size[0] / int(original_input_size)

    flops = int(model_info["FLOPs"].iloc[0])
    last_layer_flops = int(model_info["Last Layer FLOPs"].iloc[0])
    if training_strategy in ["Fine-tuning the whole model", "Full Training"]:
        return estimated_epochs * sample_count * flops * scaling * 3
    elif training_strategy == "Last Layer Learning":
        return (
            estimated_epochs * sample_count * (2 * last_layer_flops + flops * scaling)
        )
    else:
        raise ValueError(f"Unsupported training strategy: {training_strategy}")


def estimate_time(
    flops: float, gpu: str, training_strategy: str, tflops: str, gpu_df
) -> float:
    gpu_info = gpu_df[gpu_df["name"] == gpu]
    logging.info(f"Estimating time for GPU: {gpu}")
    if gpu_info.empty:
        raise ValueError(f"GPU {gpu} not found in the flops database")
    tflops_value = get_tflops_value(gpu_info.iloc[0], tflops)
    return flops / tflops_value / 1e12


def calculate_emissions(kwh: float, region: str, emissions_df) -> float:
    logging.info(f"Calculating emissions for region: {region}")
    emissions_df = emissions_df[emissions_df["region"] == region]
    emissions = emissions_df["impact"].iloc[0]
    return kwh * emissions


def calculate_price(gpu: str, region: str, time: float, pricing_df) -> float:
    logging.info(f"Calculating price for GPU: {gpu}")
    pricing_df = pricing_df[pricing_df["region"] == region]
    price = pricing_df[pricing_df["gpu"] == gpu]["price"].iloc[0] * time
    return price


def recommend_gpu_configuration(
    model,
    input_size,
    training_strategy,
    sample_count,
    estimated_epochs,
    time_coeff,
    cost_coeff,
    co2_coeff,
    tflops_type,
    max_time=None,
    max_cost=None,
    max_co2=None,
):
    logging.info("Recommending GPU configuration")
    pricing_df = pd.read_excel(
        os.path.join(get_root(), "data", "pricing", "GCP gpus pricing.xlsx")
    )
    gpu_df = pd.read_csv(os.path.join(get_root(), "data", "gpus.csv"))
    flops_df = pd.read_excel(
        os.path.join(get_root(), "data", "model_flops", "model_flops.xlsx")
    )
    emissions_df = pd.read_csv(os.path.join(get_root(), "data", "impact.csv"))

    manual_map = {
        "T4": "T4",
        "V100": "Tesla V100-PCIE-16GB",
        "P100": "Tesla P100",
        "K80": "Tesla K80",
    }

    pricing_df["Mapped_GPU"] = (
        pricing_df["gpu"].map(manual_map).fillna(pricing_df["gpu"])
    )

    total_flops = estimate_flops(
        model, input_size, training_strategy, sample_count, estimated_epochs, flops_df
    )
    results = []

    for _, price_row in pricing_df.iterrows():
        gpu_pricing = price_row["gpu"]
        gpu_model_name = price_row["Mapped_GPU"]
        region = price_row["region"]

        perf_data = gpu_df[gpu_df["name"] == gpu_model_name]

        if perf_data.empty:
            print(f"Warning: No performance data found for GPU {gpu_model_name}")
            continue

        time_seconds = estimate_time(
            total_flops, gpu_model_name, training_strategy, tflops_type, gpu_df
        )

        price = calculate_price(
            gpu_pricing, region, time_seconds / 3600, pricing_df
        )  # convert seconds to hours

        kwh = calculate_kwh_consumption(gpu_model_name, time_seconds, gpu_df)
        co2 = calculate_emissions(kwh, region, emissions_df) / 1000

        results.append(
            {
                "GPU": gpu_pricing,
                "Mapped_GPU": gpu_model_name,
                "Region": region,
                "Time": time_seconds,
                "Time (formatted)": format_time(time_seconds),
                "Cost ($)": price,
                "CO2 (kg)": co2,
            }
        )

    df = pd.DataFrame(results)

    for col in ["Time", "Cost ($)", "CO2 (kg)"]:
        df[f"Normalized_{col}"] = normalize_data(df[col])
        df[f"{col}_Score"] = (1 - df[f"Normalized_{col}"]) * 5

    df["Ranking"] = (
        df["Time_Score"] * time_coeff
        + df["Cost ($)_Score"] * cost_coeff
        + df["CO2 (kg)_Score"] * co2_coeff
    )
    
    if max_time:
        df = df[df["Time"] <= max_time]
    if max_cost:
        df = df[df["Cost ($)"] <= max_cost]
    if max_co2:
        df = df[df["CO2 (kg)"] <= max_co2]

    df.dropna(inplace=True)
    df = df.sort_values("Ranking", ascending=False)
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Recommended GPU configuration: {df}")
    return df
