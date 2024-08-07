from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Tuple, List, Optional
import pandas as pd
from src.utility import get_root
import os
import numpy as np
from datetime import timedelta

memory = SqliteSaver.from_conn_string(":memory:")

model = OllamaFunctions(
    model="phi3:mini", 
    keep_alive=-1,
    format="json",
    temperature=0,
    top_p=0.2
)

def get_all_models() -> List[str]:
    path_to_models = os.path.join(get_root(), "data", "model_flops", "model_flops.xlsx")
    models = pd.read_excel(path_to_models)['Model']
    return list(models)

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())


def get_tflops_value(perf_data, tflops_type):
    if pd.notna(perf_data.get(tflops_type, np.nan)):
        return perf_data[tflops_type]
    elif pd.notna(perf_data.get('TFLOPS32', np.nan)):
        return perf_data['TFLOPS32']
    elif pd.notna(perf_data.get('TFLOPS16', np.nan)):
        return perf_data['TFLOPS16']
    else:
        raise ValueError(f"No valid TFLOPS value found for the GPU: {perf_data['name']}")


def calculate_kwh_consumption(gpu_name, time_seconds, gpu_df):

    tdp_watts = gpu_df.loc[gpu_df['name'] == gpu_name, 'tdp_watts'].values[0]
    
    tdp_kw = tdp_watts / 1000
    
    time_hours = time_seconds / 3600
    
    energy_consumption_kwh = tdp_kw * time_hours
    
    return energy_consumption_kwh

def estimate_flops(model: str, input_size: Tuple[int, int], training_strategy: str, sample_count: int, estimated_epochs: int, flops_df: pd.DataFrame) -> float:
    model_info = flops_df[flops_df["Model"] == model]

    if model_info.empty:
        raise ValueError(f"Model {model} not found in the flops database")
    
    model_type = model_info['Type'].iloc[0]
    original_input_size = model_info['Input Size'].iloc[0].split()[0]

    if model_type == 'Vision':
        width, height = map(int, original_input_size.split('x'))
        scaling = (input_size[0] * input_size[1]) / (width * height)
    else:
        scaling = input_size[0] / int(original_input_size)
    
    flops = int(model_info['FLOPs'].iloc[0])
    last_layer_flops = int(model_info['Last Layer FLOPs'].iloc[0])
    if training_strategy in ["Fine-tuning the whole model", "Full Training"]:
        return estimated_epochs * sample_count * flops * scaling * 3
    elif training_strategy == "Last Layer Learning":
        return estimated_epochs * sample_count * (2 * last_layer_flops + flops * scaling)
    else:
        raise ValueError(f"Unsupported training strategy: {training_strategy}")

def estimate_time(flops: float, gpu: str, training_strategy: str, tflops: str, gpu_df) -> float:
    gpu_info = gpu_df[gpu_df["name"] == gpu]

    if gpu_info.empty:
        raise ValueError(f"GPU {gpu} not found in the flops database")
    tflops_value = get_tflops_value(gpu_info.iloc[0], tflops)
    return flops / tflops_value / 1e+12 


def format_time(seconds):
    delta = timedelta(seconds=seconds)
    months, days = divmod(delta.days, 30)
    hours, remaining = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    
    parts = []
    if months > 0:
        parts.append(f"{months}m")
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}min")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)

def calculate_emissions(kwh: float, region: str, emissions_df) -> float:
    emissions_df = emissions_df[emissions_df['region'] == region]
    emissions = emissions_df['impact'].iloc[0]
    return kwh * emissions

def calculate_price(gpu: str, region: str, time: float, pricing_df) -> float:
    pricing_df = pricing_df[pricing_df['region'] == region]
    price = pricing_df[pricing_df['gpu'] == gpu]['price'].iloc[0] * time
    return price

def recommend_gpu_configuration(model, input_size, training_strategy, sample_count, estimated_epochs, 
                                time_coeff, cost_coeff, co2_coeff, tflops_type,
                                max_time=None, max_cost=None, max_co2=None):

    pricing_df = pd.read_excel(os.path.join(get_root(), 'data', 'pricing', 'GCP gpus pricing.xlsx'))
    gpu_df = pd.read_csv(os.path.join(get_root(), 'data', 'gpus.csv'))
    flops_df = pd.read_excel(os.path.join(get_root(), 'data', 'model_flops', 'model_flops.xlsx'))
    emissions_df = pd.read_csv(os.path.join(get_root(), 'data', 'impact.csv'))
    
    manual_map = {
        'T4': 'T4',
        'V100': 'Tesla V100-PCIE-16GB',
        'P100': 'Tesla P100',
        'K80': 'Tesla K80',
    }

    pricing_df['Mapped_GPU'] = pricing_df['gpu'].map(manual_map).fillna(pricing_df['gpu'])

    total_flops = estimate_flops(model, input_size, training_strategy, sample_count, estimated_epochs, flops_df)
    results = []

    for _, price_row in pricing_df.iterrows():
        gpu_pricing = price_row['gpu']
        gpu_model_name = price_row['Mapped_GPU']
        region = price_row['region']

        # Find corresponding performance data
        perf_data = gpu_df[gpu_df['name'] == gpu_model_name]
        
        if perf_data.empty:
            print(f"Warning: No performance data found for GPU {gpu_model_name}")
            continue

        time_seconds = estimate_time(total_flops, gpu_model_name, training_strategy, tflops_type, gpu_df)
        
        price = calculate_price(gpu_pricing, region, time_seconds / 3600, pricing_df)  # convert seconds to hours
        
        kwh = calculate_kwh_consumption(gpu_model_name, time_seconds, gpu_df)
        co2 = calculate_emissions(kwh, region, emissions_df) / 1000
        
        results.append({
            'GPU': gpu_pricing,
            'Mapped_GPU': gpu_model_name,
            'Region': region,
            'Time': time_seconds,
            'Time (formatted)': format_time(time_seconds),
            'Cost ($)': price,
            'CO2 (kg)': co2
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    for col in ['Time', 'Cost ($)', 'CO2 (kg)']:
        df[f'Normalized_{col}'] = normalize_data(df[col])
        df[f'{col}_Score'] = (1 - df[f'Normalized_{col}']) * 5 

    df['Ranking'] = (
        df['Time_Score'] * time_coeff + 
        df['Cost ($)_Score'] * cost_coeff + 
        df['CO2 (kg)_Score'] * co2_coeff
    )

    # Apply constraints
    if max_time:
        df = df[df['Time'] <= max_time]
    if max_cost:
        df = df[df['Cost ($)'] <= max_cost]
    if max_co2:
        df = df[df['CO2 (kg)'] <= max_co2]

    df.dropna(inplace=True)
    df = df.sort_values('Ranking', ascending=False)
    df.reset_index(drop=True, inplace=True)

    return df

RANKING_PROMPT = """
You are an AI assistant specializing in assigning importance to the priorities of eco-friendliness, time efficiency, and cost efficiency. Consider the task and context provided, and provide a ranking for each priority. The bigger the value, the more important the priority.
Pay attention to the constraints and requirements of the task provided by the user. Be constructive and tell why you chose the ranking for each priority based on the user input.

1. Eco-friendliness: Rank the importance of eco-friendliness
2. Time efficiency: Rank the importance of time efficiency
3. Cost efficiency: Rank the importance of cost efficiency

Provide the values in the format of 0.0 to 1.0. For example, 0.4, 0.2, 0.

4. Reasoning: Explain why you chose the ranking for each priority without repeating the rankings.
"""


CALCULATOR_PROMPT = """
You are an AI assistant specializing in calculating price, time, and co2 emissions. Underhood, we have calculations but it requires you to fill the following information:

1. Amount of samples: Provide the number of samples in the dataset. If you don't have this information, provide an estimate.

2. Input size: Provide the size of the input, for example (256, 256, 3) to specify an image or (512, ) to specify the length of a text sequence. If this information is not available, provide an estimate. Use 3D for images and 1D for text.

3. Estimated number of epochs: Provide the estimated number of epochs to train the model. Take into account the model architecture complexity and the size of the dataset.

"""

ARCHITECTURE_PROMPT = f"""
You are an AI assistant specializing in recommending AI model architectures and training strategies. Provide concise recommendations:

Provide:

1. Recommended model architecture: Provide only one of the following list: {get_all_models()}

2. Training strategy: Choose only one of the following options and explain your choice in the response. Think if the knowledge from the model could be transferred to the new task:
   - "Fine-tuning the whole model"
   - "Full Training"
   - "Last Layer Learning"

3. Choose between TFLOPS32 or TFLOPS16: Choose the appropriate TFLOPS value for the GPU based on the user priorities and constraints. Think if a user wants to save time, or to have better precision. Output a string "TFLOPS32" or "TFLOPS16".

4. Reasoning: Explain why the chosen training strategy, model and the TFLOPS type is the best option considering the given constraints.

Respond with ONLY these three pieces of information, nothing else.
"""

SIMPLIFICATION_PROMPT = f"""
You are an AI assistant specializing in recommending AI model architectures and training strategies. The previous recommendation didn't meet the constraints, so you need to propose a simpler architecture or strategy. Provide concise recommendations:

Provide:

1. Recommended model architecture: Provide only one of the following list: {get_all_models()}
   Choose a simpler or smaller architecture compared to the previous recommendation.

2. Training strategy: Choose only one of the following options and explain your choice in the response. Consider if a less computationally intensive strategy could be used:
   - "Fine-tuning the whole model"
   - "Full Training"
   - "Last Layer Learning"

3. Choose between TFLOPS32 or TFLOPS16: Choose the appropriate TFLOPS value for the GPU based on the user priorities and constraints. Output a string "TFLOPS32" or "TFLOPS16".

4. Reasoning: Explain why the chosen training strategy, model and the TFLOPS type is the best option considering the given constraints.

Respond with ONLY these three pieces of information, nothing else.
"""


class MainState(BaseModel):
    task: str = Field(description="The task to be planned")
    data: str = Field(description="Description of the data available")
    performance_needs: str = Field(description="Performance requirements")
    time: str = Field(description="Time limit")
    budget: str = Field(description="Budget")
    eco_friendliness: str = Field(description="Eco-friendliness")
    eco_weight: float = Field(description="Ranking of eco-friendliness")
    time_weight: float = Field(description="Ranking of time efficiency")
    cost_weight: float = Field(description="Ranking of cost efficiency")
    weight_reasoning: str = Field(description="Reasoning behind the chosen ranking")
    model_architecture: str = Field(description="Recommended model architecture as a Hugging Face model name (organization/model-name)")
    training_strategy: str = Field(description="Recommended training strategy: 'Last Layer Learning', 'Full Training', or 'Fine-Tuning the whole model")
    tflops_precision: str = Field(description="TFLOPS precision for the GPU")
    architecture_reasoning: str = Field(description="Reasoning behind the chosen training strategy and model")
    dataframe: dict = Field(description="GPU and cloud data")
    max_time: Optional[float] = Field(description="Maximum time constraint")
    max_cost: Optional[float] = Field(description="Maximum cost constraint")
    max_co2: Optional[float] = Field(description="Maximum CO2 emissions constraint")
    simplification_attempts: int = Field(default=0, description="Number of attempts to simplify the architecture")
    constraints_met: bool = Field(default=True, description="Whether the constraints were met")

class RankingState(BaseModel):
    eco_weight: float = Field(description="Ranking of eco-friendliness")
    time_weight: float = Field(description="Ranking of time efficiency")
    cost_weight: float = Field(description="Ranking of cost efficiency")
    weight_reasoning: str = Field(description="Reasoning behind the chosen ranking")

class ArchitectureState(BaseModel):
    model_architecture: str = Field(description="Recommended model architecture as a Hugging Face model name (organization/model-name)")
    training_strategy: str = Field(description="Recommended training strategy: 'Last Layer Learning', 'Full Training', or 'Fine-Tuning the whole model'")
    architecture_reasoning: str = Field(description="Reasoning behind the chosen training strategy and model")
    tflops_precision: str = Field(description="TFLOPS precision for the GPU")

class TimeState(BaseModel):
    sample_count: int = Field(description="Number of samples in the dataset")
    input_size: Tuple = Field(description="Size of the input data")
    estimated_epochs: int = Field(description="Estimated number of epochs to train the model")

def ranking_node(state: MainState) -> MainState:
    messages = [
        SystemMessage(content=RANKING_PROMPT),
        HumanMessage(content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Time: {state.time}
        Budget: {state.budget}
        Eco-friendliness: {state.eco_friendliness}
        """)
    ]    
    response = model.with_structured_output(RankingState).invoke(messages)
    total_sum = response.eco_weight + response.time_weight + response.cost_weight
    if total_sum == 0:
        raise ValueError("Sum of weights is zero, cannot normalize.")
    
    response.eco_weight /= total_sum
    response.time_weight /= total_sum
    response.cost_weight /= total_sum

    return MainState(
        task=state.task,
        data=state.data,
        performance_needs=state.performance_needs,
        time=state.time,
        budget=state.budget,
        eco_friendliness=state.eco_friendliness,
        weight_reasoning=response.weight_reasoning,
        eco_weight=response.eco_weight,
        time_weight=response.time_weight,
        cost_weight=response.cost_weight,
        model_architecture=state.model_architecture,
        training_strategy=state.training_strategy,
        architecture_reasoning=state.architecture_reasoning,
        tflops_precision=state.tflops_precision,
        dataframe=state.dataframe,
        max_time=state.max_time,
        max_cost=state.max_cost,
        max_co2=state.max_co2
    )

def architecture_node(state: MainState) -> MainState:
    if state.simplification_attempts == 0:
        prompt = ARCHITECTURE_PROMPT
        human_message = f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Time: {state.time}
        Budget: {state.budget}
        Eco-friendliness: {state.eco_friendliness}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        """
    else:
        prompt = SIMPLIFICATION_PROMPT
        human_message = f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Time: {state.time}
        Budget: {state.budget}
        Eco-friendliness: {state.eco_friendliness}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        Simplification Attempts: {state.simplification_attempts}
        Previous Architecture: {state.model_architecture}
        Previous Strategy: {state.training_strategy}
        """

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_message)
    ]
    response = model.with_structured_output(ArchitectureState).invoke(messages)
    return MainState(
        task=state.task,
        data=state.data,
        performance_needs=state.performance_needs,
        time=state.time,
        budget=state.budget,
        eco_friendliness=state.eco_friendliness,
        weight_reasoning=state.weight_reasoning,
        eco_weight=state.eco_weight,
        time_weight=state.time_weight,
        cost_weight=state.cost_weight,
        model_architecture=response.model_architecture,
        training_strategy=response.training_strategy,
        tflops_precision=response.tflops_precision,
        architecture_reasoning=response.architecture_reasoning,
        dataframe=state.dataframe,
        max_time=state.max_time,
        max_cost=state.max_cost,
        max_co2=state.max_co2,
        simplification_attempts=state.simplification_attempts,
        constraints_met=state.constraints_met
    )

def calculator_node(state: MainState) -> MainState:
    messages = [
        SystemMessage(content=CALCULATOR_PROMPT),
        HumanMessage(content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Eco-friendliness: {state.eco_friendliness}
        Model Architecture: {state.model_architecture}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        """)
    ]
    response = model.with_structured_output(TimeState).invoke(messages)
    dataframe = recommend_gpu_configuration(model=state.model_architecture, input_size=response.input_size, training_strategy=state.training_strategy, sample_count=response.sample_count, estimated_epochs=response.estimated_epochs, time_coeff=state.time_weight, cost_coeff=state.cost_weight, co2_coeff=state.eco_weight, tflops_type=state.tflops_precision, max_time=state.max_time, max_cost=state.max_cost, max_co2=state.max_co2)
    return MainState(
        task=state.task,
        data=state.data,
        performance_needs=state.performance_needs,
        time=state.time,
        budget=state.budget,
        eco_friendliness=state.eco_friendliness,
        weight_reasoning=state.weight_reasoning,
        eco_weight=state.eco_weight,
        time_weight=state.time_weight,
        cost_weight=state.cost_weight,
        model_architecture=state.model_architecture,
        training_strategy=state.training_strategy,
        tflops_precision=state.tflops_precision,
        architecture_reasoning=state.architecture_reasoning,
        dataframe=dataframe.to_dict() if not dataframe.empty else {},
        max_time=state.max_time,
        max_cost=state.max_cost,
        max_co2=state.max_co2,
        simplification_attempts=state.simplification_attempts + 1 if dataframe.empty else state.simplification_attempts,
        constraints_met=not dataframe.empty
    )

def should_simplify(state: MainState):
    return not state.constraints_met and state.simplification_attempts < 3


builder = StateGraph(MainState)

builder.add_node("ranking", ranking_node)
builder.add_node("architecturer", architecture_node)
builder.add_node("calculator", calculator_node)

builder.set_entry_point("ranking")
builder.add_edge("ranking", "architecturer")
builder.add_edge("architecturer", "calculator")
builder.add_conditional_edges(
    "calculator",
    should_simplify,
    {
        True: "architecturer",
        False: END
    }
)

graph = builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream(MainState(
    task="I want to build a computer vision model that detects cars in images.",
    data="I have a dataset of 1 million car images labeled with their bounding boxes. (Images resized to 256x256 pixels for efficiency)",
    performance_needs="The model should achieve best accuracy.",
    time="I have few days to train the model.",
    budget="Budget is not a major concern. But I want to minimize the cost.",
    eco_friendliness="I don't care about eco-friendliness.",
    weight_reasoning="",
    eco_weight=0.0,
    time_weight=0.0,
    cost_weight=0.0,
    model_architecture="",
    training_strategy="",
    tflops_precision="",
    architecture_reasoning="",
    dataframe={},
    max_time=5,
    max_cost=None,
    max_co2=None,
    simplification_attempts=0,
    constraints_met=True
), thread):
    print(s)

