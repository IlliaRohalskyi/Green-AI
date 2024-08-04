from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Tuple, List
import pandas as pd
from src.utility import get_root
import os
import numpy as np

memory = SqliteSaver.from_conn_string(":memory:")

model = OllamaFunctions(
    model="phi3:mini", 
    keep_alive=-1,
    format="json",
    temperature=0,
    top_p=0.2
)

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

def recommend_gpu(scores, tflops_type):
    pricing_df = pd.read_excel(os.path.join(get_root(), 'data', 'pricing', 'GCP gpus pricing.xlsx'))
    pricing_df.columns = ['Region', 'GPU', 'Price']
    gpu_prices = pricing_df.groupby('GPU')['Price'].mean().reset_index()
    gpu_prices['Normalized_Price'] = normalize_data(gpu_prices['Price'])
    
    performance_df = pd.read_csv(os.path.join(get_root(), 'data', 'gpus.csv'))
    
    manual_map = {
        'T4': 'T4',
        'V100': 'Tesla V100-PCIE-16GB',
        'P100': 'Tesla P100',
        'K80': 'Tesla K80',
    }
    
    gpu_prices['Mapped_GPU'] = gpu_prices['GPU'].apply(lambda x: manual_map.get(x, x))
    performance_data = {row['name']: row for _, row in performance_df.iterrows()}
    
    merged_data = []
    for _, row in gpu_prices.iterrows():
        pricing_gpu = row['Mapped_GPU']
        if pricing_gpu in performance_data:
            perf_data = performance_data[pricing_gpu]
            tflops_value = get_tflops_value(perf_data, tflops_type)
            tflops_per_watt = tflops_value / perf_data['tdp_watts']
            merged_data.append({
                'GPU': row['GPU'],
                'Mapped_GPU': pricing_gpu,
                'Price': row['Price'],
                'Normalized_Price': row['Normalized_Price'],
                'TDP_Watts': perf_data['tdp_watts'],
                tflops_type: tflops_value,
                'TFLOPS_per_Watt': tflops_per_watt
            })
    
    merged_df = pd.DataFrame(merged_data)
    
    merged_df['Normalized_TFLOPS'] = normalize_data(merged_df[tflops_type])
    merged_df['Normalized_TDP'] = normalize_data(merged_df['TDP_Watts'])
    merged_df['Normalized_TFLOPS_per_Watt'] = normalize_data(merged_df['TFLOPS_per_Watt'])

    merged_df['Price_Score'] = (1 - merged_df['Normalized_Price']) * scores['price']
    merged_df['TFLOPS_Score'] = merged_df['Normalized_TFLOPS'] * scores['time']
    merged_df['TDP_Score'] = merged_df['Normalized_TFLOPS_per_Watt'] * scores['co2']

    merged_df['Total_Score'] = merged_df['Price_Score'] + merged_df['TFLOPS_Score'] + merged_df['TDP_Score']
    
    best_gpu = merged_df.sort_values('Total_Score', ascending=False).iloc[0]
    
    return best_gpu['Mapped_GPU']

def calculate_kwh_consumption(gpu_name, time_seconds):
    path_to_gpu = os.path.join(get_root(), "data", "gpus.csv")
    gpu_df = pd.read_csv(path_to_gpu)

    tdp_watts = gpu_df.loc[gpu_df['name'] == gpu_name, 'tdp_watts'].values[0]
    
    tdp_kw = tdp_watts / 1000
    
    time_hours = time_seconds / 3600
    
    energy_consumption_kwh = tdp_kw * time_hours
    
    return energy_consumption_kwh


def get_all_models() -> List[str]:
    path_to_models = os.path.join(get_root(), "data", "model_flops", "model_flops.xlsx")
    models = pd.read_excel(path_to_models)['Model']
    return list(models)

def estimate_flops(model: str, input_size: Tuple[int, int], training_strategy: str, sample_count: int, estimated_epochs: int) -> float:
    path_to_flops = os.path.join(get_root(), "data", "model_flops", "model_flops.xlsx")
    flops_df = pd.read_excel(path_to_flops)

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

    if training_strategy in ["Fine-tuning the whole model", "Full Training"]:
        return estimated_epochs * sample_count * model_info['FLOPs'].iloc[0] * scaling * 3
    elif training_strategy == "Last Layer Learning":
        return estimated_epochs * sample_count * 2 * model_info['Last Layer FLOPs'].iloc[0] + model_info['FLOPs'].iloc[0] * scaling
    else:
        raise ValueError(f"Unsupported training strategy: {training_strategy}")

def estimate_time(flops: float, gpu: str, training_strategy: str, tflops: str) -> Tuple[str, float]:
    path_to_gpu = os.path.join(get_root(), "data", "gpus.csv")
    gpu_df = pd.read_csv(path_to_gpu)
    gpu_info = gpu_df[gpu_df["name"] == gpu]

    if gpu_info.empty:
        raise ValueError(f"GPU {gpu} not found in the flops database")
    
    tflops_value = get_tflops_value(gpu_info.iloc[0], tflops)
    time_seconds = flops / tflops_value / 1e+12 if training_strategy in ["Full Training", "Fine-tuning the whole model"] else flops / tflops_value / 1e+12 / 3

    seconds = int(time_seconds % 60)
    minutes = int((time_seconds // 60) % 60)
    hours = int((time_seconds // 3600) % 24)
    days = int((time_seconds // 86400) % 30)
    months = int(time_seconds // (86400 * 30))

    output = []
    if months > 0:
        output.append(f"{months} month{'s' if months > 1 else ''}")
    if days > 0:
        output.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        output.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        output.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0 or not output:
        output.append(f"{seconds} second{'s' if seconds > 1 else ''}")

    formatted_time = ', '.join(output[:-1]) + f" and {output[-1]}" if len(output) > 1 else output[0]

    return formatted_time, time_seconds

RANKING_PROMPT = """
You are an AI assistant specializing in assigning importance to the priorities of eco-friendliness, time efficiency, and cost efficiency. Consider the task and context provided, and provide a ranking for each priority. The bigger the value, the more important the priority.
Pay attention to the constraints and requirements of the task provided by the user. Be constructive and tell why you chose the ranking for each priority based on the user input.

1. Eco-friendliness: Rank the importance of eco-friendliness
2. Time efficiency: Rank the importance of time efficiency
3. Cost efficiency: Rank the importance of cost efficiency

Provide the values in the format of 0.0 to 1.0. For example, 0.4, 0.2, 0.

4. Reasoning: Explain why you chose the ranking for each priority without repeating the rankings.
"""


TIME_PROMPT = """
You are an AI assistant specializing in time estimation. Underhood, we have calculations but it requires you to fill the following information:

1. Amount of samples: Provide the number of samples in the dataset. If you don't have this information, provide an estimate.

2. Input size: Provide the size of the input, for example (256, 256, 3) to specify an image or (512, ) to specify the length of a text sequence. If this information is not available, provide an estimate. Use 3D for images and 1D for text.

3. Estimated number of epochs: Provide the estimated number of epochs to train the model. Take into account the model architecture complexity and the size of the dataset.

4. Choose between TFLOPS32 or TFLOPS16: Choose the appropriate TFLOPS value for the GPU based on the user priorities and constraints. Output a string "TFLOPS32" or "TFLOPS16".
"""

ARCHITECTURE_PROMPT = f"""
You are an AI assistant specializing in recommending AI model architectures and training strategies. Provide concise recommendations:

Provide:

1. Recommended model architecture: Provide only one of the following list: {get_all_models()}

2. Training strategy: Choose only one of the following options and explain your choice in the response. Think if the knowledge from the model could be transferred to the new task:
   - "Fine-tuning the whole model"
   - "Full Training"
   - "Last Layer Tuning"

3. Reasoning: Explain why the chosen training strategy and the model is the best option considering the given constraints.

Respond with ONLY these three pieces of information, nothing else.
"""

GPU_AGENT_PROMPT = """
You are an AI assistant specializing in recommending GPUs for training deep learning models. Provide concise recommendations:

Provide:

1. Recommended GPU: Select a GPU from the list of GPUs: "Tesla V100-PCIE-16GB", "Tesla P100", "T4", "A100 PCIe 40/80GB"

2. Reasoning: Explain why the recommended GPU is appropriate given the task requirements, compute availability, time limit, budget constraints, and eco-efficiency considerations.

Respond with ONLY these two pieces of information, nothing else.
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
    training_strategy: str = Field(description="Recommended training strategy: 'Last Layer Tuning', 'Full Training', or 'Fine-Tuning the whole model")
    architecture_reasoning: str = Field(description="Reasoning behind the chosen training strategy and model")
    recommended_gpu: str = Field(description="Recommended GPU for training the model")
    gpu_reasoning: str = Field(description="Reasoning behind the chosen GPU recommendation")


class RankingState(BaseModel):
    eco_weight: float = Field(description="Ranking of eco-friendliness")
    time_weight: float = Field(description="Ranking of time efficiency")
    cost_weight: float = Field(description="Ranking of cost efficiency")
    weight_reasoning: str = Field(description="Reasoning behind the chosen ranking")

class ArchitectureState(BaseModel):
    model_architecture: str = Field(description="Recommended model architecture as a Hugging Face model name (organization/model-name)")
    training_strategy: str = Field(description="Recommended training strategy: 'Last Layer Tuning', 'Full Training', or 'Fine-Tuning the whole model'")
    architecture_reasoning: str = Field(description="Reasoning behind the chosen training strategy and model")

class GpuState(BaseModel):
    recommended_gpu: str = Field(description="Recommended GPU for training the model")
    gpu_reasoning: str = Field(description="Reasoning behind the chosen GPU recommendation")

class TimeState(BaseModel):
    sample_count: int = Field(description="Number of samples in the dataset")
    input_size: Tuple = Field(description="Size of the input data")
    estimated_epochs: int = Field(description="Estimated number of epochs to train the model")
    tflops_precision: str = Field(description="TFLOPS precision for the GPU")

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
        model_architecture="",
        training_strategy="",
        architecture_reasoning="",
        recommended_gpu="",
        gpu_reasoning=""
    )

def architecture_node(state: MainState) -> MainState:
    messages = [
        SystemMessage(content=ARCHITECTURE_PROMPT),
        HumanMessage(content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Time: {state.time}
        Budget: {state.budget}
        Eco-friendliness: {state.eco_friendliness}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        """)
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
        architecture_reasoning=response.architecture_reasoning,
        recommended_gpu=state.recommended_gpu,
        gpu_reasoning=state.gpu_reasoning
    )

def gpu_recommendation_node(state: MainState) -> MainState:
    messages = [
        SystemMessage(content=GPU_AGENT_PROMPT),
        HumanMessage(content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Time: {state.time}
        Budget: {state.budget}
        Eco-friendliness: {state.eco_friendliness}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        
        Recommended Model Architecture: {state.model_architecture}
        Training Strategy: {state.training_strategy}
        """)
    ]
    response = model.with_structured_output(GpuState).invoke(messages)
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
        architecture_reasoning=state.architecture_reasoning,
        recommended_gpu=response.recommended_gpu,
        gpu_reasoning=response.gpu_reasoning
    )

def time_node(state: MainState) -> MainState:
    messages = [
        SystemMessage(content=TIME_PROMPT),
        HumanMessage(content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Eco-friendliness: {state.eco_friendliness}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        """)
    ]
    response = model.with_structured_output(TimeState).invoke(messages)
    print(response)
    flops = estimate_flops(state.model_architecture, response.input_size, state.training_strategy, response.sample_count, response.estimated_epochs)
    print(flops)
    time = estimate_time(flops, state.recommended_gpu, state.training_strategy, response.tflops_precision)
    print(response)
    print("FLOPs:", flops)
    print(time)
    energy_consumption = calculate_kwh_consumption(state.recommended_gpu, time[1])
    print("Energy Consumption (kWh):", energy_consumption)
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
        architecture_reasoning=state.architecture_reasoning,
        recommended_gpu=state.recommended_gpu,
        gpu_reasoning=state.gpu_reasoning
    )


builder = StateGraph(MainState)

builder.add_node("ranking", ranking_node)
builder.add_node("architecturer", architecture_node)
builder.add_node("gpu_recommender", gpu_recommendation_node)
builder.add_node("time_estimator", time_node)

builder.set_entry_point("ranking")
builder.add_edge("ranking", "architecturer")
builder.add_edge("architecturer", "gpu_recommender")
builder.add_edge("gpu_recommender", "time_estimator")
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
    architecture_reasoning="",
    recommended_gpu="",
    gpu_reasoning=""
), thread):
    print(s)
