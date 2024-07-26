from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Tuple

memory = SqliteSaver.from_conn_string(":memory:")

model = OllamaFunctions(
    model="phi3:mini", 
    keep_alive=-1,
    format="json"
)
RANKING_PROMPT = """
You are an AI assistant specializing in assigning importance to the priorities of eco-friendliness, time efficiency, and cost efficiency. Consider the task and context provided, and provide a ranking for each priority such that the total sum is 1.0. The bigger the value, the more important the priority.

1. Eco-friendliness: Rank the importance of eco-friendliness
2. Time efficiency: Rank the importance of time efficiency
3. Cost efficiency: Rank the importance of cost efficiency

Provide the values in the format of 0.0 to 1.0. Ensure the total sum is 1.0.

4. Reasoning: Explain why you chose the ranking for each priority.
"""


TIME_PROMPT = """
You are an AI assistant specializing in time estimation. Underhood, we have calculations but it requires you to fill the following information:

1. Amount of samples: Provide the number of samples in the dataset. If you don't have this information, provide an estimate.

2. Input size: Provide the size of the input, for example (256, 256, 3) to specify an image or (1024) to specify the length of a text sequence. If this information is not available, provide an estimate. Use 3D for images and 1D for text.

"""

ARCHITECTURE_PROMPT = """
You are an AI assistant specializing in recommending AI model architectures and training strategies. Provide concise recommendations:

Provide:

1. Recommended model architecture: Provide only the Hugging Face model name in the format 'organization/model-name'. For example: 'meta-llama/Meta-Llama-3-8B-Instruct'

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

1. Recommended GPU: Provide the GPU name that best suits the task.

2. Reasoning: Explain why the recommended GPU is appropriate given the task requirements, compute availability, time limit, budget constraints, and eco-efficiency considerations.

Respond with ONLY these two pieces of information, nothing else.
"""

class MainState(BaseModel):
    task: str = Field(description="The task to be planned")
    data: str = Field(description="Description of the data available")
    performance_needs: str = Field(description="Performance requirements")
    compute: str = Field(description="Available compute")
    time: str = Field(description="Time limit")
    budget: str = Field(description="Budget")
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

def ranking_node(state: MainState) -> MainState:
    messages = [
        SystemMessage(content=RANKING_PROMPT),
        HumanMessage(content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Compute: {state.compute}
        Time: {state.time}
        Budget: {state.budget}
        """)
    ]    
    response = model.with_structured_output(RankingState).invoke(messages)
    return MainState(
        task=state.task,
        data=state.data,
        performance_needs=state.performance_needs,
        compute=state.compute,
        time=state.time,
        budget=state.budget,
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
        Compute: {state.compute}
        Time: {state.time}
        Budget: {state.budget}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        """)
    ]
    response = model.with_structured_output(ArchitectureState).invoke(messages)
    return MainState(
        task=state.task,
        data=state.data,
        performance_needs=state.performance_needs,
        compute=state.compute,
        time=state.time,
        budget=state.budget,
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
        Compute: {state.compute}
        Time: {state.time}
        Budget: {state.budget}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        
        Recommended Model Architecture: {state.model_architecture}
        Training Strategy: {state.training_strategy}
        """)
    ]
    # Invoke the model to get GPU recommendation
    response = model.with_structured_output(GpuState).invoke(messages)
    return MainState(
        task=state.task,
        data=state.data,
        performance_needs=state.performance_needs,
        compute=state.compute,
        time=state.time,
        budget=state.budget,
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
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        """)
    ]
    # Invoke the model to get time estimation
    response = model.with_structured_output(TimeState).invoke(messages)
    print(response)
    return MainState(
        task=state.task,
        data=state.data,
        performance_needs=state.performance_needs,
        compute=state.compute,
        time=state.time,
        budget=state.budget,
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
    compute="I can use any compute from the cloud providers",
    time="I have few weeks to train the model.",
    budget="Budget is not a major concern. But I want to minimize the cost.",
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
