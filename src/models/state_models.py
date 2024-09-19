from typing import Optional, List

from langchain_core.pydantic_v1 import BaseModel, Field


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
    model_architecture: str = Field(
        description="Recommended model architecture as a Hugging Face model name (organization/model-name)"
    )
    training_strategy: str = Field(
        description="Recommended training strategy: 'Last Layer Learning', 'Full Training', or 'Fine-Tuning the whole model"
    )
    tflops_precision: str = Field(description="TFLOPS precision for the GPU")
    architecture_reasoning: str = Field(
        description="Reasoning behind the chosen training strategy and model"
    )
    dataframe: dict = Field(description="GPU and cloud data")
    max_time: Optional[float] = Field(description="Maximum time constraint")
    max_cost: Optional[float] = Field(description="Maximum cost constraint")
    max_co2: Optional[float] = Field(description="Maximum CO2 emissions constraint")
    simplification_attempts: int = Field(
        default=0, description="Number of attempts to simplify the architecture"
    )
    constraints_met: bool = Field(
        default=True, description="Whether the constraints were met"
    )


class RankingState(BaseModel):
    eco_weight: float = Field(description="Ranking of eco-friendliness")
    time_weight: float = Field(description="Ranking of time efficiency")
    cost_weight: float = Field(description="Ranking of cost efficiency")
    weight_reasoning: str = Field(description="Reasoning behind the chosen ranking")


class ArchitectureState(BaseModel):
    model_architecture: str = Field(
        description="Recommended model architecture as a Hugging Face model name (organization/model-name)"
    )
    training_strategy: str = Field(
        description="Recommended training strategy: 'Last Layer Learning', 'Full Training', or 'Fine-Tuning the whole model'"
    )
    architecture_reasoning: str = Field(
        description="Reasoning behind the chosen training strategy and model"
    )
    tflops_precision: str = Field(description="TFLOPS precision for the GPU")


class TimeState(BaseModel):
    sample_count: int = Field(description="Number of samples in the dataset")
    input_size: List = Field(description="Size of the input data")
    estimated_epochs: int = Field(
        description="Estimated number of epochs to train the model"
    )
