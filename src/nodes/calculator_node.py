from langchain_core.messages import HumanMessage, SystemMessage

from models.llm_model import get_llm_model
from models.state_models import MainState, TimeState
from prompts.calculator_prompt import CALCULATOR_PROMPT
from utils.gpu_calculations import recommend_gpu_configuration

from src.logger import logging


def calculator_node(state: MainState) -> MainState:
    logging.info("Calculator node is executing")
    model = get_llm_model()
    messages = [
        SystemMessage(content=CALCULATOR_PROMPT),
        HumanMessage(
            content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Eco-friendliness: {state.eco_friendliness}
        Model Architecture: {state.model_architecture}
        Priorities (Weights 0-1): Eco-friendly {state.eco_weight}, Time-efficient {state.time_weight}, Cost-efficient {state.cost_weight}
        """
        ),
    ]
    response = model.with_structured_output(TimeState).invoke(messages)
    dataframe = recommend_gpu_configuration(
        model=state.model_architecture,
        input_size=response.input_size,
        training_strategy=state.training_strategy,
        sample_count=response.sample_count,
        estimated_epochs=response.estimated_epochs,
        time_coeff=state.time_weight,
        cost_coeff=state.cost_weight,
        co2_coeff=state.eco_weight,
        tflops_type=state.tflops_precision,
        max_time=state.max_time,
        max_cost=state.max_cost,
        max_co2=state.max_co2,
    )
    logging.info(f"Calculator node response: {response}")
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
        simplification_attempts=(
            state.simplification_attempts + 1
            if dataframe.empty
            else state.simplification_attempts
        ),
        constraints_met=not dataframe.empty,
    )
