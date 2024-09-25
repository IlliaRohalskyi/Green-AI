from langchain_core.messages import HumanMessage, SystemMessage

from src.models.llm_model import get_llm_model
from src.models.state_models import ArchitectureState, MainState
from src.prompts.architecture_prompt import ARCHITECTURE_PROMPT
from src.prompts.simplification_prompt import SIMPLIFICATION_PROMPT
from src.logger import logging

def architecture_node(state: MainState) -> MainState:
    logging.info("Architecture node is executing")   
    model = get_llm_model()
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

    messages = [SystemMessage(content=prompt), HumanMessage(content=human_message)]
    response = model.with_structured_output(ArchitectureState).invoke(messages)
    logging.info(f"Architecture node response: {response}")
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
        constraints_met=state.constraints_met,
    )
