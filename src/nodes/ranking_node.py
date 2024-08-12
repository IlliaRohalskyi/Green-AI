from langchain_core.messages import HumanMessage, SystemMessage

from src.models.llm_model import get_llm_model
from src.models.state_models import MainState, RankingState
from src.prompts.ranking_prompt import RANKING_PROMPT
from src.logger import logging


def ranking_node(state: MainState) -> MainState:
    logging.info("Ranking node is executing")
    model = get_llm_model()
    messages = [
        SystemMessage(content=RANKING_PROMPT),
        HumanMessage(
            content=f"""
        Task: {state.task}
        Data: {state.data}
        Performance Needs: {state.performance_needs}
        Time: {state.time}
        Budget: {state.budget}
        Eco-friendliness: {state.eco_friendliness}
        """
        ),
    ]
    response = model.with_structured_output(RankingState).invoke(messages)
    total_sum = response.eco_weight + response.time_weight + response.cost_weight
    if total_sum == 0:
        raise ValueError("Sum of weights is zero, cannot normalize.")

    response.eco_weight /= total_sum
    response.time_weight /= total_sum
    response.cost_weight /= total_sum
    
    logging.info(f"Ranking node response: {response}")
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
        max_co2=state.max_co2,
    )
