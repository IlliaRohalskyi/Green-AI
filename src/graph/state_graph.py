from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.config import MAX_SIMPLIFICATION_ATTEMPTS
from src.models.state_models import MainState
from src.nodes.architecture_node import architecture_node
from src.nodes.calculator_node import calculator_node
from src.nodes.ranking_node import ranking_node

from src.logger import logging

memory = SqliteSaver.from_conn_string(":memory:")


def should_simplify(state: MainState):
    logging.info("Checking if simplification is needed")
    return (
        not state.constraints_met
        and state.simplification_attempts < MAX_SIMPLIFICATION_ATTEMPTS
    )


def create_graph():
    logging.info("Creating graph")
    builder = StateGraph(MainState)

    builder.add_node("ranking", ranking_node)
    builder.add_node("architecturer", architecture_node)
    builder.add_node("calculator", calculator_node)

    builder.set_entry_point("ranking")
    builder.add_edge("ranking", "architecturer")
    builder.add_edge("architecturer", "calculator")
    builder.add_conditional_edges(
        "calculator", should_simplify, {True: "architecturer", False: END}
    )
    logging.info("Graph created successfully")
    return builder.compile(checkpointer=memory)
