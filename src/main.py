"""
This module initializes and runs a state graph.

Classes:
    MainState: Represents the state of the graph with various attributes.

Functions:
    main(): Initializes the graph and streams states through it, printing each state.

Usage:
    Run this module as a script to execute the main function.
"""

from src.graph.state_graph import create_graph
from src.models.state_models import MainState
from src.logger import logging


def main():
    """
    Main function that creates a graph and iterates over the stream of MainState objects.
    Each MainState object represents a task.
    The function prints each MainState object.

    Parameters:
        None
    Returns:
        None
    """
    logging.info("Starting main function")


    graph = create_graph()

    logging.info("Graph created successfully")

    thread = {"configurable": {"thread_id": "1"}}
    logging.info(f"Thread initialized: {thread}")

    for s in graph.stream(
        MainState(
            task="I want to build a computer vision model that detects cars in images.",
            data=(
                "I have a dataset of 1 million car images labeled with their bounding boxes. "
                "(Images resized to 256x256 pixels for efficiency)"
            ),
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
            max_time=1e19,
            max_cost=None,
            max_co2=None,
            simplification_attempts=0,
            constraints_met=True,
        ),
        thread,
    ):
        print(s)


if __name__ == "__main__":
    main()
