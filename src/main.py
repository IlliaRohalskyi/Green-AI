import pandas as pd
from src.graph.state_graph import create_graph
from src.models.state_models import MainState
from src.logger import logging

def main(task, data, performance_needs, time, budget, eco_friendliness):
    """
    Main function that creates a graph and iterates over the stream of MainState objects.
    Each MainState object represents a task.
    The function returns the result as a Pandas DataFrame.

    Parameters:
        task (str): The task description.
        data (str): Information about the dataset.
        performance_needs (str): Performance requirements.
        time (str): Time constraints.
        budget (str): Budget constraints.
        eco_friendliness (str): Eco-friendliness considerations.
    Returns:
        str: The resulting dataframe in JSON format.
    """
    logging.info("Starting main function")

    graph = create_graph()

    logging.info("Graph created successfully")

    thread = {"configurable": {"thread_id": "1"}}
    logging.info(f"Thread initialized: {thread}")

    result_data = []
    for s in graph.stream(
        MainState(
            task=task,
            data=data,
            performance_needs=performance_needs,
            time=time,
            budget=budget,
            eco_friendliness=eco_friendliness,
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
        # pass
        print(s)
    if len(s) == 1:
        node_key = next(iter(s))
        node_response = s[node_key]
        result_data = node_response['dataframe']
    
    # Konvertiere die Daten in ein Pandas DataFrame
    df = pd.DataFrame(result_data)
    
    # Gebe das DataFrame als JSON zurück
    return df.to_json()

if __name__ == "__main__":
    # Testaufruf der main-Funktion
    result = main(
        "I want to build a computer vision model that detects cars in images.",
        "I have a dataset of 1 million car images labeled with their bounding boxes.",
        "The model should achieve best accuracy.",
        "I have few days to train the model.",
        "Budget is not a major concern.",
        "I don't care about eco-friendliness."
    )
    print(result)
