import pandas as pd
from src.graph.state_graph import create_graph
from src.models.state_models import MainState
from src.logger import logging

def main(task, data, performance_needs, time, budget, eco_friendliness, max_time=1e19, max_cost=None, max_co2=None):
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
        max_time (float): Maximum allowed time for training.
        max_cost (float): Maximum allowed cost.
        max_co2 (float): Maximum allowed CO2 emissions.
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
            max_time=max_time,
            max_cost=max_cost,
            max_co2=max_co2,
            simplification_attempts=0,
            constraints_met=True,
        ),
        thread,
    ):
        # Example print for each state (optional)
        print(s)
        
    # Assuming result_data is collected in the stream loop
    if len(s) == 1:
        node_key = next(iter(s))
        node_response = s[node_key]
        result_data = node_response['dataframe']
    
    # Convert result data into Pandas DataFrame
    df = pd.DataFrame(result_data)
    
    # Return the DataFrame as JSON
    return df.to_json()

if __name__ == "__main__":
    # Test invocation of the main function
    result = main(
        "I want to build a computer vision model that detects cars in images.",
        "I have a dataset of 1 million car images labeled with their bounding boxes.",
        "The model should achieve best accuracy.",
        "I have few days to train the model.",
        "Budget is not a major concern.",
        "I don't care about eco-friendliness.",
        max_time=100000,
        max_cost=5000,
        max_co2=50
    )
    print(result)
