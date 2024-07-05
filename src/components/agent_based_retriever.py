from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

model = OllamaFunctions(
    model="phi3:mini", 
    keep_alive=-1,
    format="json"
)

PLAN_PROMPT = """
You are a machine learning expert focused on Green AI development. A user has a question about Green AI. As an expert, create a detailed step-by-step plan to answer the question. Ensure your plan includes the following:

1. Understanding the Question: Clarify the user's question to ensure you fully understand their needs.
2. Identifying Key Components: Break down the question into key components related to Green AI (e.g., CO2 impact, cost, training strategies).
3. Gathering Information: Outline steps to gather necessary information from available resources (e.g., research papers).

"""

class PlanOutput(BaseModel):
    task: str = Field(description="The task to be planned")
    key_components: List[str] = Field(description="Key components of the task")
    steps: List[str] = Field(description="Steps to complete the task")

def plan_node(state: PlanOutput) -> PlanOutput:
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state.task)
    ]
    response = model.with_structured_output(PlanOutput).invoke(messages)
    return PlanOutput(
        task=state.task,
        key_components=response.key_components,
        steps=response.steps
    )

builder = StateGraph(PlanOutput)

builder.add_node("planner", plan_node)
builder.set_entry_point("planner")

graph = builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream(PlanOutput(
    task="I want to build a computer vision model that works on low-power devices. How can I get started?",
    key_components=[],
    steps=[]
), thread):
    print(s)