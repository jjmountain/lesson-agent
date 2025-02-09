from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from agent.utils.nodes import (
    teaching_router, give_context, present_word, 
    check_meaning, give_hint, model_pronunciation, 
    evaluate_pronunciation
)
from agent.utils.state import State


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]
    
# Define a new graph
workflow = StateGraph(State, config_schema=GraphConfig)

# Add nodes
workflow.add_node("router", teaching_router)
workflow.add_node("give_context", give_context)
workflow.add_node("present_word", present_word)
workflow.add_node("check_meaning", check_meaning)
workflow.add_node("give_hint", give_hint)
workflow.add_node("model_pronunciation", model_pronunciation)
workflow.add_node("evaluate_pronunciation", evaluate_pronunciation)

# Add edges
workflow.add_conditional_edges(
    "router",
    lambda x: x["decision"],
    {
        "give_context": "give_context",
        "present_word": "present_word",
        "check_meaning": "check_meaning",
        "give_hint": "give_hint",
        "model_pronunciation": "model_pronunciation",
        "evaluate_pronunciation": "evaluate_pronunciation",
        "redirect": "router",
        "next_word": "router",
        "end_lesson": END,
        "end": END,
        "": END
    }
)

# Connect nodes back to router
for node in ["give_context", "present_word", "check_meaning", 
                "give_hint", "model_pronunciation", "evaluate_pronunciation"]:
    workflow.add_edge(node, "router")

workflow.set_entry_point("router")


graph = workflow.compile()

# Define initial state
initial_state = {
    "messages": [],
    "current_word": {
        "word": "食洗器",
        "reading": "しょくせんき",
        "meaning": "dishwasher"
    },
    "pronunciation_attempts": 0,
    "meaning_attempts": 0,
    "last_user_input": "",
    "disruption_history": [],
    "current_step": "give_context",
    "decision": "give_context",
    "output": "",
    "remaining_words": [
        {
            "word": "オーブン",
            "reading": "おーぶん",
            "meaning": "oven"
        }
    ]
}

# for output in graph.stream(initial_state):
#     print(output)


# # Define the two nodes we will cycle between
# workflow.add_node("agent", call_model)
# workflow.add_node("action", tool_node)

# # Set the entrypoint as `agent`
# # This means that this node is the first one called
# workflow.set_entry_point("agent")

# # We now add a conditional edge
# workflow.add_conditional_edges(
#     # First, we define the start node. We use `agent`.
#     # This means these are the edges taken after the `agent` node is called.
#     "agent",
#     # Next, we pass in the function that will determine which node is called next.
#     should_continue,
#     # Finally we pass in a mapping.
#     # The keys are strings, and the values are other nodes.
#     # END is a special node marking that the graph should finish.
#     # What will happen is we will call `should_continue`, and then the output of that
#     # will be matched against the keys in this mapping.
#     # Based on which one it matches, that node will then be called.
#     {
#         # If `tools`, then we call the tool node.
#         "continue": "action",
#         # Otherwise we finish.
#         "end": END,
#     },
# )

# # We now add a normal edge from `tools` to `agent`.
# # This means that after `tools` is called, `agent` node is called next.
# workflow.add_edge("action", "agent")

# # Finally, we compile it!
# # This compiles it into a LangChain Runnable,
# # meaning you can use it as you would any other runnable
# graph = workflow.compile()