from ..state import PostState


# def orchestrator_node(state: PostState) -> PostState:
#     input_flags = state["input_json"]
#     next_steps = []
#     if input_flags.get('ner'):
#         next_steps.append("ner")

#     return {**state, "next_steps":next_steps}

def orchestrator_node(state:PostState)->PostState:
    return state