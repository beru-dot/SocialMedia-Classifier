from langgraph.graph import StateGraph,START,END
from .node.ner_node import ner_node
from .node.sentiment_node import sentiment_node
from .node.translation_node import translation_node
from .node.summary_node import summary_node
from .state import PostState
from .node.orchestrator_node import orchestrator_node

def routing_function(state: PostState) -> str:
    flags = state["input_json"]
    
    if flags[0].get("ner", False):
        return "NER"
    if flags[0].get("sentiment",False):
        return "sentiment"
    if flags[0].get("translation",False):
        return "translation"
    if flags[0].get("summary",False):
        return "summary"
    return "__end__"

def synthesizer_node(state: PostState) -> PostState:
    print(state.get("summary_result"))
    combined = {
        "entities": state.get("ner_result", {}).get("entities", []),
        "sentiment": state.get("sentiment_result", {}).get("sentiment", ""),
        "translation": state.get("translation_result", ""),
        "summary": state.get("summary_result","")
    }
    return {**state, "combined_result": combined}

def make_graph():
    graph = StateGraph(state_schema=PostState)
    graph.add_node("NER", ner_node)
    graph.add_node("Sentiment", sentiment_node)
    graph.add_node("Translation", translation_node)
    graph.add_node("Summary",summary_node)
    graph.add_node("Synthesizer", synthesizer_node)

    # All processing nodes flow into the synthesizer simultaneously (fan-in)
    graph.add_edge(START, "NER")
    graph.add_edge("NER", "Sentiment")
    graph.add_edge("Sentiment", "Translation")
    graph.add_edge("Translation", "Summary")
    graph.add_edge("Summary", "Synthesizer")
    graph.add_edge("Synthesizer", END)

    return graph.compile()