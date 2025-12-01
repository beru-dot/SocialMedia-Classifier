import json
from ollama_agent.agent import make_graph
from ollama_agent.state import PostState

json_data = [
    {
        "post_content": "लोग आजकल सोशल मीडिया पर बहुत एक्टिव हैं।",
        "ner": True,
        "sentiment": True,
        "translation": True,
        "summary": True,
    }
]



if __name__ == '__main__':
    initial_state : PostState = {'input_json':json_data}
    graph = make_graph()
    result_state = graph.invoke(initial_state,{"recursion_limit":100})
    print("*"*10)
    print(result_state)

