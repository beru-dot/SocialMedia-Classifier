from ..state import PostState
import json
import torch
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama

# summaryrelationship definition (simplified for clarity)
class summary:
    def __init__(self, modelName, baseURL):
        try:
            self.modelName = modelName
            self.baseURL = baseURL
            self.client = ChatOllama(model=self.modelName, baseURL=self.baseURL, temperature=0)
        except Exception as e:
            print(f"Failed to initialise model: {str(e)}")

    def execute_qa_summary(self, inputText):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=200, separator=' ')
            chunks = text_splitter.split_text(inputText)
            summary_template = ChatPromptTemplate.from_template("""
            You are a neutral summarizer for social media posts. Produce a concise (200 words) 
            summary capturing the main claims, entities, locations, allegations or calls to action. 
            Keep it fact-forward and avoid invented facts. Do not include URLs. Be precise.
            Sentence: {text}\n\n
            """)
            for _chunk in chunks:
                summary_prompt = summary_template.format(text=_chunk)
                summary_result = self.client.invoke(summary_prompt)
                data = {"summary":summary_result.content.strip()}
                print(data)
                return data
        except Exception as e:
            print(f"error {str(e)}")
            return {}

# Instantiate a global summary model instance
summary_model = summary(modelName="llama3.2:latest", baseURL="http://127.0.0.1:11434")

def summary_node(state: PostState) -> PostState:
    input_text = state.get("translation_result", "")
    
    summary_results = summary_model.execute_qa_summary(input_text['translation'])

    return {**state, "summary_result": summary_results}
