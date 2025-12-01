from ..state import PostState
import json
import torch
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama

# translationrelationship definition (simplified for clarity)
class translation:
    def __init__(self, modelName, baseURL):
        try:
            self.modelName = modelName
            self.baseURL = baseURL
            self.client = ChatOllama(model=self.modelName, baseURL=self.baseURL, temperature=0)
        except Exception as e:
            print(f"Failed to initialise model: {str(e)}")

    def execute_qa_translation(self, inputText):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=200, separator=' ')
            chunks = text_splitter.split_text(inputText)
            translation_template = ChatPromptTemplate.from_template("""
                Please translate the following text into fluent, natural English. The input may contain code-mixed languages, transliterations, mentions (@username), hashtags (#topic), informal expressions, and misspellings. Preserve @username mentions and relevant hashtags as per context.
                Text to translate:
                {text}
                Translation:                
            """
        )
            for _chunk in chunks:
                translation_prompt = translation_template.format(text=_chunk)
                translation_result = self.client.invoke(translation_prompt)
                data = {"translation":translation_result.content.strip()}
                
                return data
        except Exception as e:
            print(f"error {str(e)}")
            return {}

# Instantiate a global translation model instance
translation_model = translation(modelName="llama3.2:latest", baseURL="http://127.0.0.1:11434")

def translation_node(state: PostState) -> PostState:
    input_text = state["input_json"][0].get("post_content", "")
    translation_results = translation_model.execute_qa_translation(input_text)
    return {**state, "translation_result": translation_results}
