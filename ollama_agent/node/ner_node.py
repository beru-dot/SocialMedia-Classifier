from ..state import PostState
import json
import torch
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama

# NERrelationship definition (simplified for clarity)
class NERrelationship:
    def __init__(self, modelName, baseURL):
        try:
            self.modelName = modelName
            self.baseURL = baseURL
            self.client = ChatOllama(model=self.modelName, baseURL=self.baseURL, temperature=0)
        except Exception as e:
            print(f"Failed to initialise model: {str(e)}")

    def execute_qa_ner(self, inputText):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=200, separator=' ')
            chunks = text_splitter.split_text(inputText)
            ner_template = ChatPromptTemplate.from_template("""
                You are an advanced AI model tasked with generating professional, concise, and well-structured NER (Named Entity Recognition) based on the provided text.
                Here is the input Text:
                {text}

                Here is the output format:     
                    Location: Names of places such as cities, states, countries, districts, towns, or addresses.
                    Organization: Names of companies, institutions, groups, or NGOs.
                    Date: Any dates mentioned.
                    Person: Names of individuals.
                    Address: Specific addresses mentioned.
                Respond ONLY with JSON in this format:
                      "Location": [...],
                      "Organization": [...],
                      "Date": [...],
                      "Person": [...],
                      "Address": [...]

                Ensure that each entity is correctly classified into one category only.
                Do not create subcategories or add any notes.
                Preserve the exact spelling of person names as they appear in the original text.
                """)
            for _chunk in chunks:
                ner_prompt = ner_template.format(text=_chunk)
                ner_result = self.client.invoke(ner_prompt)
                data = json.loads(ner_result.content)
                return data
        except Exception as e:
            print(f"error {str(e)}")
            return {}

# Instantiate a global NER model instance
ner_model = NERrelationship(modelName="llama3.2:latest", baseURL="http://127.0.0.1:11434")

def ner_node(state: PostState) -> PostState:
    input_text = state["input_json"][0].get("post_content", "")
    ner_results = ner_model.execute_qa_ner(input_text)
    return {**state, "ner_result": ner_results}
