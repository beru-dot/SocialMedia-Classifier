from ..state import PostState
import json
import torch
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama

# sentimentrelationship definition (simplified for clarity)
class Sentiment:
    def __init__(self, modelName, baseURL):
        try:
            self.modelName = modelName
            self.baseURL = baseURL
            self.client = ChatOllama(model=self.modelName, baseURL=self.baseURL, temperature=0)
        except Exception as e:
            print(f"Failed to initialise model: {str(e)}")

    def execute_qa_sentiment(self, inputText):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=200, separator=' ')
            chunks = text_splitter.split_text(inputText)
            sentiment_template = ChatPromptTemplate.from_template("""
            Classify the sentiment of the following sentence as one of: Positive, Negative, or Neutral.\n
            Respond with only the class.\n
            If unclear, lean toward Neutral.\n\n
            Sentence: {text}\n\n
            """)
            for _chunk in chunks:
                sentiment_prompt = sentiment_template.format(text=_chunk)
                sentiment_result = self.client.invoke(sentiment_prompt)
                data = {"sentiment":sentiment_result.content.strip()}
                
                return data
        except Exception as e:
            print(f"error {str(e)}")
            return {}

# Instantiate a global sentiment model instance
sentiment_model = Sentiment(modelName="llama3.2:latest", baseURL="http://127.0.0.1:11434")

def sentiment_node(state: PostState) -> PostState:
    input_text = state["input_json"][0].get("post_content", "")
    sentiment_results = sentiment_model.execute_qa_sentiment(input_text)
    return {**state, "sentiment_result": sentiment_results}
