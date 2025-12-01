# Social Media Content Analyzer

This project is a modular social media content analysis system built using LangGraph and Ollama large language models. It performs multiple NLP tasks on social media posts, including Named Entity Recognition (NER), Sentiment Analysis, Translation, and Summarization.

## Features

- **NER Node**: Extracts key entities like locations, organizations, persons from posts.
- **Sentiment Node**: Classifies the sentiment of posts as Positive, Negative, or Neutral.
- **Translation Node**: Translates code-mixed or non-English posts into fluent English.
- **Summary Node**: Provides concise and factual summaries of translated texts.
- Built with a **sequential LangGraph workflow** for robust state passing.
- Leverages **Ollama LLMs** for all language processing tasks with custom prompt templates.

## Folder Structure
social_media_analyzer/
├── src/
│ ├── init.py
│ ├── graph.py # LangGraph workflow definition
│ ├── state.py # State schemas and types
│ └── nodes/
│ ├── init.py
│ ├── ner_node.py # NER node implementation
│ ├── sentiment_node.py # Sentiment node implementation
│ ├── translation_node.py # Translation node implementation
│ └── summary_node.py # Summarization node implementation
├── config/
│ ├── init.py
│ └── settings.py # Configuration for models and environment
├── app.py # Entry point to run the LangGraph analyzer
├── requirements.txt # Python dependencies
└── README.md # This documentation file


## Setup & Installation

1. Clone the repo and navigate into it:
    ```
    git clone <repo-url>
    cd social_media_analyzer
    ```

2. Create and activate a Python virtual environment:
    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
    ```

3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Ensure **Ollama** is installed and running, and models are downloaded as specified in config/settings.py.

## Usage

Run the app which initializes the LangGraph pipeline and processes social media posts:
    python app.py


The system invokes the sequential nodes: NER → Sentiment → Translation → Summary → Synthesizer.

## How it Works

- Social media posts or text inputs feed into the LangGraph `START` node.
- The state moves sequentially through each NLP task node, augmented by Ollama LLM calls.
- Output results are aggregated into a combined state at the `Synthesizer` node for final reporting.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to fork and submit pull requests.

---

Built with ❤️ using LangGraph and Ollama LLMs.



