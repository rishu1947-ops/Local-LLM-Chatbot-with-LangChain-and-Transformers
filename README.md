This project is a simple demonstration of how to use the LangChain framework to interact with a locally-run language model from the Hugging Face Hub. The script loads a model (e.g., `distilgpt2`), wraps it in a LangChain pipeline, and uses a custom prompt template to generate a response in the style of a pirate.

## Features

-   Loads any Causal Language Model from the Hugging Face Hub.
-   Uses the `transformers` library to create a text-generation pipeline.
-   Integrates the Hugging Face pipeline into a LangChain chain using `HuggingFacePipeline`.
-   Utilizes a `PromptTemplate` to format user input with specific instructions.
-   Chains components together using the LangChain Expression Language (LCEL).
-   Outputs the model's formatted response to the console.

## Prerequisites

-   Python 3.8 or higher
-   `pip` (Python package installer)

## Setup and Installation

1.  **Clone the repository (or save the files):**
    If this were a Git repository, you would clone it. For now, just make sure you have `main.py` and `requirements.txt` in the same directory.

2.  **Create a virtual environment (recommended):**
    It's good practice to create a virtual environment to manage project dependencies.
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    Use pip to install all the necessary libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    This will download the models and dependencies, which may take a few minutes.

## How to Run

Execute the `main.py` script from your terminal:

```bash
python main.py
```

## Expected Output

When you run the script, it will first download the `distilgpt2` model if you don't have it cached. Then, it will process the question and print the output.

```
Loading model: distilgpt2

User asks: What's the best way to find treasure?

Pirate LLM says:
Answer this question in the style of a pirate: What's the best way to find treasure? First, you have to find out what's the best way to find treasure.

The best way to find treasure is to find out what's the best way to find treasure.

You have to find out what's the best way to find treasure.

You have to find out what's the best way to find treasure.

The best way to find treasure is to find out what's the best way to find treasure.

You have to find out what's the best way to find treasure.
...
```

**Note:** The generated answer from `distilgpt2` will vary and may not be highly coherent, as it's a very small model. The purpose of this example is to demonstrate the pipeline, not the quality of a specific model.

## Customization

You can easily modify the `main.py` script to experiment:

-   **Use a different model:** Change the `model_repo_id` in the `load_model_pipeline` function call.
    ```python
    # Example: Using a different model from Microsoft
    local_llm = load_model_pipeline(model_repo_id="microsoft/phi-2")
    ```
    *Note: Larger models will require more RAM and a more powerful machine.*

-   **Change the prompt style:** Modify the `my_template` string to change the persona or instructions.
    ```python
    my_template = "Explain this to a five-year-old: {question}"
    ```

-   **Ask a different question:** Update the `question_to_ask` variable.
    ```python
    question_to_ask = "Why is the sky blue?"
    ```
