from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_model_pipeline(model_repo_id="distilgpt2"):
    """ A helper function to load everything we need from Hugging Face. """
    print(f"Loading model: {model_repo_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    model = AutoModelForCausalLM.from_pretrained(model_repo_id)
    
    # Setting up the pipeline for our model
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=100
    )
    
    # Using the LangChain wrapper to make it usable in a chain
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    return hf_llm


# Define the model we're using
local_llm = load_model_pipeline()

# My custom prompt template
my_template = "Answer this question in the style of a pirate: {question}"
my_prompt = PromptTemplate.from_template(my_template)

# Creating the chain to process the prompt and get a response
llm_chain = my_prompt | local_llm | StrOutputParser()

# Let's test it out
question_to_ask = "What's the best way to find treasure?"
print(f"\nUser asks: {question_to_ask}")

# Run the chain
pirate_answer = llm_chain.invoke({"question": question_to_ask})

print("\nPirate LLM says:")
print(pirate_answer)