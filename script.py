import os
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import RelevanceEvaluator, ViolenceEvaluator
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
aoai_key = os.getenv("AOAI_KEY")
api_version = os.getenv("API_VERSION")
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
project_name = os.getenv("PROJECT_NAME")
resource_group = os.getenv("RESOURCE_GROUP")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_key = os.getenv("SEARCH_KEY")
subscription_id = os.getenv("SUBSCRIPTION_ID")


azure_ai_project = {
    "subscription_id": subscription_id,
    "resource_group_name": resource_group,
    "project_name": project_name,
}
 
# ViolenceEvaluator
violence_eval = ViolenceEvaluator(azure_ai_project)
violence_score = violence_eval(question="What is the capital of France?", answer="Paris.")



# Initialize Azure OpenAI Connection with your environment variables
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=endpoint,
    api_key=aoai_key,
    azure_deployment=deployment,
    api_version=api_version,
)
 
# Initialzing Relevance Evaluator
relevance_eval = RelevanceEvaluator(model_config)
# Running Relevance Evaluator on single input row
# relevance_score = relevance_eval(
#     response="The Alpine Explorer Tent is the most waterproof.",
#     context="From the our product list,"
#     " the alpine explorer tent is the most waterproof."
#     " The Adventure Dining Table has higher weight.",
#     query="Which tent is the most waterproof?",
# )
relevance_score = relevance_eval(context="From the our product list the alpine explorer tent is the most waterproof. The Adventure Dining Table has higher weight.", question="Which tent is the most waterproof?", answer="The Alpine Explorer Tent is the most waterproof.")
print(relevance_score)