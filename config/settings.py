# Settings

dependencies = [
    "langgraph==1.2.9",
    "langchain_openai==1.4.1",
    "python-dotenv==1.2.2",
    "dill==0.3.9",
]
max_iterations = 40  # Maximum number of steps the meta system should perform (e.g., LLM calls)

allowed_target_models = [{"wrapper": "openai", "model_name": "gpt-5.6-luna"}]  # Allowed models for the target system
