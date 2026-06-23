# Settings

dependencies = [
    "langgraph==0.4.8",
    "langchain_openai==0.3.32",
    "python-dotenv==1.0.1",
    "dill==0.3.9",
]
max_iterations = 60  # Maximum number of steps the meta system should perform (e.g., LLM calls)

allowed_target_models = [{"wrapper": "openai", "model_name": "gpt-4.1-mini"}]  # Allowed models for the target system
