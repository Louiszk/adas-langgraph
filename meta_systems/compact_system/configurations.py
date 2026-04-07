# Configurations
ACTION_CUTOFF = 4  # Number of previous actions preserved in context
RECURSION_LIMIT = 20 # Maximum number of supersteps the target system can perform
MAX_HARDENING_STEPS = 5 # Maximum number of times to loop back and generate harder tests.
meta_agent_wrapper = "openai"
meta_agent_model = "gpt-4.1-mini"
meta_agent_reasoning_effort = None

validation_wrapper = "openai"
validation_model = "gpt-5-mini"