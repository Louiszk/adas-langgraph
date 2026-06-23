# Define the task or problem for the meta system to work on in a variable named "problem_statement"
# or pass a --problem "..." flag to run_design.py

# Example format of a problem statement:
# problem_statement = """
# Design a system to solve 'Project Euler' tasks.
# Project Euler challenges participants to solve complex mathematical and computational problems
# using programming skills and mathematical insights.
#
# The system should consist of only one agent and one tool.
# The tool should allow execution of Python code, so that the agent can solve any problem.
# The state must contain the attribute "solution": "str", where only the final solution is saved.
# """

# Example: use one of the benchmark prompts
with open("generated_systems/GSMHard/prompts.txt", "r") as prompt_file:
    problem_statement = prompt_file.read()
