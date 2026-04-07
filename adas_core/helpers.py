from typing import List, Optional, Tuple, Any, Dict
from langchain_core.messages import AIMessage, HumanMessage
import subprocess
import ast
import io
import re

def get_filtered_packages(exclude_packages: Optional[List[str]] = None) -> List[str]:
    if exclude_packages is None:
        exclude_packages = []
    
    result = subprocess.run(['pip', 'list', '--not-required'], 
                          capture_output=True, text=True)
    
    packages = []
    for line in result.stdout.strip().split('\n')[2:]:  # Skip header lines
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                package_name = parts[0]
                version = parts[1]
                
                if package_name not in exclude_packages:
                    packages.append(f"{package_name} {version}")
    return packages

def validate_node_router_signature(function_code: str) -> Tuple[bool, Optional[str]]:
    """
    Validates the signature of a node or router function.
    It should accept exactly one argument named 'state'.
    """

    try:
        tree = ast.parse(function_code.strip())
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"

    # Find the function definition node
    func_def_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def_node = node
            break
    
    if not func_def_node:
        return False, f"No function definition found in the provided code."

    # Check the arguments
    args = func_def_node.args
    
    num_pos_args = len(args.args)
    has_vararg = args.vararg is not None
    has_kwarg = args.kwarg is not None
    num_kwonly_args = len(args.kwonlyargs)

    if num_pos_args == 1 and args.args[0].arg == 'state' and not has_vararg and not has_kwarg and num_kwonly_args == 0:
        return True, None
    else:
        error_parts = []
        if num_pos_args != 1:
            error_parts.append(f"Expected 1 positional argument, but found {num_pos_args}.")
        elif args.args[0].arg != 'state':
            error_parts.append(f"Expected the positional argument to be named 'state', but found '{args.args[0].arg}'.")
        
        if has_vararg:
            error_parts.append(f"Unexpected *args (variable positional arguments) found: '{args.vararg.arg}'.")
        if has_kwarg:
            error_parts.append(f"Unexpected **kwargs (variable keyword arguments) found: '{args.kwarg.arg}'.")
        if num_kwonly_args > 0:
            kwonly_names = [kw.arg for kw in args.kwonlyargs]
            error_parts.append(f"Unexpected keyword-only arguments found: {', '.join(kwonly_names)}.")
            
        return False, f"Invalid signature for '{func_def_node.name}'. Nodes and Routers must accept exactly one argument named 'state'. Issues: {' '.join(error_parts)}"


class TruncatingStringIO(io.StringIO):
    """A custom StringIO that truncates each individual write operation by removing the middle."""
    def __init__(self, limit: int = 1200, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.limit = limit

    def write(self, s: str) -> int:
        """Overrides the default write method to truncate before writing it to the buffer."""
        truncated_message = f"\n...[OUTPUT TRUNCATED]...\n"
        if len(s) > (self.limit + len(truncated_message)):
            start_chunk = s[:(self.limit // 2)]
            end_chunk = s[-(self.limit // 2):]
            s = start_chunk + truncated_message + end_chunk
        return super().write(s)
    
def clean_messages(messages: List[Any]) -> List[Any]:
    allowed_attributes = {'type', 'content', 'tool_calls', 'invalid_tool_calls'}
    allowed_tool_call_keys = {'name', 'args'}

    for message in messages:
        for attr_name in list(vars(message).keys()):
            if attr_name not in allowed_attributes:
                try:
                    delattr(message, attr_name)
                except AttributeError:
                    pass
        
        if isinstance(message, AIMessage):
            for tool_call_list_name in ['tool_calls', 'invalid_tool_calls']:
                if hasattr(message, tool_call_list_name):
                    cleaned_calls = []
                    original_calls = getattr(message, tool_call_list_name)
                    if not original_calls:
                        continue
                        
                    for call in original_calls:
                        if isinstance(call, dict):
                            cleaned_call = {k: v for k, v in call.items() if k in allowed_tool_call_keys}
                            cleaned_calls.append(cleaned_call)
                    
                    setattr(message, tool_call_list_name, cleaned_calls)
    return messages

def truncate_state(state: Dict[str, Any], max_chars: int = 1200) -> Optional[Dict[str, Any]]:
    if not state:
        return None
    
    truncated_state = {}
    truncated_message_template = "...[VALUE FOR '{}' (Type: {}) HAS BEEN TRUNCATED]..."
    msg_content_truncated_template = "...[MESSAGE CONTENT TRUNCATED]..."
    
    for key, value in state.items():
        if key == "messages" and isinstance(value, list):
            cleaned_msgs = clean_messages(value)
            
            # Truncate the content of each message
            for msg in cleaned_msgs:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if len(msg.content) > (max_chars+ len(msg_content_truncated_template)):
                        start_chunk = msg.content[:(max_chars // 2)]
                        end_chunk = msg.content[-(max_chars // 2):]
                        msg.content = start_chunk + msg_content_truncated_template + end_chunk

            truncated_state[key] = cleaned_msgs
        else:
            value_str = str(value)
            truncated_message = truncated_message_template.format(key, type(value).__name__)
            if len(value_str) > (max_chars + len(truncated_message)):
                start_chunk = value_str[:(max_chars // 2)]
                end_chunk = value_str[-(max_chars // 2):]
                truncated_value_str = start_chunk + truncated_message + end_chunk
                truncated_state[key] = truncated_value_str
            else:
                truncated_state[key] = value
                
    return truncated_state
    
def remove_old_test_results(start_index, messages):
    test_report_pattern = re.compile(r"Test suite completed\..*?</ValidatorResult>", re.DOTALL)
    
    def create_summary(match_obj):
        report_text = match_obj.group(0)
        score_match = re.search(r"The system passed (\d+)/(\d+) tests", report_text)
        if score_match:
            passed, total = score_match.groups()
            return f"[Test executed. The system passed {passed}/{total} tests.]"
        else:
            return "[Previous test result condensed.]"

    for i in range(start_index, len(messages)):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            msg.content = test_report_pattern.sub(create_summary, msg.content)