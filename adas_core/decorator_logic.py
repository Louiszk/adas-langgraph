import io
import inspect
import tokenize
import textwrap
from langgraph.graph import START, END
from typing import List, Callable, Dict, Optional, Tuple, Any, Union, Iterable, get_origin
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
import re

def extract_parenthesized_content(lines: List[str], start_line_idx: int, start_pos: Optional[int] = None) -> Tuple[Optional[str], Optional[int]]:
    """
    Extracts content inside matching parentheses using a robust, fully token-based approach.
    This version correctly handles malformed/invalid source code by catching TokenError.
    """
    source_code = "\n".join(lines)
    
    try:
        all_tokens = list(tokenize.generate_tokens(io.StringIO(source_code).readline))
    except tokenize.TokenError:
        return None, None

    if start_pos is None:
        start_pos = lines[start_line_idx].find("(")
    
    start_token_idx = -1
    start_ln, start_col = start_line_idx + 1, start_pos

    # 1. Find the index of our starting parenthesis token
    for i, tok in enumerate(all_tokens):
        if tok.type == tokenize.OP and tok.string == '(' and tok.start == (start_ln, start_col):
            start_token_idx = i
            break
    
    if start_token_idx == -1:
        return None, None

    # 2. Find the matching closing parenthesis by tracking levels
    paren_level = 1
    content_tokens = []
    
    for i in range(start_token_idx + 1, len(all_tokens)):
        tok = all_tokens[i]
        
        if tok.type == tokenize.OP:
            if tok.string == '(':
                paren_level += 1
            elif tok.string == ')':
                paren_level -= 1

        if paren_level == 0:
            end_line = tok.start[0] - 1
            return tokenize.untokenize(content_tokens).strip(), end_line

        content_tokens.append(tok)
    
    return None, None

def find_code_blocks(markdown: str) -> List[Dict[str, Union[str, int]]]:
    """
    Finds Python code blocks in markdown, using the built-in `tokenize` module
    to correctly handle Python's own syntax.
    """
    lines = markdown.splitlines()
    found_blocks = []
    
    in_code_block = False
    current_block_content = None
    current_block_start_line = None
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()

        if not in_code_block:
            if stripped_line.startswith('```'):
                in_code_block = True
                current_block_content = []
                current_block_start_line = i + 1
        else:
            if stripped_line == '```':
                block_so_far = '\n'.join(current_block_content)
                
                try:
                    list(tokenize.generate_tokens(io.StringIO(block_so_far).readline))
                    
                    in_code_block = False
                    found_blocks.append({
                        "content": block_so_far,
                        "start_line": current_block_start_line,
                        "end_line": i + 1
                    })
                    current_block_content = None
                    
                except tokenize.TokenError:
                    current_block_content.append(line)
            else:
                current_block_content.append(line)

    return found_blocks


def parse_arguments(args_str: Optional[str]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    pos_args, kw_args = (), {}
    
    if args_str:
        eval_context = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "ToolMessage": ToolMessage,
            "START": START, 
            "END": END
        }
        
        results_from_exec = {} 
        exec_str = f"def _parsing_temp_func(*args, **kwargs): return args, kwargs\npos_args, kw_args = _parsing_temp_func({args_str})"
        
        try:
            exec(exec_str, eval_context, results_from_exec)
            
            pos_args = results_from_exec.get('pos_args', ())
            kw_args = results_from_exec.get('kw_args', {})
        except NameError as ne:
            error_msg = (f"Invalid argument: a name '{ne.name}' was used in the tool arguments "
                         f"'{args_str}' but it is not defined. Do not use '{ne.name}'.")
            raise ValueError(error_msg) from ne
        except SyntaxError as se:
            error_msg = (f"Syntax error in tool arguments: '{args_str}'. Please ensure the "
                         f"arguments are correctly formatted. Details: {se}")
            raise ValueError(error_msg) from se
        except Exception as e:
            raise ValueError(f"Failed to parse tool arguments '{args_str}': {e}") from e
            
    return pos_args, kw_args

def parse_decorator_tool_calls(block_content: str, code_related_tools: Dict[str, str]) -> List[Dict[str, Any]]:
    """Parse all decorator-style tool calls from a single markdown block."""
    tool_calls = []
    lines = block_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('@@'):
            call_match = re.match(r'@@([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if call_match:
                decorator_name = call_match.group(1)
                start_pos = call_match.end()
                open_paren_pos = line.find('(', start_pos)
                
                if open_paren_pos != -1:
                    args_str, end_line_idx = extract_parenthesized_content(lines, i, open_paren_pos)
                    pos_args, kw_args = parse_arguments(args_str)
                    tool_name = "".join([part.capitalize() for part in decorator_name.split("_")]) # camelfy
                    
                    if decorator_name in code_related_tools:
                        code_start_line = end_line_idx + 1
                        
                        # Find the next decorator or the end of the block
                        next_decorator_idx = -1
                        for j in range(code_start_line, len(lines)):
                            if lines[j].strip().startswith('@@'):
                                next_decorator_idx = j
                                break
                        
                        code_end_line = next_decorator_idx if next_decorator_idx != -1 else len(lines)
                        
                        content = '\n'.join(lines[code_start_line:code_end_line])
                        param_name = code_related_tools[decorator_name]
                        kw_args[param_name] = content
                        
                        i = code_end_line
                    else:
                        i = end_line_idx + 1
                    
                    tool_call = {
                        'name': tool_name,
                        'decorator_name': decorator_name,
                        'pos_args': pos_args,
                        'kw_args': kw_args
                    }
                    tool_calls.append(tool_call)
                    continue
        i += 1
    
    return tool_calls

def execute_decorator_tool_calls(response_content: Union[str, List[str]], available_tools: Dict[str, Any], code_related_tools: Dict[str, str], state: Any) \
    -> Tuple[Optional[HumanMessage], List[Tuple[str, Any]]]:
    """Execute decorator-style tool calls found in the text"""
    tool_messages = []
    tool_results = []
    
    if not isinstance(response_content, str):
        if isinstance(response_content, list):
            response_content = " ".join(response_content)
        else:
            response_content = ""

    code_blocks = [found_block['content'] for found_block in find_code_blocks(response_content) if found_block['content']]
    
    all_decorator_calls = []
    parsing_error_message = None
    
    def add_skipped_calls_message(current_index, message):
        remaining_calls = len(all_decorator_calls) - current_index - 1
        if remaining_calls > 0:
            tool_messages.append(f"{message} {remaining_calls} subsequent decorator call(s) in this response were skipped. You can make new decorator calls in your next response.")
    
    for i, code_block in enumerate(code_blocks):
        try:
            parsed_calls = parse_decorator_tool_calls(code_block, code_related_tools)
            all_decorator_calls.extend(parsed_calls)
        except Exception as e:
            parsing_error_message = f"Error parsing code block {i}: {repr(e)}"
            break

    for idx, tool_call in enumerate(all_decorator_calls):
        tool_name = tool_call['name']
        decorator_name = tool_call['decorator_name']
        pos_args = tool_call.get('pos_args', ())
        kw_args = tool_call.get('kw_args', {})
        kw_args["state"] = state
        
        if tool_name in available_tools:
            try:
                tool = available_tools[tool_name]
                result = tool.func(*pos_args, **kw_args) if hasattr(tool, 'func') and callable(tool.func) else tool.invoke(kw_args)
                
                result_str = str(result) if result else f"Decorator `@@{decorator_name}` executed successfully."
                tool_messages.append(result_str)
                tool_results.append((tool_name, result))

                if state.get("design_completed"):
                    add_skipped_calls_message(idx, "Execution halted. Design completed.")
                    break

                if "ERROR:" in result_str.split("</Metrics>")[-1]:
                    add_skipped_calls_message(idx, "Execution halted due to error.")
                    break
                
            except Exception as e:
                error_message = f"Error executing decorator `@@{decorator_name}`: {repr(e)}"
                tool_messages.append(error_message)
                tool_results.append((tool_name, error_message))
                
                add_skipped_calls_message(idx, "Execution halted due to exception.")
                break
        else:
            add_skipped_calls_message(idx, f"Decorator `@@{decorator_name}` not found. Execution halted.")
            break
            
    if parsing_error_message:
        tool_messages.append(parsing_error_message)
        
    human_message = HumanMessage(content = "\n\n".join(tool_messages)) if tool_messages else None
                
    return human_message, tool_results


def build_decorator_signatures(tools: Iterable[Callable[..., Any]], code_related_tools: Dict[str, str]) -> str:
    """
    Builds a formatted string of decorator signatures and docstrings for the meta-agent's prompt.
    - Omits specified code-related parameters from the signature.
    - Skips return type annotations.
    - Formats both simple and complex/nested type hints cleanly.
    """
    signature_blocks = []

    for func in tools:
        if not callable(func):
            continue

        decorator_name = func.__name__
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)

        param_strings = []
        code_param_to_exclude = code_related_tools.get(decorator_name)
        framework_params_to_exclude = {'state'}

        for param in sig.parameters.values():
            if (param.name == code_param_to_exclude or
                param.name in framework_params_to_exclude):
                continue

            param_str = param.name
            
            if param.annotation != inspect.Parameter.empty:
                # Use get_origin to robustly check if the type is a generic from 'typing'
                if get_origin(param.annotation) is not None:
                    annotation_str = str(param.annotation).replace('typing.', '')
                else:
                    annotation_str = param.annotation.__name__
                
                param_str += f": {annotation_str}"
            
            # Add default value if it exists
            if param.default != inspect.Parameter.empty:
                param_str += f" = {repr(param.default)}"
            
            param_strings.append(param_str)
        
        # Format the complete signature line (without return value)
        params_joined = ", ".join(param_strings)
        signature_line = f"@@{decorator_name}({params_joined})"

        # Format the docstring
        doc_block = ""
        if doc:
            dedented_doc = textwrap.dedent(doc).strip()
            indented_doc = textwrap.indent(dedented_doc, "    ")
            doc_block = f'    """\n{indented_doc}\n    """'

        full_block = f"{signature_line}\n{doc_block}" if doc_block else signature_line
        signature_blocks.append(full_block)

    # Assemble the final prompt string
    header = "You only have the following decorators available for designing the system:"
    all_signatures = "\n\n".join(signature_blocks)
    
    return f"{header}\n```\n{all_signatures}\n```"