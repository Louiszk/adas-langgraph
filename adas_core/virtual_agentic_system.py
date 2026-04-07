import re
import ast
import textwrap
import collections
import inspect
from typing import TypedDict, Tuple, Dict, List, Any, Callable, Optional
from langgraph.graph import START, END
from adas_core.helpers import validate_node_router_signature

def _extract_top_level_names(ast_module: ast.Module) -> set[str]:
    names = set()
    for node in ast_module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names

class RemoveDefinitionsTransformer(ast.NodeTransformer):
    def __init__(self, names_to_remove: set[str]):
        self.names_to_remove = names_to_remove
        super().__init__()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name in self.names_to_remove:
            return None
        return self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST | None:
        if node.name in self.names_to_remove:
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.names_to_remove:
                return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if isinstance(node.target, ast.Name) and node.target.id in self.names_to_remove:
            return None
        return self.generic_visit(node)

class VirtualAgenticSystem:
    """
    A virtual representation of an agentic system with nodes, tools, and edges.
    This class provides a way to define the structure of an agentic system
    without actually compiling it.
    """
    
    def __init__(self, system_name: str = 'Default') -> None:
        self.system_name = system_name
        self.escaped_name = system_name.replace("/", "").replace("\\", "").replace(":", "")
        
        self.nodes = {}  # node_name -> {'description': str, 'source_code': str}
        self.tools = {}  # tool_name -> {'description': str, 'source_code': str}
        
        self.edges = []  # List[(source, target)]
        self.conditional_edges = {}  # source_node -> {condition_code: str, path_map: dict}
        
        
        self.packages_info = ["langchain-core-0.3.75", "langgraph 0.4.8"]
        self.installed_packages = {}
        self.base_imports = [
            "from adas_core.llm_wrapper import LargeLanguageModel, execute_tool_calls",
            "from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict, Iterable, Sequence, Annotated",
            "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage, trim_messages",
            "from langgraph.graph import StateGraph, START, END",
            "from langgraph.graph.message import add_messages",
            "from langchain_core.tools import tool",
            "import os"
            ]
        self.imports = list(self.base_imports)
        self.state_attributes = {"messages": "Annotated[List[AnyMessage], add_messages]"}
        
        self.utility_code = ""
        
    def set_state_attributes(self, attrs: Dict[str, str]) -> bool:
        self.state_attributes = {"messages": "Annotated[List[AnyMessage], add_messages]"}
        for name, type_annotation in attrs.items():
            self.state_attributes[name] = type_annotation
        return True
    
    def set_state_from_node(self, class_def_node: ast.ClassDef) -> str:
        """Validates the AgentState definition from a code string, then sets it."""
        attributes_found = {}
        if not class_def_node:
            return "ERROR: No 'AgentState' class definition was found in the provided code block."
        try:    
            for item in class_def_node.body:
                if isinstance(item, ast.AnnAssign):
                    attributes_found[item.target.id] = ast.unparse(item.annotation)
            
            if not attributes_found:
                return "WARNING: The AgentState definition was empty. No attributes were set."

            state_exec_globals = {"__builtins__": __builtins__}
            combined_code_parts = []
            combined_code_parts.extend(self.imports)
            combined_code_parts.append(textwrap.dedent(self.utility_code))
            combined_code_parts.append(ast.unparse(class_def_node))
            
            full_code_to_exec = "\n".join(combined_code_parts)
            exec(full_code_to_exec, state_exec_globals)
        except Exception as e:
            return f"ERROR: Validation failed for AgentState definition. {repr(e)}"

        self.set_state_attributes(attributes_found)
        return "AgentState defined successfully."
    
    def _parse_from_import(self, imp_str: str):
        """Parse strings like 'from module import a, b as c' into (module, set(names))"""
        clean = imp_str.strip()
        if not clean.startswith('from ') or ' import ' not in clean:
            return None, None
        try:
            parts = clean.split(' import ', 1)
            left = parts[0].strip()
            module = left[len('from '):].strip() if left.startswith('from ') else left
            names_part = parts[1].strip()
            names = {name.split(' as ')[0].strip() for name in names_part.split(',') if name.strip()}
            return module, names
        except Exception:
            return None, None

    # It currently does not merge separate imports from the same module
    def deduplicate_imports(self, new_import_statements: List[str]) -> List[str]:
        # Build existing map for from-imports based ONLY on base_imports
        existing_imports_map = {}
        for imp_str in self.base_imports:
            module, names = self._parse_from_import(imp_str)
            if module:
                existing_imports_map.setdefault(module, set()).update(names)

        truly_new_imports = []
        for new_imp_str in new_import_statements:
            clean_new = new_imp_str.strip()

            if clean_new in self.base_imports: 
                continue

            module, new_names = self._parse_from_import(clean_new)
            if module and new_names is not None:
                if "*" in existing_imports_map.get(module, set()):
                    continue
                if new_names.issubset(existing_imports_map.get(module, set())):
                    continue
                truly_new_imports.append(clean_new)
            else:
                truly_new_imports.append(clean_new)

        return sorted(set(truly_new_imports))
    
    def set_imports(self, import_statements: List[str]) -> str:
        """Validates and sets the import statements for the system."""
        if not import_statements:
            return "WARNING: No import statements were found. No changes were made."
        try:
            new_unique_imports = self.deduplicate_imports(import_statements)
            import_exec_globals = {}
            import_code = "\n".join(new_unique_imports)
            exec(import_code, import_exec_globals)
        except Exception as e:
            return f"ERROR: Validation failed for import statements. {repr(e)}"

        self.imports = self.base_imports + new_unique_imports
        return f"Imports set successfully with {len(import_statements)} statements."
        
    def create_node(self, name: str, description: str, func: Callable, func_source_code: Optional[str] = None) -> bool:
        if name in ["START", "__start__", START, "END", "__end__", END]:
            raise ValueError(f"START and END are reserved names for the endpoints of the graph.")
            
        if not func_source_code:
            func_source_code = textwrap.dedent(inspect.getsource(func))
        
        if name in self.nodes and self.nodes[name]["source_code"] == func_source_code:
            return False
        
        self.nodes[name] = {"description": description, "source_code": func_source_code}
        return True

    def create_tool(self, name: str, description: str, func: Callable, func_source_code: Optional[str] = None) -> bool:
        """Create a tool function that can be used by nodes."""
        if func.__doc__ is None or func.__doc__.strip() == "":
            raise ValueError("Tool function must contain a detailed docstring.")

        if not func_source_code:
            func_source_code = textwrap.dedent(inspect.getsource(func))

        if name in self.tools and self.tools[name]["source_code"] == func_source_code:
            return False
        
        self.tools[name] = {"description": description, "source_code": func_source_code}
        return True
    
    def _infer_path_map(self, function_code: str) -> Dict[str, Any]:
        """Infer possible return values from conditional edge function using regex."""
        path_map = {}
        
        string_literal_pattern = r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\''
        for match in re.finditer(string_literal_pattern, function_code):
            potential_target_str = match.group(1) if match.group(1) is not None else match.group(2)
            if potential_target_str:
                path_map[potential_target_str] = potential_target_str
        
        # Check for the END constant
        end_pattern = r'\bEND\b'
        if re.search(end_pattern, function_code):
            path_map["END"] = END
        
        return path_map

    def create_edge(self, source: Any, target: Any) -> bool:
        """Create a standard edge between nodes."""
        if source in ["START", "__start__"]:
            source = START
        if target in ["END", "__end__"]:
            target = END
            
        # Validate source and target nodes
        if source != START and source not in self.nodes:
            raise ValueError(f"Invalid source node: '{source}' does not exist")
            
        if target != END and target not in self.nodes:
            raise ValueError(f"Invalid target node: '{target}' does not exist")

        if source == target:
            raise ValueError(
                f"Standard edges from a node to itself are not allowed. Cannot create an edge from '{source}' to itself. "
                f"This would create an unconditional infinite loop, as standard edges lack an exit condition. "
                f"If you intend for a node to loop back to itself, you must use a conditional edge (a router)."
            )
        
        # Check for existing standard edge from source
        if (source, target) in self.edges:
            return False

        self.edges.append((source, target))
        return True

    def create_conditional_edge(self, source: str, condition: Callable, condition_code: Optional[str] = None, path_map: Optional[Dict[str, Any]] = None) -> bool:
        """Create a conditional edge with a router function."""
        if source in ["END", "__end__", END]:
            raise ValueError(f"Invalid source node: Routers from END are not allowed.")
        if source not in self.nodes:
            raise ValueError(f"Invalid source node: '{source}' does not exist")

        # Get or set condition code
        if not condition_code:
            condition_code = textwrap.dedent(inspect.getsource(condition))

        if source in self.conditional_edges and self.conditional_edges[source]["condition_code"] == condition_code:
            return False
        
        edge_info: Dict[str, Any] = {"condition_code": condition_code}
        
        # Get path map
        inferred_path_map = self._infer_path_map(condition_code)
        final_path_map = path_map if path_map is not None else inferred_path_map

        # Save path map if available
        if final_path_map:
            edge_info["path_map"] = final_path_map.copy()
        
        self.conditional_edges[source] = edge_info
        return True
    
    def delete_node(self, name: str) -> bool:
        """Delete a node and all associated edges."""
        if name in ["START" "__start__", START, "END", "__end__", END]:
            raise ValueError(f"Deletion of endpoints is not allowed")
        if name not in self.nodes:
            return False
        
        del self.nodes[name]
        
        if name in self.conditional_edges:
            del self.conditional_edges[name]
        
        self.edges = [(s, t) for s, t in self.edges if s != name and t != name]
        
        return True
    
    def delete_tool(self, name: str) -> bool:
        """Delete a tool."""
        if name not in self.tools:
            return False
        
        del self.tools[name] 
        return True

    def delete_edge(self, source: Any, target: Any) -> bool:
        """Delete a standard edge."""
        if source in ["START", "__start__"]:
            source = START
        if target in ["END", "__end__"]:
            target = END
            
        edge = (source, target)
        if edge not in self.edges:
            return False
        
        self.edges.remove(edge)
        return True

    def delete_conditional_edge(self, source: str) -> bool:
        """Delete a conditional edge."""
        if source not in self.conditional_edges:
            return False
        
        del self.conditional_edges[source]
        return True
    
    def get_function(self, function_code: str, component_type: Optional[str]) -> Tuple[Optional[Callable], str]:
        """
        Safely extracts and validates a single function definition from a code string.
        It uses AST parsing to isolate only the function's source code, ignoring other statements.
        """
        try:
            tree = ast.parse(textwrap.dedent(function_code))
            
            func_def_node = None
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_def_node = node
                    break
            
            if not func_def_node:
                return None, "ERROR: No function definition found in the provided code."

            function_name = func_def_node.name
            isolated_function_code = ast.unparse(func_def_node)

        except SyntaxError as e:
            return None, f"ERROR: Invalid Python syntax in the component's code block: {e}"
        
        if component_type and component_type.lower() in ["node", "router"]:
            is_valid_signature, sig_error_msg = validate_node_router_signature(isolated_function_code)
            if not is_valid_signature:
                return None, f"ERROR: for {component_type} '{function_name}': {sig_error_msg}"

        try:
            func_exec_globals = {
                "__builtins__": __builtins__,
                "tools": {},
                "AgentState": TypedDict
            }
            
            combined_code_parts = []
            combined_code_parts.extend(self.imports)
            combined_code_parts.append(textwrap.dedent(self.utility_code))
            combined_code_parts.append(isolated_function_code)
            
            full_code_to_exec = "\n".join(combined_code_parts)

            exec(full_code_to_exec, func_exec_globals)

        except Exception as e:
            return None, f"ERROR: executing function or import code for '{function_name}': {repr(e)}"
            
        if function_name in func_exec_globals and callable(func_exec_globals[function_name]):
            new_function = func_exec_globals[function_name]
            return new_function, isolated_function_code
        else:
            return None, f"ERROR: Function '{function_name}' not found after execution"

    def upsert_utility_code(self, new_code: str) -> str:
        """Adds or updates functions, classes, and constants in the utility code using AST parsing."""
        dedented_new_code = textwrap.dedent(new_code).strip()
        final_utility_code = ""
        if not dedented_new_code:
            return "Utilities updated successfully."

        try:
            new_ast = ast.parse(dedented_new_code)
            graph_definition_names = {"tools", "agentic_system_graph", "workflow"}
            filtered_body = []

            for node in new_ast.body:
                is_graph_related = False
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in graph_definition_names:
                            is_graph_related = True
                            break

                        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in graph_definition_names:
                            is_graph_related = True
                            break

                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name) and node.target.id in graph_definition_names:
                        is_graph_related = True

                elif isinstance(node, ast.ClassDef) and node.name == 'AgentState':
                    is_graph_related = True
                
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)) and not is_graph_related:
                    filtered_body.append(node)

            if not filtered_body:
                return "WARNING: Your submitted utility code is identical to the existing code. **No update was performed.**"
            new_ast.body = filtered_body
            names_to_replace = _extract_top_level_names(new_ast)

        except SyntaxError as e:
            raise ValueError(f"Syntax error in new utility code: {e}") from e

        dedented_existing_code = textwrap.dedent(self.utility_code).strip()
        if not dedented_existing_code:
            final_utility_code = ast.unparse(new_ast)
        else:
            try:
                existing_ast = ast.parse(dedented_existing_code)
            except SyntaxError as e:
                raise ValueError(f"Syntax error in existing utility code, cannot merge: {e}") from e

            transformer = RemoveDefinitionsTransformer(names_to_replace)
            modified_existing_ast = transformer.visit(existing_ast)

            if not isinstance(modified_existing_ast, ast.Module):
                raise TypeError("AST transformation did not return a Module node.")

            final_body = modified_existing_ast.body + new_ast.body
            final_ast = ast.Module(body=final_body, type_ignores=[])

            try:
                new_utility_code = ast.unparse(final_ast)
                if new_utility_code == self.utility_code:
                    return "WARNING: Your submitted utility code is identical to the existing code. **No update was performed.**"
                final_utility_code = new_utility_code
            except Exception as e:
                raise RuntimeError(f"Failed to unparse combined utility code AST: {e}") from e
        
        # Validation
        try:    
            utils_exec_globals = {"__builtins__": __builtins__}
            combined_code_parts = []
            combined_code_parts.extend(self.imports)
            combined_code_parts.append(textwrap.dedent(final_utility_code))
            
            full_code_to_exec = "\n".join(combined_code_parts)
            exec(full_code_to_exec, utils_exec_globals)
        except Exception as e:
            return f"ERROR: Validation failed for Utility Code. {repr(e)}"

        self.utility_code = final_utility_code
        return "Utilities updated successfully."

    def validate_graph(self) -> List[str]:
        """Validate graph structure and return list of errors."""
        errors = []

        # Get all defined nodes
        all_defined_nodes = set(self.nodes.keys())
        
        # Check edges for invalid connections
        for s, t in self.edges:
            if s != START and s not in all_defined_nodes:
                errors.append(f"Edge source '{s}' in ('{s}', '{t}') is not a defined node and not START.")
            if t != END and t not in all_defined_nodes:
                errors.append(f"Edge target '{t}' in ('{s}', '{t}') is not a defined node and not END.")
            if t == START: 
                errors.append(f"Edge ('{s}', '{t}') targets START, which is invalid.")
            if s == END: 
                errors.append(f"Edge ('{s}', '{t}') originates from END, which is invalid.")

        # Check conditional edges for invalid configurations
        for s, edge_info in self.conditional_edges.items():
            if s not in all_defined_nodes:
                errors.append(f"Conditional edge source '{s}' is not a defined node.")
            
            path_map = edge_info.get("path_map", {})
            if not path_map:
                errors.append(f"Conditional edge from source '{s}' has no correct return values. Must return str, List[str] or END.")

        # Check reachability from START
        reachable_nodes = set()
        q = collections.deque()
        
        # Verify START has outgoing edges
        start_has_outgoing = any(s == START for s, _ in self.edges)
        if not start_has_outgoing and all_defined_nodes:
            errors.append("No entry point: START has no outgoing edges.")
        
        # Perform BFS from START
        if start_has_outgoing or not all_defined_nodes:
            q.append(START)
            reachable_nodes.add(START)

            while q:
                curr = q.popleft()
                # Follow standard edges
                for s_edge, t_edge in self.edges:
                    if s_edge == curr and t_edge not in reachable_nodes:
                        reachable_nodes.add(t_edge)
                
                        if t_edge != END: 
                            q.append(t_edge)
                
                # Follow conditional edges
                if curr in self.conditional_edges:
                    path_map = self.conditional_edges[curr].get("path_map", {})
                    for target_node_candidate in path_map.values():
                        if target_node_candidate not in reachable_nodes and (target_node_candidate == END or target_node_candidate in all_defined_nodes):
                            reachable_nodes.add(target_node_candidate)
                            if target_node_candidate != END: 
                                q.append(target_node_candidate)
            
            # Check for unreachable nodes
            for node_name in all_defined_nodes:
                if node_name not in reachable_nodes:
                    errors.append(f"Node '{node_name}' is unreachable from START.")
            
            # Check if END is reachable
            if all_defined_nodes and END not in reachable_nodes:
                errors.append("END node is unreachable from START.")

        # Check if all nodes can reach END
        if END in reachable_nodes:
            can_reach_end = {END}
            q_rev = collections.deque([END])
            
            while q_rev:
                curr_target = q_rev.popleft()
                
                # Find nodes that can reach current target via standard edges
                for s_edge, t_edge in self.edges:
                    if (t_edge == curr_target and s_edge in reachable_nodes 
                            and s_edge not in can_reach_end):
                        can_reach_end.add(s_edge)
                        if s_edge != START: 
                            q_rev.append(s_edge)
                
                # Find nodes that can reach current target via conditional edges
                for cond_s, edge_info in self.conditional_edges.items():
                    if (cond_s in reachable_nodes and cond_s not in can_reach_end):
                        path_map = edge_info.get("path_map", {})
                        if any(path_val == curr_target for path_val in path_map.values()):
                            can_reach_end.add(cond_s)
                            q_rev.append(cond_s)
            
            # Find dead-end nodes
            for node_name in all_defined_nodes:
                if node_name in reachable_nodes and node_name not in can_reach_end:
                    errors.append(f"Node '{node_name}' is reachable from START but cannot reach END (forms a dead-end path).")

        # Check for explicit dead ends (nodes with no outgoing edges)
        for node_name in all_defined_nodes:
            if node_name in reachable_nodes and node_name != END:
                has_outgoing = (any(s_edge == node_name for s_edge, _ in self.edges) or 
                               node_name in self.conditional_edges)
                               
                if not has_outgoing:
                    errors.append(f"Node '{node_name}' has no outgoing edges.")

        # Check for cycles in standard edges
        standard_edge_graph = collections.defaultdict(list)
        for s, t in self.edges:
            standard_edge_graph[s].append(t)

        visited = set()
        recursion_stack = set()
        
        def _detect_standard_edge_cycle(node):
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in standard_edge_graph.get(node, []):
                if neighbor == END:
                    continue
                
                if neighbor not in visited:
                    if _detect_standard_edge_cycle(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node)
            return False

        for node in all_defined_nodes:
            if node not in visited:
                if _detect_standard_edge_cycle(node):
                    errors.append("The standard edges form a cycle, resulting in an infinite loop without an exit condition.")
                    break

        return sorted(list(set(errors)))
    