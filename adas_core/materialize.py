import re
import os
from langgraph.graph import START, END
from adas_core.virtual_agentic_system import VirtualAgenticSystem

def get_function_name(func_source: str) -> str:
    match = re.search(r'def\s+([^\s(]+)', func_source)
    if not match:
        raise ValueError("Could not find function definition in source.")
    
    return match.group(1)

def materialize_system(system: VirtualAgenticSystem, output_dir: str ="generated_systems") -> str:
    """Generate Python code representation of the system."""
    nodes_count = len(system.nodes)
    tool_count = len(system.tools)
    escaped_name = system.escaped_name
    
    code_lines = [
        f"# Total nodes: {nodes_count}",
        f"# Total tools: {tool_count}",
        ""
    ]

    code_lines.append("# Already installed packages")
    for sp in system.packages_info:
        code_lines.append(f"# {sp}")
    code_lines.append("")

    if system.imports:
        for imp in system.imports:
            code_lines.append(imp)

    if system.utility_code:
        code_lines.extend([
            "",
            "# ===== Utilities =====",
            system.utility_code,
            ""
        ])
        
        
    code_lines.extend([
        "",
        "# ===== Agentic System =====",
        "# Define state attributes for the system",
        "class AgentState(TypedDict):",
    ])
    
    for attr_name, attr_type in system.state_attributes.items():
        code_lines.append(f"    {attr_name}: {attr_type}")
    
    code_lines.extend([
        "",
        "# Initialize the agentic system graph with the specified state",
        "agentic_system_graph = StateGraph(AgentState)",
        "",
    ])
    
    # Tool definitions
    if system.tools:
        code_lines.append("# ===== Tool Definitions =====")
        code_lines.append("tools = {}")
        
        for tool_name, tool_info in system.tools.items():
            description = tool_info["description"]
            func_source = tool_info["source_code"]
            original_name = get_function_name(func_source)
            
            code_lines.extend([
                f"# Tool: {tool_name}",
                f"# Description: {description}",
                func_source,
                "",
                f"tools[\"{tool_name}\"] = tool(runnable={original_name}, name_or_callable=\"{tool_name}\")",
                ""
            ])
    
    # Node definitions
    if system.nodes:
        code_lines.append("# ===== Node Definitions =====")
    
        for node_name, node_info in system.nodes.items():
            description = node_info["description"]
            func_source = node_info["source_code"]
            original_name = get_function_name(func_source)
            
            code_lines.extend([
                f"# Node: {node_name}",
                f"# Description: {description}",
                func_source,
                "",
                f"agentic_system_graph.add_node(\"{node_name}\", {original_name})",
                ""
            ])
    
    # Standard edges
    if system.edges:
        code_lines.append("# ===== Standard Edges =====")
        
        for source, target in system.edges:
            source_name = "START" if source in ["START", START, "__start__"] else f'"{source}"'
            target_name = "END" if target in ["END", END, "__end__"] else f'"{target}"'
            
            code_lines.extend([
                f"agentic_system_graph.add_edge({source_name}, {target_name})",
                ""
            ])
    
    # Conditional edges
    if system.conditional_edges:
        code_lines.append("# ===== Conditional Edges =====")
        
        for source, edge_info in system.conditional_edges.items():
            source_name = "START" if source in ["START", START, "__start__"] else f'"{source}"'
            func_source = edge_info["condition_code"]
            original_name = get_function_name(func_source)
            
            code_lines.extend([
                f"# Conditional Router from: {source_name}",
                func_source,
                "",
                f"agentic_system_graph.add_conditional_edges({source_name}, {original_name})",
                ""
            ])

    # Entry/Exit Configuration
    code_lines.extend([
        "# ===== Compilation =====",
        "workflow = agentic_system_graph.compile()",
        ""
    ])

    main_code = "\n".join(code_lines)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, escaped_name + ".py")
        with open(filename, "w", encoding='utf-8') as f:
            f.write(main_code)
    
    return main_code