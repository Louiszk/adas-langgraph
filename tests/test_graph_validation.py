"""
Comprehensive, specification-driven tests for graph structural guardrails and validation.
Tests 6 distinct graph topologies against invariant specifications.
"""

import textwrap

from langgraph.graph import END, START

from adas_core.virtual_agentic_system import VirtualAgenticSystem
from tests.conftest import add_conditional_edge_to_system, add_node_to_system


class TestGraphValidationTopologies:
    def test_graph_1_valid_pipeline_with_conditional_loop(self):
        """
        Graph 1: Multi-Agent Workflow with Conditional Feedback Loop.
        Topology:
            START -> researcher -> writer -> reviewer -> (conditional: pass=END, fail=writer)
        Specification Reasoning:
            Even though writer <-> reviewer can loop multiple times, the loop is governed by a conditional edge
            that can exit to END. Every node is reachable from START and can reach END.
            Expected Validation Errors: [] (0 errors).
        """
        system = VirtualAgenticSystem("Graph1_ConditionalLoop")
        add_node_to_system(system, "researcher", "def researcher(state: dict) -> dict:\n    return state", "Researcher")
        add_node_to_system(system, "writer", "def writer(state: dict) -> dict:\n    return state", "Writer")
        add_node_to_system(system, "reviewer", "def reviewer(state: dict) -> dict:\n    return state", "Reviewer")

        system.create_edge(START, "researcher")
        system.create_edge("researcher", "writer")
        system.create_edge("writer", "reviewer")

        condition_code = textwrap.dedent("""
            def review_condition(state: dict) -> str:
                if state.get("passed"):
                    return "pass"
                return "fail"
        """).strip()
        path_map = {"pass": END, "fail": "writer"}
        condition, parsed = system.get_function(condition_code, "conditional_edge")
        assert condition is not None, parsed
        assert condition({"passed": True}) in path_map
        assert condition({"passed": False}) in path_map
        add_conditional_edge_to_system(system, "reviewer", condition_code, path_map)

        errors = system.validate_graph()
        assert errors == [], f"Expected 0 errors for valid conditional feedback graph, got: {errors}"

    def test_graph_2_valid_fanout_fanin_parallel_graph(self):
        """
        Graph 2: Parallel Branching and Aggregation (Fan-Out / Fan-In).
        Topology:
            START -> dispatcher -(conditional)-> [branch_a, branch_b]
            branch_a -> aggregator -> END
            branch_b -> aggregator -> END
        Specification Reasoning:
            Every node in both branches converges into aggregator and reaches END.
            Expected Validation Errors: [] (0 errors).
        """
        system = VirtualAgenticSystem("Graph2_FanOutFanIn")
        add_node_to_system(system, "dispatcher", "def dispatcher(state: dict) -> dict:\n    return state", "Dispatcher")
        add_node_to_system(system, "branch_a", "def branch_a(state: dict) -> dict:\n    return state", "Branch A")
        add_node_to_system(system, "branch_b", "def branch_b(state: dict) -> dict:\n    return state", "Branch B")
        add_node_to_system(system, "aggregator", "def aggregator(state: dict) -> dict:\n    return state", "Aggregator")

        system.create_edge(START, "dispatcher")
        condition_code = textwrap.dedent("""
            def dispatch_condition(state: dict) -> str:
                return state.get("branch", "a")
        """).strip()
        path_map = {"a": "branch_a", "b": "branch_b"}
        condition, parsed = system.get_function(condition_code, "conditional_edge")
        assert condition is not None, parsed
        assert condition({"branch": "a"}) in path_map
        assert condition({"branch": "b"}) in path_map
        add_conditional_edge_to_system(system, "dispatcher", condition_code, path_map)

        system.create_edge("branch_a", "aggregator")
        system.create_edge("branch_b", "aggregator")
        system.create_edge("aggregator", END)

        errors = system.validate_graph()
        assert errors == [], f"Expected 0 errors for valid fanout/fanin graph, got: {errors}"

    def test_graph_3_invalid_unreachable_orphan_node(self):
        """
        Graph 3: Graph containing a disconnected orphan node.
        Topology:
            START -> primary_node -> END
            orphan_node -> END (no edge entering orphan_node from START or any reachable node)
        Specification Reasoning:
            'orphan_node' can never be visited during graph execution from START.
            Expected Validation Errors: Must contain unreachable node error for 'orphan_node'.
        """
        system = VirtualAgenticSystem("Graph3_OrphanNode")
        add_node_to_system(
            system, "primary_node", "def primary_node(state: dict) -> dict:\n    return state", "Primary"
        )
        add_node_to_system(system, "orphan_node", "def orphan_node(state: dict) -> dict:\n    return state", "Orphan")

        system.create_edge(START, "primary_node")
        system.create_edge("primary_node", END)
        system.create_edge("orphan_node", END)

        errors = system.validate_graph()
        assert any("orphan_node" in err and "unreachable" in err for err in errors), (
            f"Expected unreachable error for orphan_node, got: {errors}"
        )

    def test_graph_4_invalid_dead_end_black_hole_node(self):
        """
        Graph 4: Graph containing a dead-end sink node that cannot reach END.
        Topology:
            START -> decision_node -> main_worker -> END
            decision_node -> black_hole_node (no outgoing edge from black_hole_node)
        Specification Reasoning:
            Any execution entering 'black_hole_node' halts without reaching END or returning a terminal graph state.
            Expected Validation Errors: Must detect that 'black_hole_node' cannot reach END and has no outgoing edges.
        """
        system = VirtualAgenticSystem("Graph4_DeadEnd")
        add_node_to_system(
            system, "decision_node", "def decision_node(state: dict) -> dict:\n    return state", "Decision"
        )
        add_node_to_system(
            system, "main_worker", "def main_worker(state: dict) -> dict:\n    return state", "Main Worker"
        )
        add_node_to_system(
            system, "black_hole_node", "def black_hole_node(state: dict) -> dict:\n    return state", "Black Hole"
        )

        system.create_edge(START, "decision_node")
        system.create_edge("decision_node", "main_worker")
        system.create_edge("decision_node", "black_hole_node")
        system.create_edge("main_worker", END)

        errors = system.validate_graph()
        assert any(
            "black_hole_node" in err and ("cannot reach END" in err or "no outgoing edges" in err) for err in errors
        ), f"Expected dead-end errors for black_hole_node, got: {errors}"

    def test_graph_5_invalid_deterministic_infinite_loop(self):
        """
        Graph 5: Unconditional cycle on standard edges.
        Topology:
            START -> step_1 -> step_2 -> step_1
        Specification Reasoning:
            Without a conditional edge or transition to END, execution will loop infinitely between step_1 and step_2.
            Expected Validation Errors: Must detect cycle in standard edges AND unreachable END node.
        """
        system = VirtualAgenticSystem("Graph5_InfiniteLoop")
        add_node_to_system(system, "step_1", "def step_1(state: dict) -> dict:\n    return state", "Step 1")
        add_node_to_system(system, "step_2", "def step_2(state: dict) -> dict:\n    return state", "Step 2")

        system.create_edge(START, "step_1")
        system.create_edge("step_1", "step_2")
        system.create_edge("step_2", "step_1")

        errors = system.validate_graph()
        assert any("cycle" in err for err in errors), f"Expected standard edge cycle error, got: {errors}"
        assert any("END" in err and "unreachable" in err for err in errors), (
            f"Expected END unreachable error, got: {errors}"
        )

    def test_graph_6_invalid_edge_endpoints_and_undefined_nodes(self):
        """
        Graph 6: Edges involving illegal endpoints (START as target, END as source, undefined ghost node).
        Topology:
            END -> START (illegal)
            START -> ghost_node (undefined target)
            ghost_node -> END (undefined source)
        Specification Reasoning:
            START cannot be an edge target. END cannot be an edge source.
            Any node referenced in an edge must exist in system.nodes.
            Expected Validation Errors: Must flag illegal START/END usage and undefined 'ghost_node'.
        """
        system = VirtualAgenticSystem("Graph6_IllegalEndpoints")

        system.edges.append((END, START))
        system.edges.append((START, "ghost_node"))
        system.edges.append(("ghost_node", END))

        errors = system.validate_graph()
        assert any("START" in err and "invalid" in err.lower() for err in errors)
        assert any("END" in err and "invalid" in err.lower() for err in errors)
        assert any("ghost_node" in err or "not a defined node" in err for err in errors)


class TestConditionalEdgePathMapSpecification:
    def test_conditional_edges_require_an_explicit_path_map(self):
        """Conditional edges must not infer destinations from condition-function source code."""
        system = VirtualAgenticSystem("InferenceTest")
        add_node_to_system(system, "decision_source", "def decision_source(state: dict) -> dict:\n    return state")
        condition_code = textwrap.dedent("""
            def choose_next(state: dict) -> str:
                if state.get("ok"):
                    return END
                return "worker_node"
        """)
        condition, parsed = system.get_function(condition_code, "conditional_edge")
        assert condition is not None, parsed

        try:
            system.create_conditional_edge("decision_source", condition, parsed)
        except ValueError as error:
            assert "explicit path_map" in str(error)
        else:
            raise AssertionError("Conditional edge creation unexpectedly inferred a path map.")

    def test_explicit_path_map_normalizes_end_marker(self):
        """The textual END marker is materialized as the LangGraph END constant."""
        system = VirtualAgenticSystem("ExplicitPathMapTest")
        add_node_to_system(system, "decision_source", "def decision_source(state: dict) -> dict:\n    return state")
        condition_code = "def choose_next(state: dict) -> str:\n    return 'complete'"

        assert add_conditional_edge_to_system(system, "decision_source", condition_code, {"complete": "END"})
        assert system.conditional_edges["decision_source"]["path_map"] == {"complete": END}

    def test_validate_graph_rejects_undefined_path_map_destination(self):
        """Every explicitly declared route must lead to a node that exists or to END."""
        system = VirtualAgenticSystem("InvalidPathMapTarget")
        add_node_to_system(system, "decision_source", "def decision_source(state: dict) -> dict:\n    return state")
        system.create_edge(START, "decision_source")
        condition_code = "def choose_next(state: dict) -> str:\n    return 'missing'"
        add_conditional_edge_to_system(system, "decision_source", condition_code, {"missing": "not_a_node"})

        errors = system.validate_graph()
        assert any("undefined node 'not_a_node'" in error for error in errors)
