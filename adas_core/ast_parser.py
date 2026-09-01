"""AST helpers for reading and editing top-level utility definitions."""

import ast


def extract_top_level_names(ast_module: ast.Module) -> set[str]:
    """Return names declared by supported top-level utility definitions."""
    names = set()
    for node in ast_module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def get_top_level_definitions(ast_module: ast.Module) -> set[tuple[str, str]]:
    """Return typed identifiers for deletable top-level utility definitions."""
    definitions = set()
    for node in ast_module.body:
        if isinstance(node, ast.FunctionDef):
            definitions.add(("function", node.name))
        elif isinstance(node, ast.ClassDef):
            definitions.add(("class", node.name))
        elif isinstance(node, ast.Assign):
            definitions.update(("assignment", target.id) for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions.add(("assignment", node.target.id))
    return definitions


class RemoveDefinitionsTransformer(ast.NodeTransformer):
    """Remove supported top-level definitions whose names are being replaced."""

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


class RemoveTypedUtilityDefinitionsTransformer(ast.NodeTransformer):
    """Remove top-level utility definitions matching an explicit kind and name."""

    def __init__(self, definitions_to_remove: set[tuple[str, str]]):
        self.definitions_to_remove = definitions_to_remove
        super().__init__()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        return None if ("function", node.name) in self.definitions_to_remove else node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST | None:
        return None if ("class", node.name) in self.definitions_to_remove else node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        remaining_targets = [
            target
            for target in node.targets
            if not (isinstance(target, ast.Name) and ("assignment", target.id) in self.definitions_to_remove)
        ]
        if not remaining_targets:
            return None
        node.targets = remaining_targets
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if isinstance(node.target, ast.Name) and ("assignment", node.target.id) in self.definitions_to_remove:
            return None
        return node
