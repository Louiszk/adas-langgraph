"""Unit tests for AST helpers used to edit utility definitions."""

import ast

from adas_core.ast_parser import (
    RemoveDefinitionsTransformer,
    RemoveTypedUtilityDefinitionsTransformer,
    extract_top_level_names,
    get_top_level_definitions,
)


def test_extracts_supported_top_level_utility_definitions():
    module = ast.parse("""
VALUE = 1
MAX_RETRIES: int = 3

class RetryPolicy:
    pass

def normalize(value: str) -> str:
    return value
""")

    assert extract_top_level_names(module) == {"VALUE", "MAX_RETRIES", "RetryPolicy", "normalize"}
    assert get_top_level_definitions(module) == {
        ("assignment", "VALUE"),
        ("assignment", "MAX_RETRIES"),
        ("class", "RetryPolicy"),
        ("function", "normalize"),
    }


def test_transformers_preserve_typed_deletion_and_name_replacement_behavior():
    module = ast.parse("""
setting = 1
setting: int = 2

def setting() -> int:
    return 3
""")

    typed_result = RemoveTypedUtilityDefinitionsTransformer({("assignment", "setting")}).visit(module)
    assert isinstance(typed_result, ast.Module)
    assert ast.unparse(typed_result) == "def setting() -> int:\n    return 3"

    replacement_result = RemoveDefinitionsTransformer({"setting"}).visit(ast.parse("setting = 1\n"))
    assert isinstance(replacement_result, ast.Module)
    assert replacement_result.body == []
