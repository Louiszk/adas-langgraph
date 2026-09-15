from config.logging import get_logger, setup_logging


def test_logging_implementation_is_exposed_from_config_package():
    assert get_logger("example").name == "adas.example"
    assert callable(setup_logging)
