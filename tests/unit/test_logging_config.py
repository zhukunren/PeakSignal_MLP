import logging

from ml_trader.logging_config import setup_logging


def test_setup_logging_writes_file_and_is_idempotent(tmp_path):
    log_file = tmp_path / "app.log"
    config = {
        "level": "INFO",
        "format": "%(levelname)s:%(name)s:%(message)s",
        "file": str(log_file),
        "max_bytes": 1024,
        "backup_count": 1,
        "console": False,
    }

    root_logger = setup_logging(config, force=True)
    managed_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_peak_signal_managed_handler", False)
    ]
    assert len(managed_handlers) == 1

    logging.getLogger("tests.logging").info("hello logging")
    for handler in managed_handlers:
        handler.flush()

    assert "INFO:tests.logging:hello logging" in log_file.read_text(encoding="utf-8")

    setup_logging(config)
    managed_handlers_after_second_setup = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_peak_signal_managed_handler", False)
    ]
    assert len(managed_handlers_after_second_setup) == 1
