"""Central logging setup for the application and core library."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

from ml_trader.config_loader import get_config


DEFAULT_LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/app.log",
    "max_bytes": 10 * 1024 * 1024,
    "backup_count": 5,
    "console": True,
}

_MANAGED_HANDLER_ATTR = "_peak_signal_managed_handler"
_CONFIGURED_ATTR = "_peak_signal_logging_configured"


def _coerce_level(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return getattr(logging, value.upper(), logging.INFO)
    return logging.INFO


def _merged_config(logging_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    configured = get_config("logging", {}) if logging_config is None else dict(logging_config)
    return {**DEFAULT_LOGGING_CONFIG, **configured}


def setup_logging(logging_config: Mapping[str, Any] | None = None, *, force: bool = False) -> logging.Logger:
    """Configure root logging once and return the root logger.

    Streamlit reruns the app module often, so setup is idempotent by default.
    Tests and command-line tools can pass ``force=True`` to rebuild handlers.
    """
    config = _merged_config(logging_config)
    level = _coerce_level(config.get("level"))
    root_logger = logging.getLogger()

    if getattr(root_logger, _CONFIGURED_ATTR, False) and not force:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            if getattr(handler, _MANAGED_HANDLER_ATTR, False):
                handler.setLevel(level)
        return root_logger

    for handler in list(root_logger.handlers):
        if force or getattr(handler, _MANAGED_HANDLER_ATTR, False):
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.setLevel(level)
    formatter = logging.Formatter(str(config.get("format")))

    log_file = Path(str(config.get("file")))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=int(config.get("max_bytes", DEFAULT_LOGGING_CONFIG["max_bytes"])),
        backupCount=int(config.get("backup_count", DEFAULT_LOGGING_CONFIG["backup_count"])),
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_HANDLER_ATTR, True)
    root_logger.addHandler(file_handler)

    if bool(config.get("console", True)):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _MANAGED_HANDLER_ATTR, True)
        root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    setattr(root_logger, _CONFIGURED_ATTR, True)
    root_logger.debug("Logging configured", extra={"log_file": str(log_file)})
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a project logger."""
    return logging.getLogger(name)
