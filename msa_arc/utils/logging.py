"""Logging setup shared by every entry point."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Attach a stream handler, and a file handler when ``log_file`` is given."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)


__all__ = ["configure_logging"]
