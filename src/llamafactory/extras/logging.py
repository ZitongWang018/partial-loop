# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from logging import Logger
from typing import Optional


class LoggerHandler(logging.Handler):
    r"""
    A handler for logging to the Web UI.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        super().__init__()
        self.logs: Optional[str] = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self.log_path = os.path.join(output_dir, "llamaboard.log")
        else:
            self.log_path = None

    def emit(self, record: "logging.LogRecord") -> None:
        if record.name == "llamafactory":
            log_message = f"{record.levelname}: {record.getMessage()}"
            if self.log_path is not None:
                with open(self.log_path, "a") as f:
                    f.write(log_message + "\n")

            if self.logs is None:
                self.logs = log_message
            else:
                self.logs += "\n" + log_message


_loggers: dict = {}
_handler = LoggerHandler()


def _get_logger(level: int = logging.INFO) -> "Logger":
    if _loggers.get("llamafactory"):
        return _loggers["llamafactory"]

    _logger = logging.getLogger("llamafactory")
    _logger.setLevel(level)
    _logger.addHandler(_handler)

    _loggers["llamafactory"] = _logger
    return _logger


def add_handler(handler: "logging.Handler") -> None:
    _logger = _get_logger()
    _logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> "Logger":
    if name is not None:
        return logging.getLogger(name)

    return _get_logger()


def reset_logging() -> None:
    r"""
    Resets logging handlers.
    """
    for key in list(_loggers.keys()):
        _logger = _loggers[key]
        for handler in _logger.handlers[:]:
            _logger.removeHandler(handler)
            handler.close()

        _loggers.pop(key)

    _handler.logs = None


def set_logger(level: int = logging.INFO) -> None:
    _get_logger(level)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
