"""
pyDAC - File-backed run logging utilities.

This module provides lightweight context managers that tee stdout and stderr
to per-step or arbitrary log files while optionally mirroring the output to the
terminal.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import contextlib
import sys
import os

class DualLogger:
    def __init__(self, file_path, also_stdout=False, mode="w"):
        self.log = open(file_path, mode, encoding="utf-8")
        self.terminal = sys.stdout if also_stdout else None

    def write(self, message):
        if self.terminal:
            self.terminal.write(message)
            self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        if self.terminal:
            self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

@contextlib.contextmanager
def log_to(step_name, base_folder="simulation_results", also_stdout=False, append=False):
    log_dir = os.path.join(base_folder, step_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")

    logger = DualLogger(log_path, also_stdout=also_stdout, mode="a" if append else "w")
    try:
        with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
            yield
    finally:
        logger.close()


@contextlib.contextmanager
def log_file(log_path, also_stdout=False, append=False):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = DualLogger(log_path, also_stdout=also_stdout, mode="a" if append else "w")
    try:
        with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
            yield
    finally:
        logger.close()




@contextlib.contextmanager
def terminal_only(also_stdout=False):
    """Like ``log_to`` but without a log file: output reaches the terminal only if asked."""
    if also_stdout:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
