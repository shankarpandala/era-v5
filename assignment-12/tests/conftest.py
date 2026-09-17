"""Load Assignment 12 by its exact path, independent of other assignments."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def sim():
    return load_module("assignment12_zero_simulator", HERE / "zero_simulator.py")


@pytest.fixture(scope="session")
def auditor():
    return load_module("assignment12_audit", HERE / "audit.py")
