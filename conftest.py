from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

import pytest


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "asyncio: mark test to run using asyncio event loop")


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    marker = pyfuncitem.get_closest_marker("asyncio")
    if marker is None:
        return None

    loop = pyfuncitem._request.getfixturevalue("event_loop")
    func: Callable[..., Any] = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None

    signature = inspect.signature(func)
    kwargs = {
        name: pyfuncitem._request.getfixturevalue(name)
        for name in signature.parameters.keys()
    }
    loop.run_until_complete(func(**kwargs))
    return True
