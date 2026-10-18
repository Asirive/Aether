from __future__ import annotations

import pytest

from aether.utils.device import select_device


@pytest.fixture(scope="session", autouse=True)
def _select_cuda_device():
    select_device()
