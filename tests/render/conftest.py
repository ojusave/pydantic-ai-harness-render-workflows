from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Pydantic AI's agent runtime currently requires an asyncio event loop."""
    return 'asyncio'
