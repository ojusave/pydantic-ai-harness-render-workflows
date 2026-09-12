from __future__ import annotations

from typing import Any

from render.workflows import TaskDefinition

from pydantic_ai_harness.render._context import activate_task_context, current_task_context


class StubTaskContext:
    async def run(self, task: TaskDefinition[..., Any], *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def test_context_is_owner_scoped_and_nestable() -> None:
    first_owner = object()
    second_owner = object()
    first_context = StubTaskContext()
    nested_context = StubTaskContext()
    second_context = StubTaskContext()

    assert current_task_context(first_owner) is None
    with activate_task_context(first_owner, first_context):
        assert current_task_context(first_owner) is first_context
        assert current_task_context(second_owner) is None
        with activate_task_context(second_owner, second_context):
            assert current_task_context(first_owner) is first_context
            assert current_task_context(second_owner) is second_context
            with activate_task_context(first_owner, nested_context):
                assert current_task_context(first_owner) is nested_context
        assert current_task_context(first_owner) is first_context
        assert current_task_context(second_owner) is None
    assert current_task_context(first_owner) is None


def test_context_resets_after_error() -> None:
    owner = object()
    context = StubTaskContext()

    try:
        with activate_task_context(owner, context):
            raise RuntimeError('boom')
    except RuntimeError:
        pass

    assert current_task_context(owner) is None
