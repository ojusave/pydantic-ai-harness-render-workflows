"""Render task identity for the toolsets a capability contributes."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from render.workflows import Workflows

from pydantic_ai_harness import RenderWorkflows


class Notes(AbstractCapability[None]):
    """Contributes a toolset it builds itself, which is what leaves the toolset unnamed.

    A capability like this is the reason Render derives toolset ids: the user who writes
    `capabilities=[Notes()]` never touches the `FunctionToolset` and cannot name it.
    """

    def __init__(self, *, id: str | None = 'notes', toolset_id: str | None = None) -> None:
        self.id = id
        self._toolset_id = toolset_id

    def get_toolset(self) -> AbstractToolset[None]:
        async def note(text: str) -> str:
            return text

        return FunctionToolset[None]([note], id=self._toolset_id)


def build_agent(
    capability: AbstractCapability[None],
    *,
    toolsets: list[AbstractToolset[None]] | None = None,
) -> Workflows:
    workflows = Workflows()
    Agent[None, str](
        TestModel(),
        name='support',
        deps_type=type(None),
        toolsets=toolsets,
        capabilities=[capability, RenderWorkflows[None](workflows, deps_type=type(None))],
    )
    return workflows


def test_a_capability_toolset_registers_tasks_under_the_capability_id() -> None:
    workflows = build_agent(Notes())

    assert 'support__function_toolset__notes.call_tool' in workflows._registry.get_task_names()


def test_a_toolset_that_brought_its_own_id_keeps_it() -> None:
    workflows = build_agent(Notes(toolset_id='handwritten'))

    names = workflows._registry.get_task_names()

    assert 'support__function_toolset__handwritten.call_tool' in names
    assert 'support__function_toolset__notes.call_tool' not in names


def test_an_id_another_toolset_already_uses_falls_back_to_a_numbered_variant() -> None:
    async def recall(topic: str) -> str:
        return topic

    workflows = build_agent(Notes(), toolsets=[FunctionToolset[None]([recall], id='notes')])

    names = workflows._registry.get_task_names()

    # Both toolsets are registered, and neither one's tasks answer for the other.
    assert 'support__function_toolset__notes.call_tool' in names
    assert 'support__function_toolset__notes.2.call_tool' in names


def test_an_unnamed_capability_keeps_the_unnamed_toolset_error() -> None:
    # Nothing stable to derive from, so the agent still tells the user to set an id.
    with pytest.raises(UserError, match='need to have a unique `id`'):
        build_agent(Notes(id=None))
