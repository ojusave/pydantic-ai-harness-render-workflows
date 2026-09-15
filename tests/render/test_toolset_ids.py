"""Render task identity for the toolsets a capability contributes."""

from __future__ import annotations

import inspect

import pytest
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from render.workflows import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext


class Notes(AbstractCapability[None]):
    """Contributes a toolset it builds itself, which is what leaves the toolset unnamed.

    A capability like this is the reason a toolset reaches Render with no id: the user who
    writes `capabilities=[Notes()]` never touches the `FunctionToolset` and cannot name it.
    Whether this one names its own toolset is the variable each test below sets.
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
    call_tools: list[str] | None = None,
) -> tuple[Agent[None, str], RenderWorkflows[None]]:
    workflows = Workflows()
    render_workflows = RenderWorkflows[None](workflows, deps_type=type(None))
    agent = Agent[None, str](
        TestModel() if call_tools is None else TestModel(call_tools=call_tools),
        name='support',
        deps_type=type(None),
        toolsets=toolsets,
        capabilities=[capability, render_workflows],
    )
    return agent, render_workflows


async def recorded_task_names(
    agent: Agent[None, str],
    render_workflows: RenderWorkflows[None],
    prompt: str,
) -> list[str]:
    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending_result = run_agent.func(context, prompt)
    assert inspect.isawaitable(pending_result)
    await pending_result
    return context.task_names


@pytest.mark.anyio
async def test_a_toolset_that_brought_its_own_id_registers_its_tasks_under_it() -> None:
    agent, render_workflows = build_agent(Notes(toolset_id='handwritten'), call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    # Task names are persisted journal data, so the name an `id` produces is pinned here:
    # a rename strands in-flight workflows recorded against the old one.
    assert 'support__function_toolset__handwritten.call_tool' in names


def test_an_unnamed_capability_toolset_is_refused_by_name() -> None:
    """Render cannot name it, and will not invent a name for persisted task identity.

    The capability's own id is stable and tempting, but it is not the toolset's id, and
    writing it into the toolset means writing a private field. Saying so names the
    capability the user has to fix, which the framework's generic message cannot do.
    """
    with pytest.raises(UserError, match=r'Notes \(FunctionToolset\)'):
        build_agent(Notes())


def test_the_refusal_does_not_depend_on_the_capability_having_an_id() -> None:
    # A capability with no id is no more nameable, and gets the same answer.
    with pytest.raises(UserError, match='no stable name to register under'):
        build_agent(Notes(id=None))


@pytest.mark.anyio
async def test_a_toolset_the_user_owns_is_unaffected_by_what_a_capability_contributes() -> None:
    async def recall(topic: str) -> str:
        return topic

    agent, render_workflows = build_agent(
        Notes(toolset_id='notes'),
        toolsets=[FunctionToolset[None]([recall], id='recall')],
        call_tools=['note', 'recall'],
    )

    names = await recorded_task_names(agent, render_workflows, 'remember this')

    # Both toolsets are registered, and neither one's tasks answer for the other.
    assert 'support__function_toolset__notes.call_tool' in names
    assert 'support__function_toolset__recall.call_tool' in names


def test_two_toolsets_under_one_id_are_still_the_framework_id_collision() -> None:
    async def recall(topic: str) -> str:
        return topic

    # Nothing here derives or disambiguates ids any more, so a genuine clash reaches
    # Pydantic AI's own uniqueness check rather than being renamed out of the way.
    with pytest.raises(UserError, match='Two toolsets have the same `id`'):
        build_agent(Notes(toolset_id='notes'), toolsets=[FunctionToolset[None]([recall], id='notes')])
