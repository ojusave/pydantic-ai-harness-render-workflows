"""Render task identity for the toolsets a capability contributes."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset
from pydantic_ai.toolsets.external import ExternalToolset
from render.workflows import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext
from .test_render_workflows import RegistrationRecordingWorkflows

# Built in a fresh interpreter twice, so the names below are the ones two worker processes
# agree on rather than the ones one process happens to produce. Task names are persisted
# workflow identity: a name that depended on anything process-local would strand a
# recovering task run against a name nothing in the new process registers.
_TASK_NAME_PROBE = """
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from render.workflows import Workflows

from pydantic_ai_harness import RenderWorkflows, ToolOutputLimits
from pydantic_ai_harness.subagents import SubAgent, SubAgents

names = []


class RecordingWorkflows(Workflows):
    def task(self, func=None, *, name=None, retry=None, timeout_seconds=None, plan=None):
        decorator = super().task(name=name, retry=retry, timeout_seconds=timeout_seconds, plan=plan)

        def record(target):
            definition = decorator(target)
            names.append(definition.name)
            return definition

        return record if func is None else record(func)


worker = Agent(TestModel(), name='worker', description='Does the work')
Agent(
    TestModel(),
    name='support',
    deps_type=type(None),
    capabilities=[
        SubAgents(agents=[SubAgent(worker)], agent_folders=None),
        ToolOutputLimits(),
        RenderWorkflows(RecordingWorkflows(), deps_type=type(None)),
    ],
)
print('\\n'.join(sorted({name for name in names if '__function_toolset__' in name})))
"""


def test_importing_render_workflows_does_not_add_an_optional_mcp_import() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import sys; '
                'from pydantic_ai.agent import AbstractAgent; '
                'before = "pydantic_ai.mcp" in sys.modules; '
                'from pydantic_ai_harness import RenderWorkflows; '
                'assert AbstractAgent and RenderWorkflows; '
                'assert ("pydantic_ai.mcp" in sys.modules) == before'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


class Notes(AbstractCapability[None]):
    """Contributes a toolset it builds itself, which is what leaves the toolset unnamed.

    A capability like this is the reason a toolset reaches Render with no id: the user who
    writes `capabilities=[Notes()]` never touches the `FunctionToolset` and cannot name it.
    What each test below varies is whether this one names its own toolset, and whether it
    has an id of its own for Render to name the toolset from.
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
    app: Workflows | None = None,
    toolsets: list[AbstractToolset[None]] | None = None,
    call_tools: list[str] | None = None,
) -> tuple[Agent[None, str], RenderWorkflows[None]]:
    render_workflows = RenderWorkflows[None](app if app is not None else Workflows(), deps_type=type(None))
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


def probe_task_names() -> list[str]:
    """Return the function-toolset task names a fresh interpreter registers."""
    completed = subprocess.run(
        [sys.executable, '-c', _TASK_NAME_PROBE],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.split()


@pytest.mark.anyio
async def test_a_toolset_that_brought_its_own_id_registers_its_tasks_under_it() -> None:
    agent, render_workflows = build_agent(Notes(toolset_id='handwritten'), call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    # Task names are persisted journal data, so the name an `id` produces is pinned here:
    # a rename strands in-flight workflows recorded against the old one. A toolset that
    # was named keeps its own name; the capability's id never overrides it.
    assert 'support__function_toolset__handwritten.call_tool' in names


@pytest.mark.anyio
async def test_an_unnamed_capability_toolset_is_named_from_the_capability() -> None:
    """The capability's `id` is the one name available that is stable across processes.

    It is unique per agent because Pydantic AI registers capabilities by it, and the worker
    builds the same agent from the same module, so it names the same toolset there.
    """
    agent, render_workflows = build_agent(Notes(id='web_search'), call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    assert 'support__function_toolset__web_search.call_tool' in names


@pytest.mark.anyio
async def test_a_derived_name_gives_way_to_the_toolset_already_holding_it() -> None:
    """A derived name never displaces one a user chose, and never merges two toolsets.

    The suffix is decided in toolset traversal order, so the pair below gets the same two
    names in any process that builds this agent.
    """

    async def recall(topic: str) -> str:
        return topic

    agent, render_workflows = build_agent(
        Notes(id='notes'),
        toolsets=[FunctionToolset[None]([recall], id='notes')],
        call_tools=['note', 'recall'],
    )

    names = await recorded_task_names(agent, render_workflows, 'remember this')

    assert 'support__function_toolset__notes.call_tool' in names
    assert 'support__function_toolset__notes.2.call_tool' in names


class ExternallyAnsweredNotes(Notes):
    """Contributes one toolset Render registers tasks for and one it does not.

    An external toolset's results are produced outside the run, so Render never registers
    a task for it and it needs no name. Only the leaf kinds that do get tasks are named.
    """

    def get_toolset(self) -> AbstractToolset[None]:
        async def note(text: str) -> str:
            return text

        return CombinedToolset(
            [
                FunctionToolset[None]([note]),
                ExternalToolset[None]([ToolDefinition(name='answered_elsewhere')]),
            ]
        )


@pytest.mark.anyio
async def test_a_contributed_toolset_render_registers_nothing_for_needs_no_name() -> None:
    agent, render_workflows = build_agent(ExternallyAnsweredNotes(id='notes'), call_tools=['note'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    assert 'support__function_toolset__notes.call_tool' in names
    assert not [name for name in names if 'notes.2' in name]


class TwoToolsetNotes(Notes):
    """One capability, two toolsets, and one `id` between them to derive from."""

    def get_toolset(self) -> AbstractToolset[None]:
        async def note(text: str) -> str:
            return text

        async def recall(topic: str) -> str:
            return topic

        return CombinedToolset([FunctionToolset[None]([note]), FunctionToolset[None]([recall])])


@pytest.mark.anyio
async def test_two_unnamed_leaves_under_one_capability_are_numbered_in_traversal_order() -> None:
    """Both leaves need their own task names, and only one of them can be the plain `id`.

    Traversal order is a property of the toolset tree the agent was built from, so the two
    leaves keep the same two names in the worker process that rebuilds it.
    """
    agent, render_workflows = build_agent(TwoToolsetNotes(id='notes'), call_tools=['note', 'recall'])

    names = await recorded_task_names(agent, render_workflows, 'take a note')

    assert 'support__function_toolset__notes.call_tool' in names
    assert 'support__function_toolset__notes.2.call_tool' in names


@pytest.mark.anyio
async def test_the_derived_suffix_keeps_counting_past_the_first_taken_name() -> None:
    """Every name already in use is skipped, in order, rather than the count restarting."""

    async def recall(topic: str) -> str:
        return topic

    agent, render_workflows = build_agent(
        Notes(id='notes'),
        toolsets=[
            FunctionToolset[None]([recall], id='notes'),
            FunctionToolset[None]([], id='notes.2'),
        ],
        call_tools=['note', 'recall'],
    )

    names = await recorded_task_names(agent, render_workflows, 'remember this')

    assert 'support__function_toolset__notes.3.call_tool' in names


def test_a_toolset_the_user_attached_without_an_id_is_refused_on_its_own_terms() -> None:
    """Nothing derives a name for a toolset the user holds: they can pass one.

    The capability-derived name exists because the toolset it names is unreachable. This
    one is reachable, so the refusal says to name it rather than inventing a name for it.
    """

    async def recall(topic: str) -> str:
        return topic

    app = RegistrationRecordingWorkflows()

    with pytest.raises(UserError, match='needs a unique `id`'):
        build_agent(Notes(toolset_id='notes'), app=app, toolsets=[FunctionToolset[None]([recall])])

    assert app.registered_task_names == []


class UnreadableIdNotes(Notes):
    """A capability whose toolset reports an `id` that is not the one it was given.

    Stands in for a Pydantic AI release that stops reading a leaf toolset's `id` from
    `_id`. The integration performs one private write, so it has to refuse rather than
    register task names the toolset does not answer to.
    """

    def get_toolset(self) -> AbstractToolset[None]:
        class DetachedIdToolset(FunctionToolset[None]):
            @property
            def id(self) -> str | None:
                return None

        return DetachedIdToolset([])


def test_a_toolset_that_does_not_report_the_assignment_is_refused_before_registration() -> None:
    app = RegistrationRecordingWorkflows()

    with pytest.raises(UserError, match='still reports the `id`'):
        build_agent(UnreadableIdNotes(), app=app)

    assert app.registered_task_names == []


def test_an_unnamed_toolset_under_an_unnamed_capability_is_refused_before_registration() -> None:
    """With no capability `id` there is nothing stable to derive, so the refusal stands.

    It names the capability the user has to fix, and it happens before the app holds any
    task: Render's public API cannot unregister one, so a rejected agent would otherwise
    leave the app carrying a partial task set for the life of the process.
    """
    app = RegistrationRecordingWorkflows()

    with pytest.raises(UserError, match=r'Notes \(FunctionToolset\)'):
        build_agent(Notes(id=None), app=app)

    assert app.registered_task_names == []


def test_two_toolsets_that_already_share_an_id_are_still_a_collision() -> None:
    async def recall(topic: str) -> str:
        return topic

    # Both of these were named by whoever owns them, so there is nothing to derive and
    # nothing to disambiguate: a genuine clash is Pydantic AI's answer, given earlier.
    app = RegistrationRecordingWorkflows()

    with pytest.raises(UserError, match='Two toolsets have the same `id`'):
        build_agent(Notes(toolset_id='notes'), app=app, toolsets=[FunctionToolset[None]([recall], id='notes')])

    assert app.registered_task_names == []


def test_two_independent_agents_register_the_same_derived_task_names() -> None:
    """Two agents built the same way are the same workflow identity, twice.

    Nothing about the derived name comes from the instance, so a second agent in the same
    process, on its own `Workflows` app, registers the names the first one did.
    """
    first, _ = build_agent(Notes(id='web_search'), app=(first_app := RegistrationRecordingWorkflows()))
    second, _ = build_agent(Notes(id='web_search'), app=(second_app := RegistrationRecordingWorkflows()))

    assert first is not second
    assert 'support__function_toolset__web_search.call_tool' in first_app.registered_task_names
    assert first_app.registered_task_names == second_app.registered_task_names


def test_the_shared_capabilities_register_the_same_names_in_a_fresh_process() -> None:
    """`SubAgents` and `ToolOutputLimits` are the two capabilities that name no toolset.

    Both are built with no Render-specific argument, and the names their tasks register
    under come from their own default ids. Two interpreters are compared rather than one
    interpreter twice, because a worker recovering a task run is a different process.
    """
    names = probe_task_names()

    assert names == probe_task_names()
    assert 'support__function_toolset__sub_agents.call_tool' in names
    assert 'support__function_toolset__tool_output_limits.call_tool' in names
