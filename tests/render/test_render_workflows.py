"""Public-path integration tests for Render Workflows durability."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterable, Callable
from typing import Any, ParamSpec, TypeVar

import anyio
import pytest
from pydantic_ai import Agent, FunctionToolset, RunContext, ToolsetTool
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import DynamicToolset
from render.workflows import Retry, TaskContext, TaskDefinition, Workflows

from pydantic_ai_harness import RenderWorkflows, ToolOutputLimits
from pydantic_ai_harness.subagents import SubAgent, SubAgents

from .conftest import RecordingTaskContext

P = ParamSpec('P')
R = TypeVar('R')


class RegistrationRecordingWorkflows(Workflows):
    """Record every task registered through the public `Workflows.task` decorator.

    A `Workflows` app publishes no registry, so the decorator is the only public place to
    observe that a registration happened at all. Both documented call styles are recorded,
    because the app that carries an agent's operation tasks is also the app a user registers
    their own workflow entry points on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.registered_task_names: list[str] = []

    def task(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        retry: Retry | None = None,
        timeout_seconds: int | None = None,
        plan: str | None = None,
    ) -> Any:
        decorator = super().task(name=name, retry=retry, timeout_seconds=timeout_seconds, plan=plan)

        def recording_decorator(target: Callable[..., Any]) -> TaskDefinition[..., Any]:
            definition = decorator(target)
            self.registered_task_names.append(definition.name)
            return definition

        if func is None:
            return recording_decorator
        return recording_decorator(func)


class UnidentifiedAudit(AbstractCapability[None]):
    """A capability that contributes a durable operation and carries no `id`."""

    @durable_operation(name='record')
    async def record(self, ctx: RunContext[None], message: str) -> str:
        del ctx
        return f'recorded:{message}'


class IdentifiedAudit(UnidentifiedAudit):
    """The same capability with the explicit `id` a durable engine requires."""

    id = 'audit'


class FanOutRecordingTaskContext(RecordingTaskContext):
    """Record the maximum number of overlapping tool operation tasks."""

    def __init__(self) -> None:
        super().__init__()
        self.active_tool_tasks = 0
        self.max_active_tool_tasks = 0

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        is_tool_call = task.name.endswith('.call_tool')
        if is_tool_call:
            self.active_tool_tasks += 1
            self.max_active_tool_tasks = max(self.max_active_tool_tasks, self.active_tool_tasks)
        try:
            return await super().run(task, *args, **kwargs)
        finally:
            if is_tool_call:
                self.active_tool_tasks -= 1


class FakeMCPToolset(MCPToolset[None]):
    """In-memory MCP toolset used to exercise the public durable wrapper."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.max_retries = None
        self.cache_tools = True
        self.include_instructions = False
        self.include_return_schema = None
        self.id = 'remote-tools'

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        tool_def = ToolDefinition(
            name='remote_lookup',
            parameters_json_schema={
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': ['query'],
            },
        )
        return {'remote_lookup': self.tool_for_tool_def(tool_def, ctx=ctx)}

    async def get_instructions(self, ctx: RunContext[None]) -> None:
        del ctx

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[None],
        tool: ToolsetTool[None],
    ) -> str:
        del ctx, tool
        self.calls.append((name, tool_args))
        return 'remote result'


class SuspendedModel(Model):
    """A local model that exposes public continuation cancellation behavior."""

    def __init__(self, suspended: anyio.Event) -> None:
        super().__init__()
        self.suspended = suspended
        self.cancelled: list[ModelResponse] = []

    @property
    def model_name(self) -> str:
        return 'suspended-model'

    @property
    def system(self) -> str:
        return 'test'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        del messages, model_settings, model_request_parameters
        return ModelResponse(
            parts=[TextPart('still working')],
            model_name=self.model_name,
            provider_response_id='job-123',
            state='suspended',
        )

    def continuation_delay(self, response: ModelResponse) -> float:
        assert response.provider_response_id == 'job-123'
        self.suspended.set()
        return 60

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append(response)


def build_agent() -> tuple[Agent[None, str], RenderWorkflows[None], list[str]]:
    calls: list[str] = []

    async def lookup(query: str) -> str:
        calls.append(query)
        return f'result for {query}'

    workflows = Workflows()
    render_workflows = RenderWorkflows(workflows)
    agent = Agent(
        TestModel(call_tools=['lookup']),
        name='support',
        tools=[lookup],
        capabilities=[render_workflows],
    )
    return agent, render_workflows, calls


@pytest.mark.anyio
async def test_agent_runs_inline_outside_render_task() -> None:
    agent, render_workflows, calls = build_agent()

    result = await agent.run('find it')

    assert isinstance(result.output, str)
    assert calls == ['a']
    assert render_workflows.in_durable_context is False


@pytest.mark.anyio
async def test_agent_dispatches_model_and_static_tool_through_render_tasks() -> None:
    agent, render_workflows, calls = build_agent()
    durable_states: list[bool] = []

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        durable_states.append(render_workflows.in_durable_context)
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending_result = run_agent.func(context, 'find it')
    assert inspect.isawaitable(pending_result)
    result = await pending_result

    assert isinstance(result, str)
    assert calls == ['a']
    assert durable_states == [True]
    assert context.task_names.count('support__model.request') == 2
    assert 'support__function_toolset__<agent>.call_tool' in context.task_names


@pytest.mark.anyio
async def test_independent_tool_calls_fan_out_as_render_child_tasks() -> None:
    both_started = anyio.Event()
    started = 0

    async def wait_for_peer() -> str:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()
        return 'done'

    async def first() -> str:
        return await wait_for_peer()

    async def second() -> str:
        return await wait_for_peer()

    workflows = Workflows()
    render_workflows = RenderWorkflows(workflows)
    agent = Agent(
        TestModel(call_tools=['first', 'second']),
        name='parallel-support',
        tools=[first, second],
        capabilities=[render_workflows],
    )

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = FanOutRecordingTaskContext()
    with anyio.fail_after(5):
        pending_result = run_agent.func(context, 'run both')
        assert inspect.isawaitable(pending_result)
        await pending_result

    assert context.max_active_tool_tasks == 2


@pytest.mark.anyio
async def test_dynamic_tool_discovery_and_call_run_as_render_child_tasks() -> None:
    calls: list[str] = []
    tools = FunctionToolset[None]()

    @tools.tool_plain
    async def dynamic_lookup(query: str) -> str:
        calls.append(query)
        return f'found {query}'

    def resolve_tools(ctx: RunContext[None]) -> FunctionToolset[None]:
        del ctx
        return tools

    workflows = Workflows()
    render_workflows = RenderWorkflows[None](workflows, deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['dynamic_lookup']),
        name='dynamic-support',
        deps_type=type(None),
        toolsets=[DynamicToolset(resolve_tools, id='dynamic-tools')],
        capabilities=[render_workflows],
    )

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    result = run_agent.func(context, 'find it')
    assert inspect.isawaitable(result)
    assert isinstance(await result, str)

    assert calls == ['a']
    assert 'dynamic-support__dynamic_toolset__dynamic-tools.get_tools' in context.task_names
    assert 'dynamic-support__dynamic_toolset__dynamic-tools.call_tool' in context.task_names


@pytest.mark.anyio
async def test_dynamic_tool_cannot_opt_out_of_render_child_task() -> None:
    tools = FunctionToolset[None]()

    @tools.tool_plain
    async def dynamic_lookup(query: str) -> str:
        return f'found {query}'

    def resolve_tools(ctx: RunContext[None]) -> FunctionToolset[None]:
        del ctx
        return tools

    workflows = Workflows()
    render_workflows = RenderWorkflows[None](
        workflows,
        deps_type=type(None),
        resolve_tool_options=lambda _operation_id, _tool, _tool_name: False,
    )
    agent = Agent[None, str](
        TestModel(call_tools=['dynamic_lookup']),
        name='dynamic-support',
        deps_type=type(None),
        toolsets=[DynamicToolset(resolve_tools, id='dynamic-tools')],
        capabilities=[render_workflows],
    )

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    pending_result = run_agent.func(RecordingTaskContext(), 'find it')
    assert inspect.isawaitable(pending_result)
    with pytest.raises(UserError, match='only for function tools'):
        await pending_result


def build_delegating_agent() -> tuple[Agent[None, str], RenderWorkflows[None], Workflows]:
    """A parent that delegates once to a sub-agent carrying its own model.

    The delegate tool lives in a toolset the test names, which is what a durable engine
    needs to register it: task names are persisted workflow identity.
    """
    worker = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('worker result')])),
        name='worker',
        description='Does the work',
    )

    async def delegate_task(ctx: RunContext[None], task: str) -> str:
        # Delegation reads the parent model from the projected run context, so a projection
        # that cannot answer for the model fails the delegation outright.
        assert ctx.model is not None
        return (await worker.run(task)).output

    steps = {'n': 0}

    def parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        steps['n'] += 1
        if steps['n'] == 1:
            args: dict[str, Any] = {'task': 'do it'}
            return ModelResponse(parts=[ToolCallPart('delegate_task', args, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('all done')])

    workflows = Workflows()
    render_workflows = RenderWorkflows[None](workflows, deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(parent_model),
        name='support',
        deps_type=type(None),
        toolsets=[FunctionToolset[None]([delegate_task], id='delegates')],
        capabilities=[render_workflows],
    )
    return agent, render_workflows, workflows


@pytest.mark.anyio
async def test_a_named_toolset_registers_its_render_tasks_under_its_id() -> None:
    agent, render_workflows, _ = build_delegating_agent()

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending_result = run_agent.func(context, 'go')
    assert inspect.isawaitable(pending_result)
    await pending_result

    # Task names are persisted journal data, so they are pinned here: a rename strands
    # in-flight workflows recorded against the old name. `TaskDefinition.name` is the
    # public observation: Workflows does not expose its registry, and this run dispatches
    # the call as a child task.
    assert 'support__function_toolset__delegates.call_tool' in context.task_names


def test_a_capability_that_names_no_toolset_cannot_be_registered() -> None:
    """`SubAgents` builds its own toolset and names it nothing, and nothing public can.

    Recorded as a test because it is a real loss rather than a hypothetical: an agent that
    delegates through `SubAgents` cannot be constructed with `RenderWorkflows` until the
    capability accepts an `id` for the toolset it creates. The alternative was registering
    persisted task names derived from the capability's own id, which meant writing the
    toolset's private `_id`.
    """
    worker = Agent(TestModel(), name='worker', description='Does the work')

    with pytest.raises(UserError, match='SubAgents'):
        Agent[None, str](
            TestModel(),
            name='support',
            deps_type=type(None),
            capabilities=[
                SubAgents[None](agents=[SubAgent(worker)], agent_folders=None),
                RenderWorkflows[None](Workflows(), deps_type=type(None)),
            ],
        )


def test_tool_output_limits_is_refused_as_an_unnamed_capability_toolset() -> None:
    """`ToolOutputLimits` is the other documented unnamed FunctionToolset, and it is refused too.

    Construction raises before any task definition is registered: the capability contributes
    `read_tool_result` through a FunctionToolset the user never holds, so there is no public
    place to give that toolset an `id`.
    """
    with pytest.raises(UserError, match='ToolOutputLimits'):
        Agent[None, str](
            TestModel(),
            name='support',
            deps_type=type(None),
            capabilities=[
                ToolOutputLimits[None](),
                RenderWorkflows[None](Workflows(), deps_type=type(None)),
            ],
        )


def test_a_capability_contributing_operations_without_an_id_is_rejected_before_registration() -> None:
    """Pydantic AI reaches this check only after an engine has registered its tasks.

    A `Workflows` app has no public way to unregister one, so the app a rejected agent was
    given would keep a partial task set for the life of the process. The wording is pinned
    to core's because this is the same refusal, only earlier.
    """
    app = RegistrationRecordingWorkflows()

    with pytest.raises(UserError) as rejection:
        Agent[None, str](
            TestModel(),
            name='support',
            deps_type=type(None),
            capabilities=[RenderWorkflows[None](app, deps_type=type(None)), UnidentifiedAudit()],
        )

    assert str(rejection.value) == (
        "Capability 'UnidentifiedAudit' contributes durable operations and needs an explicit "
        '`id` because persisted operation identity and worker-side recovery must remain stable. '
        "Construct it as `UnidentifiedAudit(id='...')`."
    )
    assert app.registered_task_names == []


@pytest.mark.anyio
async def test_the_same_capability_with_an_id_registers_and_runs_its_operation_task() -> None:
    """The control for the rejection above: the spy does see registration when one happens.

    The `id` is the only difference from the rejected capability, so the operation is also run
    once, through an entry point registered with the app's own decorator rather than the
    capability's, to show the recorded task is the one that carries it.
    """
    app = RegistrationRecordingWorkflows()
    render_workflows = RenderWorkflows[None](app, deps_type=type(None))
    audit = IdentifiedAudit()
    agent = Agent[None, str](
        TestModel(),
        name='support',
        deps_type=type(None),
        capabilities=[render_workflows, audit],
    )

    @agent.instructions
    async def audited_instructions(ctx: RunContext[None]) -> str:
        return await audit.record(ctx, 'hello')

    @app.task
    async def entry_point(ctx: TaskContext) -> str:
        with render_workflows.activate(ctx):
            return (await agent.run('go')).output

    assert 'support__capability__audit.record' in app.registered_task_names
    assert 'support__model.request' in app.registered_task_names
    assert 'entry_point' in app.registered_task_names

    context = RecordingTaskContext()
    with anyio.fail_after(5):
        assert isinstance(await entry_point.func(context), str)

    assert 'support__capability__audit.record' in context.task_names


def test_two_render_workflows_capabilities_sharing_one_app_are_rejected() -> None:
    workflows = Workflows()

    with pytest.raises(UserError, match='exactly one `RenderWorkflows`'):
        Agent(
            TestModel(),
            name='support',
            capabilities=[RenderWorkflows(workflows), RenderWorkflows(workflows)],
        )


def test_two_render_workflows_capabilities_for_different_apps_are_rejected() -> None:
    """Two apps is the case a shared-app name collision would never catch.

    Each capability registers a full task set on its own app, so nothing collides and the
    agent is left with two of everything and no defined answer for which app a run reaches.
    """
    with pytest.raises(UserError, match='exactly one `RenderWorkflows`'):
        Agent(
            TestModel(),
            name='support',
            capabilities=[RenderWorkflows(Workflows()), RenderWorkflows(Workflows())],
        )


@pytest.mark.anyio
async def test_delegation_to_a_sub_agent_with_its_own_model_runs_in_a_render_task() -> None:
    agent, render_workflows, _ = build_delegating_agent()

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending_result = run_agent.func(context, 'go')
    assert inspect.isawaitable(pending_result)
    result = await pending_result

    # The delegate tool runs inside a child task, against a projection of the parent's run
    # context. Delegation reads the parent model from it, so a projection that cannot answer
    # for the model fails the delegation outright.
    assert result == 'all done'
    assert context.task_names.count('support__function_toolset__delegates.call_tool') == 1


@pytest.mark.anyio
async def test_a_tool_in_a_child_task_reads_the_run_model_from_its_own_process() -> None:
    model = TestModel(call_tools=['inspect_model'])
    seen: list[object] = []

    async def inspect_model(ctx: RunContext[None]) -> str:
        seen.append(ctx.model)
        return 'noted'

    workflows = Workflows()
    render_workflows = RenderWorkflows[None](workflows, deps_type=type(None))
    agent = Agent[None, str](
        model,
        name='support',
        deps_type=type(None),
        tools=[inspect_model],
        capabilities=[render_workflows],
    )

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    pending_result = run_agent.func(RecordingTaskContext(), 'look')
    assert inspect.isawaitable(pending_result)
    await pending_result

    # The model instance itself never crossed the boundary: the child task resolved the run's
    # model id against the registry the agent module built in this process. It is the plain
    # model, not the durable wrapper the workflow side holds, so the tool's own model calls
    # stay inside the task it is already running in.
    assert seen == [model]


@pytest.mark.anyio
async def test_mcp_tool_cannot_opt_out_of_render_child_task() -> None:
    workflows = Workflows()
    render_workflows = RenderWorkflows[None](
        workflows,
        deps_type=type(None),
        resolve_tool_options=lambda _operation_id, _tool, _tool_name: False,
    )
    agent = Agent[None, str](
        TestModel(call_tools=['remote_lookup']),
        name='mcp-support',
        deps_type=type(None),
        toolsets=[FakeMCPToolset()],
        capabilities=[render_workflows],
    )

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    pending_result = run_agent.func(RecordingTaskContext(), 'find it')
    assert inspect.isawaitable(pending_result)
    with pytest.raises(UserError, match='only for function tools'):
        await pending_result


@pytest.mark.anyio
async def test_mcp_tool_discovery_and_call_run_as_render_child_tasks() -> None:
    workflows = Workflows()
    render_workflows = RenderWorkflows[None](workflows, deps_type=type(None))
    toolset = FakeMCPToolset()
    agent = Agent[None, str](
        TestModel(call_tools=['remote_lookup']),
        name='mcp-support',
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=[render_workflows],
    )

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending = run_agent.func(context, 'find it')
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)

    assert toolset.calls == [('remote_lookup', {'query': 'a'})]
    assert 'mcp-support__mcp_server__remote-tools.get_tools' in context.task_names
    assert 'mcp-support__mcp_server__remote-tools.call_tool' in context.task_names


@pytest.mark.anyio
async def test_event_stream_handler_and_buffered_stream_run_as_render_child_tasks() -> None:
    seen: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        del ctx
        async for event in stream:
            seen.append(event)

    workflows = Workflows()
    render_workflows = RenderWorkflows[None](
        workflows,
        deps_type=type(None),
        event_stream_handler=handler,
    )
    agent = Agent[None, str](
        TestModel(call_tools=['streamed_lookup']),
        name='streaming-support',
        deps_type=type(None),
        capabilities=[render_workflows],
    )

    @agent.tool_plain
    async def streamed_lookup() -> str:
        return 'streamed result'

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending = run_agent.func(context, 'stream it')
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)

    assert any(isinstance(event, FunctionToolResultEvent) for event in seen)
    assert any(isinstance(event, FinalResultEvent) for event in seen)
    assert 'streaming-support__model.request_stream' in context.task_names
    assert 'streaming-support__event_stream_handler' in context.task_names


@pytest.mark.anyio
async def test_cancelling_a_suspended_response_runs_cleanup_as_a_render_child_task() -> None:
    suspended = anyio.Event()
    model = SuspendedModel(suspended)
    workflows = Workflows()
    render_workflows = RenderWorkflows[None](workflows, deps_type=type(None))
    agent = Agent[None, str](
        model,
        name='cancellable-support',
        deps_type=type(None),
        capabilities=[render_workflows],
    )

    @render_workflows.task
    async def run_agent(ctx: TaskContext, prompt: str) -> str:
        del ctx
        return (await agent.run(prompt)).output

    context = RecordingTaskContext()
    pending = run_agent.func(context, 'start it')
    assert inspect.isawaitable(pending)
    # Both waits are open-ended: a regression that never suspends, or one whose cancellation
    # never settles, would otherwise hang the suite instead of failing this test.
    with anyio.fail_after(5):
        run_task = asyncio.ensure_future(pending)
        await suspended.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    assert [response.provider_response_id for response in model.cancelled] == ['job-123']
    assert 'cancellable-support__model.cancel_suspended_response' in context.task_names
