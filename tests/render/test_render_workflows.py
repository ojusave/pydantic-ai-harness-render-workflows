"""Public-path integration tests for Render Workflows durability."""

from __future__ import annotations

import inspect
from typing import Any, ParamSpec, TypeVar

import anyio
import pytest
from pydantic_ai import Agent, FunctionToolset, RunContext, ToolsetTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import DynamicToolset
from render.workflows import TaskContext, TaskDefinition, Workflows

from pydantic_ai_harness import RenderWorkflows
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

P = ParamSpec('P')
R = TypeVar('R')


class RecordingTaskContext(TaskContext):
    """Execute child tasks locally while recording the public Render task boundary."""

    def __init__(self) -> None:
        self.task_names: list[str] = []

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.task_names.append(task.name)
        result = task.func(self, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


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
        self._id = 'remote-tools'
        self.max_retries = None
        self.cache_tools = True
        self.include_instructions = False
        self.include_return_schema = None

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
        del name, tool_args, ctx, tool
        return 'remote result'


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
    assert calls
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
    assert calls
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

    Both capabilities contribute a leaf `FunctionToolset`, which a durable engine can only
    register when it has an `id`. Neither is wrapped by anything here: registering the
    capabilities as they ship is the point.
    """
    worker = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('worker result')])),
        name='worker',
        description='Does the work',
    )

    steps = {'n': 0}

    def parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        steps['n'] += 1
        if steps['n'] == 1:
            args: dict[str, Any] = {'agent_name': 'worker', 'task': 'do it'}
            return ModelResponse(parts=[ToolCallPart('delegate_task', args, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('all done')])

    workflows = Workflows()
    render_workflows = RenderWorkflows[None](workflows, deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(parent_model),
        name='support',
        deps_type=type(None),
        capabilities=[
            SubAgents[None](agents=[SubAgent(worker)], agent_folders=None),
            ToolOutputLimits[None](),
            render_workflows,
        ],
    )
    return agent, render_workflows, workflows


def test_capability_toolsets_register_render_tasks_under_their_capability_id() -> None:
    _, _, workflows = build_delegating_agent()

    names = set(workflows._registry.get_task_names())

    # Task names are persisted journal data, so they are pinned here: a rename strands
    # in-flight workflows recorded against the old name.
    assert {
        'support__function_toolset__sub_agents.call_tool',
        'support__function_toolset__sub_agents.validate_args',
        'support__function_toolset__tool_output_limits.call_tool',
        'support__function_toolset__tool_output_limits.validate_args',
    } <= names


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
    assert context.task_names.count('support__function_toolset__sub_agents.call_tool') == 1


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
