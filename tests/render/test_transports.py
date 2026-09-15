from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from render.workflows import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext


@dataclass
class Deps:
    tenant: str


@pytest.mark.anyio
async def test_model_and_function_transports_round_trip_public_run_context() -> None:
    seen: list[tuple[Deps, str | None, RunUsage, UsageLimits | None, object]] = []
    model = TestModel(call_tools=['lookup'])
    runtime = RenderWorkflows[Deps](Workflows(), deps_type=Deps)
    agent = Agent[Deps, str](
        model,
        name='transport-agent',
        deps_type=Deps,
        capabilities=[runtime],
    )

    @agent.tool
    async def lookup(ctx: RunContext[Deps], value: str) -> str:
        seen.append((ctx.deps, ctx.run_id, ctx.usage, ctx.usage_limits, ctx.model))
        return f'{ctx.deps.tenant}:{value}'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        result = await agent.run(
            'look up status',
            deps=Deps('acme'),
            usage=RunUsage(requests=2, input_tokens=10),
            usage_limits=UsageLimits(request_limit=8),
        )
        return result.output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)

    assert seen
    deps, run_id, usage, usage_limits, child_model = seen[0]
    assert deps == Deps('acme')
    assert run_id is not None
    assert usage.requests >= 3
    assert usage_limits == UsageLimits(request_limit=8)
    assert child_model is model
    assert 'transport-agent__model.request' in context.task_names
    assert 'transport-agent__function_toolset__<agent>.call_tool' in context.task_names


async def _run_in_workflow(
    agent: Agent[None, str], runtime: RenderWorkflows[None], context: TaskContext, *, model: str | None = None
) -> str:
    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('inspect', model=model) if model else await agent.run('inspect')).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    return await pending


@pytest.mark.anyio
async def test_child_task_model_is_guarded_when_the_worker_registers_no_instance() -> None:
    # A model-name string is deliberately not resolved when the agent is bound, so nothing is
    # registered under `default` and a child task has no instance to attach to its run context.
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str]('test', name='string-model-agent', deps_type=type(None), capabilities=[runtime])

    @agent.tool
    async def inspect_model(ctx: RunContext[None]) -> str:
        return str(ctx.model)

    with pytest.raises(UserError, match="'model' is not available on 'RenderRunContext'"):
        await _run_in_workflow(agent, runtime, RecordingTaskContext())


@pytest.mark.anyio
async def test_registered_model_id_resolves_to_its_instance_in_the_child_task() -> None:
    alternate = TestModel(custom_output_text='from the alternate model')
    runtime = RenderWorkflows[None](Workflows(), models={'alternate': alternate})
    agent = Agent[None, str](
        TestModel(call_tools=['inspect_model']),
        name='registry-agent',
        deps_type=type(None),
        capabilities=[runtime],
    )
    seen: list[object] = []

    @agent.tool
    async def inspect_model(ctx: RunContext[None]) -> str:
        seen.append(ctx.model)
        return 'ok'

    output = await _run_in_workflow(agent, runtime, RecordingTaskContext(), model='alternate')

    assert output == 'from the alternate model'
    assert seen == [alternate]


@pytest.mark.anyio
async def test_unregistered_model_instance_is_rejected_before_dispatch() -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](TestModel(), name='instance-agent', deps_type=type(None), capabilities=[runtime])
    context = RecordingTaskContext()

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('inspect', model=TestModel(custom_output_text='unregistered'))).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    # An instance cannot cross the task boundary, so the run fails before any task is started
    # rather than rebuilding a different model from its name on the worker.
    with pytest.raises(UserError, match='was not registered with `RenderWorkflows`'):
        await pending
    assert context.task_names == []
