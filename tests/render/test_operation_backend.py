from __future__ import annotations

import inspect

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from render.workflows import Options, Retry, TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext


def build_agent(*, options: Options | None = None) -> tuple[Agent[None, str], RenderWorkflows[None]]:
    app = Workflows()
    runtime = RenderWorkflows[None](app, deps_type=type(None), model_options=options)
    agent = Agent[None, str](
        TestModel(),
        name='support',
        deps_type=type(None),
        capabilities=[runtime],
    )
    return agent, runtime


async def _run_in_workflow(agent: Agent[None, str], runtime: RenderWorkflows[None], context: TaskContext) -> str:
    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('remote')).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    return await pending


@pytest.mark.anyio
async def test_runs_inline_outside_render_context() -> None:
    agent, runtime = build_agent()
    assert isinstance((await agent.run('inline')).output, str)
    assert runtime.in_durable_context is False


@pytest.mark.anyio
async def test_dispatches_registered_task_and_activates_child_context() -> None:
    agent, runtime = build_agent()
    context = RecordingTaskContext()

    assert isinstance(await _run_in_workflow(agent, runtime, context), str)
    assert context.task_names == ['support__model.request']


@pytest.mark.anyio
async def test_invocation_options_that_differ_from_registration_are_rejected() -> None:
    runtime = RenderWorkflows[None](
        Workflows(),
        deps_type=type(None),
        tool_options=Options(timeout_seconds=60),
        resolve_tool_options=lambda operation_id, tool, tool_name: Options(timeout_seconds=120),
    )
    agent = Agent[None, str](
        TestModel(call_tools=['ping']),
        name='per-call-options',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def ping(ctx: RunContext[None]) -> str:
        del ctx
        return 'pong'

    # Render fixes a task's retry, timeout, and plan when the task is registered, so options
    # resolved later for one tool cannot take effect and are rejected instead of ignored.
    with pytest.raises(UserError, match='fixed when an agent is bound'):
        await _run_in_workflow(agent, runtime, RecordingTaskContext())


@pytest.mark.anyio
async def test_registered_task_options_are_snapshotted_when_the_agent_is_bound() -> None:
    """Registration copies the caller's options, so later mutation cannot change a live task.

    `TaskDefinition` publishes only `name` and `func`, so the registered retry, timeout, and plan
    cannot be read back through the public Render SDK. The snapshot is observable instead through
    the invocation check: the resolver hands back the very object that was registered, and the
    call is rejected only because registration kept a copy of its earlier values.
    """
    options = Options(retry=Retry(max_retries=2, wait_duration_ms=100), timeout_seconds=90, plan='flex')
    runtime = RenderWorkflows[None](
        Workflows(),
        deps_type=type(None),
        tool_options=options,
        resolve_tool_options=lambda operation_id, tool, tool_name: options,
    )
    agent = Agent[None, str](
        TestModel(call_tools=['ping']),
        name='snapshot-options',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def ping(ctx: RunContext[None]) -> str:
        del ctx
        return 'pong'

    assert options.retry is not None
    options.retry.max_retries = 99
    options.timeout_seconds = 1

    with pytest.raises(UserError, match='fixed when an agent is bound'):
        await _run_in_workflow(agent, runtime, RecordingTaskContext())
