from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic_ai.durable_exec import CapabilityOperationId, RoleBasedOperationConfig
from pydantic_ai.exceptions import ModelRetry, UserError
from render.workflows import Options, Retry, TaskDefinition, Workflows

from pydantic_ai_harness.render._capability import RenderWorkflows
from pydantic_ai_harness.render._compat import DurableOperation, JSONObject, ParameterTransport
from pydantic_ai_harness.render._context import current_task_context
from pydantic_ai_harness.render._operation_backend import RenderBoundOperation, RenderOperationBackend


class ObjectTransport(ParameterTransport[JSONObject, JSONObject]):
    def dump(self, params: JSONObject) -> JSONObject:
        return params

    def load(self, payload: JSONObject, *, runtime: object) -> JSONObject:
        return payload


class StringCodec:
    def dump(self, value: str) -> object:
        return value

    def load(self, payload: object) -> str:
        if not isinstance(payload, str):
            raise TypeError('expected a string')
        return payload


class NoCacheIdentity:
    def project(self, params: JSONObject) -> tuple[object, ...]:
        return ()


class LocalTaskContext:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def run(self, task: TaskDefinition[..., Any], *args: Any, **kwargs: Any) -> Any:
        assert not kwargs
        self.calls.append((task.name, args[0]))
        value = task.func(self, *args)
        if inspect.isawaitable(value):
            return await value
        return value


def make_backend(
    handler: Any,
    *,
    options: Options | None = None,
) -> tuple[RenderWorkflows[object], RenderOperationBackend, RenderBoundOperation[JSONObject, JSONObject, str]]:
    app = Workflows()
    runtime = RenderWorkflows[object](app, capability_options=options)
    config = RoleBasedOperationConfig[Options | None](
        model=options,
        event=options,
        tool=options,
        capability=options,
    )
    backend = RenderOperationBackend(app, runtime=runtime, agent_name='support', config=config)
    operation = DurableOperation[JSONObject, JSONObject, str](
        operation_id=CapabilityOperationId('demo', operation='execute'),
        handler=handler,
        parameter_transport=ObjectTransport(),
        cache_identity=NoCacheIdentity(),
        result_codec=StringCodec(),
        config_role='capability',
    )
    bound = backend.bind(operation)
    assert isinstance(bound, RenderBoundOperation)
    return runtime, backend, bound


@pytest.mark.anyio
async def test_runs_inline_outside_render_context() -> None:
    async def handler(params: JSONObject) -> str:
        return str(params['value'])

    _, _, bound = make_backend(handler)

    assert await bound({'value': 'inline'}) == 'inline'


@pytest.mark.anyio
async def test_dispatches_registered_task_and_activates_child_context() -> None:
    runtime: RenderWorkflows[object]

    async def handler(params: JSONObject) -> str:
        assert current_task_context(runtime._owner_token) is context
        return str(params['value'])

    runtime, backend, bound = make_backend(handler)
    context = LocalTaskContext()

    with runtime.activate(context):
        result = await bound({'value': 'remote'})

    assert result == 'remote'
    assert context.calls == [
        (
            'support__capability__demo.execute',
            {
                'version': 1,
                'operation': 'support__capability__demo.execute',
                'payload': {'value': 'remote'},
            },
        )
    ]
    assert bound.task.name == 'support__capability__demo.execute'
    assert backend.registrations() == []


@pytest.mark.anyio
async def test_maps_options_at_registration() -> None:
    async def handler(params: JSONObject) -> str:
        return 'ok'

    options = Options(
        retry=Retry(max_retries=4, wait_duration_ms=250, backoff_scaling=2),
        timeout_seconds=90,
        plan='2c-4g',
    )
    runtime, _, bound = make_backend(handler, options=options)
    context = LocalTaskContext()

    task_info = runtime.app._registry.get_task(bound.task.name)
    assert task_info is not None
    assert task_info.options == options

    with runtime.activate(context):
        with pytest.raises(UserError, match='fixed when an agent is bound'):
            await bound({}, config=Options(timeout_seconds=1))
    assert context.calls == []


@pytest.mark.anyio
async def test_registration_snapshots_mutable_options() -> None:
    async def handler(params: JSONObject) -> str:
        return 'ok'

    options = Options(
        retry=Retry(max_retries=2, wait_duration_ms=100),
        timeout_seconds=90,
        plan='flex',
    )
    runtime, _, bound = make_backend(handler, options=options)
    context = LocalTaskContext()

    options.timeout_seconds = 1
    assert options.retry is not None
    options.retry.max_retries = 99

    task_info = runtime.app._registry.get_task(bound.task.name)
    assert task_info is not None
    assert task_info.options.timeout_seconds == 90
    assert task_info.options.retry is not None
    assert task_info.options.retry.max_retries == 2

    with runtime.activate(context):
        with pytest.raises(UserError, match='fixed when an agent is bound'):
            await bound({}, config=options)
    assert context.calls == []


@pytest.mark.anyio
async def test_expected_control_flow_crosses_as_tagged_result() -> None:
    async def handler(params: JSONObject) -> str:
        raise ModelRetry('choose another value')

    runtime, _, bound = make_backend(handler)
    context = LocalTaskContext()

    with runtime.activate(context):
        with pytest.raises(ModelRetry, match='choose another value'):
            await bound({})


@pytest.mark.anyio
async def test_unexpected_error_is_left_for_render_retry_policy() -> None:
    async def handler(params: JSONObject) -> str:
        raise RuntimeError('transient failure')

    runtime, _, bound = make_backend(handler)
    context = LocalTaskContext()

    with runtime.activate(context):
        with pytest.raises(RuntimeError, match='transient failure'):
            await bound({})
