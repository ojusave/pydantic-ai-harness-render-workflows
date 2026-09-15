"""Registered Pydantic AI operations backed by Render Workflows tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic_ai.durable_exec import JournalOperationNamer, RegisteredOperationBackend, RoleBasedOperationConfig
from pydantic_ai.exceptions import UserError
from render import Options, Retry, TaskContext, Workflows
from render.workflows import TaskDefinition

from ._compat import BoundDurableOperation, DurableOperation, dump_operation_params, load_operation_params
from ._protocol import (
    OperationRequest,
    OperationResult,
    control_flow_error,
    make_request,
    permanent_error,
    read_request,
    read_result,
    success,
)

if TYPE_CHECKING:
    from ._capability import RenderWorkflows

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
RuntimeDepsT = TypeVar('RuntimeDepsT')


def _snapshot_options(options: Options) -> Options:
    retry = options.retry
    return Options(
        retry=(
            Retry(
                max_retries=retry.max_retries,
                wait_duration_ms=retry.wait_duration_ms,
                backoff_scaling=retry.backoff_scaling,
            )
            if retry is not None
            else None
        ),
        timeout_seconds=options.timeout_seconds,
        plan=options.plan,
    )


class RenderBoundOperation(
    BoundDurableOperation[ParamsT, WireT, ResultT],
    Generic[ParamsT, WireT, ResultT, RuntimeDepsT],
):
    """Dispatch one operation through its statically registered Render task."""

    def __init__(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        task: TaskDefinition[[OperationRequest], OperationResult],
        operation_name: str,
        options: Options | None,
        runtime: RenderWorkflows[RuntimeDepsT],
    ) -> None:
        self._operation = operation
        self.task = task
        self._operation_name = operation_name
        self._options = options
        self._runtime = runtime

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        context = self._runtime.current_task_context
        if context is None:
            return await self._operation.handler(params)

        self._check_static_config(config)
        wire_params = dump_operation_params(self._operation, params)
        request = make_request(self._operation_name, wire_params)
        result = await context.run(self.task, request)
        payload = read_result(result)
        return self._operation.result_codec.load(payload)

    def _check_static_config(self, config: object | None) -> None:
        if config is not None and config != self._options:
            raise UserError(
                'Render Workflows task options are fixed when an agent is bound and cannot vary per invocation.'
            )


class RenderOperationBackend(RegisteredOperationBackend[Options | None], Generic[RuntimeDepsT]):
    """Register each supported Pydantic AI operation on a Workflows app."""

    def __init__(
        self,
        app: Workflows,
        *,
        runtime: RenderWorkflows[RuntimeDepsT],
        agent_name: str,
        config: RoleBasedOperationConfig[Options | None],
    ) -> None:
        super().__init__(namer=JournalOperationNamer(agent_name), config=config)
        self._app = app
        self._runtime = runtime

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: Options | None,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        async def operation_task(context: TaskContext, request: OperationRequest) -> OperationResult:
            try:
                wire_params = read_request(request, expected_operation=name)
                params = load_operation_params(operation, wire_params, runtime=self._runtime)
            except Exception as exc:
                # Retrying cannot repair persisted request bytes or worker-side decoding.
                return permanent_error('invalid-request', exc)

            try:
                with self._runtime.activate(context):
                    value = await operation.handler(params)
            except Exception as exc:
                try:
                    expected_error = control_flow_error(exc)
                except Exception as encoding_error:
                    return permanent_error('invalid-result', encoding_error)
                if expected_error is not None:
                    return expected_error
                raise

            try:
                return success(operation.result_codec.dump(value))
            except Exception as exc:
                # The handler may already have committed an external side effect.
                return permanent_error('invalid-result', exc)

        registered_config = _snapshot_options(config) if config is not None else None
        options = registered_config or Options()
        # A `None` field intentionally lets the public Workflows API resolve its app default
        # when this task is registered; only explicit per-operation values are snapshotted here.
        task = self._app.task(
            name=name,
            retry=options.retry,
            timeout_seconds=options.timeout_seconds,
            plan=options.plan,
        )(operation_task)
        bound = RenderBoundOperation(
            operation,
            task=task,
            operation_name=name,
            options=registered_config,
            runtime=self._runtime,
        )
        # `app.task` has already registered the task definition. The generic backend's second
        # return value is for worker-registration callables, not task runs, so it is empty here.
        return bound, ()
