"""Registered Pydantic AI operations backed by Render Workflows tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic_ai.durable_exec import JournalOperationNamer, RegisteredOperationBackend, RoleBasedOperationConfig
from pydantic_ai.exceptions import UserError
from render import Options, Retry, TaskContext, Workflows
from render.workflows import TaskDefinition

from ._compat import BoundDurableOperation, DurableOperation, dump_operation_params, load_operation_params
from ._context import activate_task_context, current_task_context
from ._protocol import (
    OperationRequest,
    OperationResult,
    control_flow_error,
    make_request,
    read_request,
    read_result,
    success,
)

if TYPE_CHECKING:
    from ._capability import RenderWorkflows

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')


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


class RenderBoundOperation(BoundDurableOperation[ParamsT, WireT, ResultT], Generic[ParamsT, WireT, ResultT]):
    """Dispatch one operation through its statically registered Render task."""

    def __init__(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        task: TaskDefinition[[OperationRequest], OperationResult],
        operation_name: str,
        options: Options | None,
        runtime: RenderWorkflows,
    ) -> None:
        self._operation = operation
        self.task = task
        self._operation_name = operation_name
        self._options = options
        self._runtime = runtime
        self._owner_token = runtime._owner_token  # pyright: ignore[reportPrivateUsage]

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        context = current_task_context(self._owner_token)
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


class RenderOperationBackend(RegisteredOperationBackend[Options | None]):
    """Register each supported Pydantic AI operation on a Workflows app."""

    def __init__(
        self,
        app: Workflows,
        *,
        runtime: RenderWorkflows,
        agent_name: str,
        config: RoleBasedOperationConfig[Options | None],
    ) -> None:
        super().__init__(namer=JournalOperationNamer(agent_name), config=config)
        self._app = app
        self._runtime = runtime
        self._owner_token = runtime._owner_token  # pyright: ignore[reportPrivateUsage]

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: Options | None,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        async def operation_task(context: TaskContext, request: OperationRequest) -> OperationResult:
            wire_params = read_request(request, expected_operation=name)
            params = load_operation_params(operation, wire_params, runtime=self._runtime)
            try:
                with activate_task_context(self._owner_token, context):
                    value = await operation.handler(params)
                return success(operation.result_codec.dump(value))
            except Exception as exc:
                expected_error = control_flow_error(exc)
                if expected_error is not None:
                    return expected_error
                raise

        registered_config = _snapshot_options(config) if config is not None else None
        options = registered_config or Options()
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
        # Render registers the TaskDefinition directly on `app`. The generic backend contract
        # only accepts callable worker registrations, so there is no additional handle to return.
        return bound, ()
