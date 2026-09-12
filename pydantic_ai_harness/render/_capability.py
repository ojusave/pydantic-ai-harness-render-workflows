"""Render Workflows durability capability."""

from __future__ import annotations

try:
    import render  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `render` package to use the Render Workflows capability, '
        'for example with `pip install "pydantic-ai-harness[render]"`.'
    ) from _import_error

import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, ClassVar, Concatenate, Literal, ParamSpec, TypeVar, overload

from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationBackend,
    DurableOperationId,
    RoleBasedOperationConfig,
    ToolsetCallToolId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset
from render import Options, Retry, TaskContext, Workflows
from render.workflows import TaskDefinition
from render.workflows.task import BoundTaskDecorator

from ._compat import (
    CapabilityMethodDeclaration,
    RenderRunContextCodec,
)
from ._context import activate_task_context, current_task_context
from ._operation_backend import RenderOperationBackend
from ._transports import (
    RenderCancelTransport,
    RenderCapabilityOperationTransport,
    RenderCompactMessagesTransport,
    RenderDynamicCallTransport,
    RenderDynamicGetToolsTransport,
    RenderEventStreamHandlerTransport,
    RenderFunctionCallTransport,
    RenderGetToolsTransport,
    RenderMCPCallTransport,
    RenderModelRequestTransport,
)

P = ParamSpec('P')
R = TypeVar('R')

ToolOptionsResolver = Callable[
    [DurableOperationId, object | None, str],
    Options | Literal[False] | None,
]

Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None


@dataclass(init=False)
class RenderWorkflows(BaseDurabilityCapability[AgentDepsT]):
    """Route an agent's durable operations through an explicit Workflows app.

    Outside a Render task, the capability is transparent. Use this instance's
    `task` decorator for each workflow entry point that calls the agent.
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name='Render Workflows',
        durable_unit_noun='task',
        durable_container_noun='workflow',
        codec=JSON_CODEC,
        wrapped_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        toolset_lifecycles={
            'function': 'enter-outside-durable',
            'mcp': 'enter-outside-durable',
            'dynamic': 'enter-never',
        },
        journal_discovery=True,
        sequential_tools_in_durable_context=False,
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        tool_config_key='render_workflows',
    )

    def __init__(
        self,
        app: Workflows,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        deps_type: type[AgentDepsT] | None = None,
        model_options: Options | None = None,
        tool_options: Options | None = None,
        event_options: Options | None = None,
        capability_options: Options | None = None,
        resolve_tool_options: ToolOptionsResolver | None = None,
    ) -> None:
        """Create a capability and register its operation tasks on `app` when bound.

        Args:
            app: The exact Render `Workflows` app started by the worker.
            models: Additional models keyed by their durable ID.
            event_stream_handler: Optional handler for agent stream events.
            name: Stable prefix for generated Render task names. Defaults to the agent name.
            deps_type: Dependency type used for task-boundary serialization. Defaults to the
                agent's dependency type when the capability is bound.
            model_options: Options for model operation tasks.
            tool_options: Options for tool operation tasks.
            event_options: Options for event handler tasks.
            capability_options: Options for other capabilities' durable operation tasks.
            resolve_tool_options: Optional resolver for a tool-specific opt-out. Returning
                `None` keeps `tool_options`, and `False` executes a supported function tool
                inline. Render fixes task options at registration, so returning different
                `Options` later raises a `UserError` rather than silently ignoring them.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self.app = app
        self._deps_type = deps_type
        base_tool_options = tool_options or Options()

        def resolve_options(
            operation_id: DurableOperationId, tool: object | None, tool_name: str
        ) -> Options | Literal[False]:
            if tool is None or resolve_tool_options is None:
                return base_tool_options
            resolved = resolve_tool_options(operation_id, tool, tool_name)
            if resolved is False and (
                not isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
                or operation_id.toolset_kind != 'function'
            ):
                raise UserError(
                    '`resolve_tool_options` may return `False` only for function tools; '
                    'MCP and dynamic tools must run as Render child tasks.'
                )
            return base_tool_options if resolved is None else resolved

        self._operation_config = RoleBasedOperationConfig[Options | None](
            model=model_options or Options(),
            tool=base_tool_options,
            event=event_options or Options(),
            capability=capability_options or Options(),
            resolve_tool=resolve_options if resolve_tool_options is not None else None,
        )
        self._operation_backend: RenderOperationBackend | None = None
        self._context_codec: RenderRunContextCodec[Any] | None = None
        # BaseDurabilityCapability binds a shallow copy. The original task decorator and the
        # bound runtime intentionally share this token while remaining isolated from other apps.
        self._owner_token = object()

    @property
    def in_durable_context(self) -> bool:
        return current_task_context(self._owner_token) is not None

    def activate(self, context: TaskContext) -> AbstractContextManager[None]:
        """Activate a Render task context for an adapter or local test harness."""
        return activate_task_context(self._owner_token, context)

    def _check_bindable(self) -> None:
        if self.in_durable_context:
            raise UserError(
                'An agent with `RenderWorkflows` must be constructed outside a Render workflow so '
                'its operation tasks are registered before the worker starts.'
            )

    # The async overload comes first so a coroutine function does not bind `R` to its coroutine.
    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], Awaitable[R]],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], R],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        *,
        name: str | None = ...,
        retry: Retry | None = ...,
        timeout_seconds: int | None = ...,
        plan: str | None = ...,
    ) -> BoundTaskDecorator: ...

    def task(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        retry: Retry | None = None,
        timeout_seconds: int | None = None,
        plan: str | None = None,
    ) -> Any:
        """Register a task that activates this capability's Render context."""

        def decorator(task_func: Callable[..., Any]) -> TaskDefinition[..., Any]:
            if inspect.iscoroutinefunction(task_func):

                @functools.wraps(task_func)
                async def async_wrapper(context: TaskContext, *args: Any, **kwargs: Any) -> Any:
                    with self.activate(context):
                        return await task_func(context, *args, **kwargs)

                wrapped = async_wrapper
            else:

                @functools.wraps(task_func)
                def sync_wrapper(context: TaskContext, *args: Any, **kwargs: Any) -> Any:
                    with self.activate(context):
                        return task_func(context, *args, **kwargs)

                wrapped = sync_wrapper
            return self.app.task(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
                wrapped,
                name=name,
                retry=retry,
                timeout_seconds=timeout_seconds,
                plan=plan,
            )

        if func is None:
            return decorator
        return decorator(func)

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        if self._deps_type is None:
            self._deps_type = agent.deps_type
        self._context_codec = RenderRunContextCodec(deps_type=self._deps_type, agent=agent)
        self._operation_backend = RenderOperationBackend(
            self.app,
            runtime=self,  # pyright: ignore[reportArgumentType]
            agent_name=self.name,
            config=self._operation_config,
        )
        if self._event_stream_handler is not None:
            self._bound_event_operation = self._bind_event_operation(self._operation_backend)
        super()._bind_to_agent(agent)

    def get_durable_operation_backend(self) -> DurableOperationBackend[Options | None]:
        backend = self._operation_backend
        assert backend is not None
        return backend

    def _codec(self) -> RenderRunContextCodec[Any]:
        codec = self._context_codec
        assert codec is not None
        return codec

    def _capability_operation_parameter_transport(
        self, declaration: CapabilityMethodDeclaration
    ) -> RenderCapabilityOperationTransport[AgentDepsT]:
        return RenderCapabilityOperationTransport(self._codec(), declaration)

    def _function_call_parameter_transport(
        self, toolset: FunctionToolset[AgentDepsT]
    ) -> RenderFunctionCallTransport[AgentDepsT]:
        return RenderFunctionCallTransport(self._codec(), toolset)

    def _get_tools_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> RenderGetToolsTransport[AgentDepsT]:
        del toolset
        return RenderGetToolsTransport(self._codec(), result_type=dict[str, ToolDefinition])

    def _get_instructions_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> RenderGetToolsTransport[AgentDepsT]:
        del toolset
        return RenderGetToolsTransport(self._codec(), result_type=Instructions)

    def _dynamic_get_tools_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> RenderDynamicGetToolsTransport[AgentDepsT]:
        del toolset
        return RenderDynamicGetToolsTransport(self._codec())

    def _dynamic_call_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> RenderDynamicCallTransport[AgentDepsT]:
        del toolset
        return RenderDynamicCallTransport(self._codec())

    def _mcp_call_parameter_transport(self, toolset: AbstractToolset[AgentDepsT]) -> RenderMCPCallTransport[AgentDepsT]:
        return RenderMCPCallTransport(self._codec(), toolset)

    def _model_request_parameter_transport(self, result_type: object) -> RenderModelRequestTransport[AgentDepsT]:
        return RenderModelRequestTransport(self._codec(), result_type=result_type)

    def _cancel_suspended_response_parameter_transport(
        self,
    ) -> RenderCancelTransport[AgentDepsT]:
        return RenderCancelTransport(self._codec())

    def _compact_messages_parameter_transport(self) -> RenderCompactMessagesTransport[AgentDepsT]:
        return RenderCompactMessagesTransport(self._codec())

    def _event_stream_handler_parameter_transport(
        self,
    ) -> RenderEventStreamHandlerTransport[AgentDepsT]:
        return RenderEventStreamHandlerTransport(self._codec())
