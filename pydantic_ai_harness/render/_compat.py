"""Compatibility boundary for Pydantic AI durable execution internals.

Pydantic AI does not currently publish the semantic parameter and registered
operation types needed by a cross-process backend. Keep those imports in this
module so an upstream API change has one repair point.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast, overload

from pydantic import TypeAdapter
from pydantic_ai._run_context import AnchoredEvidence, CapabilityEventT, CustomEventT
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import JSON_CODEC
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    ModelRequestContextProjection,
    capability_operation_result_type,
    collect_capability_operations,
)
from pydantic_ai.durable_exec._operation import (
    DurableOperation,
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ParameterTransport,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DynamicToolsResult,
    EnqueueGuard,
    enqueue_not_supported_message,
    validation_context_from_agent,
)
from pydantic_ai.durable_exec._utils import StreamedActivityResult
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent, CustomEvent, ModelMessage
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_ai.usage import RunUsage, UsageLimits
from typing_extensions import TypeVar as TypeVarExtensions

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = (
    'BoundDurableOperation',
    'CallToolResult',
    'CapabilityMethodDeclaration',
    'CapabilityOperationParams',
    'DynamicToolsResult',
    'DynamicToolsetCallToolParams',
    'DurableOperation',
    'EventStreamHandlerParams',
    'JSONObject',
    'JSONValue',
    'ModelCancelSuspendedResponseParams',
    'ModelCompactMessagesParams',
    'ModelRequestContextProjection',
    'ModelRequestParams',
    'ParameterTransport',
    'RenderRunContext',
    'RenderRunContextCodec',
    'StreamedActivityResult',
    'ToolsetCallToolParams',
    'ToolsetGetToolsParams',
    'capability_operation_result_type',
    'get_capability_operation_declaration',
    'dump_json_object',
    'dump_operation_params',
    'function_tool_original_name',
    'load_json_object',
    'load_json_type',
    'load_operation_params',
    'make_model_request_context',
    'model_settings_from_json',
    'model_settings_to_json',
    'resolve_tool_for_definition',
    'to_json_object',
)

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list['JSONValue'] | dict[str, 'JSONValue']
# Object values remain `object` statically because Python 3.10 cannot declare a
# named recursive alias that Pydantic 2.12 can rebuild reliably on Python 3.14.
# `to_json_object` performs the real recursive JSON validation at runtime.
JSONObject: TypeAlias = dict[str, object]

T = TypeVar('T')
ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
AgentDepsT = TypeVarExtensions('AgentDepsT', default=object)
ToolDepsT = TypeVar('ToolDepsT')


def to_json_object(value: object) -> JSONObject:
    """Validate an already encoded value as a JSON object."""
    # A real JSON encode/decode avoids Pydantic's recursive-alias schema bug on
    # Python 3.14 while proving this value can cross the Render boundary.
    normalized = cast(object, json.loads(json.dumps(value, allow_nan=False)))
    if not isinstance(normalized, dict):
        raise TypeError(f'Expected a JSON object, got {type(normalized).__name__}.')
    return cast(JSONObject, normalized)


def dump_json_object(type_form: object, value: object) -> JSONObject:
    """Encode a typed value using Pydantic AI's JSON durability codec."""
    return to_json_object(JSON_CODEC.dump(type_form, value))


def dump_operation_params(operation: DurableOperation[ParamsT, WireT, ResultT], params: ParamsT) -> JSONObject:
    """Dump one operation through a transport that promises a JSON-object wire."""
    return to_json_object(operation.parameter_transport.dump(params))


def load_operation_params(
    operation: DurableOperation[ParamsT, WireT, ResultT],
    payload: JSONObject,
    *,
    runtime: object,
) -> ParamsT:
    """Load one operation from the common Render JSON-object wire.

    Pydantic's generic registered-backend contract permits any `WireT`, while
    this integration only installs transports whose wire is `JSONObject`.
    The cast is contained here so the backend remains strictly typed.
    """
    return operation.parameter_transport.load(cast(WireT, payload), runtime=runtime)


def load_json_type(type_form: type[T], payload: object) -> T:
    """Decode a concrete runtime type from a JSON value.

    `DurabilityCodec.load` is intentionally untyped because Pydantic accepts
    arbitrary type forms. The runtime check contains that unavoidable boundary.
    """
    value = JSON_CODEC.load(type_form, payload)
    if not isinstance(value, type_form):  # pragma: no cover - Pydantic validates concrete classes
        raise TypeError(f'Expected {type_form.__name__}, got {type(value).__name__}.')
    return value


def load_json_object(payload: JSONObject) -> dict[str, Any]:
    """Decode a JSON object for a private semantic parameter with `Any` values."""
    value = JSON_CODEC.load(dict[str, Any], payload)
    if not isinstance(value, dict):  # pragma: no cover - Pydantic validates the declared mapping
        raise TypeError(f'Expected dict, got {type(value).__name__}.')
    # `DurabilityCodec` returns `Any`; Pydantic validated this exact open mapping type.
    return cast(dict[str, Any], value)


def model_settings_to_json(value: ModelSettings | None) -> JSONObject | None:
    """Preserve provider-specific model settings as an open JSON mapping."""
    if value is None:
        return None
    return dump_json_object(dict[str, Any], value)


def model_settings_from_json(value: JSONObject | None) -> ModelSettings | None:
    """Restore the open mapping accepted by the `ModelSettings` TypedDict API."""
    if value is None:
        return None
    # Model settings subclasses add provider keys. Treating the decoded open
    # mapping as the base TypedDict preserves those keys without narrowing them.
    return cast(ModelSettings, load_json_object(value))


def function_tool_original_name(tool: ToolsetTool[ToolDepsT]) -> str | None:
    """Read function-tool identity without leaking its private concrete type."""
    if isinstance(tool, FunctionToolsetTool):
        return tool.original_name
    return None


def resolve_tool_for_definition(
    toolset: object,
    tool_def: ToolDefinition,
    *,
    ctx: RunContext[ToolDepsT],
    original_name: str | None = None,
) -> ToolsetTool[ToolDepsT]:
    """Call the durable reconstruction hook implemented by function and MCP toolsets."""
    method = getattr(toolset, 'tool_for_tool_def', None)
    if not callable(method):
        raise TypeError(f'{type(toolset).__name__} cannot rebuild a tool from its definition.')
    # Both FunctionToolset and the MCP toolset expose this method, but there is
    # no shared public protocol. This call is the contained compatibility edge.
    if original_name is None:
        return cast(ToolsetTool[ToolDepsT], method(tool_def, ctx=ctx))
    return cast(ToolsetTool[ToolDepsT], method(tool_def, ctx=ctx, original_name=original_name))


def make_model_request_context(
    *,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    model_id: str | None,
    streaming: bool,
) -> ModelRequestContext:
    """Build the compaction context whose live model is restored by the handler."""
    # The durable handler resolves the registered model before invoking
    # `model.compact_messages`. This mirrors Pydantic AI's Temporal transport.
    context = ModelRequestContext(
        model=cast(Model, None),
        messages=messages,
        model_settings=model_settings,
        model_request_parameters=model_request_parameters,
    )
    context.model_id = model_id
    context.streaming = streaming
    return context


def get_capability_operation_declaration(
    capability: AbstractCapability[ToolDepsT], operation: str
) -> CapabilityMethodDeclaration:
    """Resolve one declaration through Pydantic AI's private collector."""
    try:
        return collect_capability_operations(capability)[operation]
    except KeyError as exc:
        raise ValueError(f'Capability {type(capability).__name__!r} has no operation {operation!r}.') from exc


_STR_SET_ADAPTER: TypeAdapter[set[str]] = TypeAdapter(set[str])
_REHYDRATORS: tuple[tuple[str, type[Any], TypeAdapter[Any]], ...] = (
    ('usage', dict, TypeAdapter(RunUsage)),
    ('usage_limits', dict, TypeAdapter(UsageLimits)),
    ('loaded_capability_ids', list, _STR_SET_ADAPTER),
    ('discovered_tool_names', list, _STR_SET_ADAPTER),
    ('available_tool_names', list, _STR_SET_ADAPTER),
    ('active_capability_ids', list, _STR_SET_ADAPTER),
    ('_deferred_capability_ids', list, _STR_SET_ADAPTER),
    ('_anchored_evidence', dict, TypeAdapter(AnchoredEvidence)),
)

_NONE_UNLESS_ATTACHED = (
    'agent',
    'root_capability',
    'pending_messages',
    'validation_context',
    'tool_manager',
    'realtime_session',
    '_durable_operations',
    '_run_capabilities_by_id',
)
_DEFAULTED_UNLESS_CARRIED: tuple[tuple[str, Any], ...] = (('_anchored_evidence', AnchoredEvidence()),)
_RENAMED_FIELDS: tuple[tuple[str, str], ...] = (
    ('capability_loaded', 'capability_active'),
    ('available_capability_ids', 'active_capability_ids'),
)
_GUARDED_FIELDS = frozenset(RunContext.__dataclass_fields__) - {'deps', *_NONE_UNLESS_ATTACHED}


class RenderRunContext(RunContext[AgentDepsT]):
    """Restricted run context reconstructed inside a Render child task."""

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
        for old_name, new_name in _RENAMED_FIELDS:
            if old_name in self.__dict__:
                self.__dict__.setdefault(new_name, self.__dict__.pop(old_name))
        for name in _NONE_UNLESS_ATTACHED:
            self.__dict__.setdefault(name, None)
        for name, default in _DEFAULTED_UNLESS_CARRIED:
            self.__dict__.setdefault(name, default)
        for name, wire_type, adapter in _REHYDRATORS:
            if isinstance(value := self.__dict__.get(name), wire_type):
                self.__dict__[name] = adapter.validate_python(value)
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getattribute__(self, name: str) -> Any:
        if name in _GUARDED_FIELDS and name not in object.__getattribute__(self, '__dataclass_fields__'):
            raise UserError(f'{name!r} is not available on {self.__class__.__name__!r} inside a Render child task.')
        return super().__getattribute__(name)

    def _expose_field(self, name: str) -> None:
        """Mark a framework-restored field as readable inside the child task."""
        instance_fields = object.__getattribute__(self, '__dataclass_fields__')
        instance_fields[name] = RunContext.__dataclass_fields__[name]

    @property
    def available_tool_names(self) -> set[str]:
        if (snapshot := self.__dict__.get('available_tool_names')) is not None:
            return snapshot
        return super().available_tool_names

    @property
    def active_capability_ids(self) -> set[str]:
        if (snapshot := self.__dict__.get('active_capability_ids')) is not None:
            return snapshot
        return super().active_capability_ids

    @property
    def _deferred_capability_ids(self) -> set[str]:
        if (snapshot := self.__dict__.get('_deferred_capability_ids')) is not None:
            return snapshot
        return super()._deferred_capability_ids

    @overload
    async def emit(self, event: CustomEventT, /) -> CustomEventT: ...

    @overload
    async def emit(self, event: CapabilityEventT, /) -> CapabilityEventT: ...

    async def emit(self, event: CustomEvent | CapabilityEvent, /) -> CustomEvent | CapabilityEvent:
        raise UserError(
            'Emitting events from a tool or event stream handler is not supported inside a Render child task.'
        )

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Project the serializable context state needed by child operations."""
        return {
            'run_id': ctx.run_id,
            'conversation_id': ctx.conversation_id,
            'metadata': ctx.metadata,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'tool_call_approved': ctx.tool_call_approved,
            'tool_call_metadata': ctx.tool_call_metadata,
            'retry': ctx.retry,
            'max_retries': ctx.max_retries,
            'run_step': ctx.run_step,
            'partial_output': ctx.partial_output,
            'trace_include_content': ctx.trace_include_content,
            'instrumentation_version': ctx.instrumentation_version,
            'usage': ctx.usage,
            'usage_limits': ctx.usage_limits,
            'loaded_capability_ids': ctx.loaded_capability_ids,
            'discovered_tool_names': ctx.discovered_tool_names,
            '_anchored_evidence': ctx._anchored_evidence,
            'available_tool_names': ctx.available_tool_names,
            'active_capability_ids': ctx.active_capability_ids,
            '_deferred_capability_ids': ctx._deferred_capability_ids,
            'capability_active': ctx.capability_active,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: Mapping[str, Any], deps: AgentDepsT) -> RenderRunContext[AgentDepsT]:
        """Rebuild a restricted context from its JSON projection."""
        return cls(**ctx, deps=deps)


class RenderRunContextCodec(Generic[AgentDepsT]):
    """Encode a run context and dependencies for a fresh Render process."""

    def __init__(
        self,
        *,
        deps_type: type[AgentDepsT],
        agent: AbstractAgent[AgentDepsT, Any] | None,
        run_context_type: type[RenderRunContext[Any]] = RenderRunContext,
    ) -> None:
        self._deps_type = deps_type
        self._agent = agent
        self._run_context_type = run_context_type

    def dump(self, ctx: RunContext[AgentDepsT]) -> JSONObject:
        context = self._run_context_type.serialize_run_context(ctx)
        return {
            'version': 1,
            'context': dump_json_object(dict[str, Any], context),
            'deps': cast(
                JSONValue, json.loads(json.dumps(JSON_CODEC.dump(self._deps_type, ctx.deps), allow_nan=False))
            ),
        }

    def load(self, payload: JSONObject) -> RunContext[AgentDepsT]:
        if payload.get('version') != 1:
            raise ValueError(f'Unsupported Render run-context version: {payload.get("version")!r}.')
        context_payload = payload.get('context')
        if not isinstance(context_payload, dict):
            raise TypeError('Render run-context payload requires a JSON object in `context`.')
        if 'deps' not in payload:
            raise TypeError('Render run-context payload requires `deps`.')

        context = load_json_object(to_json_object(payload['context']))
        deps_value = JSON_CODEC.load(self._deps_type, payload['deps'])
        if not isinstance(deps_value, self._deps_type):
            raise TypeError(f'Expected dependencies of type {self._deps_type.__name__}.')
        ctx = self._run_context_type.deserialize_run_context(context, deps=deps_value)
        if self._agent is not None:
            ctx.__dict__['agent'] = self._agent
            ctx.__dict__['root_capability'] = self._agent.root_capability
            ctx.__dict__['validation_context'] = validation_context_from_agent(self._agent)(ctx)
        ctx.__dict__['pending_messages'] = EnqueueGuard(enqueue_not_supported_message('task', 'workflow'))
        return ctx
