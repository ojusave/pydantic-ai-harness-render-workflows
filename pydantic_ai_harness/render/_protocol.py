"""JSON wire protocol used by Render Workflows operation tasks."""

from __future__ import annotations

import json
from typing import Any, Literal, TypeAlias, TypedDict

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.durable_exec import JSON_CODEC
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    ToolFailed,
    UserError,
)
from pydantic_ai.messages import ModelResponse

from ._compat import JSONObject as JsonObject
from ._compat import JSONValue as JsonValue
from ._compat import dump_json_object, load_json_object, load_json_type, normalize_json_value

PROTOCOL_VERSION = 1
MAX_ARGUMENT_BYTES = 4 * 1024 * 1024

_OBJECT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


class OperationRequest(TypedDict):
    version: int
    operation: str
    payload: JsonObject


class OperationSuccess(TypedDict):
    version: int
    status: Literal['ok']
    payload: JsonValue


class OperationControlFlowError(TypedDict):
    version: int
    status: Literal['control-flow']
    error: JsonObject


class OperationPermanentError(TypedDict):
    version: int
    status: Literal['error']
    error: JsonObject


OperationResult: TypeAlias = OperationSuccess | OperationControlFlowError | OperationPermanentError


class RenderProtocolError(ValueError):
    """Raised when a Render operation receives an invalid wire envelope."""


class RenderPayloadTooLargeError(RenderProtocolError):
    """Raised before dispatch when task arguments exceed Render's limit."""


def make_request(operation: str, payload: object) -> OperationRequest:
    """Build and preflight the single positional argument sent to a Render task."""
    json_payload = _as_json_object(payload, label='operation payload')
    request: OperationRequest = {'version': PROTOCOL_VERSION, 'operation': operation, 'payload': json_payload}
    _check_argument_size(request)
    return request


def read_request(value: object, *, expected_operation: str) -> JsonObject:
    """Validate a task request and return its encoded operation parameters."""
    envelope = _object(value, label='operation request')
    _check_keys(envelope, required={'version', 'operation', 'payload'}, label='operation request')
    _check_version(envelope)
    operation = envelope['operation']
    if operation != expected_operation:
        raise RenderProtocolError(
            f'Render operation request was routed to {expected_operation!r}, but names operation {operation!r}.'
        )
    return _as_json_object(envelope['payload'], label='operation payload')


def success(payload: object) -> OperationSuccess:
    return {
        'version': PROTOCOL_VERSION,
        'status': 'ok',
        'payload': _as_json_value(payload, label='operation result'),
    }


def control_flow_error(exc: Exception) -> OperationControlFlowError | None:
    """Encode expected Pydantic AI control flow as a successful Render task result."""
    error: JsonObject
    if isinstance(exc, ModelRetry):
        error = {'kind': 'model-retry', 'message': exc.message}
    elif isinstance(exc, ToolFailed):
        error = {'kind': 'tool-failed', 'message': exc.message}
    elif isinstance(exc, ApprovalRequired):
        error = {'kind': 'approval-required', 'metadata': _as_json_value(exc.metadata, label='approval metadata')}
    elif isinstance(exc, CallDeferred):
        error = {'kind': 'call-deferred', 'metadata': _as_json_value(exc.metadata, label='deferred-call metadata')}
    elif isinstance(exc, SkipModelRequest):
        error = {'kind': 'skip-model-request', 'response': dump_json_object(ModelResponse, exc.response)}
    elif isinstance(exc, SkipToolValidation):
        error = {
            'kind': 'skip-tool-validation',
            'validated_args': dump_json_object(dict[str, Any], exc.validated_args),
        }
    elif isinstance(exc, SkipToolExecution):
        error = {
            'kind': 'skip-tool-execution',
            'result': _as_json_value(JSON_CODEC.dump(Any, exc.result), label='skipped tool result'),
        }
    else:
        return None
    return {'version': PROTOCOL_VERSION, 'status': 'control-flow', 'error': error}


def permanent_error(kind: Literal['invalid-request', 'invalid-result'], exc: Exception) -> OperationPermanentError:
    """Finish a task when retrying cannot repair its boundary data."""
    message = str(exc).strip() or type(exc).__name__
    error: JsonObject = {'kind': kind, 'message': message[:500]}
    return {'version': PROTOCOL_VERSION, 'status': 'error', 'error': error}


def read_result(value: object) -> JsonValue:
    """Decode a task result, recreating expected Pydantic AI control flow."""
    envelope = _object(value, label='operation result')
    _check_version(envelope)
    status = envelope.get('status')
    if status == 'ok':
        _check_keys(envelope, required={'version', 'status', 'payload'}, label='successful operation result')
        return _as_json_value(envelope['payload'], label='operation result payload')
    if status == 'control-flow':
        _check_keys(envelope, required={'version', 'status', 'error'}, label='control-flow operation result')
        _raise_control_flow(envelope['error'])
    if status == 'error':
        _check_keys(envelope, required={'version', 'status', 'error'}, label='failed operation result')
        _raise_permanent_error(envelope['error'])
    raise RenderProtocolError(f'Render operation result has unknown status {status!r}.')


def _raise_control_flow(value: object) -> None:
    error = _object(value, label='control-flow error')
    kind = error.get('kind')
    if kind == 'model-retry':
        _check_keys(error, required={'kind', 'message'}, label='model-retry error')
        raise ModelRetry(_string(error['message'], label='model-retry message'))
    if kind == 'tool-failed':
        _check_keys(error, required={'kind', 'message'}, label='tool-failed error')
        raise ToolFailed(_string(error['message'], label='tool-failed message'))
    if kind == 'approval-required':
        _check_keys(error, required={'kind', 'metadata'}, label='approval-required error')
        raise ApprovalRequired(metadata=_metadata(error['metadata'], label='approval metadata'))
    if kind == 'call-deferred':
        _check_keys(error, required={'kind', 'metadata'}, label='call-deferred error')
        raise CallDeferred(metadata=_metadata(error['metadata'], label='deferred-call metadata'))
    if kind == 'skip-model-request':
        _check_keys(error, required={'kind', 'response'}, label='skip-model-request error')
        raise SkipModelRequest(
            load_json_type(
                ModelResponse,
                _as_json_object(error['response'], label='skip-model-request response'),
            )
        )
    if kind == 'skip-tool-validation':
        _check_keys(error, required={'kind', 'validated_args'}, label='skip-tool-validation error')
        raise SkipToolValidation(
            load_json_object(_as_json_object(error['validated_args'], label='skip-tool-validation arguments'))
        )
    if kind == 'skip-tool-execution':
        _check_keys(error, required={'kind', 'result'}, label='skip-tool-execution error')
        raise SkipToolExecution(JSON_CODEC.load(Any, _as_json_value(error['result'], label='skipped tool result')))
    raise RenderProtocolError(f'Render operation result has unknown control-flow kind {kind!r}.')


def _raise_permanent_error(value: object) -> None:
    error = _object(value, label='operation error')
    _check_keys(error, required={'kind', 'message'}, label='operation error')
    kind = error['kind']
    if kind not in ('invalid-request', 'invalid-result'):
        raise RenderProtocolError(f'Render operation result has unknown error kind {kind!r}.')
    message = _string(error['message'], label='operation error message')
    raise UserError(f'Render operation {kind.replace("-", " ")}: {message}')


def _check_argument_size(request: OperationRequest) -> None:
    # TaskContext.run serializes positional arguments as a JSON list. Measure that final shape,
    # including JSON punctuation and whitespace, instead of only measuring the semantic payload.
    encoded = _json_bytes([request], label='operation request')
    if len(encoded) > MAX_ARGUMENT_BYTES:
        raise RenderPayloadTooLargeError(
            f'Render operation arguments are {len(encoded)} bytes, exceeding the {MAX_ARGUMENT_BYTES}-byte limit.'
        )


def _as_json_value(value: object, *, label: str) -> JsonValue:
    try:
        return normalize_json_value(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RenderProtocolError(f'{label.capitalize()} must be JSON serializable: {exc}') from exc


def _as_json_object(value: object, *, label: str) -> JsonObject:
    json_value = _as_json_value(value, label=label)
    if not isinstance(json_value, dict):
        raise RenderProtocolError(f'{label.capitalize()} must be a JSON object.')
    return _OBJECT_ADAPTER.validate_python(json_value, strict=True)


def _json_bytes(value: object, *, label: str) -> bytes:
    normalized = _as_json_value(value, label=label)
    try:
        return json.dumps(normalized, allow_nan=False).encode()
    except (OverflowError, TypeError, ValueError) as exc:
        raise RenderProtocolError(f'{label.capitalize()} must be JSON serializable: {exc}') from exc


def _object(value: object, *, label: str) -> dict[str, object]:
    try:
        return _OBJECT_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise RenderProtocolError(f'{label.capitalize()} must be a JSON object.') from exc


def _check_keys(value: dict[str, object], *, required: set[str], label: str) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        details: list[str] = []
        if missing:
            details.append(f'missing {missing!r}')
        if extra:
            details.append(f'unexpected {extra!r}')
        raise RenderProtocolError(f'{label.capitalize()} has invalid fields: {", ".join(details)}.')


def _check_version(value: dict[str, object]) -> None:
    version = value.get('version')
    if version != PROTOCOL_VERSION:
        raise RenderProtocolError(
            f'Render operation protocol version {version!r} is unsupported; expected {PROTOCOL_VERSION}.'
        )


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RenderProtocolError(f'{label.capitalize()} must be a string.')
    return value


def _metadata(value: object, *, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    return _object(value, label=label)
