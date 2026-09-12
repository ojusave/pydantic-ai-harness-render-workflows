from __future__ import annotations

import pytest
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    ToolFailed,
)
from pydantic_ai.messages import ModelResponse, TextPart

from pydantic_ai_harness.render._protocol import (
    MAX_ARGUMENT_BYTES,
    RenderPayloadTooLargeError,
    RenderProtocolError,
    control_flow_error,
    make_request,
    read_request,
    read_result,
    success,
)


def test_request_round_trip() -> None:
    request = make_request('agent.model.request', {'messages': ['hello']})

    assert request == {
        'version': 1,
        'operation': 'agent.model.request',
        'payload': {'messages': ['hello']},
    }
    assert read_request(request, expected_operation='agent.model.request') == {'messages': ['hello']}


def test_request_rejects_wrong_operation() -> None:
    request = make_request('one', {})

    with pytest.raises(RenderProtocolError, match="routed to 'two', but names operation 'one'"):
        read_request(request, expected_operation='two')


def test_request_rejects_non_json_payload() -> None:
    with pytest.raises(RenderProtocolError, match='Operation payload must be JSON serializable'):
        make_request('operation', object())


def test_request_rejects_oversized_final_argument_shape() -> None:
    with pytest.raises(RenderPayloadTooLargeError, match=str(MAX_ARGUMENT_BYTES)):
        make_request('operation', {'value': 'x' * MAX_ARGUMENT_BYTES})


def test_success_round_trip() -> None:
    assert read_result(success({'answer': 42})) == {'answer': 42}


@pytest.mark.parametrize(
    ('source', 'expected_type', 'expected_value'),
    [
        (ModelRetry('try again'), ModelRetry, 'try again'),
        (ToolFailed('not found'), ToolFailed, 'not found'),
        (ApprovalRequired(metadata={'ticket': 7}), ApprovalRequired, {'ticket': 7}),
        (CallDeferred(metadata={'queue': 'slow'}), CallDeferred, {'queue': 'slow'}),
    ],
)
def test_control_flow_round_trip(
    source: Exception, expected_type: type[Exception], expected_value: str | dict[str, object]
) -> None:
    envelope = control_flow_error(source)
    assert envelope is not None

    with pytest.raises(expected_type) as exc_info:
        read_result(envelope)

    if isinstance(exc_info.value, ModelRetry | ToolFailed):
        assert exc_info.value.message == expected_value
    else:
        assert isinstance(exc_info.value, ApprovalRequired | CallDeferred)
        assert exc_info.value.metadata == expected_value


def test_unexpected_exception_is_not_encoded() -> None:
    assert control_flow_error(RuntimeError('boom')) is None


def test_skip_model_request_round_trip() -> None:
    response = ModelResponse(parts=[TextPart(content='cached')])
    envelope = control_flow_error(SkipModelRequest(response))
    assert envelope is not None

    with pytest.raises(SkipModelRequest) as exc_info:
        read_result(envelope)

    assert exc_info.value.response == response


def test_skip_tool_validation_round_trip() -> None:
    envelope = control_flow_error(SkipToolValidation({'city': 'San Francisco'}))
    assert envelope is not None

    with pytest.raises(SkipToolValidation) as exc_info:
        read_result(envelope)

    assert exc_info.value.validated_args == {'city': 'San Francisco'}


def test_skip_tool_execution_round_trip() -> None:
    envelope = control_flow_error(SkipToolExecution({'forecast': ['sunny', 19]}))
    assert envelope is not None

    with pytest.raises(SkipToolExecution) as exc_info:
        read_result(envelope)

    assert exc_info.value.result == {'forecast': ['sunny', 19]}


@pytest.mark.parametrize(
    'envelope',
    [
        {'version': 2, 'status': 'ok', 'payload': None},
        {'version': 1, 'status': 'mystery'},
        {'version': 1, 'status': 'ok', 'payload': None, 'extra': True},
        {'version': 1, 'status': 'control-flow', 'error': {'kind': 'mystery'}},
    ],
)
def test_invalid_result_envelope(envelope: object) -> None:
    with pytest.raises(RenderProtocolError):
        read_result(envelope)
