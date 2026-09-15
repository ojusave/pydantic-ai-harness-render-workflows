from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, PartStartEvent, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_ai_harness.render._compat import (
    CapabilityOperationParams,
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams,
    JSONObject,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    RenderRunContext,
    RenderRunContextCodec,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
    get_capability_operation_declaration,
    to_json_object,
)
from pydantic_ai_harness.render._transports import (
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


@dataclass
class Deps:
    tenant: str


class ExampleCapability(AbstractCapability[Deps]):
    id = 'example'

    @durable_operation(name='echo')
    async def echo(self, ctx: RunContext[Deps], value: str) -> str:
        return f'{ctx.deps.tenant}:{value}'


class DefinitionResolver:
    def __init__(self, tool: ToolsetTool[Deps]) -> None:
        self.tool = tool

    def tool_for_tool_def(self, tool_def: object, *, ctx: RunContext[Deps]) -> ToolsetTool[Deps]:
        del tool_def, ctx
        return self.tool


def _json_round_trip(payload: JSONObject) -> JSONObject:
    return to_json_object(json.loads(json.dumps(payload)))


def _run_context(*, model_id: str | None = None) -> tuple[RunContext[Deps], RenderRunContextCodec[Deps]]:
    model = TestModel()
    agent = Agent(model, name='transport-test', deps_type=Deps)
    ctx = RunContext(
        deps=Deps(tenant='acme'),
        model=model,
        _model_id=model_id,
        usage=RunUsage(requests=2, input_tokens=10),
        usage_limits=UsageLimits(request_limit=8),
        agent=agent,
        run_id='run-1',
        conversation_id='conversation-1',
        metadata={'region': 'oregon'},
        retries={'lookup': 1},
        tool_name='lookup',
        tool_call_id='call-1',
        run_step=3,
    )
    return ctx, RenderRunContextCodec(deps_type=Deps, agent=agent)


class TestRenderTransports:
    def test_model_request_crosses_json_and_rebuilds_context(self) -> None:
        ctx, codec = _run_context()
        transport = RenderModelRequestTransport(codec, result_type=ModelResponse)
        params = ModelRequestParams(
            'tenant-model',
            messages=[ModelRequest(parts=[UserPromptPart('hello')])],
            model_settings=ModelSettings(temperature=0.2),
            model_request_parameters=ModelRequestParameters(),
            run_context=ctx,
        )

        loaded = transport.load(_json_round_trip(transport.dump(params)), runtime=object())

        assert loaded.model_id == 'tenant-model'
        assert loaded.messages == params.messages
        assert loaded.model_settings == params.model_settings
        assert loaded.run_context.deps == Deps(tenant='acme')
        assert loaded.run_context.run_id == 'run-1'
        assert loaded.run_context.usage == RunUsage(requests=2, input_tokens=10)
        assert loaded.run_context.usage_limits == UsageLimits(request_limit=8)
        assert loaded.run_context.agent is ctx.agent
        assert isinstance(loaded.run_context, RenderRunContext)
        loaded.run_context.model = ctx.model
        loaded.run_context._expose_field('model')
        assert loaded.run_context.model is ctx.model

    @pytest.mark.anyio
    async def test_static_function_tool_crosses_json_and_remains_callable(self) -> None:
        ctx, codec = _run_context()
        toolset = FunctionToolset[Deps](id='functions')

        @toolset.tool
        async def greet(context: RunContext[Deps], name: str) -> str:
            return f'{context.deps.tenant}:{name}'

        tool = (await toolset.get_tools(ctx))['greet']
        transport = RenderFunctionCallTransport(codec, toolset)
        params = ToolsetCallToolParams(
            'greet',
            tool_args={'name': 'Lais'},
            ctx=ctx,
            tool=tool,
        )

        loaded = transport.load(_json_round_trip(transport.dump(params)), runtime=object())

        assert loaded.tool is not None
        assert loaded.tool.tool_def == tool.tool_def
        assert loaded.ctx.deps == Deps(tenant='acme')
        assert await toolset.call_tool(loaded.name, loaded.tool_args, loaded.ctx, loaded.tool) == 'acme:Lais'

    @pytest.mark.anyio
    async def test_tool_discovery_dynamic_call_and_mcp_call_cross_json(self) -> None:
        ctx, codec = _run_context()
        get_tools = RenderGetToolsTransport(codec, result_type=dict[str, object])
        loaded_get = get_tools.load(
            _json_round_trip(get_tools.dump(ToolsetGetToolsParams(ctx))),
            runtime=object(),
        )
        assert loaded_get.ctx.deps == Deps(tenant='acme')

        dynamic_definition = ToolDefinition(
            name='dynamic',
            parameters_json_schema={
                'type': 'object',
                'properties': {'count': {'type': 'integer'}},
                'required': ['count'],
            },
        )
        dynamic = RenderDynamicCallTransport(codec)
        loaded_dynamic = dynamic.load(
            _json_round_trip(
                dynamic.dump(
                    DynamicToolsetCallToolParams(
                        'dynamic',
                        tool_args={'count': 4},
                        ctx=ctx,
                        tool_def=dynamic_definition,
                    )
                )
            ),
            runtime=object(),
        )
        assert loaded_dynamic.tool_args == {'count': 4}
        assert loaded_dynamic.tool_def == dynamic_definition

        dynamic_get = RenderDynamicGetToolsTransport(codec)
        assert dynamic_get.load(
            _json_round_trip(dynamic_get.dump(ToolsetGetToolsParams(ctx))), runtime=object()
        ).ctx.deps == Deps(tenant='acme')

        function_toolset = FunctionToolset[Deps](id='mcp-fixture')

        @function_toolset.tool
        async def lookup(context: RunContext[Deps], key: str) -> str:
            return f'{context.deps.tenant}:{key}'

        tool = (await function_toolset.get_tools(ctx))['lookup']
        mcp = RenderMCPCallTransport(codec, DefinitionResolver(tool))
        loaded_mcp = mcp.load(
            _json_round_trip(
                mcp.dump(ToolsetCallToolParams('lookup', tool_args={'key': 'status'}, ctx=ctx, tool=tool))
            ),
            runtime=object(),
        )
        assert loaded_mcp.tool is tool
        assert loaded_mcp.ctx.deps == Deps(tenant='acme')

    def test_model_auxiliary_operations_and_event_cross_json(self) -> None:
        ctx, codec = _run_context()
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('summarize')])]
        request_context = ModelRequestContext(
            model=TestModel(),
            messages=messages,
            model_settings=ModelSettings(max_tokens=30),
            model_request_parameters=ModelRequestParameters(),
            model_id='summary-model',
            streaming=True,
        )
        compact = RenderCompactMessagesTransport(codec)
        loaded_compact = compact.load(
            _json_round_trip(
                compact.dump(
                    ModelCompactMessagesParams(
                        'summary-model',
                        request_context=request_context,
                        instructions='Keep decisions.',
                        run_context=ctx,
                    )
                )
            ),
            runtime=object(),
        )
        assert loaded_compact.instructions == 'Keep decisions.'
        assert loaded_compact.request_context.messages == messages
        assert loaded_compact.request_context.model_id == 'summary-model'
        assert loaded_compact.request_context.streaming is True

        response = ModelResponse(parts=[TextPart('paused')], model_name='test-model')
        cancel = RenderCancelTransport(codec)
        loaded_cancel = cancel.load(
            _json_round_trip(
                cancel.dump(ModelCancelSuspendedResponseParams('summary-model', response=response, run_context=ctx))
            ),
            runtime=object(),
        )
        assert loaded_cancel.response == response
        assert loaded_cancel.run_context is not None
        assert loaded_cancel.run_context.deps == Deps(tenant='acme')

        loaded_without_context = cancel.load(
            _json_round_trip(
                cancel.dump(ModelCancelSuspendedResponseParams('summary-model', response=response, run_context=None))
            ),
            runtime=object(),
        )
        assert loaded_without_context.run_context is None

        event = PartStartEvent(index=0, part=TextPart('streamed'))
        event_transport = RenderEventStreamHandlerTransport(codec)
        loaded_event = event_transport.load(
            _json_round_trip(event_transport.dump(EventStreamHandlerParams(event, run_context=ctx))),
            runtime=object(),
        )
        assert loaded_event.event == event
        assert loaded_event.run_context.deps == Deps(tenant='acme')

    def test_capability_operation_crosses_json(self) -> None:
        ctx, codec = _run_context()
        declaration = get_capability_operation_declaration(ExampleCapability(), 'echo')
        transport = RenderCapabilityOperationTransport(codec, declaration)

        loaded = transport.load(
            _json_round_trip(
                transport.dump(CapabilityOperationParams(ctx, arguments={'value': 'hello'}, model_id='tenant-model'))
            ),
            runtime=object(),
        )

        assert loaded.arguments == {'value': 'hello'}
        assert loaded.model_id == 'tenant-model'
        assert loaded.run_context.deps == Deps(tenant='acme')

    def test_child_context_reports_the_model_registered_under_the_run_model_id(self) -> None:
        ctx, _ = _run_context(model_id='tenant-model')
        replacement = TestModel()
        codec = RenderRunContextCodec(
            deps_type=Deps,
            agent=ctx.agent,
            resolve_model=lambda model_id: replacement if model_id == 'tenant-model' else None,
        )
        transport = RenderGetToolsTransport(codec, result_type=dict[str, ToolDefinition])

        loaded = transport.load(_json_round_trip(transport.dump(ToolsetGetToolsParams(ctx))), runtime=object())

        assert loaded.ctx.model_id == 'tenant-model'
        assert loaded.ctx.model is replacement

    def test_child_context_keeps_the_model_guarded_when_the_id_resolves_to_nothing(self) -> None:
        ctx, _ = _run_context()
        codec = RenderRunContextCodec(deps_type=Deps, agent=ctx.agent, resolve_model=lambda _model_id: None)
        transport = RenderGetToolsTransport(codec, result_type=dict[str, ToolDefinition])

        loaded = transport.load(_json_round_trip(transport.dump(ToolsetGetToolsParams(ctx))), runtime=object())

        # A model this process cannot name is better than a wrong one: the restriction stands
        # and says so.
        with pytest.raises(UserError, match="'model' is not available"):
            _ = loaded.ctx.model

    def test_model_response_result_type_is_declared(self) -> None:
        _, codec = _run_context()
        transport = RenderModelRequestTransport(codec, result_type=ModelResponse)

        assert transport.result_type is ModelResponse
        assert ModelResponse(parts=[TextPart('ok')]).parts == [TextPart('ok')]
