"""Process-isolated Render Workflows fixture used by the local-runtime test."""

from __future__ import annotations

import os

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from render.workflows import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows


class RuntimeDeps(BaseModel):
    """Serializable values transported to isolated operation tasks."""

    prefix: str
    controller_pid: int


class IsolatedLookupResult(BaseModel):
    """Evidence returned after one Pydantic AI model-requested tool retry."""

    controller_pid: int
    deps_prefix: str
    model_pid: int
    model_retry_count: int
    tool_pid: int
    value: str


class AgentOutput(BaseModel):
    """Final model text payload reconstructed from serialized history."""

    final_model_pid: int
    tool_result: IsolatedLookupResult


class RootTaskResult(BaseModel):
    """Public root-task return value observed through the local CLI."""

    agent_output: AgentOutput
    root_pid: int


def runtime_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Drive one Pydantic AI ModelRetry from serialized message history."""
    del info
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart):
                payload = AgentOutput(
                    final_model_pid=os.getpid(),
                    tool_result=IsolatedLookupResult.model_validate(part.content),
                )
                return ModelResponse(parts=[TextPart(payload.model_dump_json())])
            if isinstance(part, RetryPromptPart):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'isolated_lookup',
                            {'value': 'retry-value', 'model_pid': os.getpid()},
                            tool_call_id='lookup-retry',
                        )
                    ]
                )

    return ModelResponse(
        parts=[
            ToolCallPart(
                'isolated_lookup',
                {'value': 'initial-value', 'model_pid': os.getpid()},
                tool_call_id='lookup-initial',
            )
        ]
    )


workflows = Workflows()
render_workflows = RenderWorkflows[RuntimeDeps](workflows, deps_type=RuntimeDeps)
agent = Agent[RuntimeDeps, str](
    FunctionModel(runtime_model, model_name='local-runtime-model'),
    name='local-runtime-agent',
    deps_type=RuntimeDeps,
    retries=1,
    capabilities=[render_workflows],
)


@agent.tool
async def isolated_lookup(ctx: RunContext[RuntimeDeps], value: str, model_pid: int) -> IsolatedLookupResult:
    """Return process evidence after one Pydantic AI ModelRetry."""
    if ctx.retry == 0:
        raise ModelRetry('retry once across the child-task boundary')
    return IsolatedLookupResult(
        controller_pid=ctx.deps.controller_pid,
        deps_prefix=ctx.deps.prefix,
        model_pid=model_pid,
        model_retry_count=ctx.retry,
        tool_pid=os.getpid(),
        value=value,
    )


@render_workflows.task(name='run-local-runtime-agent')
async def run_local_runtime_agent(ctx: TaskContext, prompt: str, deps: RuntimeDeps) -> dict[str, object]:
    """Run the public agent entry point inside a Render root task."""
    del ctx
    result = await agent.run(prompt, deps=RuntimeDeps.model_validate(deps))
    payload = RootTaskResult(
        agent_output=AgentOutput.model_validate_json(result.output),
        root_pid=os.getpid(),
    )
    dumped = TypeAdapter(RootTaskResult).dump_python(payload, mode='json')
    return TypeAdapter(dict[str, object]).validate_python(dumped)


if __name__ == '__main__':
    workflows.start()
