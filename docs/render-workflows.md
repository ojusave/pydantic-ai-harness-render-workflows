---
title: Render Workflows
description: Run Pydantic AI model requests and tool calls as registered Render Workflows tasks.
---

# Render Workflows

Use this capability when a Pydantic AI agent should run its model requests, tool calls, event handling, and capability operations as tasks on a [Render Workflows](https://render.com/docs/workflows) app. `RenderWorkflows` registers those tasks with Render and supplies their Pydantic AI operation boundaries and JSON transport.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[render]"
```

The extra installs the Render Python SDK. Use the top-level import:

```python
from pydantic_ai_harness import RenderWorkflows
```

## Define the app and agent

Create one `Workflows` app, pass that exact object to `RenderWorkflows`, and export it for the Render worker. Construct the agent at module load time so its generated operation tasks register before the worker starts.

```python {title="app.py" test="skip"}
from pydantic_ai import Agent
from render import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows


async def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'


app = Workflows()
durability = RenderWorkflows(app)
agent = Agent(
    'openai:gpt-5.6-sol',
    name='support',
    tools=[get_weather],
    capabilities=[durability],
)


@durability.task
async def support(ctx: TaskContext, prompt: str) -> str:
    del ctx
    return (await agent.run(prompt)).output
```

Start the exported app with Render's normal worker command:

```bash
py-cli render-workflows app:app
```

For local execution, start that command through the Render CLI development server:

```bash
py-cli render workflows dev -- render-workflows app:app
```

`@durability.task` delegates registration to `app.task` and activates the matching `TaskContext`. Agent operations in that scope use `ctx.run(...)` to reach generated child tasks.

> **Execution boundary:** Attaching the capability alone does not route every call through Render. A direct `agent.run(...)` outside the matching `@durability.task` scope runs inline. A plain `@app.task` does not activate this capability.

Pass executable tools and toolsets when constructing the agent. Later or per-run executable toolsets bypass pre-registration and are rejected inside a Render workflow. Give the agent and each toolset stable, unique names because generated task names are persisted workflow identity.

## Generated tasks

The capability registers the operation types used by the agent:

- model requests, buffered stream requests, compaction, and suspended-response cleanup;
- static function-tool argument validation and calls;
- MCP and dynamic-toolset discovery, instructions, validation, and calls;
- `event_stream_handler` delivery;
- durable operations contributed by other capabilities.

Outside a `durability.task` scope, these operations call their original Pydantic AI handlers inline. The same agent can therefore run in local tests or non-workflow code.

Each operation consumes one registered task. Render currently limits a workflow service to 500 tasks, so include generated agent, model, toolset, event, and capability tasks when estimating service size.

## Task options and tool opt-out

Use Render `Options` for the generated model, tool, event, and capability tasks:

```python {test="skip"}
from render import Options, Retry, Workflows

from pydantic_ai_harness.render import RenderWorkflows


def resolve_tool_options(_operation_id, _tool, tool_name):
    if tool_name == 'read_local_cache':
        return False
    return None


app = Workflows()
durability = RenderWorkflows(
    app,
    model_options=Options(
        retry=Retry(max_retries=3, wait_duration_ms=1_000),
        timeout_seconds=300,
        plan='flex',
    ),
    tool_options=Options(timeout_seconds=120, plan='2c-4g'),
    event_options=Options(timeout_seconds=60),
    capability_options=Options(timeout_seconds=120),
    resolve_tool_options=resolve_tool_options,
)
```

Configure the workflow entry task separately, for example `@durability.task(timeout_seconds=600, plan='flex')`.

Render fixes task options during registration. A generated call task can serve multiple tools in a toolset, so retry, timeout, and plan cannot vary per invocation. Returning different `Options` from the resolver raises `UserError`. Returning `None` keeps `tool_options`. Returning `False` runs a supported static function tool inside the workflow entry task, without an independent child-task retry, timeout, plan, or task record.

## JSON and dependency boundary

A child task may run in a fresh process. The capability sends one versioned JSON object and reconstructs the supported Pydantic AI state on the receiving side.

- Dependencies must round-trip through Pydantic's JSON codec. `deps_type` defaults to the agent's dependency type.
- Messages, model settings, metadata, tool definitions and arguments, events, capability arguments, and results that cross the boundary must be JSON encodable.
- Model instances do not cross. The default model and entries in `models={...}` are registered by ID and resolved in the child task.
- Render task arguments have a 4 MiB limit. The capability checks the final JSON envelope before dispatch.
- Live in-process objects are unavailable unless the reconstructed run context explicitly supports them.

Treat the complete agent run and tool side effects as at least once. A retry of an individual child task can repeat its external side effect if interruption occurs before Render records the result. A retry of the workflow entry task starts `agent.run(...)` again and can repeat model requests and tool calls that completed during the earlier attempt. The current Render SDK does not let this integration assign stable idempotency keys to child task calls or resume the Pydantic agent loop from a checkpoint.

## Capability toolsets, models, and task lineage

Render identifies a leaf toolset's tasks by its `id`, and a capability that builds its own toolset leaves that toolset unnamed. Binding names each unnamed capability-contributed leaf after the capability that owns it, so `SubAgents(id='sub_agents')` registers `<agent>__function_toolset__sub_agents.call_tool`. A toolset that came with an `id` keeps it, an `id` another toolset already uses gets a numbered variant, and an unnamed leaf under a capability with no `id` still raises the error that says how to name it. Task names are persisted journal data, so changing a capability's `id` strands runs recorded under the old name.

Model instances do not cross the boundary, but their ids do. A child task resolves the run's model id against the models registered in its own process (the agent's default model plus the `models={...}` entries) and reports that instance as `ctx.model`, which is what lets a tool, a delegated sub-agent, or another capability read the model inside a child task. It resolves to the plain model rather than the workflow side's durable wrapper, so that work stays in the task already running it. A plain-string default that each run resolves for itself has no registered instance, and `ctx.model` remains unavailable in a child task.

Render owns task-run lineage. This integration spawns children through `TaskContext.run()` and cannot assign `parentTaskRunId` or `rootTaskRunId` itself. Code that draws a run graph should read `rootTaskRunId` where the platform populates it, keep `parentTaskRunId` to work out depth, and page through every task-run listing. Where the root field comes back empty, scope the listing to the Workflow and walk parent links instead.

Coverage for this area works in two layers: deterministic tests for toolset ids, task registration, and the JSON boundary, then an opt-in `render workflows dev` run that starts a root task and checks the model and tool child tasks across a real process boundary.

## Streaming and cancellation

Streaming is buffered at the model-task boundary. The child task consumes the provider stream and returns its completed response and captured events. The workflow-side agent can replay them after the child task finishes, but provider tokens do not stream live across `ctx.run(...)`. An `event_stream_handler` follows the durable operation path and does not change this boundary.

Render task-run cancellation remains a native client and control-plane action. Pydantic AI cancellation tokens are unsupported inside a Render workflow because they are in-process handles. Suspended-model cleanup uses its own generated child task. Neither form makes external tool side effects transactional.

## Render remains the workflow runtime

Use Render's native SDK and platform behavior for synchronous and asynchronous clients, task queues, retries, timeouts, compute plans, task fan-out, cancellation, and observability. Use a Render cron job when a schedule should trigger a root task. Generated Pydantic AI operations are ordinary Render tasks and appear in Render's status, logs, metrics, and task views.

The capability emits no additional OpenTelemetry spans. Pydantic AI's model and tool instrumentation continues to trace the agent operations, while Render records the child task runs, retries, logs, and metrics at the workflow boundary.

## Pydantic AI compatibility boundary

`RenderWorkflows` uses the public `BaseDurabilityCapability` and durable backend contracts. Pydantic AI does not yet publish every semantic parameter, transport, and bound-operation type needed by a cross-process registered backend. The integration contains those private imports in `pydantic_ai_harness/render/_compat.py`.

That containment does not make the private API stable. A Pydantic AI release can require a corresponding Harness update. In production, use a Pydantic AI and Harness combination tested together, and run the Render integration tests before upgrading either dependency independently.
