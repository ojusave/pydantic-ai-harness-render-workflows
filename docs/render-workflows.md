---
title: Render Workflows
description: Run long-running, distributed Pydantic AI agents (research, batch document processing, monitoring, scheduled work) as independently retried model, tool, and capability task runs with managed orchestration and on-demand compute.
---

# Render Workflows

`RenderWorkflows` runs a Pydantic AI agent on a [Render Workflows](https://render.com/docs/workflows) app: constructing the agent registers one Render task definition per supported agent operation, and inside a workflow each supported invocation starts a child task run of that definition, with the retry policy, timeout, and compute plan fixed at registration. Use it when the agent's own job is long-running or distributed, so its model and tool calls need managed orchestration and on-demand task compute rather than one in-process call stack. Render retries those task runs; it does not replay the agent loop.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/)

Not every agent shape composes today. The harness's `SubAgents` capability is refused at agent construction, so native sub-agent delegation is not available on this integration today. [What this integration refuses today](#what-this-integration-refuses-today) covers why and what works instead.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[render]"
```

The extra installs the Render Python SDK. Use the top-level import:

```python
from pydantic_ai_harness import RenderWorkflows
```

A complete Blueprint-backed research agent is in [`pydantic-render-workflows-validation`](https://github.com/ojusave/pydantic-render-workflows-validation).

## Define the app and agent

Create one `Workflows` app, pass that exact object to `RenderWorkflows`, and export it for the Render worker. Construct the agent at module load time so its operation task definitions register before the worker starts.

```python {title="app.py" test="skip"}
from pydantic_ai import Agent
from render import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows


async def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'


app = Workflows()
workflows = RenderWorkflows(app)
agent = Agent(
    'openai:gpt-5.6-sol',
    name='support',
    tools=[get_weather],
    capabilities=[workflows],
)


@workflows.task
async def support(ctx: TaskContext, prompt: str) -> str:
    del ctx
    return (await agent.run(prompt)).output
```

Start the exported app with Render's normal worker command:

```bash
py-cli render-workflows app:app
```

`@workflows.task` delegates registration to `app.task` and activates the matching `TaskContext`. Agent operations in that scope use `ctx.run(...)` to start child task runs.

Attach exactly one `RenderWorkflows` to an agent. Each instance registers a full set of task definitions against the app it was given, so a second leaves the agent with two sets and nothing to say which app a run dispatches to. A second one is refused at construction.

Pass executable tools and toolsets when constructing the agent. Later or per-run executable toolsets bypass registration and are rejected inside a Render workflow. Give the agent and each toolset stable, unique names because registered task names are persisted workflow identity.

## When should a Pydantic AI agent use this?

Use it when the agent job itself needs long-running or distributed task compute. The agent shapes that fit:

- a research agent that reads a list of sources over minutes and writes a report;
- a document pipeline that grinds through a batch of files, repeating the same model and tool calls across many inputs;
- a monitor or other scheduled job, triggered by a Render cron job that starts a root task run;
- an agent that fans a question out to several delegates in parallel, written as a named `FunctionToolset` (see [What this integration refuses today](#what-this-integration-refuses-today));
- any run that needs a failed step retried on its own, or background work that should outlive the HTTP request that started it.

Do not wait until someone asks to deploy a web service on Render. A static site or ordinary web service does not register these tasks. Do not use this for Temporal-style replay, crash recovery, or resuming `agent.run` from a checkpoint: Render retries task runs, it does not replay the agent loop.

## How do I run model and tool calls as separate task runs?

Wrap the agent in `@workflows.task` and attach `RenderWorkflows` to the same `Workflows` app. Inside that scope, model requests, tool calls, event handling, and `@durable_operation` methods dispatch as child task runs. Attaching the capability alone does not route every `agent.run` through Render.

## Task definitions and child task runs

Registration and invocation are separate events, and they scale differently.

**At construction**, binding the capability registers one task definition per supported operation:

- model requests, buffered stream requests, compaction, and suspended-response cleanup;
- per named function toolset, argument validation and tool calls;
- per MCP and dynamic toolset, discovery, instructions, validation, and calls;
- `event_stream_handler` delivery;
- each method another capability declares with `@durable_operation`.

Definitions register once, whatever the agent later does. Render currently limits a workflow service to 500 task definitions, so count the workflow entry task plus the generated model, toolset, event, and capability definitions when estimating service size.

**At run time**, inside a `workflows.task` scope, each supported invocation starts a child task run of the already-registered definition: one child task run per model request, per tool call, per event delivery, per durable-operation call. Child task runs are unbounded by the definition count and are what Render retries, times out, schedules, and bills.

Outside a `workflows.task` scope, these operations call their original Pydantic AI handlers inline, so the same agent still runs in local tests and in non-workflow code.

## What this integration refuses today

Render task names are persisted workflow identity, so every registered leaf toolset needs a stable `id`. A capability that builds its own toolset internally leaves that toolset unnamed, and Pydantic AI publishes no stable-ID assignment API for naming it afterwards. Rather than derive a name from the capability or write the toolset's private field, `RenderWorkflows` raises a `UserError` at agent construction that names each offending capability and says how to fix it.

Today that error covers the harness's `SubAgents` and `ToolOutputLimits`, and any other capability that contributes an unnamed `FunctionToolset` or `DynamicToolset`. The error is raised at construction, before any task definition is registered, so the failure is a build-time message rather than a half-registered service.

Neither `SubAgents` nor `ToolOutputLimits` exposes a public toolset-id knob: the `id` field on each names the capability, not the toolset it builds, so there is nothing to pass through to fix the refusal. The way forward is to own the toolset yourself and attach the tools to the agent as `toolsets=[FunctionToolset(..., id='...')]`, which puts the registered name under your control. That gives you a named toolset, not the refused capability: a delegate tool you write does not reproduce `SubAgents`' delegation contract, and tools you name yourself do not reproduce `ToolOutputLimits`' measurement, spill, and read-back behavior.

A toolset that already carries an `id` registers under it. Two toolsets sharing one `id` reach Pydantic AI's own uniqueness check and raise there; nothing is renamed or disambiguated for you.

Delegation you write yourself as a named `FunctionToolset` registers and runs, and the boundary above is what makes its distributed form narrower than an in-process one. Because a delegate executes in its own task run and only its JSON return crosses back, expect its usage delta not to reach the parent, its buffered child events not to be replayed to the parent, and `max_calls` to have no shared counter across distributed delegates. Those are consequences of the documented JSON boundary rather than behavior measured here: the harness's `SubAgents` is refused at construction, so its distributed behavior has not been exercised at all. Carry the accounting and telemetry you need in the delegate's own return value.

## Task options and tool opt-out

Use Render `Options` for the model, tool, event, and capability task definitions:

```python {test="skip"}
from render import Options, Retry, Workflows

from pydantic_ai_harness import RenderWorkflows


def resolve_tool_options(_operation_id, _tool, tool_name):
    if tool_name == 'read_local_cache':
        return False
    return None


app = Workflows()
workflows = RenderWorkflows(
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

Configure the workflow entry task separately, for example `@workflows.task(timeout_seconds=600, plan='flex')`.

Render fixes task options at registration. One call task definition serves every tool in its toolset, so retry, timeout, and plan cannot vary per invocation within a toolset. Returning different `Options` from the resolver raises `UserError`. Returning `None` keeps `tool_options`.

To give static function tools distinct definitions and distinct options, put each one in its own named `FunctionToolset`: the split comes from how you group the tools, not automatically per tool inside a single toolset.

Returning `False` runs a supported static function tool inside the workflow entry task, with no child task run of its own, and so no independent retry, timeout, plan, or task record. It is rejected for MCP and dynamic tools.

## JSON and dependency boundary

A child task run may execute in a fresh process. The capability sends one versioned JSON object and reconstructs the supported Pydantic AI state on the receiving side.

- Dependencies must round-trip through Pydantic's JSON codec. `deps_type` defaults to the agent's dependency type.
- Messages, model settings, metadata, tool definitions and arguments, events, capability arguments, and results that cross the boundary must be JSON encodable.
- Model instances do not cross. The default model and entries in `models={...}` are registered by ID and resolved in the child task.
- Render caps the total arguments of one task run at 4 MB ([additional limits](https://render.com/docs/workflows-limits#additional-limits)). The capability sizes the final JSON envelope and raises before dispatch rather than sending an oversized call.
- Live in-process objects are unavailable unless the reconstructed run context explicitly supports them.

## What Render retains

Everything that crosses the task boundary is task state that Render stores: prompts, model responses, tool arguments and results, dependencies, and operation metadata. Render keeps task state for 30 days and then deletes it (see [task state retention](https://render.com/docs/workflows-limits#task-state-retention)).

Keep secrets out of it. Read API keys, tokens, and credentials from environment variables inside the child task instead of passing them through `deps`, tool arguments, or task results.

## Toolset ids, models, and task-run lineage

Render identifies a leaf toolset's task definitions by its `id`, as described in [What this integration refuses today](#what-this-integration-refuses-today).

Model instances do not cross the boundary, but their ids do. A child task resolves the run's model id against the models registered in its own process (the agent's default model plus the `models={...}` entries) and reports that instance as `ctx.model`, which is what lets a tool or another capability read the model inside a child task. It resolves to the plain model rather than the workflow-side model wrapper, so that work stays in the task run already executing it. A plain-string default that each run resolves for itself has no registered instance, and `ctx.model` remains unavailable in a child task.

Render owns task-run lineage. This integration spawns children through `TaskContext.run()` and cannot assign `parentTaskRunId` or `rootTaskRunId` itself. Code that draws a run graph should read `rootTaskRunId` where the platform populates it, keep `parentTaskRunId` to work out depth, and page through every task-run listing. Where the root field comes back empty, scope the listing to the Workflow and walk parent links instead.

Coverage for this area is the deterministic tests for toolset ids, task registration, and the JSON boundary, plus the opt-in local-runtime test below.

## Why are no child task runs appearing?

The usual cause is that `agent.run(...)` is not inside the matching `@workflows.task` scope. A direct call outside that decorator runs inline. A plain `@app.task` does not activate this capability. Another cause is `resolve_tool_options` returning `False` for a static function tool: that tool runs inside the entry task and has no independent child-task record.

## What happens on retry?

Child task runs retry independently under the Render `Retry` options set at registration. A retry of one model or tool task run can repeat that task's side effects if interruption happens before Render records the result.

A retry of the workflow entry task starts `agent.run(...)` again. Work that already finished in the earlier attempt can run a second time. The current Render SDK does not let this integration assign stable idempotency keys to child calls or resume the Pydantic agent loop from a checkpoint. Retries are not replay or checkpoint resume: treat the complete agent run as at least once.

## How do I run this locally?

Start the same worker command through the Render CLI development server:

```bash
py-cli render workflows dev -- render-workflows app:app
```

That process loads the exported `Workflows` app and registers the task definitions. Direct `agent.run(...)` in ordinary tests still runs inline, which is useful when you are not exercising the workflow boundary.

The repository carries an opt-in test that drives the same local runtime end to end. It is skipped by default and needs the `render` CLI at version 2.16.0 or later, but no Render API key:

```bash
PYDANTIC_AI_HARNESS_RENDER_LOCAL_RUNTIME=1 uv run pytest tests/render/test_local_runtime.py
```

It proves exactly this much: the entry task and the generated model and tool task definitions register on a real local Render server; one root task run produces three model child task runs and two tool child task runs, all completed and all parented to the root run; the controller, the root task, the two model executions, and the tool execution report five distinct operating-system process IDs; serializable `deps` arrive intact in the tool process after a JSON round trip; and one `ModelRetry` is handled across the child-task boundary. It proves nothing about replay, checkpoint resume, or hosted Render behavior.

## Streaming and cancellation

Streaming is buffered at the model-task boundary. The child task consumes the provider stream and returns its completed response and captured events. The workflow-side agent delivers them after the child task run finishes, but provider tokens do not stream live across `ctx.run(...)`. An `event_stream_handler` follows the same operation-task path and does not change this boundary.

Render task-run cancellation remains a native client and control-plane action. Pydantic AI cancellation tokens are unsupported inside a Render workflow because they are in-process handles. Suspended-model cleanup uses its own child task run. Neither form makes external tool side effects transactional.

## Render remains the workflow runtime

Use Render's native SDK and platform behavior for synchronous and asynchronous clients, task queues, retries, timeouts, compute plans, task fan-out, cancellation, and observability. Use a Render cron job when a schedule should trigger a root task run. The registered Pydantic AI operations are ordinary Render tasks and appear in Render's status, logs, metrics, and task views.

The capability emits no additional OpenTelemetry spans. Pydantic AI's model and tool instrumentation continues to trace the agent operations, while Render records the child task runs, retries, logs, and metrics at the workflow boundary.

## Pydantic AI compatibility boundary

`RenderWorkflows` uses the public `BaseDurabilityCapability` and registered backend contracts. Pydantic AI does not yet publish every semantic parameter, transport, and bound-operation type needed by a cross-process registered backend. The integration contains those private imports in `pydantic_ai_harness/render/_compat.py`.

That containment does not make the private API stable. A Pydantic AI release can require a corresponding Harness update. In production, use a Pydantic AI and Harness combination tested together, and run the Render integration tests before upgrading either dependency independently.

## API reference

::: pydantic_ai_harness.render.RenderWorkflows
