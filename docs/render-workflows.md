---
title: Render Workflows
description: Run long-running, distributed Pydantic AI agents (research, batch document processing, monitoring, scheduled work) as independently retried model, tool, and capability task runs with managed orchestration and on-demand compute.
---

# Render Workflows

`RenderWorkflows` runs a Pydantic AI agent on a [Render Workflows](https://render.com/docs/workflows) app: constructing the agent registers task definitions for supported agent operations, and inside a workflow each supported invocation starts a child task run with retry policy, timeout, and compute plan fixed at registration. Statically known function tools can receive separate definitions and options. Use it when the agent's own job is long-running or distributed, so its model and tool calls need managed orchestration and on-demand task compute rather than one in-process call stack. Render retries those task runs; it does not replay the agent loop.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/)

The supported contract is narrower than the whole agent surface. Task definitions use stable agent, toolset, and optional static-tool names, and everything an operation needs crosses as JSON. [Task names for capability toolsets](#task-names-for-capability-toolsets) and [JSON and dependency boundary](#json-and-dependency-boundary) are where that changes a design.

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
- an agent that delegates work to explicitly configured child agents whose model and tool operations run as Render tasks (see [Sub-agent delegation](#sub-agent-delegation));
- any run that needs a failed step retried on its own, or background work that should outlive the HTTP request that started it.

Do not wait until someone asks to deploy a web service on Render. A static site or ordinary web service does not register these tasks. Do not use this for Temporal-style replay, crash recovery, or resuming `agent.run` from a checkpoint: Render retries task runs, it does not replay the agent loop.

## How do I run model and tool calls as separate task runs?

Wrap the agent in `@workflows.task` and attach `RenderWorkflows` to the same `Workflows` app. Inside that scope, model requests, tool calls, event handling, and `@durable_operation` methods dispatch as child task runs. Attaching the capability alone does not route every `agent.run` through Render.

## Task definitions and child task runs

Registration and invocation are separate events, and they scale differently.

**At construction**, binding the capability registers one task definition per supported operation:

- model requests, buffered stream requests, compaction, and suspended-response cleanup;
- per function toolset by default, or per statically known function tool when its resolved options differ, argument validation and tool calls;
- per MCP and dynamic toolset, discovery, instructions, validation, and calls;
- `event_stream_handler` delivery;
- each method another capability declares with `@durable_operation`.

Definitions register once, whatever the agent later does. Render currently limits a workflow service to 500 task definitions, so count the workflow entry task plus the generated model, toolset, event, and capability definitions when estimating service size.

**At run time**, inside a `workflows.task` scope, each supported invocation starts a child task run of the already-registered definition: one child task run per model request, per tool call, per event delivery, per durable-operation call. Child task runs are unbounded by the definition count and are what Render retries, times out, schedules, and bills.

Outside a `workflows.task` scope, these operations call their original Pydantic AI handlers inline, so the same agent still runs in local tests and in non-workflow code.

## Task names for capability toolsets

Render task names are persisted workflow identity, so every registered leaf toolset needs a stable `id`.

An `id` you set yourself always wins: that toolset registers under the name you gave it and nothing is derived for it. Two toolsets sharing one `id` reach Pydantic AI's own uniqueness check and raise there; nothing is renamed or disambiguated for you.

A capability that builds its own toolset internally can leave that toolset unnamed. Pydantic AI has no public API for assigning an `id` after construction, so `RenderWorkflows` does not mutate one into place. An unnamed capability-owned leaf stays inline in the workflow entry task and receives no independent retry, timeout, compute, logs, or run history. A capability-owned toolset that already has an explicit `id` registers normally.

An unnamed supported leaf you attach yourself is rejected before any task definition registers. Give user-owned function, MCP, and dynamic toolsets stable IDs at construction.

## Sub-agent delegation

`SubAgents` leaves its internal `delegate_task` toolset unnamed, so delegation itself stays inline in the workflow entry task. This preserves its parent-side `max_calls` check and event handling without assigning private Pydantic state.

To run a delegate's supported model and tool operations as Render task runs, construct that child `Agent` with its own `RenderWorkflows` instance using the same `Workflows` app as the parent. The app-scoped active `TaskContext` is then available to the explicitly configured child. Its model and named or agent-owned function tools register at construction and dispatch through `TaskContext.run()` during delegation. A child without `RenderWorkflows`, a child using another app, or a child built later from disk stays inline.

Successful operation results carry the child's usage delta and buffered custom or capability events in the versioned JSON envelope. The caller applies the delta once and re-emits events in order. Effects from a failed call or `ModelRetry` attempt are discarded. `SubAgent.max_calls` is enforced for concurrent delegations within the active parent task run because delegation stays in that process; it is not a global budget across retries or separate root task runs.

Immediate capability events make a synchronous decision before their emitter continues. That decision cannot be buffered across a child task, so the integration fails those events closed. Keep a tool that emits an immediate event inline.

## Large tool outputs

`ToolOutputLimits` also contributes an unnamed helper toolset, so that helper remains inline. It measures and reduces a tool return after the registered tool task returns to the workflow entry task.

Its `Spill` mode is the part to design around. A spill writes the full payload to a filesystem-backed store and hands the model a handle that a later `read_tool_result` call reads back. Task runs are separate processes and can execute on separate, isolated filesystems, so a handle written during one task run cannot be assumed readable by the task run that reads it, and nothing at the Render boundary shares that store for you.

Inside a workflow, have tools return bounded JSON and keep a large artifact in storage that outlives a single task run: object storage, a database, or another durable service both sides can reach. Return its key and read the artifact back through a tool that fetches it.

## Memory

Pass `Memory(...)` in the agent constructor alongside `RenderWorkflows`. Its static toolset stays registered while each run resolves its own memory scope. Snapshot loading and memory tool calls execute in child tasks.

Use a `MemoryStore` backed by a shared external service that every task instance can reach. The default `InMemoryStore` is process-local; a local file or SQLite database is not shared across hosted task instances. A `store_resolver` and callable `namespace` must reconstruct the same scope from JSON dependencies in each worker. Workflows does not make the memory backend persistent.

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

Render fixes task options at registration. The resolver first receives `tool=None` and `tool_name=''` for the toolset default. Each statically known function tool is then resolved with its concrete tool and name. Returning `None` keeps `tool_options`.

When every static tool resolves to the shared default, the toolset keeps its existing shared call and validation task definitions. When at least one resolves different `Options` or `False`, each eligible static tool receives definitions named from the agent, toolset, tool, and operation. Those definitions can have independent retries, timeouts, compute, logs, and run history. Invocation-time options must equal the registered snapshot.

Returning `False` for a static function tool registers no task for that tool and runs it inside the workflow entry task. `False` is rejected for MCP and dynamic tools because their concrete tools are not known when the Workflow service registers definitions.

## JSON and dependency boundary

A child task run may execute in a fresh process. The capability sends one versioned JSON object and reconstructs the supported Pydantic AI state on the receiving side.

Current callers write protocol v2. Workers also accept v1 requests and answer with effect-free v1 results, which lets a new worker finish work submitted by an older caller. New callers accept those v1 results.

- Dependencies must round-trip through Pydantic's JSON codec. `deps_type` defaults to the agent's dependency type.
- Messages, model settings, metadata, tool definitions and arguments, usage deltas, buffered events, capability arguments, and results that cross the boundary must be JSON encodable.
- Model instances do not cross. The default model and entries in `models={...}` are registered by ID and resolved in the child task.
- Render caps the total arguments of one task run at 4 MB ([additional limits](https://render.com/docs/workflows-limits#additional-limits)). The capability sizes the final JSON envelope and raises before dispatch rather than sending an oversized call.
- Live in-process objects are unavailable unless the reconstructed run context explicitly supports them.

## What Render retains

Everything that crosses the task boundary is task state that Render stores: prompts, model responses, tool arguments and results, dependencies, and operation metadata. Render keeps task state for 30 days and then deletes it (see [task state retention](https://render.com/docs/workflows-limits#task-state-retention)).

Keep secrets out of it. Read API keys, tokens, and credentials from environment variables inside the child task instead of passing them through `deps`, tool arguments, or task results.

## Models and task-run lineage

Render identifies a registered leaf toolset's task definitions by its explicit `id`, as described in [Task names for capability toolsets](#task-names-for-capability-toolsets).

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

The repository carries an opt-in test that drives the same local runtime end to end. It is skipped by default and needs the `render` CLI at version 2.28.0 or later, but no Render API key:

```bash
PYDANTIC_AI_HARNESS_RENDER_LOCAL_RUNTIME=1 uv run pytest tests/render/test_local_runtime.py
```

It proves exactly this much: an entry task plus parent, child, and grandchild operation definitions register on a real local Render server; one root task run produces 12 completed operation task runs parented to that root; eight reported operating-system process IDs are distinct; serializable `deps` survive the JSON boundary; sibling child tools return additive usage and ordered events exactly once; and one `ModelRetry` is handled across a grandchild tool-task boundary. A second local-runtime case verifies constructor-supplied Memory, write/read calls through separate task processes using a shared local SQLite file, and a custom `ctx.tracer` span reaching the tool worker's exporter. These tests prove nothing about replay, checkpoint resume, or hosted Render behavior. The shared local SQLite file is a test fixture, not a hosted storage design.

## Streaming and cancellation

Streaming is buffered at the model-task boundary. The child task consumes the provider stream and returns its completed response and captured events. The workflow-side agent delivers them after the child task run finishes, but provider tokens do not stream live across `ctx.run(...)`. An `event_stream_handler` follows the same operation-task path and does not change this boundary.

Render task-run cancellation remains a native client and control-plane action. Pydantic AI cancellation tokens are unsupported inside a Render workflow because they are in-process handles. Suspended-model cleanup uses its own child task run. Neither form makes external tool side effects transactional.

## Render remains the workflow runtime

Use Render's native SDK and platform behavior for synchronous and asynchronous clients, task queues, retries, timeouts, compute plans, task fan-out, cancellation, and observability. Use a Render cron job when a schedule should trigger a root task run. The registered Pydantic AI operations are ordinary Render tasks and appear in Render's status, logs, metrics, and task views.

The capability emits no additional OpenTelemetry spans. Pydantic AI's model and tool instrumentation continues to trace the agent operations, while Render records the child task runs, retries, logs, and metrics at the workflow boundary.

`ctx.tracer` is available inside child tasks. It is a no-op when tracing is disabled and otherwise uses the worker's Pydantic AI instrumentation settings, including `agent.instrument`, `Agent.instrument_all(...)`, and registered instrumented models. Configure instrumentation when each worker loads the app; tracer objects are not serialized. The caller's `trace_include_content` setting is preserved. This restores tool-local spans but does not propagate OpenTelemetry parent span context across `TaskContext.run`; those spans can be separate traces.

Resolving effective agent/global instrumentation currently uses a private Pydantic AI settings getter inside `_compat.py`, covered by the same version-compatibility tests as context reconstruction.

## Pydantic AI compatibility boundary

`RenderWorkflows` uses the public `BaseDurabilityCapability` and registered backend contracts. Pydantic AI does not yet publish every semantic parameter, transport, and bound-operation type needed by a cross-process registered backend. The integration contains those private imports in `pydantic_ai_harness/render/_compat.py`.

That containment does not make the private API stable. A Pydantic AI release can require a corresponding Harness update. In production, use a Pydantic AI and Harness combination tested together, and run the Render integration tests before upgrading either dependency independently.

## API reference

::: pydantic_ai_harness.render.RenderWorkflows
