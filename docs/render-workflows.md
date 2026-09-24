---
title: Run AI agents in the background with retries and timeouts
description: Run long-running LLM tasks outside HTTP requests, retry failed tool calls, and check background job status with Pydantic AI and Render Workflows.
---

# Render Workflows

Run long-running AI agents as background jobs when their model requests and tool calls need separate retries, timeouts, or compute settings. The [Render Workflows](https://render.com/docs/workflows) integration keeps the Pydantic AI agent loop in an entry task and runs supported operations as child tasks, so you can inspect each operation's status, logs, and result.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/) | [Detailed reference](#task-definitions-and-child-task-runs)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Run long-running AI agents in the background

An agent might call a model to plan its work, then use a tool to process a large document. You can give that tool more time or computing power without changing the settings for the model calls.

If the tool fails, Render can retry it while the agent waits for the result. Retrying the task that runs the whole agent is different: the agent starts from the beginning and may repeat work it already completed.

If you only need background execution, a native Render task around `agent.run(...)` may be enough. Use `RenderWorkflows` when you also need to configure or inspect individual model and tool operations.

## Quickstart: run a background agent job locally

The example below uses Pydantic AI's `TestModel` and a mock weather tool to exercise task dispatch without an LLM API key or a Render deployment. The test model generates sample tool arguments and returns fixed text, so it neither interprets your prompt nor fetches real weather.

You need Python 3.11 or later and the [Render CLI](https://render.com/docs/cli) 2.28.0 or later. The installation below uses Pydantic AI 2.46.0 and Render SDK 1.2.0, the versions tested with these examples.

### 1. Install dependencies

In a new directory, initialize a project with [uv](https://docs.astral.sh/uv/) using `uv init --bare`; for an existing uv project, run the installation command directly. If you use pip, first create and activate a virtual environment.

Until a Harness release includes this integration, install it from the repository checkout containing this integration. Replace `/path/to/pydantic-ai-harness` with the checkout's absolute path:

```bash
pip/uv-add "/path/to/pydantic-ai-harness[render]" "pydantic-ai-slim==2.46.0" "render==1.2.0"
```

For a published Harness release that includes the integration, replace `/path/to/pydantic-ai-harness[render]` with `pydantic-ai-harness[render]`. The `render` extra installs the Python SDK and its `render-workflows` executable, while the Render CLI requires a separate installation.

### 2. Define the app and agent

Save the following as `app.py` in your project directory:

```python {title="app.py" names="defined"}
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from render import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows


async def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'


app = Workflows()
workflows = RenderWorkflows(app)
agent = Agent(
    TestModel(custom_output_text='The workflow completed.'),
    name='support',
    tools=[get_weather],
    capabilities=[workflows],
)


@workflows.task
async def support(ctx: TaskContext, prompt: str) -> str:
    del ctx
    return (await agent.run(prompt)).output
```

Construct the agent at module load time so its task definitions register before the worker starts, and pass the same `app` object to both the capability and the worker. The `@workflows.task` decorator activates child-task dispatch for `agent.run(...)`; a plain `@app.task` or a call outside that scope runs the agent inline.

### 3. Start the local task server

From the directory containing `app.py`, run:

```bash
py-cli render workflows dev -- render-workflows app:app
```

The server listens on port `8120` and lists the registered tasks, including `support` and `support__model.request`. Leave this terminal running while you submit the job from another terminal.

With pip, run `render workflows dev -- render-workflows app:app` from your activated environment. The [local development guide](https://render.com/docs/workflows-local-development) covers port options and the local server's limits.

### 4. Start a background job and check its status

In another terminal, submit the entry task:

```bash
render workflows tasks runs start support --local --input '["Check the weather."]' --confirm --output json
```

Copy the returned `id` into the following command to check the job's status and retrieve its result:

```bash
render workflows tasks runs show <RUN_ID> --local --output json
```

A successful run has `status: "completed"` and `results: ["The workflow completed."]`. If the job is still running, repeat the status command until it finishes.

To inspect the model and tool calls separately, list their task runs:

```bash
render workflows tasks runs list 'support__model.request' --local --output json
render workflows tasks runs list 'support__function_toolset__<agent>.call_tool' --local --output json
```

This example produces two model requests and one weather-tool call, each with a `parentTaskRunId` matching the entry task's ID. When you finish inspecting them, stop the server with Ctrl+C; its in-memory run history is lost on shutdown.

### 5. Use a real model

Install the provider dependency for the OpenAI example:

```bash
pip/uv-add "pydantic-ai-slim[openai]==2.46.0"
```

Set `OPENAI_API_KEY` in the worker's environment, then replace `TestModel(custom_output_text='The workflow completed.')` in `app.py` with `'openai:gpt-5.6-sol'` and remove the `TestModel` import. After restarting the server, repeat step 4 to make a real provider request. The response will vary, although the weather tool will still return mock data until you replace it with an API call.

## Avoid HTTP timeouts for long-running LLM tasks

When an agent takes longer than an HTTP request can remain open, move its work to a separate worker and return a job ID from your web endpoint. The client can then poll for the result while the agent continues independently, whether your application uses FastAPI or another Python web framework.

Deploy the exported `app` as a Workflow with the start command `uv run render-workflows app:app`, and configure the provider credentials on that Workflow. Render's [workflow setup](https://render.com/docs/workflows) and [task submission](https://render.com/docs/workflows-running) guides explain deployment and authenticated calls from application code.

Your web endpoint submits `support` through Render's task client and returns its run ID, for example in an HTTP `202 Accepted` response. Store that ID with the requesting user so your status endpoint can verify ownership before returning the job's status, result, or failure, and keep Render credentials on the server. The commands in step 4 demonstrate the same submission and result retrieval locally.

## Retry failed tool calls and set task timeouts

Pass Render task options when constructing the capability to give model requests and tool calls different retry policies, timeouts, and compute plans:

```python {names="defined"}
from render import Options, Retry, Workflows

from pydantic_ai_harness import RenderWorkflows

app = Workflows()
workflows = RenderWorkflows(
    app,
    model_options=Options(
        retry=Retry(max_retries=3, wait_duration_ms=1_000),
        timeout_seconds=300,
        plan='flex',
    ),
    tool_options=Options(
        retry=Retry(max_retries=2, wait_duration_ms=1_000),
        timeout_seconds=120,
        plan='2c-4g',
    ),
)
```

Use this `workflows` instance in both the agent and its entry decorator, configuring the entry task separately with options such as `@workflows.task(timeout_seconds=600, plan='flex')`. These settings are fixed when the tasks register, so they cannot change between invocations.

To give a statically known function tool its own options, use `resolve_tool_options`; returning `False` keeps that tool inline. The [per-tool options reference](#task-options-and-tool-opt-out) explains the resolver contract and the restrictions for MCP and dynamic tools.

Render task retries are separate from Pydantic AI's `ModelRetry`, which asks the model to correct or reconsider a call, and from any retries performed by the provider SDK. Account for all three when setting retry limits and timeouts; the [Pydantic AI retry guide](/ai/core-concepts/retries/) explains the model and provider behavior.

## What happens when an agent task fails?

A failed child task can retry under its registered policy while the entry task waits for a result. If an attempt performs an external action before its result is recorded, a retry can repeat that action, so use an application-level idempotency key or another deduplication mechanism for writes and API requests that must not happen twice.

Retrying the entry task restarts `agent.run(...)` and can repeat model or tool calls that finished in the earlier attempt. The integration does not resume from a checkpoint, replay completed steps, or guarantee exactly-once side effects; design the complete agent run to tolerate repeated execution.

## Constraints to check before deployment

- **Registration:** Attach one `RenderWorkflows` instance per agent and supply executable tools at construction. Explicit toolset IDs must be stable and unique, while unnamed capability-owned tools can remain inline without a child-task record.
- **Task inputs:** Dependencies, messages, arguments, and results must be JSON serializable, and Render limits a task run's arguments to 4 MB. Pass resource IDs across the boundary and reconstruct clients inside workers.
- **Storage:** Hosted task instances do not share local memory or files, so agent memory and large artifacts need a shared external backend. Read credentials from worker environment variables rather than passing them through task inputs or results.
- **Streaming and cancellation:** Model responses are buffered until their child task finishes, so live tokens do not stream across the task boundary. Pydantic AI cancellation tokens are unsupported in a workflow; use Render's native task cancellation instead.
- **Compatibility:** The adapter uses private Pydantic AI APIs, so test the integration before upgrading its dependencies. The local quickstart verifies task dispatch, but does not establish hosted failure recovery or performance.

Render retains task inputs and results for 30 days and allows up to 500 task definitions per workflow. Check the current [limits and pricing](https://render.com/docs/workflows-limits) when deciding how many operations to register as separate tasks.

## Task definitions and child task runs

When the agent is constructed, binding the capability registers task definitions for its supported operations:

- model requests, buffered stream requests, compaction, and suspended-response cleanup;
- per function toolset by default, or per statically known function tool when its resolved options differ, argument validation and tool calls;
- per MCP and dynamic toolset, discovery, instructions, validation, and calls;
- `event_stream_handler` delivery;
- each method another capability declares with `@durable_operation`.

Task definitions register once, even when an operation runs many times. Count the entry task and the generated model, toolset, event, and capability definitions against Render's limit of 500 definitions per workflow.

Inside a `workflows.task` scope, each supported invocation starts a child run of its registered definition: a model request, tool call, event delivery, or durable-operation call. The number of runs is separate from the definition limit, and Render schedules, retries, times out, and bills those runs individually.

Outside a `workflows.task` scope, these operations call their original Pydantic AI handlers inline, so the same agent still runs in local tests and in non-workflow code.

## Task names for capability toolsets

Render task names are persisted workflow identity, so every registered leaf toolset needs a stable `id`.

A toolset with an explicit `id` registers under that name, while two toolsets sharing an `id` fail Pydantic AI's uniqueness check. The integration does not rename duplicate IDs.

A capability can leave an internally constructed toolset unnamed, in which case that toolset stays inline in the entry task without independent retries, timeouts, compute, logs, or run history. Pydantic AI has no public API for assigning an `id` after construction, so `RenderWorkflows` only registers capability-owned toolsets that already have one.

Give user-owned function, MCP, and dynamic toolsets stable IDs at construction; an unnamed supported leaf attached directly to the agent is rejected before any task definition registers.

## Sub-agent delegation

`SubAgents` leaves its internal `delegate_task` toolset unnamed, so delegation itself stays inline in the workflow entry task. This preserves its parent-side `max_calls` check and event handling without assigning private Pydantic state.

To run a delegate's supported model and tool operations as Render task runs, construct that child `Agent` with its own `RenderWorkflows` instance using the same `Workflows` app as the parent. The app-scoped active `TaskContext` is then available to the explicitly configured child. Its model and named or agent-owned function tools register at construction and dispatch through `TaskContext.run()` during delegation. A child without `RenderWorkflows`, a child using another app, or a child built later from disk stays inline.

Successful operation results carry the child's usage delta and buffered custom or capability events in the versioned JSON envelope. The caller applies the delta once and re-emits events in order. Effects from a failed call or `ModelRetry` attempt are discarded. `SubAgent.max_calls` is enforced for concurrent delegations within the active parent task run because delegation stays in that process; it is not a global budget across retries or separate root task runs.

Immediate capability events require a synchronous decision before their emitter continues, which cannot be buffered across a child task. The integration rejects those events across the task boundary, so keep tools that emit them inline.

## Large tool outputs

`ToolOutputLimits` also contributes an unnamed helper toolset, so that helper remains inline. It measures and reduces a tool return after the registered tool task returns to the workflow entry task.

In `Spill` mode, the capability writes the full payload to a filesystem-backed store and gives the model a handle for a later `read_tool_result` call. Because task runs execute in separate processes and can have isolated filesystems, the later task might not be able to read the file behind that handle.

For large artifacts, return bounded JSON containing a key into object storage, a database, or another durable service that both tasks can reach. A later tool can then fetch the artifact from that shared store.

## Memory

Pass `Memory(...)` alongside `RenderWorkflows` in the agent constructor so its static toolset registers before execution. Each run resolves its own memory scope, with snapshot loading and memory tool calls executing in child tasks.

Use a `MemoryStore` backed by a shared external service that every task instance can reach. The default `InMemoryStore` is process-local; a local file or SQLite database is not shared across hosted task instances. A `store_resolver` and callable `namespace` must reconstruct the same scope from JSON dependencies in each worker. Workflows does not make the memory backend persistent.

## Task options and tool opt-out

Use Render `Options` for the model, tool, event, and capability task definitions:

```python {names="defined"}
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

Render fixes task options at registration, when the resolver first receives `tool=None` and `tool_name=''` to determine the toolset default. It then receives each statically known function tool and its name; returning `None` keeps `tool_options`.

When every static tool resolves to the shared default, the toolset keeps its existing shared call and validation task definitions. When at least one resolves different `Options` or `False`, each eligible static tool receives definitions named from the agent, toolset, tool, and operation. Those definitions can have independent retries, timeouts, compute, logs, and run history. Invocation-time options must equal the registered snapshot.

Returning `False` for a static function tool registers no task for that tool and runs it inside the workflow entry task. `False` is rejected for MCP and dynamic tools because their concrete tools are not known when the Workflow service registers definitions.

## JSON and dependency boundary

Because a child task can execute in a fresh process, the capability sends its inputs as a versioned JSON object and reconstructs the supported Pydantic AI state in that process.

Current callers write protocol v2, while workers also accept v1 requests and return effect-free v1 results that both old and new callers can read. This allows a new worker to finish work submitted by an older caller.

- Dependencies must round-trip through Pydantic's JSON codec. `deps_type` defaults to the agent's dependency type.
- Messages, model settings, metadata, tool definitions and arguments, usage deltas, buffered events, capability arguments, and results that cross the boundary must be JSON encodable.
- Model instances do not cross. The default model and entries in `models={...}` are registered by ID and resolved in the child task.
- Render caps the total arguments of one task run at 4 MB ([additional limits](https://render.com/docs/workflows-limits#additional-limits)). The capability sizes the final JSON envelope and raises before dispatch rather than sending an oversized call.
- Live in-process objects are unavailable unless the reconstructed run context explicitly supports them.

## What Render retains

Everything that crosses the task boundary is task state that Render stores: prompts, model responses, tool arguments and results, dependencies, and operation metadata. Render keeps task state for 30 days and then deletes it (see [task state retention](https://render.com/docs/workflows-limits#task-state-retention)).

To keep API keys, tokens, and other credentials out of retained task state, read them from environment variables inside the child task rather than passing them through `deps`, tool arguments, or task results.

## Models and task-run lineage

Render identifies a registered leaf toolset's task definitions by its explicit `id`, as described in [Task names for capability toolsets](#task-names-for-capability-toolsets).

A child task resolves the model ID received in its inputs against the models registered in its own process: the agent's default model and any `models={...}` entries. It exposes that instance as `ctx.model` for tools and other capabilities that need it. This is the plain model, rather than the workflow-side wrapper, so calls through `ctx.model` stay in the current task. A plain-string default that each run resolves for itself has no registered instance, and `ctx.model` remains unavailable in a child task.

Render owns task-run lineage. This integration spawns children through `TaskContext.run()` and cannot assign `parentTaskRunId` or `rootTaskRunId` itself. Code that draws a run graph should read `rootTaskRunId` where the platform populates it, keep `parentTaskRunId` to work out depth, and page through every task-run listing. Where the root field comes back empty, scope the listing to the Workflow and walk parent links instead.

Coverage for this area is the deterministic tests for toolset ids, task registration, and the JSON boundary, plus the opt-in local-runtime test below.

## Why are no child task runs appearing?

Check that `agent.run(...)` is inside the matching `@workflows.task` scope, since a plain `@app.task` or a direct call outside that scope runs the agent inline. A static function tool also stays in the entry task when `resolve_tool_options` returns `False`, so it has no separate child-task record.

## What happens on retry?

Child task runs retry independently under the Render `Retry` options set at registration. A retry of one model or tool task run can repeat that task's side effects if interruption happens before Render records the result.

Retrying the entry task restarts `agent.run(...)` and can repeat work that finished in the earlier attempt. The Render SDK does not let this integration assign stable idempotency keys to child calls or resume the agent loop from a checkpoint, so design the complete run to tolerate repeated execution.

## Local runtime tests

The repository carries an opt-in test that drives the same local runtime end to end. It is skipped by default and needs the `render` CLI at version 2.28.0 or later, but no Render API key:

```bash
PYDANTIC_AI_HARNESS_RENDER_LOCAL_RUNTIME=1 uv run pytest tests/render/test_local_runtime.py
```

The nested-agent test registers entry, parent, child, and grandchild operations on a local Render server, then checks that one root run produces 12 completed operation runs parented to that root. It verifies JSON dependency transport, usage accounting, ordered event delivery within the run, and a `ModelRetry` across a grandchild tool boundary. The eight distinct process IDs include the test controller and the entry task.

A second case tests constructor-supplied Memory, reads and writes through separate task processes using a shared local SQLite fixture, and a custom `ctx.tracer` span reaching the tool worker's exporter. These tests cover the local runtime; they do not establish checkpoint resume, hosted storage sharing, or hosted failure recovery.

## Streaming and cancellation

Streaming is buffered at the model-task boundary. The child task consumes the provider stream and returns its completed response and captured events. The workflow-side agent delivers them after the child task run finishes, but provider tokens do not stream live across `ctx.run(...)`. An `event_stream_handler` follows the same operation-task path and does not change this boundary.

Render task-run cancellation remains a native client and control-plane action. Pydantic AI cancellation tokens are unsupported inside a Render workflow because they are in-process handles. Suspended-model cleanup uses its own child task run. Neither form makes external tool side effects transactional.

## Execution and tracing

Use Render's native SDK and platform behavior for synchronous and asynchronous clients, task queues, retries, timeouts, compute plans, task fan-out, cancellation, and observability. Use a Render cron job when a schedule should trigger a root task run. The registered Pydantic AI operations are ordinary Render tasks and appear in Render's status, logs, metrics, and task views.

The capability emits no additional OpenTelemetry spans. Pydantic AI's model and tool instrumentation continues to trace the agent operations, while Render records the child task runs, retries, logs, and metrics at the workflow boundary.

`ctx.tracer` is available inside child tasks. It is a no-op when tracing is disabled and otherwise uses the worker's Pydantic AI instrumentation settings, including `agent.instrument`, `Agent.instrument_all(...)`, and registered instrumented models. Configure instrumentation when each worker loads the app; tracer objects are not serialized. The caller's `trace_include_content` setting is preserved. This restores tool-local spans but does not propagate OpenTelemetry parent span context across `TaskContext.run`; those spans can be separate traces.

Resolving effective agent/global instrumentation currently uses a private Pydantic AI settings getter inside `_compat.py`, covered by the same version-compatibility tests as context reconstruction.

## Pydantic AI compatibility boundary

`RenderWorkflows` uses the public `BaseDurabilityCapability` and registered backend contracts. Pydantic AI does not yet publish every semantic parameter, transport, and bound-operation type needed by a cross-process registered backend. The integration contains those private imports in `pydantic_ai_harness/render/_compat.py`.

Changes to those private APIs can require a corresponding Harness update. Use Pydantic AI and Harness versions tested together, and run the Render integration tests before upgrading either dependency independently.

## API reference

::: pydantic_ai_harness.render.RenderWorkflows
