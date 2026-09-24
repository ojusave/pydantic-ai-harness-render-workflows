# Render Workflows

Run long-running AI agents as background jobs when their model requests and tool calls need separate retries, timeouts, or compute settings. The [Render Workflows](https://render.com/docs/workflows) integration keeps the Pydantic AI agent loop in an entry task and runs supported operations as child tasks, so you can inspect each operation's status, logs, and result.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/) | [Detailed reference](https://github.com/pydantic/pydantic-ai-harness/blob/main/docs/render-workflows.md)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

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

uv:

```bash
uv add "/path/to/pydantic-ai-harness[render]" "pydantic-ai-slim==2.46.0" "render==1.2.0"
```

pip:

```bash
pip install "/path/to/pydantic-ai-harness[render]" "pydantic-ai-slim==2.46.0" "render==1.2.0"
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
render workflows dev -- uv run render-workflows app:app
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

uv:

```bash
uv add "pydantic-ai-slim[openai]==2.46.0"
```

pip:

```bash
pip install "pydantic-ai-slim[openai]==2.46.0"
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

To give a statically known function tool its own options, use `resolve_tool_options`; returning `False` keeps that tool inline. The [per-tool options reference](https://github.com/pydantic/pydantic-ai-harness/blob/main/docs/render-workflows.md#task-options-and-tool-opt-out) explains the resolver contract and the restrictions for MCP and dynamic tools.

Render task retries are separate from Pydantic AI's `ModelRetry`, which asks the model to correct or reconsider a call, and from any retries performed by the provider SDK. Account for all three when setting retry limits and timeouts; the [Pydantic AI retry guide](https://pydantic.dev/docs/ai/core-concepts/retries/) explains the model and provider behavior.

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

## Further reference

The [integration reference](https://github.com/pydantic/pydantic-ai-harness/blob/main/docs/render-workflows.md) covers task naming, JSON transport, buffered events, memory, tool-output storage, and tracing. Its [sub-agent section](https://github.com/pydantic/pydantic-ai-harness/blob/main/docs/render-workflows.md#sub-agent-delegation) explains how to configure each child with its own `RenderWorkflows` instance using the same app, since attaching the capability to the parent does not configure its children.

If an expected child run does not appear, check that `agent.run(...)` executes inside the matching `@workflows.task` scope and that the tool has not been configured to run inline.
