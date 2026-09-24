# Render Workflows

Run long-running AI agents in the background with separate retries, timeouts, and compute settings for model requests
and tool calls. The [Render Workflows](https://render.com/docs/workflows) integration runs the agent loop in an entry
task and supported operations as child tasks, each with its own status, logs, and result. For example, a tool that
processes a large document can have a longer timeout than the model calls around it.

If you only need background execution, a native Render task around `agent.run(...)` may be enough.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Run the example locally

The example uses Pydantic AI's `TestModel` and a mock weather tool, so it needs no LLM API key or Render deployment.
The model generates sample tool arguments and returns fixed text rather than interpreting the prompt.

You need Python 3.11 or later. Install the [Render CLI](https://render.com/docs/cli) 2.28.0 or later separately. The installation below uses Pydantic AI 2.46.0 and Render SDK 1.2.0, the versions tested with these examples.

### 1. Install dependencies

In a new directory, initialize a project with [uv](https://docs.astral.sh/uv/) using `uv init --bare`; for an existing uv project, run the installation command directly. If you use pip, first create and activate a virtual environment.

Until a Harness release includes this integration, install it from the repository checkout containing it. Replace `/path/to/pydantic-ai-harness` with the checkout's absolute path:

uv:

```bash
uv add "/path/to/pydantic-ai-harness[render]" "pydantic-ai-slim==2.46.0" "render==1.2.0"
```

pip:

```bash
pip install "/path/to/pydantic-ai-harness[render]" "pydantic-ai-slim==2.46.0" "render==1.2.0"
```

For a published Harness release that includes the integration, replace `/path/to/pydantic-ai-harness[render]` with `pydantic-ai-harness[render]`. The `render` extra supplies the Python SDK and its `render-workflows` executable.

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

Construct the agent and its tools at module load time, attaching one `RenderWorkflows` instance per agent so task
definitions register before the worker starts. Pass the same `app` object to the capability and the worker.
The `@workflows.task` decorator activates child-task dispatch for `agent.run(...)`; a plain `@app.task` or a call
outside that scope runs the agent inline.

### 3. Start the local task server

From the directory containing `app.py`, run:

```bash
render workflows dev -- uv run render-workflows app:app
```

The server listens on port `8120` and lists the registered tasks, including `support` and `support__model.request`.
Keep it running for the following commands.

With pip, run `render workflows dev -- render-workflows app:app` from your activated environment. The [local development guide](https://render.com/docs/workflows-local-development) covers port options and the local server's limits.

### 4. Submit a job and inspect its runs

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

For a FastAPI endpoint or another Python web application, submit `support` through Render's task client and return
its run ID in an HTTP `202 Accepted` response. This avoids keeping the HTTP request open while the agent works;
the client polls a status endpoint for the result or failure. Associate the ID with the requesting user so that endpoint
can verify ownership, and keep Render credentials on the server.

Deploy the exported `app` as a Workflow with the start command `uv run render-workflows app:app`, and configure its
provider credentials. Render's [workflow setup](https://render.com/docs/workflows) and
[task submission](https://render.com/docs/workflows-running) guides cover deployment and authenticated calls.

## Retry failed tool calls and set task timeouts

To configure model and tool tasks separately, replace the `app` and `workflows` declarations in `app.py` with:

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

Configure the entry task separately, for example with `@workflows.task(timeout_seconds=600, plan='flex')`.
All task settings are fixed at registration and cannot change between invocations.

To give a statically known function tool its own options, use `resolve_tool_options`; returning `False` keeps that tool inline. The [per-tool options reference](https://github.com/pydantic/pydantic-ai-harness/blob/main/docs/render-workflows.md#task-options-and-tool-opt-out) explains the resolver contract and the restrictions for MCP and dynamic tools.

A failed child task can retry while the entry task waits. Retrying the entry task restarts `agent.run(...)`, so model
and tool calls that already finished may run again. There is no checkpoint resume or replay of completed steps.
Either kind of retry can repeat an external action performed before its result was recorded; use application-level
idempotency keys or deduplication for writes and API requests that must not happen twice.

Render retries are separate from Pydantic AI's `ModelRetry`, which asks the model to correct or reconsider a call,
and from retries in the provider SDK. Account for all three when setting retry limits and timeouts; see the
[Pydantic AI retry guide](https://pydantic.dev/docs/ai/core-concepts/retries/).

## Constraints to check before deployment

- **Toolsets:** Explicit IDs must be stable and unique. Unnamed capability-owned tools can remain inline without a child-task record.
- **Task inputs:** Dependencies, messages, arguments, and results must be JSON serializable, and Render limits a task run's arguments to 4 MB. Pass resource IDs across the boundary and reconstruct clients inside workers.
- **Storage:** Hosted task instances do not share local memory or files, so agent memory and large artifacts need a shared external backend. Read credentials from worker environment variables rather than passing them through task inputs or results.
- **Streaming and cancellation:** Model responses are buffered until their child task finishes, so live tokens do not stream across the task boundary. Pydantic AI cancellation tokens are unsupported in a workflow; use Render's native task cancellation instead.
- **Compatibility:** The adapter uses private Pydantic AI APIs, so test the integration before upgrading its dependencies. The local quickstart verifies task dispatch, but does not establish hosted failure recovery or performance.

Render retains task inputs and results for 30 days and allows up to 500 task definitions per workflow. Check the current [limits and pricing](https://render.com/docs/workflows-limits) when deciding how many operations to register as separate tasks.

## Further reference

See the [integration reference](https://github.com/pydantic/pydantic-ai-harness/blob/main/docs/render-workflows.md)
for task naming, sub-agent setup, JSON transport, memory, large tool outputs, and tracing.
