---
name: pydantic-ai-render-workflows
description: >-
  Use when a Pydantic AI agent's own job is long-running or distributed: a
  research agent that spends minutes writing a report, a batch document
  pipeline, a monitor or scheduled job, parallel model and tool calls, or work
  that must outlive the HTTP request that started it. Covers running those
  agents on Render Workflows, where each supported operation is a registered
  task definition and each invocation is a child task run with its own retry,
  timeout, and managed compute. Use when the user mentions Render Workflows,
  background agent jobs, fan-out, or per-step retries and timeouts. Do not wait
  until they ask to deploy a web service on Render. Do not use for
  Temporal-style replay, checkpoint resume, or crash recovery.
license: MIT
---

# Pydantic AI on Render Workflows

`RenderWorkflows` runs a Pydantic AI agent on a [Render Workflows](https://render.com/docs/workflows) app.
Constructing the agent registers one Render task definition per supported operation (model requests,
per-toolset validation and calls, event delivery, `@durable_operation` methods). Inside a workflow, each
supported invocation starts a child task run of that definition. Use it when the agent's own job is
long-running or distributed; Render retries those task runs rather than replaying the agent loop.

The supported contract is narrower than the agent surface, so read
[the capability README](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/#readme)
before designing around this. The points below are the ones that change a design.

## Install

```bash
uv add "pydantic-ai-harness[render]"
```

## Required boundary

Wrap `agent.run(...)` in `@workflows.task` on the same `Workflows` app passed to `RenderWorkflows`.
Attaching the capability alone does not route calls through Render. A plain `@app.task` does not activate
it. Attach exactly one `RenderWorkflows` per agent.

```python {test="skip"}
from pydantic_ai import Agent
from render import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows

app = Workflows()
workflows = RenderWorkflows(app)
agent = Agent('openai:gpt-5.6-sol', name='support', capabilities=[workflows])


@workflows.task
async def support(ctx: TaskContext, prompt: str) -> str:
    del ctx
    return (await agent.run(prompt)).output
```

## Task names for capability toolsets

Render task names are persisted workflow identity, so every registered leaf toolset needs a stable `id`. An
`id` set on the toolset wins. For an unnamed supported leaf that a capability owns (nobody writing
`capabilities=[SubAgents(...)]` holds that toolset), `RenderWorkflows` derives the `id` from the owning
capability's `id` before anything registers, adding a deterministic numeric suffix where one derived name
would be taken twice. A capability with no `id` that owns an unnamed leaf is refused with a `UserError` at
agent construction, before any task registers: pass an `id` through the capability, or attach the tools
yourself as `toolsets=[FunctionToolset(..., id='...')]`. Two toolsets sharing an explicit `id` reach
Pydantic AI's own uniqueness check.

## Delegation and large tool outputs

Native `SubAgents` registers and runs: `delegate_task` becomes a task definition and each delegation a child
task run. The delegate's own model requests and tool calls execute inside that one task run, not as
separately registered children, because a sub-agent does not carry `RenderWorkflows` itself. Only the
delegate's JSON return crosses back, so the child's usage delta does not reach the parent, its buffered
events are not merged into the parent's event stream, and `max_calls` has no counter shared across
processes. Carry the accounting and telemetry you need in the delegate's own return value.

`ToolOutputLimits` registers and runs too, but its `Spill` mode is filesystem-backed: task runs are separate
processes on filesystems that may be isolated from each other, so a spill written in one task run cannot be
assumed readable by the task run that calls `read_tool_result`. Inside a workflow, return bounded JSON and
keep large artifacts in external durable storage, returning a key the read side can fetch.

## Other design constraints

- Task options (retry, timeout, plan) are fixed at registration. One call task definition serves every tool
  in its toolset. To give static tools distinct definitions and options, put each in its own named
  `FunctionToolset`.
- Everything crossing the boundary must be JSON encodable, including `deps`, and the arguments of one task
  run must fit Render's documented 4 MB argument cap. Model instances do not cross; their ids do.
- Render keeps task state for 30 days, so prompts, responses, tool arguments and results, and deps are
  retained. Read secrets from environment variables inside the child task instead of sending them through.
- Streaming is buffered at the model-task boundary; provider tokens do not stream live across `ctx.run(...)`.
- Retries are task-level. A child-task retry can repeat that task's side effects; an entry-task retry starts
  `agent.run` again. This is not replay or checkpoint resume.

Full guide, including local development and the opt-in local-runtime verification command:
[Render Workflows README](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/render/#readme).
