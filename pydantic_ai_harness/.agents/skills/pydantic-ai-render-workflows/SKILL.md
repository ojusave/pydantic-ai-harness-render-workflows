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

The supported contract is narrower than the agent surface, and the harness's `SubAgents` is refused at
agent construction today, so read
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

## What is refused, and what to do instead

A capability that builds its own toolset leaves that toolset unnamed, and Pydantic AI publishes no stable-ID
assignment API to name it afterwards. `RenderWorkflows` raises a `UserError` at agent construction naming
each such capability rather than registering persisted task names nobody chose.

Today that includes the harness's `SubAgents` and `ToolOutputLimits`, so native sub-agent delegation is not
available on this integration today. Neither capability exposes a public toolset-id knob: their
`id` field names the capability, not the toolset it builds. The alternative that works is a toolset you own,
attached as `toolsets=[FunctionToolset(..., id='...')]`. That is your toolset under your name, not a
re-creation of the refused capability's behavior.

Delegation written as a named `FunctionToolset` registers and runs. Because the delegate executes in its own
task run and only its JSON return crosses back, expect its usage delta and buffered child events not to
reach the parent, and `max_calls` to have no shared counter across distributed delegates. Those follow from
the JSON boundary and from `SubAgents` being refused at construction, not from measurements of distributed
delegation.

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
