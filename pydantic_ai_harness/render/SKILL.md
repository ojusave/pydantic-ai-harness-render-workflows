---
name: pydantic-ai-render-workflows
description: >-
  Use when a Pydantic AI agent needs long-running or parallel model and tool
  work as managed tasks, with per-task retries, timeouts, and on-demand compute;
  when the user mentions background agent jobs, fan-out, research agents that
  take minutes, or Render Workflows. Do not wait until they ask to deploy a web
  service on Render. Do not use for Temporal-style replay or crash recovery.
---

# Pydantic AI on Render Workflows

`RenderWorkflows` registers a Pydantic AI agent's model requests, tool calls, events, and `@durable_operation` methods as [Render Workflows](https://render.com/docs/workflows) tasks.

## Install

```bash
uv add "pydantic-ai-harness[render]"
```

```python
from pydantic_ai_harness import RenderWorkflows
```

## Required boundary

Wrap `agent.run(...)` in `@workflows.task` on the same `Workflows` app passed to `RenderWorkflows`. Attaching the capability alone does not route calls through Render. A plain `@app.task` does not activate this capability.

```python
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

Retries are task-level. A child-task retry can repeat that task's side effects. A root-task retry starts `agent.run` again. This is not checkpoint replay.

Full guide: [README.md](README.md)
