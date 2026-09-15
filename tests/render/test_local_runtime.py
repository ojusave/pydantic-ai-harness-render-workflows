"""Opt-in integration coverage against the keyless local Render runtime."""

from __future__ import annotations

import os

from pydantic import TypeAdapter

from .conftest import LocalRenderRuntime, LocalTaskRun
from .runtime_app import RootTaskResult

MODEL_TASK = 'local-runtime-agent__model.request'
TOOL_TASK = 'local-runtime-agent__function_toolset__<agent>.call_tool'
ROOT_TASK = 'run-local-runtime-agent'

_ROOT_RESULTS = TypeAdapter(list[RootTaskResult])


def _runs_for_root(runtime: LocalRenderRuntime, task_name: str, root_id: str) -> list[LocalTaskRun]:
    return [run for run in runtime.list_runs(task_name) if run.root_task_run_id == root_id]


def test_agent_operations_run_in_isolated_local_render_processes(
    local_render_runtime: LocalRenderRuntime,
) -> None:
    runtime = local_render_runtime
    registered_names = {task.name for task in runtime.list_tasks()}
    assert {ROOT_TASK, MODEL_TASK, TOOL_TASK} <= registered_names

    controller_pid = os.getpid()
    started = runtime.start_task(
        ROOT_TASK,
        f'["exercise process isolation", {{"prefix": "serializable-deps", "controller_pid": {controller_pid}}}]',
    )
    completed = runtime.wait_for_run(started.id)
    assert completed.status == 'completed', runtime.logs()
    results = _ROOT_RESULTS.validate_python(completed.results)
    assert len(results) == 1
    result = results[0]
    tool_result = result.agent_output.tool_result

    assert tool_result.controller_pid == controller_pid
    assert tool_result.deps_prefix == 'serializable-deps'
    assert tool_result.model_retry_count == 1
    assert tool_result.value == 'retry-value'

    process_ids = {
        controller_pid,
        result.root_pid,
        result.agent_output.final_model_pid,
        tool_result.model_pid,
        tool_result.tool_pid,
    }
    assert len(process_ids) == 5, process_ids

    model_runs = _runs_for_root(runtime, MODEL_TASK, started.id)
    tool_runs = _runs_for_root(runtime, TOOL_TASK, started.id)
    assert len(model_runs) == 3
    assert len(tool_runs) == 2
    assert all(run.status == 'completed' for run in [*model_runs, *tool_runs])
    assert all(run.parent_task_run_id == started.id for run in [*model_runs, *tool_runs])
