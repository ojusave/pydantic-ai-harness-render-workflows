from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

WORKFLOW = Path(__file__).parents[1] / '.github' / 'workflows' / 'main.yml'
DETECTORS = {'detect': 'localstack', 'detect-mongodb': 'mongodb', 'detect-redis': 'redis'}


def _step(step_id: str) -> str:
    changes = WORKFLOW.read_text().split('  changes:\n', 1)[1].split('\n  clai-test:', 1)[0]
    match = re.search(rf'^      - id: {step_id}\n(.*?)(?=^      - |\Z)', changes, re.MULTILINE | re.DOTALL)
    assert match is not None
    return match.group(1)


def _run_step(step_id: str, repository: Path, base: str, output: Path) -> subprocess.CompletedProcess[str]:
    script = dedent(_step(step_id).split('        run: |\n', 1)[1])
    return subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', script],
        cwd=repository,
        env={**os.environ, 'BASE_SHA': base, 'HEAD_SHA': 'HEAD', 'GITHUB_OUTPUT': str(output)},
        capture_output=True,
        text=True,
        timeout=10,
    )


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.name=CI Test', '-c', 'user.email=ci@example.invalid', *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repository = tmp_path / 'repository'
    repository.mkdir()
    _git(repository, 'init', '--quiet')
    for directory in ('localstack', 'step_persistence', 'spend'):
        path = repository / 'pydantic_ai_harness' / directory / '__init__.py'
        path.parent.mkdir(parents=True)
        path.write_text('')
    _git(repository, 'add', '.')
    _git(repository, 'commit', '--quiet', '-m', 'Initial tree')
    return repository


@pytest.mark.parametrize('new_branch', [False, True])
@pytest.mark.parametrize(
    ('changed_path', 'expected'),
    [
        ('README.md', set[str]()),
        ('uv.lock', {'localstack', 'mongodb', 'redis'}),
        ('pydantic_ai_harness/localstack/__init__.py', {'localstack'}),
        ('pydantic_ai_harness/step_persistence/__init__.py', {'mongodb'}),
        ('pydantic_ai_harness/spend/__init__.py', {'redis'}),
    ],
)
def test_changes_detects_new_branches_and_filters_existing_commits(
    repository: Path, tmp_path: Path, new_branch: bool, changed_path: str, expected: set[str]
) -> None:
    base = '0' * 40 if new_branch else _git(repository, 'rev-parse', 'HEAD')
    (repository / changed_path).write_text('# Changed\n')
    _git(repository, 'add', '.')
    _git(repository, 'commit', '--quiet', '-m', 'Change one path')

    output = tmp_path / 'base-output'
    result = _run_step('base', repository, base, output)
    assert result.returncode == 0, result.stderr
    resolved_base = output.read_text().removeprefix('sha=').strip()
    if new_branch:
        expected = set(DETECTORS.values())
    else:
        assert resolved_base == base

    for step_id, name in DETECTORS.items():
        assert 'BASE_SHA: ${{ steps.base.outputs.sha }}' in _step(step_id)
        output = tmp_path / f'{name}-output'
        result = _run_step(step_id, repository, resolved_base, output)
        assert result.returncode == 0, result.stderr
        assert output.read_text() == f'{name}={str(name in expected).lower()}\n'


def test_changes_does_not_hide_invalid_base_commits(repository: Path, tmp_path: Path) -> None:
    for step_id in DETECTORS:
        output = tmp_path / step_id
        result = _run_step(step_id, repository, 'not-a-commit', output)
        assert result.returncode != 0
        assert not output.exists()


def test_changes_selects_the_event_base_without_shell_interpolation() -> None:
    assert (
        "BASE_SHA: ${{ github.event_name == 'pull_request' && "
        'github.event.pull_request.base.sha || github.event.before }}'
    ) in _step('base')
