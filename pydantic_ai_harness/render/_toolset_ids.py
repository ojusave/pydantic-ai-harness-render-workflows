"""Give every Render-registered leaf toolset a stable `id` before any task registers."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Any, NoReturn, Protocol, TypeAlias, runtime_checkable

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset

from ._compat import CapabilityOwnedToolset, assign_toolset_id

__all__ = ('prepare_capability_toolset_ids',)

SupportedLeafToolset: TypeAlias = 'AbstractToolset[Any]'


@runtime_checkable
class _MCPToolsetShape(Protocol):
    """Identify MCP leaves without importing the optional MCP client package."""

    def list_resources(self) -> Awaitable[object]: ...


def prepare_capability_toolset_ids(toolsets: Sequence[AbstractToolset[Any]]) -> None:
    """Settle task identity for every supported leaf before the first task registration.

    A capability that builds its own toolset leaves it unnamed: nobody writing
    `capabilities=[SubAgents(...)]` ever holds the `FunctionToolset` underneath. The
    capability's own `id` is stable, unique per agent, and identical in the worker
    process, so it is what the toolset's Render task names are derived from.

    The whole assignment is planned before any of it is applied, so an agent that cannot
    be named at all is rejected without half-naming the toolsets it was built from, and
    always before the `Workflows` app is holding tasks that nothing can unregister.
    """
    nodes = _walk(toolsets)
    leaves = [node for node in nodes if _is_supported_leaf(node)]
    owners = _capability_owners(nodes)
    _reject_duplicate_ids(leaves)

    planned: list[tuple[SupportedLeafToolset, str]] = []
    # Every `id` in the tree is off limits, not only the ones Render registers tasks for:
    # a derived name that shadowed another toolset's would be a name two things answer to.
    taken = {node.id for node in nodes if node.id is not None}
    for toolset in leaves:
        if toolset.id is not None:
            continue
        capability = owners.get(id(toolset))
        if capability is None or capability.id is None:
            _reject_unnameable(toolset, capability)
        derived = _unique_id(capability.id, taken)
        planned.append((toolset, derived))
        taken.add(derived)

    for toolset, derived in planned:
        assign_toolset_id(toolset, derived)


def _reject_unnameable(toolset: SupportedLeafToolset, capability: AbstractCapability[Any] | None) -> NoReturn:
    """Refuse an unnamed leaf whose `id` no one on the public surface can supply."""
    if capability is None:
        raise UserError(f'{type(toolset).__name__} needs a unique `id` to register tasks with Render Workflows.')
    raise UserError(
        'Render Workflows registers tasks per toolset and task names are persisted workflow identity, '
        f'so a toolset contributed by a capability with no `id` has no stable name to register under: '
        f'{type(capability).__name__} ({type(toolset).__name__}). Give the capability an explicit `id` '
        f"(`{type(capability).__name__}(id='...')`), or attach the tools to the agent directly with an "
        'explicit toolset `id`.'
    )


def _unique_id(capability_id: str, taken: set[str]) -> str:
    """Derive a name from a capability `id` another toolset has not already claimed.

    The suffix counts from `.2` so the common case reads as the capability's own `id`,
    and it is resolved in traversal order so the same agent produces the same names in
    every process that builds it.
    """
    if capability_id not in taken:
        return capability_id
    suffix = 2
    while f'{capability_id}.{suffix}' in taken:
        suffix += 1
    return f'{capability_id}.{suffix}'


def _capability_owners(nodes: Sequence[AbstractToolset[Any]]) -> dict[int, AbstractCapability[Any]]:
    owners: dict[int, AbstractCapability[Any]] = {}

    for node in nodes:
        if not isinstance(node, CapabilityOwnedToolset):
            continue

        for leaf in _walk((node.wrapped,)):
            if _is_supported_leaf(leaf):
                # Nested capability wrappers are visited after their parents,
                # so the closest capability becomes the owner.
                owners[id(leaf)] = node.capability

    return owners


def _is_supported_leaf(toolset: AbstractToolset[Any]) -> bool:
    return isinstance(toolset, FunctionToolset | DynamicToolset | _MCPToolsetShape)


def _reject_duplicate_ids(leaves: Sequence[SupportedLeafToolset]) -> None:
    """Refuse an `id` two distinct toolsets already carry, rather than renaming either.

    A derived name never takes an `id` a toolset already holds, so a clash between two
    toolsets that were both named is a genuine collision and stays Pydantic AI's answer,
    only reached before this app registers anything.
    """
    seen: dict[str, SupportedLeafToolset] = {}
    for toolset in leaves:
        toolset_id = toolset.id
        if toolset_id is None:
            continue
        existing = seen.get(toolset_id)
        if existing is not None and existing is not toolset:
            raise UserError(
                f'Two toolsets have the same `id` {toolset_id!r}. Toolset `id`s must be unique among all '
                'toolsets registered with the same agent.'
            )
        seen[toolset_id] = toolset


def _walk(toolsets: Sequence[AbstractToolset[Any]]) -> list[AbstractToolset[Any]]:
    """Return every node visited by Pydantic AI's toolset traversal."""
    nodes: list[AbstractToolset[Any]] = []

    for root in toolsets:
        root.apply(nodes.append)

    return nodes
