"""Refuse Render task registration for a contributed toolset that has no `id`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset

from ._compat import CapabilityOwnedToolset

__all__ = ('reject_unnameable_capability_toolsets',)

SupportedLeafToolset: TypeAlias = 'FunctionToolset[Any] | DynamicToolset[Any] | MCPToolset[Any]'


def reject_unnameable_capability_toolsets(toolsets: Sequence[AbstractToolset[Any]]) -> None:
    """Validate task identity before registering any supported leaf toolset."""
    leaves = _supported_leaves(toolsets)
    owned = _owned_leaves(toolsets)
    unnamed = [(toolset, capability) for toolset, capability in owned.values() if toolset.id is None]
    if not unnamed:
        _reject_other_invalid_ids(leaves)
        return

    described = ', '.join(sorted(f'{type(owner).__name__} ({type(toolset).__name__})' for toolset, owner in unnamed))
    raise UserError(
        'Render Workflows registers tasks per toolset and task names are persisted workflow identity, '
        f'so a toolset contributed with no `id` has no stable name to register under: {described}. '
        'The `id` has to come from whoever owns the toolset: pass one through the capability when it '
        'accepts one, or attach the tools to the agent directly with an explicit toolset `id`.'
    )


def _owned_leaves(
    toolsets: Sequence[AbstractToolset[Any]],
) -> dict[int, tuple[SupportedLeafToolset, AbstractCapability[Any]]]:
    owners: dict[int, tuple[SupportedLeafToolset, AbstractCapability[Any]]] = {}

    for node in _walk(toolsets):
        if not isinstance(node, CapabilityOwnedToolset):
            continue

        for leaf in _walk((node.wrapped,)):
            if isinstance(leaf, FunctionToolset | DynamicToolset | MCPToolset):
                # Nested capability wrappers are visited after their parents,
                # so the closest capability becomes the owner.
                owners[id(leaf)] = (leaf, node.capability)

    return owners


def _supported_leaves(toolsets: Sequence[AbstractToolset[Any]]) -> list[SupportedLeafToolset]:
    return [
        toolset for toolset in _walk(toolsets) if isinstance(toolset, FunctionToolset | DynamicToolset | MCPToolset)
    ]


def _reject_other_invalid_ids(leaves: Sequence[SupportedLeafToolset]) -> None:
    seen: dict[str, SupportedLeafToolset] = {}
    for toolset in leaves:
        toolset_id = toolset.id
        if toolset_id is None:
            raise UserError(f'{type(toolset).__name__} needs a unique `id` to register tasks with Render Workflows.')
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
