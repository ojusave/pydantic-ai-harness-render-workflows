"""Assign stable Render task names to capability-provided toolsets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset

from ._compat import CapabilityOwnedToolset

__all__ = ('assign_render_toolset_ids',)

NameableToolset: TypeAlias = 'FunctionToolset[Any] | DynamicToolset[Any]'


def assign_render_toolset_ids(toolsets: Sequence[AbstractToolset[Any]]) -> None:
    """Give unnamed capability toolsets stable IDs before Render registers tasks.

    Pydantic AI's registered backend requires an ID for every leaf toolset.
    A capability often creates its toolset internally, so the user has no
    toolset instance to name. The capability ID is stable across the workflow
    and worker processes, making it the appropriate Render task identity.
    """
    used_ids = {toolset.id for toolset in _walk(toolsets) if toolset.id is not None}

    for toolset, capability in _owned_leaves(toolsets):
        if toolset.id is not None or capability.id is None:
            continue

        toolset_id = _next_available_id(capability.id, used_ids)

        # FunctionToolset and DynamicToolset expose a read-only `id` property,
        # backed by the constructor's `_id` field. Render assigns it here,
        # before Pydantic AI registers or executes the toolset.
        toolset._id = toolset_id  # pyright: ignore[reportPrivateUsage]
        used_ids.add(toolset_id)


def _owned_leaves(
    toolsets: Sequence[AbstractToolset[Any]],
) -> list[tuple[NameableToolset, AbstractCapability[Any]]]:
    """Pair each nameable leaf with the capability that contributed it."""
    owners: dict[int, tuple[NameableToolset, AbstractCapability[Any]]] = {}

    for node in _walk(toolsets):
        if not isinstance(node, CapabilityOwnedToolset):
            continue

        for leaf in _walk((node.wrapped,)):
            if isinstance(leaf, FunctionToolset | DynamicToolset):
                # Nested capability wrappers are visited after their parents,
                # so the closest capability becomes the owner.
                owners[id(leaf)] = (leaf, node.capability)

    return list(owners.values())


def _walk(toolsets: Sequence[AbstractToolset[Any]]) -> list[AbstractToolset[Any]]:
    """Return every node visited by Pydantic AI's toolset traversal."""
    nodes: list[AbstractToolset[Any]] = []

    for root in toolsets:
        root.apply(nodes.append)

    return nodes


def _next_available_id(preferred: str, used_ids: set[str]) -> str:
    """Return the preferred ID, adding a numeric suffix only on collision."""
    candidate = preferred
    suffix = 2
    while candidate in used_ids:
        candidate = f'{preferred}.{suffix}'
        suffix += 1
    return candidate
