"""Render task identity for the toolsets a capability contributes.

Render identifies a leaf toolset's tasks by the toolset `id` and refuses to register a leaf
without one. A capability that builds its own toolset has nowhere to take an id from, so the
id is derived from the capability here instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import count
from typing import Any, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset

from ._compat import CapabilityOwnedToolset

__all__ = ('name_capability_toolsets',)

NameableToolset: TypeAlias = 'FunctionToolset[Any] | DynamicToolset[Any]'


def name_capability_toolsets(toolsets: Sequence[AbstractToolset[Any]]) -> None:
    """Name each unnamed capability-contributed leaf after the capability that owns it.

    A capability's `id` is unique within the agent and stable across processes, which is what
    a Render task name needs: the workflow side registers the names before the worker starts,
    and the worker derives the same ones from the same agent module. Naming runs when the
    capability binds, so the ids are settled before registration reads them.

    A leaf that already has an id keeps it, and an unnamed leaf under an unnamed capability is
    left alone so registration still raises Pydantic AI's own guidance on naming it.
    """
    taken = _existing_ids(toolsets)
    for leaf, capability in _capability_leaves(toolsets):
        if leaf.id is not None or (capability_id := capability.id) is None:
            continue
        toolset_id = _unused(capability_id, taken)
        leaf._id = toolset_id  # pyright: ignore[reportPrivateUsage]
        taken.add(toolset_id)


def _capability_leaves(
    toolsets: Sequence[AbstractToolset[Any]],
) -> list[tuple[NameableToolset, AbstractCapability[Any]]]:
    """Every nameable leaf under a `CapabilityOwnedToolset`, paired with the owning capability."""
    wrappers: list[CapabilityOwnedToolset[Any]] = []

    def collect_wrapper(node: AbstractToolset[Any]) -> None:
        if isinstance(node, CapabilityOwnedToolset):
            wrappers.append(node)

    for toolset in toolsets:
        toolset.apply(collect_wrapper)

    owners: dict[int, tuple[NameableToolset, AbstractCapability[Any]]] = {}
    for wrapper in wrappers:
        # The walk runs outside in, so a capability nested inside another one's contribution
        # overwrites the outer owner and keeps the leaves it actually built.
        def claim(node: AbstractToolset[Any], capability: AbstractCapability[Any] = wrapper.capability) -> None:
            if isinstance(node, FunctionToolset | DynamicToolset):
                owners[id(node)] = (node, capability)

        wrapper.wrapped.apply(claim)
    return list(owners.values())


def _existing_ids(toolsets: Sequence[AbstractToolset[Any]]) -> set[str]:
    ids: set[str] = set()

    def collect_id(node: AbstractToolset[Any]) -> None:
        if node.id is not None:
            ids.add(node.id)

    for toolset in toolsets:
        toolset.apply(collect_id)
    return ids


def _unused(preferred: str, taken: set[str]) -> str:
    """`preferred`, or the first numbered variant no other toolset on the agent is using."""
    if preferred not in taken:
        return preferred
    return next(candidate for n in count(2) if (candidate := f'{preferred}.{n}') not in taken)
