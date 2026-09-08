"""Graph operations used by citation-aware retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Set


GraphNeighbors = Mapping[str, Mapping[str, Set[str]]]


def neighbors_for(
    citation: str,
    mode: str,
    graph_neighbors: GraphNeighbors,
) -> set[str]:
    """Return incoming, outgoing, or both citation neighbors."""

    if mode not in {"incoming", "outgoing", "both"}:
        raise ValueError("mode must be 'incoming', 'outgoing', or 'both'")

    outgoing = set(graph_neighbors.get("outgoing", {}).get(citation, set()))
    incoming = set(graph_neighbors.get("incoming", {}).get(citation, set()))
    if mode == "outgoing":
        return outgoing
    if mode == "incoming":
        return incoming
    return outgoing | incoming


def expand_neighbors(
    seeds: Set[str],
    graph_neighbors: GraphNeighbors,
    *,
    mode: str = "both",
) -> set[str]:
    """Collect one-hop neighbors for a set of seed citations."""

    expanded: set[str] = set()
    for citation in seeds:
        expanded.update(neighbors_for(citation, mode, graph_neighbors))
    return expanded
