from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from coord_harness.core.enums import TopologyPreset


@dataclass(frozen=True)
class Topology:
    preset: TopologyPreset
    agent_ids: list[str]
    edges: list[tuple[str, str]]
    root_agent: str

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def node_count(self) -> int:
        return len(self.agent_ids)

    @property
    def inbound(self) -> dict[str, list[str]]:
        mapping = {agent_id: [] for agent_id in self.agent_ids}
        for source, target in self.edges:
            mapping[target].append(source)
        return mapping

    @property
    def outbound(self) -> dict[str, list[str]]:
        mapping = {agent_id: [] for agent_id in self.agent_ids}
        for source, target in self.edges:
            mapping[source].append(target)
        return mapping

    def shortest_path_parents_to_root(self) -> dict[str, str | None]:
        parents: dict[str, str | None] = {self.root_agent: None}
        queue: deque[str] = deque([self.root_agent])
        inbound = self.inbound
        while queue:
            node = queue.popleft()
            for neighbor in sorted(inbound[node]):
                if neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if len(parents) != len(self.agent_ids):
            missing = set(self.agent_ids) - set(parents)
            raise ValueError(f"Topology is not connected to root {self.root_agent}: {sorted(missing)}")
        return parents

    def summary(self) -> dict[str, int | str]:
        inbound = self.inbound
        outbound = self.outbound
        return {
            "preset": self.preset.value,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "max_in_degree": max((len(nodes) for nodes in inbound.values()), default=0),
            "max_out_degree": max((len(nodes) for nodes in outbound.values()), default=0),
            "root_agent": self.root_agent,
        }


def _make_bidirectional(edges: set[tuple[str, str]]) -> list[tuple[str, str]]:
    bidirectional = set()
    for source, target in edges:
        if source == target:
            continue
        bidirectional.add((source, target))
        bidirectional.add((target, source))
    return sorted(bidirectional)


def build_topology(preset: TopologyPreset, agent_count: int, seed: int) -> Topology:
    if agent_count < 1:
        raise ValueError("agent_count must be >= 1")
    agent_ids = [f"agent_{idx}" for idx in range(agent_count)]
    root_agent = agent_ids[0]
    undirected_edges: set[tuple[str, str]] = set()

    if preset is TopologyPreset.STAR:
        for agent_id in agent_ids[1:]:
            undirected_edges.add((root_agent, agent_id))
    elif preset is TopologyPreset.BALANCED_TREE:
        for child_index in range(1, agent_count):
            parent_index = (child_index - 1) // 2
            undirected_edges.add((agent_ids[parent_index], agent_ids[child_index]))
    elif preset is TopologyPreset.LINEAR_CHAIN:
        for idx in range(agent_count - 1):
            undirected_edges.add((agent_ids[idx], agent_ids[idx + 1]))
    elif preset is TopologyPreset.COMPLETE_GRAPH:
        for left_index in range(agent_count):
            for right_index in range(left_index + 1, agent_count):
                undirected_edges.add((agent_ids[left_index], agent_ids[right_index]))
    elif preset is TopologyPreset.SPARSE_GRAPH:
        rng = random.Random(seed)
        for idx in range(agent_count):
            undirected_edges.add(tuple(sorted((agent_ids[idx], agent_ids[(idx + 1) % agent_count]))))
        target_edges = max(agent_count + max(agent_count // 2, 1), agent_count)
        while len(undirected_edges) < target_edges:
            source = rng.choice(agent_ids)
            target = rng.choice(agent_ids)
            if source != target:
                undirected_edges.add(tuple(sorted((source, target))))
    else:
        raise ValueError(f"Unsupported topology preset: {preset}")

    edges = _make_bidirectional(undirected_edges)
    topology = Topology(preset=preset, agent_ids=agent_ids, edges=edges, root_agent=root_agent)
    topology.shortest_path_parents_to_root()
    return topology
