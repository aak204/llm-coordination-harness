from __future__ import annotations

from coord_harness.core.enums import TopologyPreset
from coord_harness.core.topology import build_topology


def test_star_topology_has_expected_root_degree() -> None:
    topology = build_topology(TopologyPreset.STAR, agent_count=5, seed=7)
    assert topology.root_agent == "agent_0"
    assert len(topology.outbound["agent_0"]) == 4
    assert len(topology.inbound["agent_0"]) == 4


def test_all_presets_are_connected_to_root() -> None:
    for preset in TopologyPreset:
        topology = build_topology(preset, agent_count=6, seed=11)
        parent_map = topology.shortest_path_parents_to_root()
        assert len(parent_map) == 6
