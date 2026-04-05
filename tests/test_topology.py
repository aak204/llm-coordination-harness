from __future__ import annotations

from coord_harness.core.enums import TopologyPreset
from coord_harness.core.topology import build_topology


def test_star_topology_has_expected_root_degree() -> None:
    topology = build_topology(TopologyPreset.STAR, agent_count=5, seed=7)
    assert topology.root_agent == "agent_0"
    assert len(topology.outbound["agent_0"]) == 4
    assert len(topology.inbound["agent_0"]) == 4


def test_linear_chain_has_expected_endpoints_and_root_path() -> None:
    topology = build_topology(TopologyPreset.LINEAR_CHAIN, agent_count=5, seed=7)
    parent_map = topology.shortest_path_parents_to_root()
    assert parent_map["agent_4"] == "agent_3"
    assert topology.outbound["agent_0"] == ["agent_1"]
    assert topology.inbound["agent_4"] == ["agent_3"]


def test_complete_graph_connects_every_pair() -> None:
    topology = build_topology(TopologyPreset.COMPLETE_GRAPH, agent_count=5, seed=7)
    assert len(topology.outbound["agent_0"]) == 4
    assert len(topology.inbound["agent_4"]) == 4
    assert topology.edge_count == 20


def test_all_presets_are_connected_to_root() -> None:
    for preset in TopologyPreset:
        topology = build_topology(preset, agent_count=6, seed=11)
        parent_map = topology.shortest_path_parents_to_root()
        assert len(parent_map) == 6
