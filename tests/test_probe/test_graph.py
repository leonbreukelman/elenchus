"""Tests for the Deutsch Probe subgraph."""

from elenchus.probe.graph import build_probe_graph


def test_probe_graph_builds():
    graph = build_probe_graph()
    assert graph is not None
