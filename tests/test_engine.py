"""Tests for the main Elenchus engine graph."""

from elenchus.engine import build_engine_graph


def test_engine_graph_builds():
    graph = build_engine_graph()
    assert graph is not None
