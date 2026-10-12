import pytest
from graph import Graph, Node, shortest_path


def test_add_names_deduplicates_and_returns_stored_node():
    graph = Graph()
    graph.add_names(["a", "b", "a"])
    assert graph.size() == 2
    assert isinstance(graph.get_node("a"), Node)
    assert graph.get_node("a") is graph.nodes["a"]


def test_shortest_path_relaxes_again_after_better_route():
    graph = Graph()
    graph.add_names(["a", "b", "c", "d"])
    for edge in [("a", "b", 10), ("a", "c", 1), ("c", "b", 1), ("b", "d", 1)]:
        graph.add_edge(*edge)
    distance, path, distances = graph.shortest_path("a", "d")
    assert (distance, path) == (3, ["a", "c", "b", "d"])
    assert distances["b"] == 2
    assert shortest_path(graph, "d", "a")[0] == 3


def test_missing_and_disconnected_endpoints():
    graph = Graph()
    graph.add_names(["a", "b"])
    assert graph.shortest_path("a", "b") == -1
    assert graph.shortest_path("missing", "b") == -1
    assert graph.shortest_path("a", "a")[:2] == (0, ["a"])


def test_invalid_edges_rejected_without_modification():
    graph = Graph()
    graph.add_names(["a", "b"])
    with pytest.raises(KeyError):
        graph.add_edge("a", "missing", 1)
    with pytest.raises(ValueError):
        graph.add_edge("a", "b", -1)
    assert graph.get_node("a").num_neighbors() == 0
