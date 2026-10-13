class Node:
    def __init__(self, name):
        self.name = name
        self.edges = {}

    def get_weight(self, n):
        return self.edges.get(n, -1)
    def add_edge(self, n, w):
        self.edges[n] = w

    def num_neighbors(self):
        return len(self.edges)

    def get_neighbors(self):
        return list(self.edges.keys())

class Graph:
    def __init__(self):
        self.nodes = {}

    def size(self):
        return len(self.nodes)

    def add_names(self, names):
        for name in names:
            if name not in self.nodes:
                self.nodes[name] = Node(name)

    def add_edge(self, n1, n2, w):
        if n1 not in self.nodes or n2 not in self.nodes:
            raise KeyError("Both endpoints must be added before creating an edge")
        if w < 0:
            raise ValueError("Shortest paths require nonnegative edge weights")
        if n1 in self.nodes and n2 in self.nodes:
            self.nodes[n1].add_edge(n2,w)
            self.nodes[n2].add_edge(n1,w)

    def get_names(self):
        return list(self.nodes.keys())

    def get_node(self, n):
        return self.nodes[n]

    def shortest_path(self, n1, n2):
        return shortest_path(self, n1, n2)


def shortest_path(self, n1, n2):
    """Return distance, path and distances; return -1 for missing/unreachable endpoints."""
    from heapq import heappop, heappush
    from itertools import count

    if n1 not in self.nodes or n2 not in self.nodes:
        return -1
    distances = {node: float("inf") for node in self.nodes}
    distances[n1] = 0
    previous = {}
    tie_breaker = count()
    queue = [(0, next(tie_breaker), n1)]
    while queue:
        distance, _, node = heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, weight in self.nodes[node].edges.items():
            if weight < 0:
                raise ValueError("Shortest paths require nonnegative edge weights")
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heappush(queue, (candidate, next(tie_breaker), neighbor))
    if distances[n2] == float("inf"):
        return -1
    path = [n2]
    while path[-1] != n1:
        path.append(previous[path[-1]])
    path.reverse()
    return distances[n2], path, distances
