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
            if name not in names:
                self.nodes[name] = Node(name)

    def add_edge(self, n1, n2, w):
        for n1 in self.nodes and n2 in self.nodes:
            self.nodes[n1].add_edge(n2,w)
            self.nodes[n2].add_edge(n1,w)

    def get_names(self):
        return list(self.nodes.keys())

    def get_node(self, n):
        return (Node)

def shortest_path(self, n1, n2):
    if n1 not in self.nodes or n2 not in self.nodes:
        return -1

    # Initialize distances and paths
    distances = {node: float('inf') for node in self.nodes}
    distances[n1] = 0
    paths = {}

    # Iterate through nodes and update distances and paths
    for node in self.nodes:
        for neighbor in self.nodes[node].get_neighbors():
            tentative_distance = distances[node] + self.nodes[node].get_weight(neighbor)
            if tentative_distance < distances[neighbor]:
                distances[neighbor] = tentative_distance
                paths[neighbor] = node

    # Reconstruct the path
    path = [n2]
    current_node = n2
    while current_node != n1:
        current_node = paths[current_node]
        path.append(current_node)
    path.reverse()

    return distances[n2], path, distances